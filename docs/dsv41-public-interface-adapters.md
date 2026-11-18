# DeepSeek V4.1：保留的公开框架接口适配

对照基准是内部 `dsv41_sft_rl` 的
`e49ff83aa5667099a79e9f5aa5f6be8db958bfc9`，公开框架基线是 `8a29e23`。
下列路径相对于仓库根目录；模型文件简称指
`src/torchturbo/models/deepseek_v41/`，引擎文件简称指 `src/torchturbo/engine/`。

“需要适配”表示两套接口不能直接互换，不表示当前桥接写法是唯一实现。
尤其是私有依赖副本和基类选择，属于兼容组织方式，不能描述为内部原文件逐字相同。
本清单把这些差异、同语义移植接线、非必要的兼容扩展和本次明确修复的行为差异分开说明。

## 实际接口差异

| 编号 | 位置 | 内部版本与当前版本的具体差异 | 为什么保留 | 对架构、语义的影响 |
| --- | --- | --- | --- | --- |
| 1 | `model.py` 的基类、import | 内部直接继承内部 `TurboVLMModel`，并从 base 导入 `ParameterGroup` 和旧 SP 枚举；当前以公开 `TurboModel` 作基类，`ParameterGroup` 来自 engine.config。 | 两版 VLM 的输入准备、SP 边界约定不同；当前 wrapper 自己接入 V4.1 的输入 hook。基类改法是兼容选择，并非公开版没有 VLM 基类。 | 不改变 V4.1 子模块、参数名或模型 forward；不继承公开 VLM 的通用媒体输入恢复行为。当前仍是内部声明的 text-only 训练范围，不能据此声称完整视觉训练已验证。 |
| 2 | `model.py:init_context` | 内部 VLM 基类自动对 text/vision 子配置临时设置 FP32；当前 wrapper 显式包裹公开基类上下文，设置并恢复两份子配置 dtype。 | 基类选择后需要补回内部的嵌套配置初始化行为。 | 保持内部 FP32 主参数初始化，不改变计算 dtype 策略。 |
| 3 | `model.py:get_sequence_parallel_input_slicing_spec` | 内部使用 `SequenceParallelInputMode.PRE_EMBEDDING`；公开框架使用 `SequenceParallelInputSlicingSpec`，显式登记 inputs_embeds 的序列维和 attention_mask 的补零规则。 | 公开 backend 消费的是 slicing spec，不消费旧枚举。 | 保留 embedding 前切分及全局 segment mask。SP/packing 的数值等价仍需 NPU 验证。 |
| 4 | `model.py:on_prepare_inputs` | 内部模型直接消费带 segment ID 的全局 attention_mask 和段内 position_ids；公开入口传 cu_seq_lens_q/k。当前从 cu 边界重建 segment ID，并在未提供位置时重建段内位置。 | 元数据表示不同，不能把累计长度当作模型 mask。 | 改变输入表示，不改变段间隔离、段内位置或 packing 算法。保留调用方已有 position_ids。 |
| 5 | `model.py:get_moe_runtime_inputs` | 内部运行时从 2D mask 提供 padding；公开基类默认面向 cu 边界。当前 cu 路径交给公开基类，mask 路径取零值为 padding，并按 SP 切成局部 mask。 | 保证旧 mask 调用及公开 packed 调用都把同一批无效 token 告诉 router。 | 不改路由分数或 token 所有权；影响的是 padding 元数据接线。 |
| 6 | `moe_bridge.py:SourceRouter` | 内部返回 `(probs, routing_map, logits, top_indices)`；公开 MoELayer 要求 indices 在 logits 前，还传 accumulate_tokens_per_expert 参数。桥仅重排返回项并接受额外参数。 | tuple 顺序和调用签名不一致。 | 内部 router 计算体未变。额外 accumulate 参数被忽略，保留内部的统计/重计算行为；不宣称继承了公开 router 的全部统计优化。 |
| 7 | `moe_bridge.py:SourceTokenDispatcher` | 内部调用完整 token_permutation/token_unpermutation；公开框架分为 dispatch/combine 六步。首个 dispatch 执行完整内部 permutation，中间步骤传递结果，最终 combine 调用内部 unpermutation。 | 不能把公开版另一套分步通信算法当作内部算法替代。 | 保留内部 EP all-to-all、顺序、计数、概率 dtype 及 deferred weighting；空中间步骤不会追加通信。仍是原始 dispatcher 架构。 |
| 8 | `experts.py`、公开 `models/modules/moe/moe_layer.py` | 内部专家计算不在 activation 内乘 probs；公开 GroupedMLP 默认在第二个 GEMM 前乘。V4.1 子类保留公开参数构造接口，使用内部完整 forward，并声明 apply_probs_after_experts；MoELayer 在原始输出边界或 deferred unpermutation 处乘权重。 | 直接使用公开专家会改变 BF16 舍入和加权位置；同时不能改变其他公开模型默认行为。 | V4.1 恢复内部加权语义，参数形状和专家归属不变。公开构造器与内部 activation 的函数签名不同，子类做 unary/binary 调用桥接；并非整个专家类逐字相同。 |
| 9 | `parallel_plan.py` 的 MoE 构造 | 内部直接构造内部 MoELayer/GroupedMLP；当前公开 MoELayer 构造后换入 source router/dispatcher，并显式设置 layer_number、moe_permute_fusion=False。get_base_moe_config 另显式传 moe_router_pre_softmax=True，内部 helper 原本默认设 True，公开 helper 没有该默认。 | 消除构造接口与 helper 默认值差异。 | 保持内部路由、专家参数层级和配置值；不是新增路由方案。 |
| 10 | `moe_source_router.py` 的 loss scaler import | 内部 router 使用原有 MoEAuxLossAutoScaler；当前原始辅助损失函数保留在 V4.1 副本，scaler 引用公开引擎实际更新的同一类。两版 scaler 实现已对照一致。 | 若复制出独立 scaler 类，公开引擎设置的 GA loss scale 不会到达 router。 | 只统一类状态身份，保留内部损失公式及缩放行为。 |
| 11 | `parallel/fsdp.py`、`backends/fsdp2/backend.py` | 内部 FSDP plan/backend 已有 cast_forward_inputs；公开基线缺此字段。当前补齐 plan -> unit -> MixedPrecisionPolicy 传递，包括自动 ModuleList 单元。 | decoder 输入中的跨层 pre_mix 必须保留 FP32，不能被统一转换成 BF16。 | 恢复内部精度语义，不改变 FSDP 分片和 AC 边界。 |
| 12 | `engine/core.py`、backend finalize | 内部先 on_materialized，再 finalize_model；当前公开入口原先 finalize_model_load 在前，现改为同样在 materialization 后挂梯度同步 hook。 | Engram wkv/q/k 和 host 参数可能替换 meta Parameter，早挂 hook 会留在旧对象上。 | 保持内部参数生命周期及 dense no_shard AVG；host 表仍排除通用 dense hook，由原始 EP/EDP 路径处理。 |
| 13 | `engine/optimizers/plan.py`、backend optimizer hooks | 内部在 backend 准备参数组时剔除 host 表并构造 EngramOptimizer；公开基线另有 OptimizerPlan 的参数覆盖检查/分组。当前在这些入口排除 `_engram_host_offload`，并在最终 requires_grad 状态下筛表，接入原始 step/clip hooks。 | 表不能被 dense AdamW 更新，也不能被公开覆盖检查误报为遗漏。 | dense/sparse 所有权、EDP 平均、L2 norm 合并、LR/betas/eps 来源不变；ActiveRowAdam/EngramOptimizer 源码未改。 |
| 14 | `backends/fsdp2/checkpoint/manager.py` | 内部 backend 直接持有 DCP、HF saver 和 EngramOptimizer；公开版由 CheckpointCoordinator/manager 分层。manager 通过 callback 获取稍后才创建的 optimizer，把 optimizer_distcp 的父目录传给原始 Engram save/load。 | manager 创建早于 optimizer，且接收的是组件目录，不能把 sparse 文件误写到 optimizer_distcp 内。 | 层次接线改变，文件仍位于 checkpoint 根的 engram_optimizer_ep<N>.pt。两版原本都使用 huggingface、model_distcp、optimizer_distcp；目录名不是本次新差异。 |
| 15 | `backends/fsdp2/checkpoint/dcp.py` | 内部 DCP 方法接收 checkpoint 根目录和 parallel_context；公开 DcpCheckpoint 接收组件目录，EP 信息保存在实例。当前在公开 state/metadata 流程中纳入内部 trainable-host EP suffix 和 engram_layout.json。 | 公开 checkpoint API 与内部不相同，必须将相同的 shard 身份落到正确目录与 state key。 | 保留 EP geometry、训练模式和 hash 约束。公开版原有 extra_state、缺失键检查、CPUResident 分支仍保留；Engram 是普通 CPU Parameter，不走 NodeSharedCPUResident 的完整共享表路径。整套 DCP 类不是内部原类逐字副本。 |
| 16 | `checkpoint/weight_saver.py`、manager 的 HF reload | 内部 weight_saver 位于 fsdp2 根，公开版位于 checkpoint 子包并保留公开 generic exporter。当前加入原始 `_save_host_engram_tables`，host 表从通用 gather 排除；HF reload 在通用 loader 后显式调用内部 load_rows/load_params。 | 巨表必须按 EP owner 流式读写，不能通过公开 generic loader/gather 整表物化；内部专用行 I/O 本身未改。 | 保持 BF16、逻辑行数、EP owner 和 sibling 参数加载语义。公开通用 HF shard 分配/metadata 发布流程保留，普通 shard 文件编号不要求逐字等同内部；不能宣称导出目录每个字节相同。 |
| 17 | `ops/dispatcher.py:sparse_flash_mla` | 内部 facade 总是透传 custom_package/max_seqlens；公开基线 facade 无此参数。当前增加参数，仅在 True/非 None 时传递，避免给旧 provider 添加不认识的默认关键字。 | V4.1 的 TND provider 需要这些参数，同时公开 dispatcher 还服务既有 provider。 | 非默认 V4.1 调用收到相同参数；省略默认值等于 provider 原默认值，不改 MLA 算法。 |
| 18 | `ops/providers/npu/gated_activation.py` | 内部 provider 没有 clamp_gate_after_activation 关键字；公开 dispatcher/专家构造器传该关键字。当前接受 False；True 分支使用公开 native 的 clamp-after 合约。 | 不接关键字会 TypeError，忽略 True 又会破坏其他公开模型语义。 | V4.1 配置为 False，保留内部 clipped SwiGLU 路径。True 分支属于其他公开模型兼容行为，不是内部 V4.1 新功能。 |

## 为保持内部实现而隔离的依赖

这些是文件组织上的选择，不是模型算法替换。若全局覆盖公开公共模块，也能恢复内部算法，但会同时改变其他公开模型；当前选择把调用指向内部依赖副本。

| 编号 | 位置 | 内部与公开依赖的具体差异 | 为什么隔离 | 架构、语义 |
| --- | --- | --- | --- | --- |
| 19 | `source_packing_utils.py`、model/npu/native compressor import | 内部 V4 packing 生成动态完整块；公开 V4 packing 使用不同的静态容量实现。native compressor 的内部局部 import 也重定向。 | 直接复用公开同名文件会改变 V4.1 的块形状和 mask 行为。 | 原始内部代码仅换 import 地址，V4.1 packing 语义不变。 |
| 20 | `source_hyper_connections.py` | 内部 HcPost 使用 FP32 bmm，公开 V4 版本有不同的 elementwise reduction/dtype 行为。抽取内部 HC split 和 HcPost 两个类。 | 不能因名称相同就复用不同计算体。 | 保留原始类计算定义及梯度路径，组织位置改变。 |
| 21 | `source_functools.py` | 内部 cache key 检查 Tensor mutation `_version`，公开 cache 缺该检查。 | 原地更新 mask/position 后需要使内部缓存失效。 | 原始 cache 实现不变，不增加新的缓存规则。 |
| 22 | `moe_source_utils.py`、`moe_source_moe_utils.py`、router、dispatcher | 公开 MoE helper、空样本辅助损失处理及 dispatcher API 与内部不同；当前保留内部完整计算/通信体，仅换 import 并增加第 6、7、10 项桥接。 | 防止迁移时静默落入公开另一套 top-k、通信或空样本数学。 | 内部原语保留，包括内部已有的边界行为；没有用新通信方案替换。 |

## 同语义接线与非必要扩展

1. **配置和启动接线**：`mega_moe`、`extra['mega_moe_local_tokens']`、train_engram/KL/NPU fraction 的 runtime CLI、两种 AC offload 约束、adapters 中的 token capacity、启动脚本 bootstrap，均沿用内部名字和值。公开 config builder/backend/adapter 调用点不同，补的是注册/消费位置。MegaMoE 安装仍在 PEFT 后、FSDP 前；算法目录、bootstrap 和 NPU memory fraction 解析/应用本身不是重新设计的适配算法。
2. **公共 router 扩展已清理**：第 4 个 commit 中，公开 `models/modules/moe/router.py`、`utils.py` 已逐字恢复为公开基线 `8a29e23`，移除公共计算路径中的温度除法、额外归一化参数和分支。`MoEConfig.moe_router_temperature`、`moe_router_norm_topk_prob` 保留，仅由 V4.1 source router/helpers 使用。新增隔离用例验证 source 路径的非默认温度和归一化仍有效，同时公共路径保持原始 softmax/sigmoid 行为；其他模型的既有路由/梯度用例亦验证通过。
3. **测试注入**：CPU 用例仅在测试里关闭 pinned allocation，以绕过本机不可用的 MPS pinned allocator。生产 pinning、EP 表分配及 fallback 完全保留内部源码。这不是生产接口适配。

## 本次明确的行为修复：冻结优先

内部 `on_materialized()` 原来无条件执行
`weight.requires_grad_(train_engram)`。这会覆盖 PEFT/调用方在 meta 参数上设置的冻结，
也会在 freeze_policy 生效后重复调用 hook 时重新打开表梯度。

第 6 个 commit 中改为：

```python
weight.requires_grad_(train_engram and weight.requires_grad)
```

这项是用户明确要求的行为修复，不标为“与内部行为完全相同”。
训练表需要同时满足运行时开关和当前参数允许训练；显式冻结优先。
未冻结且 train_engram=True 的正常训练行为不变；train_engram=False 仍关闭训练。
不会修改 EP/EDP 拓扑、all-to-all autograd、pending gradients 结构、稀疏更新、
优化器文件或 BF16 表布局。该 hook 不负责运行中创建/重配 optimizer，也不新增自动解冻能力。

新回归先在旧代码上复现两项失败，再验证 materialize 前冻结、freeze_policy 后 hook replay、
真实 PEFT LoRA、默认开关、checkpoint 续训及 EP 稀疏更新。
NPU 下融合数值、SP/EP、真实大表内存和性能仍待上机；接口桥接的 CPU 通过不等于这些已经验证。
