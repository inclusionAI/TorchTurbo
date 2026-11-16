# 一、构建镜像CI

资源列表

| 资源类型 | runner名 | 是否可用 | 作用 |
|---|---|---|---|
| CPU |  ubuntu-22.04-arm | 是 |  构建镜像(pull基础镜像，push turbo镜像) |
| CPU |  linux-aarch64-cpu-x(4\24) | 否 | 构建镜像(pull基础镜像，push turbo镜像) |
| NPU |  linux-aarch64-a3-800i-x(2\4\6\8) | 是 | 测试用例 NPU计算 |


该部分使用CPU资源，分为三步：

1. pull cann基础镜像
2. build cann镜像为turbo镜像
3. push 镜像，以供后续PR CI使用

**pull image**

| 资源 | 阶段 | 网速 | 时间 |
|---|---|---|---|
|linux-aarch64-cpu-4 | FROM quay.io/ascend/cann:9.0.0-a3-openeuler24.03-py3.11|  <90Mb/s | 2min 12s |
| linux-aarch64-cpu-4| swr.cn-north-12.myhuaweicloud.com/ascend/torchturbo_npu:base-latest | <4Mb/s  | 14min |


**build image**

| 资源 | 阶段 | 耗时 | 占比 |
|---|---|---|---|
|linux-aarch64-cpu-4 | `#10` 编译 fla-npu（Ascend 算子） | **84.7 min** | 40.5% |
| linux-aarch64-cpu-4| `#12` 源码编译 PyTorch 2.10.0 | **68.1 min** | 32.5% |
| linux-aarch64-cpu-4| `#17` 导出并推送镜像到 SWR | **50.1 min** | 23.9% |
| linux-aarch64-cpu-4| 其余全部（拉基础镜像 + pip + 收尾） | 6.9 min | 3.1% |

**push image**

| 资源| 子阶段 | 耗时 |
|---|---|---|
| linux-aarch64-cpu-4| `exporting layers`（本地写盘） | 47.6 s |
| linux-aarch64-cpu-4 | **`pushing layers`（上传 SWR）** | **2951.6 s ≈ 49.2 min** |
| linux-aarch64-cpu-4 | push manifest（两个 tag） | 2.9 s + 0.5 s |


# 二、PR CI

该部分使用NPU pod，分为两步：

1. pull turbo镜像

2. 执行测试用例

**pull turbo镜像**

| 资源 | 阶段 | 网速 | 时间 |
|---|---|---|---|
|linux-aarch64-a3-800i-8 | FROM quay.io/ascend/cann:9.0.0-a3-openeuler24.03-py3.11|  | 42min |
| linux-aarch64-a3-800i-8| FROM 	swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-a3-openeuler24.03-py3.11| | 3min 12s  |

---

# 三、base 镜像构建

产物：`swr.cn-north-12.myhuaweicloud.com/ascend/torchturbo_npu:base-latest`（6.06 GiB）

## 3.1 构建包与引入包

| # | 构建包 | 引入包 | 风险 | 风险说明 |
|---|---|---|---|---|
| 1 | `FROM cann:9.0.0-a3` | CANN 9.0.0 toolkit、openEuler 24.03、Python 3.11 | 低 | tag 未锁 digest |
| 2 | `dnf install` ×15 | cmake、gcc、g++、make、git、curl、nfs-utils、numactl-devel、openblas-devel、patch、tmux、unzip | 低 | 无版本约束；与 #9 的 pip cmake 两套并存 |
| 4 | `triton==3.5.0` | — | 低 | 钉死 |
| 5 | `triton-ascend==3.2.2` | numpy 1.26.4、scipy 1.13.1、pandas 3.0.6、**pytest 8.3.2**、pytest-xdist、pybind11、attrs、decorator、psutil 等 | **中** | 传递依赖全浮动；覆盖 `triton/` 致元数据失真；是 numpy 1.26.4 的唯一钉子 |
| 6 | `torch_npu==2.10.0.post2` | **torch==2.10.0（PyPI 版）** | **低** |  |
| 7 | fla-npu 源码编译（**84.7 min**） | pyyaml 6.0.3、numpy、sympy、scipy、protobuf、expecttest、packaging | **低** | 已确认commit |
| 8 | `flash-linear-attention@954438d1` | **transformers 5.22.0**、einops 0.8.2、huggingface-hub、tokenizers 0.23.3、safetensors 0.8.0、regex、tqdm、rich、typer 等 20+ 包 | **高** | `transformers>=4.45.0` 无上界；依赖在 Dockerfile 中不可见；若要求 numpy>=2 会顶掉 1.26.4 |
| 9 | PyTorch v2.10.0 源码编译（**68.1 min**）<br>必需：aarch64 下 CMakeLists 用裸 `set(USE_MIMALLOC ON)` 覆盖 `-DUSE_MIMALLOC=OFF`，只能改源码；且 `torch_npu` 硬绑 `torch==2.10.0` | setuptools、cmake 4.4.4、ninja 1.13.2、requests、six、filelock、fsspec、jinja2、networkx | **低** | tag 固定 → `sed` 目标行确定，无上游漂移；残留风险仅「换 torch 版本时未同步保证 `USE_MIMALLOC=OFF`」 |
| 10 | `py-spy==0.4.2` | — | 低 | 钉死 |

## 3.2 PR CI 使用期间

| 步骤 | 包 | 风险 | 风险说明 |
|---|---|---|---|
| 拉镜像 | `base-latest` | **中** | 移动 tag，kubelet `IfNotPresent` 只比 tag 字符串 → **重建后不会自动更新**，需改 digest 或清缓存 |
| 装测试依赖 | `pytest>=7.0` | 低 | 镜像已有 8.3.2（来自 triton-ascend），实际不下载 |
| 装测试依赖 | `omegaconf>=2.3.0` | **中** | 镜像缺（唯一缺失的 pyproject 依赖）；**手工清单与 pyproject 会漂移** |
| 装测试依赖 | `safetensors>=0.4.3` | 低 | 镜像已有 0.8.0（来自 fla） |
| 装测试依赖 | `transformers`（默认不传） | **高** | 镜像已有 5.22.0（来自 fla）；默认留空即不动，传约束才升级，无上界 |
| 装本项目 | `pip install -e ".[dev]" --no-build-isolation` | 低 | 依赖由 pyproject 解析，不会漂移；实测只装 omegaconf 一个包 |
| 装本项目 | `--no-build-isolation` | 低 | 缺此参数会下载整份 torch（`build-system.requires` 里有 torch） |
| 运行测试 | torch 版本守卫 | 低 | `BEFORE/AFTER` 比对；实测 `2.10.0+cpu.nomimalloc >= 2.10.0` 成立，守卫不会触发 |

## 3.3 最需关注

| 项 | 问题 | 动作 |
|---|---|---|
| transformers 隐式引入 | 由 fla 的 `>=4.45.0` 决定，Dockerfile 里看不见 | 改显式 `pip install transformers==<ver> einops==<ver>` + `pip install --no-deps /tmp/flash-linear-attention` |
| numpy 版本 | 镜像 1.26.4 仅由 triton-ascend 传递钉住，Dockerfile / pyproject 均未声明 `numpy<2`；fla 或 transformers 任一要求 `>=2` 就会顶掉，NPU 算子栈在 kernel 加载期才报错 | Dockerfile 显式加 `pip install "numpy<2"`；PR CI 侧保持 `numpy<2` 最后安装 |

## 3.4 PyTorch 与 numpy 的版本必须一致

报错日志
~~~
File "torchturbo/engine/checkpoint/coordinator.py", line 258, in _weight_converter
    converter = collective_check(...)
File "torchturbo/engine/checkpoint/distributed.py", line 44, in collective_check
    dist.all_gather_object(errors, error)
File "torch/distributed/distributed_c10d.py", line 3172, in _tensor_to_object
    buf = tensor.numpy().tobytes()[:tensor_size]
RuntimeError: Numpy is not available
~~~

torch 的 C 扩展在**编译期**绑定 numpy ABI，运行期换大版本即报错。旧顺序里 numpy 1.26.4
由 `triton-ascend` 传递带出、且发生在 torch 编译之前；训练侧 `lf_deps.sh` 结尾的
`pip install numpy==2.0` 又把它升到 2.0.0 → SFT 启动即失败。

改法（`eaadf67`）：torch 编译块上移到所有 pip 装包之前，直接用 CANN 镜像自带 numpy；
`triton-ascend` 加 `--no-deps`，防止它把 numpy 拉回 1.26.4。`verify_env.py` 里
「numpy 必须是 1.x」的旧断言已删除，它与新契约正好相反，否则每次 CI 都会误报警告。

残留：`jinja2` / `networkx` 原本由 PyPI 版 torch 带来（`torch_npu` 先装时 pip 解析
`torch==2.10.0` 顺带装的），改序后无人安装。两者只在 `torch.compile` 与 `torch.fx`
时才 import，eager 训练和 smoke 都测不出来。**本次保留 `--no-deps` 原样先构一版**；
修复待确认后再上：去掉它让 wheel 元数据自行解析（6 个依赖都不含 numpy，不影响编译期
版本），并加 `python -c "import jinja2, networkx"` 断言。

---

# 四、遗留问题：`cann_ops_transformer` 无构建来源

**问题**：`lightning_indexer.py` / `sparse_flash_mla.py` 直接 `from cann_ops_transformer import ops`，
但本仓库 Dockerfile、pyproject、requirements、md 和脚本里**都没有它的构建或安装说明**。
L1 镜像不提供，它只存在于**客户现场打包的镜像**里，来源一旦丢失就装不回来。

**实测**（2026-11-02，A3 服务器）：

```
$ import cann_ops_transformer   → ModuleNotFoundError
$ pip list | grep ops           → npu_ops_transformer 1.0.0
```

`npu_ops_transformer` **不是它的别名**，是 CANN 9.0.0 自带的 MoE 通信包
（只有 `moe_distribute_*`、`update_context`、`deep_ep`），没有这两个算子。
也不要 `import` 它：`__init__.py` 会现场 JIT 编译 5 个 `.so`，耗时且易失败。

**后果**：两个 provider 的 `available()` 都写成 `except ImportError: return False`，异常被吞掉；
`tests/ops/npu/` 又不覆盖这两个算子，所以 smoke **17 passed 全绿**，算子却从未被选中。

**待办**：需要时在 L2 镜像里用 vllm-ascend `csrc/build_aclnn.sh` 产出 `cann-ops-transformer*.run` 安装。
当前维持现状，`verify_env.py` 每次打印 `cann_ops_transformer: not installed` 作提醒，不让 job 失败。

---

# 五、遗留问题：`jinja2` / `networkx` 缺失

**问题**：`8cdcb33` 把 torch 编译块上移到 `torch_npu` 之前后，这两个包没有任何地方安装。

**成因**：torch 的 wheel 声明了 `install_requires`（`setup.py:1696`）：
`filelock`、`typing-extensions`、`sympy`、`networkx`、`jinja2`、`fsspec`。
旧顺序里 `torch_npu` 先装、pip 需解析 `torch==2.10.0`，会从 PyPI 拉一份 torch 并
顺带装齐这些依赖；随后源码 torch 用 `--no-deps` 只覆盖文件，依赖留在原地。
改序后 `2.10.0+cpu.nomimalloc` 已满足该约束，pip 判定 already satisfied，**一跳都不装**。

**缺口只有 2 个**：`filelock` / `fsspec` 由 `huggingface-hub` 提供，`sympy` 由 fla-npu
的 `requirements.txt` 提供，`typing-extensions` 由 `requirements-build.txt` 提供。
`jinja2` 只出现在 `huggingface-hub` 的 `extra` 里（默认不装），`networkx` 无人提供。

**影响**：`jinja2` 是 `torch._inductor` 的代码生成模板引擎，`networkx` 供 `torch.fx`
passes / `torch.export`。**两者只在 `torch.compile` / `torch.fx` 时 import**，eager
算子测试与 SFT 都走不到，所以 smoke 全绿也发现不了。

**待办**：去掉 torch wheel 安装的 `--no-deps`，让 pip 按元数据自行解析（这 6 个依赖
都不含 numpy，不影响编译期 numpy 版本），并加一行
`python -c "import jinja2, networkx"` 做构建期断言。
代价：该行位于第 4 层，改动会让其后全部缓存失效，即一次完整重建（约 160 min）。

**修复**：已落地。torch wheel 安装去掉 `--no-deps`，改为按元数据自行解析——
这 6 个依赖都不含 numpy，不影响编译期 numpy 版本。

---

# 六、llamafactory 环境构建耗时基线

复核结论：运行时安装 lf 依赖只需 **360 s**，不值得为它单独做 L2 镜像。

| 阶段 | 耗时 | 说明 |
|---|---|---|
| 拉 base 镜像（冷节点） | SWR 3min12s / quay 42min | 见第二章 |
| ③ `pip install -e ".[dev]" --no-build-isolation` | ~60 s | 只装 omegaconf + pytest |
| ④ `scripts/runs/<model>/install_lf_deps.sh` | **360 s** | 15 个包 |
| 合计（不含拉镜像） | **~7 min** | |

**为什么不做 L2（ci-lf / ci-swift）**：

- 净收益仅 4~4.5 min。冷节点上 L2 多 2.5 GB，按 SWR 4 MiB/s 要多拉 1.5~2 min，抵掉一部分。
- 一次 L2 构建 + 推送要 16~18 min（base 6 min + 推送 10 min），需运行 4 次以上才回本。
- 调试期 base 还在改，`FROM` 一变 L2 整层失效，等于每次 base 调试多付一次 L2 成本。
- per-model 增量（`tiktoken` / `torchvision==0.25.0` / `qwen-vl-utils==0.0.14`）只有 1~2 个包、10~30 s，留在运行时装。
- lf 与 swift 的 `transformers` / `trl` / `datasets` / `peft` 版本互斥，本来也得拆两个镜像。

**重启条件**：base 连续 2~3 次构建无改动 + CI 进入 PR 门禁（每天 ≥5 次运行）+ 有固定
NPU 节点（镜像命中率 >80%）。届时做 2 个框架镜像，而非每模型一个。

**注意**：L2 唯一不可替代的价值是"版本冲突在构建期暴露"（如 `ms_swift 4.4.2` 要求
`datasets<4.8.5`，而 `swift_deps.sh` 装 `5.0.1`，pip 只警告不失败）。但这可以在运行期
用 `python -m pip check` 拿到，不需要为此建镜像。

---

# 七、事故：`rm -rf ~/.cache` 在 SFS 上不可行

`bailing_v3/install_{lf,swift}_deps.sh` 末行 `rm -rf ~/.cache/ ~/.triton/`，而
`/root/.cache` 在 NPU runner 上就是 SFS Turbo 的 NFS 挂载点。`rm -rf` 先删光内容、
最后再 `rmdir` 挂载点，后者必然 `EBUSY`：实测删了 7 min 24 s 后失败，脚本退出 1，
④ 挂掉、⑤ 训练一次都没跑。更严重的是它真删掉了共享盘上并发 job 共用的数据，
含 `models/Qwen/Qwen3.5-2B`（4.5 GB）和 `gsm8k`。

改法：

1. 两个脚本删掉该行（`~/.triton` 不在挂载点、镜像未预置、装依赖时还不存在，同样无意义）。
2. workflow 加 `HF_MODULES_CACHE: '/tmp/hf-modules'`，把 `trust_remote_code` 的代码副本
   挪到 pod 本地。共享 SFS 上多 pod 并发拷贝会留下半截 `.py`，而 transformers 只按
   `exists()` 判缓存，坏文件将永久保留。

效果：安装脚本正常退出、⑤ 能跑到训练；其余 6 个脚本的 `rm` 目标目录不再被创建，
自动变成空操作。
