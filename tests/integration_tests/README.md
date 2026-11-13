# 集成测试：端到端正确性验证
增加集成测试功能，为 CI/CD 提供端到端正确性验证：支持通过在 CPU 侧构建小模型快速验证训练流程，也支持在 NPU 侧真实环境验证真实分布式训练场景，并通过与基准Golden Value的比较检查数值回归。
## 关键文件结构
```text
tests/integration_tests/
├── run_tests.sh                 # Shell 入口
├── test_suite/
│   └── base_suite.py            # 用例集合recipe列表
├── scripts/
│   ├── run_tests.py             # 读取配置、准备模型、启动训练
│   ├── gen_init_checkpoint.py   # 在 CPU 上生成随机初始化的 checkpoint
│   └── compare_metrics.py       # 提取指标、比较或更新 golden
├── assets/
│   ├── tokenizer/              # 供小模型使用的 tokenizer 和 processor 等配置
│   └── sft_test/               # 供小模型使用的数据集
├── test_cases/<模型>/<用例>/
│   ├── config.yaml             # 用例的训练配置
│   └── golden_values_<平台>.json
```
启动流程：`run_tests.sh` 会启动 `run_tests.py`，读取 `test_suite` 中指定 suite 的用例列表，再加载 `test_cases` 中对应的 YAML。未提供模型路径时，调用 `gen_init_checkpoint.py` 在 CPU 上生成小模型权重；随后通过 torchrun 启动训练框架。训练成功后，`compare_metrics.py` 从本次 TensorBoard 日志提取逐步的 `loss`、`grad_norm`，与 golden 比较或按配置更新基准。
## 数值验证
### Golden Value
Golden Value 是一次经过确认的成功训练所记录的参考值。`compare_metrics.py` 从 TensorBoard 提取各训练 step 的 `loss`、`grad_norm` 并保存为 JSON。
Golden Value 与测试用例 yaml 放在同一目录，分别命名为 `golden_values_cpu.json`、`golden_values_npu.json`。
### Golden Value 配置开关
| 开关 | 行为 |
| --- | --- |
| `UPDATE_GOLDEN: false`，suite 中 `compare=True` | 比较现有基准，任一步超差即失败 |
| `UPDATE_GOLDEN: true` | 将本次指标写为新基准，不执行旧基准回归比较 |
| `UPDATE_GOLDEN: false`，`compare=False` | 只运行训练，跳过指标比较 |
建立基准后，将 YAML 的 `UPDATE_GOLDEN` 恢复为 `false`，并保持 suite 的 `compare=True`。
### atol / rtol
两个指标用于 Golden Value 的数值对比，每一步分别满足以下条件才通过：
```text
abs(actual - golden) <= atol + rtol * abs(golden)
```
- `atol`：绝对容差，允许固定大小的误差，主要用于基准接近零的情况。
- `rtol`：相对容差；`0.001` 表示 0.1%，`0.1` 表示 10%。
例如 `golden=2`、`atol=1e-5`、`rtol=0.1`，允许的绝对误差为 `0.20001`。容差配置位于 `test_suite/base_suite.py`。
CPU 要求逐步数值完全一致；NPU 使用近似比较。
## 模型生成
对新功能或 PR 进行快速端到端验证时，可以使用随机初始化的小模型：当 YAML 未提供模型路径，并配置了 `MODEL_ARGS` 时，`gen_init_checkpoint.py` 会从 TorchTurbo 注册表取得对应模型类，按指定结构在 CPU 上构建并保存权重。初始化使用 `TRAIN_ARGS.seed`。模型配置、seed 和依赖环境固定后，可重复生成完全一致的初始权重。
下面是构建小模型的核心函数：
```python
def build_cpu_model(model_cls: type, config: Any, engine_config: Any, seed: int) -> Any:
    import torch

    torch.manual_seed(seed)
    with (
        model_cls.init_context(config, engine_config=engine_config),
        torch.device("cpu"),
    ):
        model = model_cls._from_config(config)
    return model
```
这里复用模型自身的初始化流程，以 float32 精度在 CPU 上建立真实权重，不做分布式切分。
## test_cases
### 用例目录与 YAML
用例按 `<模型>/<场景>` 分类，例如 `qwen3_5_moe/sft_2npu/`。每个目录包含一份 YAML 和对应平台的 golden。
```yaml
UPDATE_GOLDEN: false

MODEL_ARGS:
  model_type: qwen3_5_moe
  text_config:
    hidden_size: 128
    num_hidden_layers: 2
    layer_types:
      - linear_attention
      - full_attention
    head_dim: 32
    ...
TRAIN_ARGS:
  model: null
  template: qwen3_5_nothink
  finetuning_type: full
  flash_attn: fa2
  max_steps: 4
  ...
ENGINE_CONFIG:
  backend: fsdp2
  mixed_precision: bf16
  parallel_plan:
    sp_size: 1
    ep_size: 1
  ...
LAUNCH_ARGS:
  nproc_per_node: 2
  nnodes: 1
  node_rank: 0
  master_addr: 127.0.0.1
  master_port: 29500
ENV_VARS:
  TOKENIZERS_PARALLELISM: 'false'
  HF_HUB_OFFLINE: '1'
  HF_DATASETS_OFFLINE: '1'
  HCCL_CONNECT_TIMEOUT: '7200'
  HCCL_EXEC_TIMEOUT: '7200'
  ...
```
具体配置项作用为：

| YAML 字段 | 作用 |
| --- | --- |
| `MODEL_ARGS` | 随机建模的结构参数；加载现有模型时保留为 `{}` |
| `TRAIN_ARGS` | 传给训练框架的参数：模型、tokenizer、步数、优化器等 |
| `ENGINE_CONFIG` | TorchTurbo 配置 yaml：SP/EP、激活重算、分块 loss 等 |
| `LAUNCH_ARGS` | torchrun 参数：节点数、每节点进程数、节点 rank 和通信端口 |
| `ENV_VARS` | 该用例使用的环境变量 |
| `UPDATE_GOLDEN` | 是否用本次成功运行更新 Golden Value |

### base_suite.py
`test_suite` 中每个 Python 文件定义一组测试用例，即一个 suite。可按 CI/CD 的 nightly、weekly 任务，或按平台、模型组织不同的 suite，通过 `--suite base_suite` 选择当前基础用例集合。
以`qwen3_5_moe/sft_2npu`为例：
```python
TestCaseSpec(
    test_name="qwen3_5_moe/sft_2npu",
    test_descr="Qwen3.5-MoE VL-template SFT on 2 NPUs",
    framework="llamafactory",
    mode="train",
    test_type="regular",
    compare=True,
    metrics={
        "loss": {"atol": 1.0e-5, "rtol": 1.0e-3},
        "grad_norm": {"atol": 1.0e-5, "rtol": 1.0e-3},
    },
)
```
| Spec 字段 | 作用 |
| --- | --- |
| `test_name` / `test_descr` | 用例目录及运行时说明 |
| `framework` / `mode` | 训练框架及执行模式；当前 mode 只支持 `train` |
| `test_type` | 初始模型或训练状态从哪里来，见下节 |
| `platform` / `ngpu` | 所需平台和设备数；CPU 用例的 ngpu 表示 rank 数 |
| `compare` / `metrics` | 是否比较指标，以及每项指标的 atol、rtol |
新增用例时，创建目录及 YAML，再加入对应的 suite
### 现有三个 case

| 用例 | 初始模型与数据 | 训练设置 |
| --- | --- | --- |
| `qwen3_5_moe/sft_2cpu` | 自己构建的 Qwen 小模型；用 `assets` 里面的简单数据集 | Swift，2 CPU ranks，fp32，15 步 |
| `qwen3_5_moe/sft_2npu` | 自己构建的 Qwen 小模型；用 `assets` 里面的简单数据集 | LF，2 NPUs，bf16，4 步 |
| `bailing_v3/sft_16npu` | 真实 Bailing tiny 模型；真实 science 数据集 | Swift，单机 16 NPUs，bf16，4 步 |
## 测试模式
`test_type` 主要用于配置测试方式，具体参考 Megatron-LM 里面的 functional tests 功能：
| 类型 | 验证内容 |
| --- | --- |
| `regular` | 从随机初始化模型开始训练，与 golden 比较；当前用例不设置模型路径 |
| `frozen-start` | 从指定 checkpoint 的权重开始一次新训练，与 golden 比较 |
| `ckpt-resume` | 保存当前运行的中间 checkpoint，恢复后与连续训练的后半段对齐 |
| `frozen-resume` | 从固定的历史 checkpoint 恢复，用于检查跨版本恢复行为 |
| `checkpoint-consistency` | 对比不同并行配置下保存的 checkpoint 运行数值，检查一致性 |
| `release` | 运行较长训练，检查收敛趋势及长期稳定性 |
**当前只支持 `regular`、`frozen-start`。** 其余类型待后续开发。
## 运行测试
以下命令均在 **TorchTurbo 根目录**执行。
| 参数 | 含义 |
| --- | --- |
| `--suite` | suite 模块名，当前为 `base_suite` |
| `--platform` | 选择 `cpu` 或 `npu` 用例 |
| `--ngpu` | 可用设备数，CPU 时表示可用 rank 数 |
| `--test_name` | 精确用例名；默认 `all`，运行该平台下设备数满足要求的用例 |
实际进程数取自 YAML 的 `LAUNCH_ARGS.nproc_per_node`；仅改 `--ngpu` 不会改变训练规模。当前串行执行用例，遇到失败即停止。
### CPU
需要自行准备 cpu 侧运行环境：
```bash
bash tests/integration_tests/run_tests.sh --suite base_suite --platform cpu --ngpu 2 --test_name qwen3_5_moe/sft_2cpu
```
### NPU
脚本会自行调用仓库里的 `internal/scripts/runs/<模型>/install_*_deps.sh` 安装依赖。
Bailing 单机 16 卡：
```bash
bash tests/integration_tests/run_tests.sh --suite base_suite --platform npu --ngpu 16 --test_name bailing_v3/sft_16npu
```
Golden Value 比较通过后会打印输出 `Regression passed` 来说明数值对比通过。
## 现开发情况

### CPU 支持与初始化一致性

本分支为支持 CPU 执行和权重生成对少量源码进行了临时性修改：

- CPU 使用 eager attention；NPU/CUDA 保持 FlashAttention 路径。
- 为 `causal_conv1d` 和 gated delta rule 提供 CPU 原生实现。
- 显式将 Bailing KDA 的 `dt_bias` 初始化为零，消除未初始化权重导致小模型生成时初始权重差异。

### 模式支持与后续扩展

当前已跑通随机小模型和指定模型的 SFT 数值回归。后续可增加：

- `ckpt-resume` / `frozen-resume`：验证模型、optimizer、scheduler 和训练步数的恢复。
- `checkpoint-consistency`：验证不同并行规模或保存格式之间的状态一致性。
- 以及其他需要增加的功能。