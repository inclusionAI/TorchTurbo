# fla-npu 新 wheel（v26.9.1）引入 e2e 测试报告

测试目的：验证新 wheel 下 KDA 算子的端到端收益（AscendC vs Triton），以及 GDN 路径 wheel 升级（26.6.0 → 26.9.1）的精度回归与性能变化。

## 一、KDA 测试（bailing v3flash）

### 测试条件

| 项 | 值 |
|---|---|
| 模型 | Bailing-v3flash |
| 硬件 | Ascend 910C，4 节点 × 16 NPU |
| 环境 | torchturbo 训练镜像；AscendC = fla-npu-a3 26.9.1+dev36e67d2，Triton 基线 = fla 0.6.0 |
| 配置 | SEQ_LEN 32768 varlen packed、sp4 / ep8，64 rank、bf16、bs 1、GA 1、lr 1.25e-5、seed 42、数据集 science |

### 结果

| backend | 首步 loss | 稳态 sec/step | 统计窗口 | 加速比 |
|---|---|---|---|---|
| AscendC (v26.9.1) | 0.5235 | **27.7** | step 191-335（n=144，平台期延续至 436 步） | **1.27x** |
| Triton (fla-org) | 0.5235 | 35.3 | step 50-335（全程平稳） | baseline |

精度：首步 loss 逐位一致；215 步轨迹平均相对偏差 0.01%~0.04%（bf16 噪声级），两实现数值等价。
备注：AscendC 有 step 59-115 瞬态慢相（编译缓存填充），含慢相的窗口（50-190）会低估为 1.08x。

## 二、GDN 测试（Qwen3.5-MoE 35B-A3B，wheel 新旧对比）

### 测试条件

| 项 | 值 |
|---|---|
| 模型 | Qwen3.5-35B-A3B |
| 硬件 | Ascend 910C，1 节点 × 16 NPU |
| 环境 | 镜像 cann9.1.0_aarch64；旧栈 = wheel 26.6.0 + 原绑定，新栈 = wheel 26.9.1+dev36e67d2 + 适配绑定（同容器切换，唯一变量） |
| 配置 | sp2 / ep8、SEQ_LEN 1024 varlen packed、stage sft、bf16、bs 1、GA 1、lr 1.25e-5、seed 42、数据集 science、50 步 |
| 脚本 | scripts/runs/qwen3_5_moe/lf_text_sft.sh |

### 结果

| 栈 | 首步 loss | 稳态 sec/step | 统计窗口 | 提速 |
|---|---|---|---|---|
| 旧栈 (26.6.0) | 0.5469 | 4.23 | step 2-45（n=43） | — |
| 新栈 (26.9.1+适配) | 0.5469 | 3.86 | 同 | **~1.10x** |

精度：首步 loss 逐位一致；45 步轨迹平均相对偏差 0.09%~0.14%，符号平衡无增长，新旧栈数值等价。

