
## 结论速览

这次 run 总耗时 **3 h 29 m 50 s（209.8 min）**，其中 **96.9% 集中在 3 个步骤**：

| 排名 | 阶段 | 耗时 | 占比 |
|---|---|---|---|
| 1 | `#10` 编译 fla-npu（Ascend 算子） | **84.7 min** | 40.5% |
| 2 | `#12` 源码编译 PyTorch 2.10.0 | **68.1 min** | 32.5% |
| 3 | `#17` 导出并推送镜像到 SWR | **50.1 min** | 23.9% |
| — | 其余全部（拉基础镜像 + pip + 收尾） | 6.9 min | 3.1% |

---

## 一、Job 级步骤耗时

| # | 步骤 | 起 → 止 (UTC) | 耗时 |
|---|---|---|---|
| 1 | Set up job | 10:53:43 → 10:53:44 | 1 s |
| 2 | ① 检出 | 10:53:44 → 10:53:45 | 1 s |
| 3 | ② 环境自检 | 10:53:45 | <1 s |
| 4 | ③ 腾磁盘 | — | **skipped**（`free_disk=false`） |
| 5 | ④ buildx builder | 10:53:45 → 10:53:50 | 5 s |
| 6 | ⑤ 计算 tag | 10:53:50 | <1 s |
| 7 | ⑥ 登录 SWR | 10:53:50 → 10:53:53 | 3 s |
| 8 | ⑦ 前置检查 | 10:53:53 → 10:53:56 | 3 s |
| 9 | **⑧ 构建并推送** | 10:53:56 → 14:23:18 | **209.4 min** |
| 10 | ⑨ 校验产物 | 14:23:18 → 14:23:23 | 5 s |
| 11 | ⑩ Job Summary | 14:23:23 | <1 s |
| 12 | ⑪ 清理 builder | 14:23:23 → 14:23:31 | 8 s |
| 13 | Post checkout / Complete | 14:23:31 → 14:23:33 | 2 s |

> 除 ⑧ 之外的所有步骤加起来只有 **约 30 秒**，CI 编排本身没有开销问题。

---

## 二、Dockerfile 各层耗时（buildkit 实测）

| Buildkit | Dockerfile 指令 | 耗时 | 累计时刻 |
|---|---|---|---|
| `#4` | `FROM quay.io/ascend/cann:9.0.0-a3` | 122.5 s | 10:55:59 |
| `#5` | `dnf install`（gcc/cmake/openblas…） | 96.3 s | 10:57:35 |
| `#6` | ossutil | 2.8 s | 10:57:38 |
| `#7` | `pip install triton==3.5.0` | 69.9 s | 10:58:48 |
| `#8` | `pip install triton-ascend==3.2.2` | 43.5 s | 10:59:31 |
| `#9` | `pip install torch_npu==2.10.0.post2` | 27.5 s | 10:59:59 |
| **`#10`** | **编译 fla-npu（ascend910_93）** | **5083.2 s** | 12:24:42 |
| `#11` | flash-linear-attention（纯 Python） | 22.8 s | 12:25:05 |
| **`#12`** | **源码编译 PyTorch 2.10.0** | **4084.2 s** | 13:33:09 |
| `#13` | py-spy | 3.9 s | 13:33:13 |
| `#14–16` | 清理 / sed / sshd | ~0 s | 13:33:13 |
| **`#17`** | **export + push 到 SWR** | **3004.7 s** | 14:23:18 |


---

## 三、三个瓶颈的细节拆解

### 1. `#10` fla-npu 编译 — 84.7 min（最大头）

```
11:00:33  Building wheel for flash-linear-attention-npu-a3: started
11:01:33  still running...   ← 之后每 ~60 s 一条心跳，无任何进度信息
   ...    （连续 82 分钟只有 "still running..."）
12:24:34  Created wheel ... size=41479866 (39.5 MiB)
```

- 前置动作（clone 2.2 s、pip 依赖 6.7 s、`g++` 编 `libfla_npu_stable.so` 33 s）合计不到 1 分钟。
- **真正耗时 5041 s 全在 `scripts/build_wheel.py` 里**，即 Ascend C 算子（`FLA_NPU_SOC=ascend910_93`）的编译。
- 日志完全不可观测：82 分钟里没有任何编译单元进度，出问题只能等超时。

### 2. `#12` PyTorch 编译 — 68.1 min

| 子阶段 | 耗时 |
|---|---|
| clone + 全部 submodule（shallow） | ~150 s |
| `pip install -r requirements-build.txt`（ninja/cmake） | ~3 s |
| CMake configure | 15.5 s |
| **`cmake --build -j 4`（5473 个 target）** | **3875 s ≈ 64.6 min** |
| `bdist_wheel` 打包 + 安装 | ~40 s |

- 4 vCPU 跑 5473 个编译单元，`-j4` 已打满；`MAX_JOBS=8` 会因 15 GiB 内存 OOM（workflow 注释已记录）。
- 编译速率约 **1.41 target/s**，属于纯 CPU 受限，没有 IO 或网络瓶颈。

### 3. `#17` 推送 — 50.1 min

| 子阶段 | 耗时 |
|---|---|
| `exporting layers`（本地写盘） | 47.6 s |
| **`pushing layers`（上传 SWR）** | **2951.6 s ≈ 49.2 min** |
| push manifest（两个 tag） | 2.9 s + 0.5 s |

- 按 workflow 注释的产物 6~8 GiB 反推，**实际有效带宽只有 2.1~2.8 MiB/s**，比注释里假设的 4.00 MiB/s 慢 **30~45%**，所以预估的 26~34 min 变成了 49 min。
- 这是纯网络受限，与 CPU 无关。

---

## 四、优化优先级建议

| 优先级 | 目标 | 手段 | 预期收益 |
|---|---|---|---|
| **P0** | 消除重复编译 | 开 `--cache-from type=registry,ref=...:base-cache` + `--cache-to type=registry,mode=max`（比 `type=gha` 更适合 6~8 GiB 大层） | 二次构建可跳过 `#10`/`#12`，**直接省 150 min** |
| **P0** | 拆分镜像层 | 把 fla-npu、PyTorch 各自做成独立 stage 并单独推 tag，L1 只做 `FROM` 引用 | 改一行依赖不必重编 84 min 的算子 |
| **P1** | 缩短 PyTorch 编译 | 换 8 vCPU / 32 GiB runner + `MAX_JOBS=8`；或直接复用官方 aarch64 wheel（仅需保留 nomimalloc 补丁） | 64.6 min → 约 30 min 或 0 |
| **P1** | 缩短推送 | 精简镜像（`rm -rf` 编译中间产物、`pip cache purge`、`/tmp/pytorch` 已删但 build 目录可能残留）；SWR 与 runner 同区域 | 49 min → 30 min 量级 |
| **P2** | 提升可观测性 | `#10` 里给 `build_wheel.py` 加 `-j`/verbose，或分片编译，避免 82 分钟黑盒 | 排障成本 |
| **P2** | 降低超时风险 | 当前 209 min / 350 min 上限，余量 40%；若开缓存后仍接近上限，考虑 `timeout-minutes` 上调或拆 job | — |

> 一句话：**这次 3.5 小时里，只有 30 秒是 CI 框架的开销，剩下全是"编译两次 + 上传一次"**。开 registry 缓存 + 拆 stage 是唯一能带来数量级改善的动作。