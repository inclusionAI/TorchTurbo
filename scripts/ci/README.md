# CI 架构与本地校验

CI 只有一条数据流：

```text
lint -> Planner -> PlanArtifact -> Executor -> JobResult -> Gate
```

Planner 读取两类输入：

```text
CiConfig   用户意图：资源、venv 映射、PR/nightly 范围
TestFacts  仓库事实：pytest 文件、device/rank/framework、integration manifest
```

Executor 只消费 Plan 中的一个 Job；Gate 只消费 Plan 与 JobResult。两者都不重新选测。

## 目录

```text
scripts/ci/
  config/
    ci.yml            唯一 CI 配置：resources / workloads / scopes
    envs/             Python 依赖清单
  contracts/
    artifacts.py      schema v2 artifact 编解码与校验
  producer/
    ci_config.py      CiConfig loader
    facts.py          pytest 扫描 + integration suite manifest
    scan.py           pytest AST 事实
    planner.py        CiConfig + TestFacts -> PlanArtifact
    change_scope.py   文档变更分类
  consumer/
    executor.py       执行一个 Job -> JobResult
    gate.py           Plan + JobResults -> Decision
    runtime/
      env.py          CPU/NPU/GPU 探针与清理
      venv.py         内容寻址 venv
  tools/
    lint.py           通用 linter + CI 契约检查
    image.py          基础镜像解析
    report.py         非阻断 timing summary
  tests/              config / facts / contracts / planner / executor / gate
```

## 边界

| 阶段 | 输入 | 输出 | 不做什么 |
|---|---|---|---|
| lint | 源码、workflow、ci.yml、tests | annotations | 不选择用例 |
| Planner | CiConfig、TestFacts、scope、change scope、镜像覆盖 | `plan.json` | 不安装环境、不执行测试 |
| Executor | PlanArtifact + 一个 `job_id` | `job-result.json` | 不读取 CiConfig/TestFacts |
| Gate | PlanArtifact + 全部 JobResult | `decision.json` | 不重新选测 |

Planner 按 CPU/NPU/GPU 输出 matrix。matrix 项只包含 `job_id`、`image_ref`、
`runner_variable`、`runner`、`timeout_minutes`，完整 Job 只存在于 PlanArtifact。

Executor 执行前先写 `infra_error/executor-incomplete` 占位结果，结束时原子覆盖。即使
runner 被外部杀死，Gate 仍能区分“缺结果”和“执行未完成”。

## 用例模型

只有两种 Job kind：

```text
pytest       单元、组件、adapter 测试
integration  tests/integration_tests 的训练回归
```

pytest suite 固定为三层：

| suite | 规则 | PR | nightly |
|---|---|---|---|
| `core` | 通用 engine/ops/runtime 等 | 全量 | 全量 |
| `model` | 模型族依赖测试 | 不跑 | 全量 |
| `adapter` | LLaMA-Factory / ms-swift 适配测试 | 不跑 | 全量 |

integration case 的事实源是 suite runner 的 `--list-cases --json`。一个 case 一个 Job，
不把 case 的 framework、ranks、config、golden 再抄进 `ci.yml`。

## 运行范围

范围只由 `config/ci.yml` 的 `scopes` 决定：

| scope | pytest | integration | Job |
|---|---|---|---:|
| `pull_request` | `core=all` | `sft_2cpu`、`sft_2npu` | 4 |
| `nightly` | 全部 | 全部 3 case | 12 |
| docs-only | 无 | 无 | 0 |

PR 不跑模型 pytest，因为不同模型族依赖的 venv 不同，安装成本和稳定性不可控。PR 使用
两个轻量 integration case 看护 CPU 和 NPU 主路径；完整模型和 16 卡 case 留给 nightly。

纯文档包括 `docs/**`、`*.md`、`*.rst`、`*.adoc`、`LICENSE`、`NOTICE`。空 diff、
分类失败或读取失败按代码变更处理。

## 扩展点

- 新增镜像或 lane：改 `resources.images` / `resources.lanes`。Planner 选择容量最小且满足
  ranks 的 lane，不在 consumer 写平台分支。
- 新增 venv：改 `resources.venvs`，可用 `extends` 复用依赖链；依赖文件内容计入 digest。
- 新增 pytest：补 device marker 和依赖事实即可。必要时增加 `model_venvs`、
  `framework_venvs`、`combo_venvs` 或文件 override。
- 新增 adapter：补 framework venv 映射和测试事实。PR 是否运行由 `scopes` 决定。
- 新增 integration case：在 suite 中增加 case；PR 使用 `explicit`，不会自动纳入。
- 新增 integration suite：在 `workloads.integration.suites` 增加 runner 和默认超时。
- 新增 platform：增加 image、lane 和 workload timeout；integration case 的 platform 来自
  manifest，不在配置中重复。

资源完全不存在于 CI 能力范围时，Planner 把路由写入 `coverage.unsupported_routes`；
资源存在但无法满足 lane/ranks 时 Planner 直接失败，不会静默丢路由。

## 常用命令

```bash
# 配置与测试事实
python3 -m scripts.ci.producer.ci_config check
python3 -m scripts.ci.producer.facts check

# PR / nightly / docs-only Plan 均可构建
python3 -m scripts.ci.producer.planner --verify

# 静态检查
python3 -m scripts.ci.tools.lint
python3 -m scripts.ci.tools.lint --only contract
python3 -m scripts.ci.tools.lint --only config

# 生成 PlanArtifact 和平台 matrix
python3 -m scripts.ci.producer.planner \
  --scope pull_request \
  --output /tmp/plan.json \
  --github-output /tmp/github-output

# 单 Job 诊断
python3 -m scripts.ci.producer.planner \
  --scope pull_request \
  --target-job integration-base-suite-qwen3-5-moe-sft-2cpu \
  --output /tmp/plan.json

# 执行一个 Job
python3 -m scripts.ci.consumer.executor \
  --plan /tmp/plan.json \
  --job-id <job-id> \
  --output /tmp/ci-result/job-result.json

# 判定完整 Plan
python3 -m scripts.ci.consumer.gate \
  --plan /tmp/plan.json \
  --results /tmp/ci-results \
  --output /tmp/decision.json \
  --summary /tmp/summary.md

# 全部 CI 单元与契约测试
python3 -m pytest scripts/ci/tests -q
```

## 约束

- integration Job 只能有一个 case，step spec 的 digest 必须与 runner manifest 一致。
- CI 执行 integration 时固定 `--skip-install --forbid-update-golden`。
- pytest 必须收集到测试；integration evidence 的 suite/case 必须与 Plan 一致。
- skip 必须带原因，且原因出现在 Plan 的 `allowed_skip_reasons` 中。
- 跨阶段只传带 `schema_version` 和内容 digest 的 JSON artifact。
