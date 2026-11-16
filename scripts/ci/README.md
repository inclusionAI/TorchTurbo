# CI 架构与本地校验

CI 只保留一条数据流：

```text
lint -> Producer -> PlanArtifact -> Consumer -> JobResult -> Gate
```

对应的代码边界：

```text
Planner (Catalog + Policy) -> PlanArtifact
Executor (one Job)         -> JobResult
Gate (Plan + JobResults)   -> Decision
```

Planner 负责选择工作，Executor 不读取 Catalog/Policy，Gate 不重新选测。所有跨阶段
输入都是带有 `schema_version` 和内容摘要的 JSON artifact。

## 目录

```text
scripts/ci/
  config/
    catalog.yml       镜像、venv、runner lane、pytest 与 E2E 静态事实
    policies.yml      pull_request / nightly 的工作范围
    envs/             Python 依赖清单
  contracts/
    artifacts.py      PlanArtifact / JobResult / Decision 契约与编解码
  producer/
    catalog.py        Catalog loader、测试扫描与 E2E case 解析
    policy.py         Policy loader 与配置引用校验
    planner.py        Catalog + Policy -> PlanArtifact
    scan.py           pytest AST 事实：device、rank、framework、model family
    change_scope.py   文档变更分类
  consumer/
    executor.py       执行 Plan 中的一个 Job -> JobResult
    gate.py           Plan + JobResult -> Decision
    runtime/
      env.py          CPU/NPU/GPU 环境探针、torch guard、清理
      venv.py         Plan 中自包含 venv 的创建与复用
      bin/torchrun    训练脚本使用的 torchrun wrapper
      model_e2e/      模型准备、脚本修补、训练 orchestration
  tools/
    lint.py           通用 linter + 仓库级 CI 契约检查
    image.py          基础镜像判断、构建、校验、发布与解析
    report.py         CI 计时与模型 E2E summary
  tests/              contracts / planner / executor / gate / catalog 测试
```

## 各阶段职责

| 阶段 | 输入 | 输出 | 不做什么 |
|---|---|---|---|
| lint | 源码、workflow、Catalog | exit code + annotations | 不选择用例 |
| Planner | Catalog、Policy、change scope、镜像覆盖 | `plan.json` PlanArtifact | 不安装环境、不执行用例 |
| Executor | PlanArtifact + 一个 `job_id` | `job-result.json` | 不读取 Catalog/Policy |
| Gate | PlanArtifact + 全部 JobResult | `decision.json` | 不重新选测 |

Planner 会按 CPU/NPU/GPU 输出 matrix。matrix 项只包含执行所需字段：
`job_id`、`image_ref`、`runner_variable`、`runner`、`timeout_minutes`，不嵌入完整 Job。

Executor 在真正执行前先写一份 `infra_error/executor-incomplete` 占位结果，结束时原子
覆盖。即使进程被外部杀死，Gate 仍能区分“缺结果”和“执行未完成”。

Gate 校验结果完整性、runtime/spec 摘要、pytest 证据、E2E case 证据和 skip allowlist。
缺少结果、重复结果、额外结果或伪造的 spec 都会判红。

## 运行范围

范围只由 `config/policies.yml` 决定：

- `pull_request`：所有 core pytest，一个模型 guard；模型 pytest 和 adapter pytest 不跑。
- `nightly`：所有 pytest suite 和所有 E2E case。
- 纯文档变更：生成零 Job Plan，不进入 pytest、NPU 或模型执行。

纯文档包括 `docs/**`、`*.md`、`*.rst`、`*.adoc`、`LICENSE`、`NOTICE`。空 diff、
分类失败或读取失败一律按代码变更处理。

## 扩展点

### 增加镜像或 runner lane

在 `config/catalog.yml` 的 `resources.images` 与 `resources.lanes` 增加条目。同一个
device 只能有一个 default image；Planner 根据 `device + image + ranks` 选择容量最小
且足够的 lane，因此 2/4/8/16 卡资源不需要在 consumer 中写分支。

### 增加 venv

在 `resources.venvs` 增加 Python 版本和 requirements 文件；需要继承时使用 `extends`。
Catalog 会解析继承链、检查循环引用，并把依赖文件内容计入 venv digest。

### 增加 pytest suite 或 adapter

pytest 文件的事实由 `producer/scan.py` 从源码静态扫描，不需要 import 测试模块。suite
分类、family/framework venv 选择和显式 override 位于 `producer/catalog.py`；Policy 只
决定 `all` 或 `none`。新增 adapter 时同时补 marker/device 事实和所需 override。

### 增加模型 E2E case

在 `workloads.model_e2e.models.<model>.cases` 增加 case，声明 framework、venv、安装脚本、
训练脚本、模型仓库、数据集、rank 与 SP/EP。PR 默认只选
`bailing_hybrid-lf-sft`；nightly 自动选择全部 case。新增 case 不修改 Executor。

一个 case 的步骤仍是：

```text
install -> prepare model/config -> train -> report
```

## 常用命令

```bash
# 配置与测试扫描事实
python3 -m scripts.ci.producer.catalog check
python3 -m scripts.ci.producer.policy check

# PR / nightly / docs-only Plan 均可构建
python3 -m scripts.ci.producer.planner --verify

# 静态检查
python3 -m scripts.ci.tools.lint
python3 -m scripts.ci.tools.lint --only contract
python3 -m scripts.ci.tools.lint --only shellcheck

# 生成 PlanArtifact 和平台 matrix
python3 -m scripts.ci.producer.planner \
  --policy pull_request \
  --output /tmp/plan.json \
  --github-output /tmp/github-output

# 单 Job 诊断：Plan 中只保留一个 Job
python3 -m scripts.ci.producer.planner \
  --policy pull_request \
  --target-job pytest-core-npu-npu-default-default \
  --output /tmp/plan.json \
  --github-output /tmp/github-output

# 执行一个 Job
python3 -m scripts.ci.consumer.executor \
  --plan /tmp/plan.json \
  --job-id <job-id> \
  --output /tmp/ci-result/job-result.json

# 平台维护窗口：写入结构化 skip 结果
python3 -m scripts.ci.consumer.executor \
  --plan /tmp/plan.json \
  --job-id <job-id> \
  --record-skip maintenance \
  --skip-message "NPU maintenance window" \
  --output /tmp/ci-result/job-result.json

# 判定完整 Plan
python3 -m scripts.ci.consumer.gate \
  --plan /tmp/plan.json \
  --results /tmp/ci-results \
  --output /tmp/decision.json \
  --summary /tmp/summary.md

# 环境与模型目录诊断
python3 -m scripts.ci.consumer.runtime.env probe --device cpu --file /tmp/torch.json
python3 -m scripts.ci.consumer.runtime.model_e2e.prepare_model probe \
  --dir /root/.cache/models/<repo> [--layers N]

# 全部 CI 单元与契约测试
python3 -m pytest scripts/ci/tests -q
```

## 约定

- 人读日志走 stderr；机器读取的 `KEY=value` 走文件或显式输出。
- 事实写入文件，不依赖 step output；失败运行时 step output 往往为空。
- skip 必须带原因，且原因必须出现在 Plan 的 `allowed_skip_reasons` 中。
- 静态检查先跑通用工具，再跑仓库契约：`ruff`、`shellcheck`、`actionlint`、
  `hadolint`、workflow 拓扑、Catalog、pytest marker。
