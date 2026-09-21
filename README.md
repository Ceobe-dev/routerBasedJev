# Capability-Based LLM Router

一个保持“纯路由”边界的 Python MVP：离线从公开 Benchmark 构建模型能力画像，在线通过 Jev 预测任务需求，并选择最低成本的合格模型。Router 只返回决策，不调用推荐模型。

## 技术架构分层

分层用于约束依赖方向，不增加新的业务模块。核心仍然只有 `Capability Builder`、`Model Registry`、`Requirement Predictor` 和 `Router` 四部分。

```text
                         程序入口层
                main.py / 后续 SDK / HTTP API
                              │
                              ▼
┌──────────────────────── 在线路由层 ────────────────────────┐
│  Prompt → Requirement Predictor → requirements             │
│                                  │                         │
│                                  ▼                         │
│  Model Registry ──────────────→ Router / Routing Policy     │
│                                  │                         │
│                                  ▼                         │
│                             RouteDecision                  │
└────────────────────────────────────────────────────────────┘
                              ▲
                              │ 读取预计算画像
┌──────────────────────── 离线构建层 ────────────────────────┐
│  Benchmark facts → Capability Builder → Model Registry     │
│  benchmarks.yaml   capabilities.yaml    models.yaml        │
└────────────────────────────────────────────────────────────┘

外部适配：JevPredictor → TypeSafe /v1/systemone
横切契约：models.py 数据对象；errors.py 统一错误封装
```

### 1. 程序入口层

`main.py` 负责解析 CLI 参数、装配依赖、加载 `.env` 并序列化最终结果。入口层只调用 Router 的公开接口；核心代码不依赖 CLI、HTTP 或 UI，因此后续可以在不修改路由逻辑的情况下增加 SDK 或 API。

### 2. 在线路由层

- `predictor.py`：把 Prompt 转换为 `RequirementPrediction`，默认实现是 `JevPredictor`。
- `policy.py`：完成 Provider、成本、延迟和能力数据过滤，计算 shortfall，并执行 tier 与最低成本策略。
- `router.py`：只负责串联 Predictor、Model Registry 和 Routing Policy，最终返回 `RouteDecision`，不调用被推荐的模型。

### 3. 离线构建层

- `capability.py`：读取每个模型的非空 Benchmark 成绩，自动计数并通过注入的 Aggregator 计算能力均值。
- `registry.py`：严格加载 YAML、连接 Benchmark 事实和模型元数据、保存预计算的 `ModelProfile`。
- 离线链路不参与每次在线请求；Benchmark 或聚合配置变化后重新生成 `models.yaml` 即可。

### 4. 数据与外部适配层

| 数据或适配器 | 职责 | 是否事实源 |
|---|---|---|
| `data/benchmarks.yaml` | Benchmark 参数、来源、协议及模型原始成绩 | 是 |
| `data/capabilities.yaml` | Benchmark 到 capability 的配置映射 | 是 |
| `data/models.yaml` | Provider、价格、延迟和预计算能力画像 | 能力分数为可重建缓存 |
| `JevPredictor` | TypeSafe API 请求、重试、校验和归一化 | 否 |

依赖方向固定为“入口 → 在线核心 → Registry/领域对象”和“Benchmark 配置 → 离线 Builder → Registry”。YAML 加载、Jev HTTP、CLI 参数等基础设施细节不能反向侵入 Router Core。

### 统一错误边界

配置、Registry、Predictor 和 Routing 的预期异常统一转换为 `RouteError`，并放入最终 `RouteDecision.error`。成功时 `error=null`；未知编程错误不会被静默伪装成业务失败。

## 安装

需要 Python 3.12 或更高版本：

```bash
python -m pip install -e .
```

## 配置 Jev

项目会自动读取根目录 `.env`。将其中的占位密钥替换为 TypeSafe API Key：

```dotenv
TYPESAFE_API_KEY=replace-with-typesafe-api-key
JEV_BASE_URL=https://api.typesafe.ai
JEV_MODEL=jev-latest
JEV_TIMEOUT_SECONDS=30
JEV_MAX_RETRIES=2
JEV_RETRY_BACKOFF_SECONDS=0.5
```

`.env` 已被 Git 忽略；可提交的模板是 `.env.example`。Jev 使用官方 `POST /v1/systemone`，每项 capability 对应一个 10 级 score 问题，返回值除以 9 归一化到 `0..1`。

## 离线构建能力画像

`data/benchmarks.yaml` 保存原始公开成绩与评测参数，`data/capabilities.yaml` 决定如何聚合，`data/models.yaml` 是在线 Router 读取的缓存画像。重新构建并保存：

```python
from llm_router import ModelRegistry

registry = ModelRegistry.build_from_yaml(
    "data/models.yaml",
    "data/benchmarks.yaml",
    "data/capabilities.yaml",
)
registry.save("data/models.yaml")
```

Capability Builder 会自动统计配置中的唯一 Benchmark 总数以及每项 capability 的 Benchmark 数量。构建模型画像时，只统计该模型实际存在的非空成绩，`MeanAggregator` 使用这些成绩的实际数量作为分母。结果保存在 `ModelProfile.capability_benchmark_counts`：

```python
from llm_router import CapabilityBuilder
from llm_router.registry import load_benchmark_observations

benchmark_ids, _ = load_benchmark_observations("data/benchmarks.yaml")
builder = CapabilityBuilder.from_yaml(
    "data/capabilities.yaml",
    benchmark_ids=benchmark_ids,
)
print(builder.configured_benchmark_count)   # 20
print(builder.configured_benchmark_counts) # reasoning=8, coding=7, debugging=2, tool_use=3
```

缺失值表示“没有可追溯的公开成绩”，不会被当成零，也不会进入平均值分母。当前样本的延迟均为 `null`，因为没有统一、可靠的一手公开 p50 数据。

## 在线路由

```bash
python main.py "分析下面 Python 代码为什么会死锁" --tier balanced
python main.py "分析 SQL 性能问题" --provider openai --tier intelligence
```

Python API：

```python
from llm_router import ModelRegistry, Router

registry = ModelRegistry.from_yaml("data/models.yaml")
with Router(registry) as router:
    decision = router.route("分析下面 Python 代码为什么会死锁")
print(decision.model_dump())
```

成功和预期异常都返回同一个 `RouteDecision`。成功时 `error` 为 `null`；配置、Jev、Registry 或路由异常时为：

```json
{
  "error": {
    "code": "predictor_error",
    "message": "Jev request failed",
    "retryable": true,
    "details": {}
  }
}
```

## 可替换依赖

只保留三个策略接口，并通过构造函数注入：

- `RequirementPredictor`：默认 `JevPredictor`
- `CapabilityAggregator`：默认 `MeanAggregator`
- `RoutingPolicy`：默认 `ShortfallPolicy`

Shortfall 为 `sum(max(0, requirement - capability))`。档位阈值为 efficiency `0.10`、balanced `0.05`、intelligence `0.00`；合格模型按成本最低选择，无合格模型时先取 shortfall 最小者，再以成本打破并列。

默认 Predictor 只询问当前 Registry 至少有一个模型具备公开数据的 capability。模型缺少任务实际需要的能力时会被排除，而不是把“未知”伪装成 0 分；若没有可比较模型，返回结构化路由错误。

## Benchmark 数据范围

YAML 共登记 20 个公开 Benchmark：

- Reasoning（8）：GPQA Diamond、Humanity's Last Exam、MMLU-Pro、BIG-Bench Hard、ARC Challenge、DROP、GSM8K、MATH-500。
- Coding（7）：HumanEval、MBPP、LiveCodeBench、SciCode、BigCodeBench、DS-1000、CRUXEval。
- Debugging（2）：SWE-bench Verified、DebugBench。
- Tool use（3）：tau2-bench、BFCL、ToolBench。

每个定义都保存指标、范围、归一化、版本、协议、来源与限制。当前模型成绩来自 [OpenAI simple-evals](https://github.com/openai/simple-evals) 与 [SWE-bench/OpenAI Verified 报告](https://openai.com/index/introducing-swe-bench-verified/)；没有锁定一致评测协议的数据只保留定义，不填造数。
