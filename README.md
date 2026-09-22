# Capability-Based LLM Router

一个保持“纯路由”边界的 Python MVP：离线构建模型能力画像，在线预测任务需求，并从满足要求的候选模型中选择成本最低的模型。

## 本项目解决什么问题

同一个大语言模型并不适合所有任务。能力更强的模型通常价格更高，而便宜的模型又可能无法完成高难度的推理、编码、调试或工具调用任务。如果所有请求都固定发送给最强模型，会造成不必要的成本；如果只追求低价，则可能牺牲任务质量。

本项目在模型调用之前增加一个独立的路由决策层，用来解决三个问题：

- 如何根据公开 Benchmark 的原始成绩建立可追溯的模型能力画像；
- 如何把用户任务转换为归一化的能力需求；
- 如何在 Provider、成本和延迟等约束下，选择能力合格且成本最低的模型。


## Benchmark

## Quickstart

### 1. 安装

需要 Python 3.12 或更高版本：

```bash
python -m pip install -e .
```

### 2. 配置 Jev

先复制 `.env.example` 为 `.env`。

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

Windows CMD：

```cmd
copy .env.example .env
```

Linux / macOS：

```bash
cp .env.example .env
```

然后打开 `.env`，将占位密钥替换为 TypeSafe API Key：

```dotenv
TYPESAFE_API_KEY=replace-with-typesafe-api-key
JEV_BASE_URL=https://api.typesafe.ai
JEV_MODEL=jev-latest
JEV_TIMEOUT_SECONDS=30
JEV_MAX_RETRIES=2
JEV_RETRY_BACKOFF_SECONDS=0.5
JEV_REQUESTION_CONFIDENCE_THRESHOLD=0.6
```

项目会自动读取根目录下的 `.env`。该文件已被 Git 忽略，仓库只提交 `.env.example`，不要提交真实凭据。

### 3. 运行路由

最基本的命令：

```bash
python main.py "分析下面 Python 代码为什么会死锁"
```

指定路由档位：

```bash
python main.py "分析下面 Python 代码为什么会死锁" --tier balanced
```

限定 Provider、成本或延迟：

```bash
python main.py "分析 SQL 性能问题" --provider openai --tier intelligence
python main.py "生成一个 Python 工具" --providers openai anthropic --max-cost 5
python main.py "修复这段代码" --max-latency 2
```

可用参数：

| 参数 | 说明 |
|---|---|
| `task` 或 `--task` | 要分析和路由的任务文本 |
| `--tier` | `efficiency`、`balanced` 或 `intelligence`，默认 `balanced` |
| `--provider` | 只允许一个指定 Provider |
| `--providers` | 允许多个 Provider |
| `--max-cost` | 最大路由成本 |
| `--max-latency` | 最大 p50 延迟（秒） |
| `--registry` | 自定义 `models.yaml` 路径 |

CLI 输出一个 JSON 格式的 `RouteDecision`。`confidences` 按能力给出 Jev 最终采用的置信度；成功时 `error` 为 `null`。可预期的配置、Registry、Jev 或路由异常也使用相同对象返回，并在 `error` 中提供结构化错误信息。

Jev 首轮会预测全部能力。某项能力的 confidence 低于 `JEV_REQUESTION_CONFIDENCE_THRESHOLD` 时，Predictor 最多再质询两轮；每轮只询问仍低置信度的能力。最终逐能力选择 confidence 最高的一轮，并确保 requirement/confidence 来自同一轮，避免低置信度后续结果覆盖更可靠的早期结果。第二、三轮的逐能力 instructions 配置在 `data/jev_requestion_prompt.yaml`，空值会沿用首轮原始 instructions。

### 4. 通过 Python 使用

```python
from llm_router import ModelRegistry, Router

registry = ModelRegistry.from_yaml("data/models.yaml")

with Router(registry) as router:
    decision = router.route(
        "分析下面 Python 代码为什么会死锁",
        tier="balanced",
        provider="openai",
    )

print(decision.model_dump())
```

### 5. 重新构建模型能力画像

当 Benchmark 原始成绩或 capability 聚合配置发生变化时，重新生成在线路由使用的 `data/models.yaml`：

```python
from llm_router import ModelRegistry

registry = ModelRegistry.build_from_yaml(
    "data/models.yaml",
    "data/benchmarks.yaml",
    "data/capabilities.yaml",
)
registry.save("data/models.yaml")
```

## How it works

系统只有两步：离线准备模型能力数据，在线为任务选择模型。

```text
离线：Benchmark 成绩 → 聚合为模型能力画像

在线：用户任务 → 预测能力需求 → 过滤候选模型 → 选择最低成本的合格模型
```

- Benchmark 原始成绩始终保留，缺失成绩不会按 0 计算。
- 在线路由可按 Provider、最大成本和最大延迟过滤模型。
- Router 比较任务需求与模型能力：优先选择满足当前 tier 的最低成本模型；没有模型达标时，返回能力差距最小的 fallback。
- Router 只返回 `RouteDecision`，不会调用最终选中的模型。

## 项目结构与分层

核心业务只包含四个模块：`Capability Builder`、`Model Registry`、`Requirement Predictor` 和 `Router`。

```text
routerBasedJev/
├── main.py                       # CLI 入口与依赖装配
├── llm_router/
│   ├── capability.py             # Capability Builder 与聚合策略
│   ├── registry.py               # YAML 加载、校验与 Model Registry
│   ├── predictor.py              # Requirement Predictor 与 Jev 适配
│   ├── policy.py                 # Routing Policy 与 shortfall 策略
│   ├── router.py                 # 在线路由编排
│   ├── models.py                 # 领域数据对象
│   ├── errors.py                 # 统一错误类型
│   └── __init__.py               # 公共 Python API
├── data/
│   ├── benchmarks.yaml           # Benchmark 定义与原始成绩
│   ├── capabilities.yaml         # Benchmark 到 capability 的映射
│   ├── jev_requestion_prompt.yaml # 第二、三轮逐能力提示词
│   └── models.yaml               # 模型元数据与预计算画像
├── .env.example                  # Jev 运行参数模板
└── pyproject.toml                # 项目与依赖配置
```

架构分层与代码位置：

| 分层 | 代码位置 | 职责 |
|---|---|---|
| 程序入口层 | `main.py` | 解析 CLI 参数、装配依赖并输出结果 |
| 在线路由层 | `llm_router/router.py`、`predictor.py`、`policy.py` | 预测任务需求、过滤候选模型并完成选择 |
| 离线构建与 Registry | `llm_router/capability.py`、`registry.py` | 从 YAML 构建、校验和加载模型能力画像 |
| 领域对象与错误 | `llm_router/models.py`、`errors.py` | 定义统一的数据和错误结构 |
| 数据层 | `data/*.yaml` | 保存 Benchmark、能力映射和模型画像 |

依赖方向保持为“程序入口 → 在线路由 → Registry/领域对象”。核心模块不依赖 CLI、UI 或传输协议，因此后续可以独立增加 SDK 或 HTTP API。
