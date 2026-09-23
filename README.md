# Capability-Based LLM Router

一个保持“纯路由”边界的 Python MVP：离线构建模型能力画像，在线预测任务需求，并从满足要求的候选模型中选择成本最低的模型。

## 本项目解决什么问题

同一个大语言模型并不适合所有任务。能力更强的模型通常价格更高，而便宜的模型又可能无法完成高难度的推理、编码、指令遵循或工具调用任务。如果所有请求都固定发送给最强模型，会造成不必要的成本；如果只追求低价，则可能牺牲任务质量。

本项目在模型调用之前增加一个独立的路由决策层，用来解决三个问题：

- 如何根据公开 Benchmark 的原始成绩建立可追溯的模型能力画像；
- 如何把用户任务转换为归一化的能力需求；
- 如何在 Provider、成本和延迟等约束下，选择能力合格且成本最低的模型。


但是是有问题的，需要更大成本测试和解决
问题1.confidence具体的阈值是多少？多少才算是模型对自己的回答相当有自信，然后能够直接固定住判断出来的能力指标？自信度一直低说明模型判不准，不断提高也不能说明它是正常的，然后balance调节判断阈值的具体阈值也是未经验证的
问题2.模型能力指标和提示词的能力指标的差别，benchmark里的模型测试的任务都是高难度的，而要求jev做的任务不一定都这么难，判断模型能力的benchmark分数很有可能导致对模型能力指标判断偏低，简单来讲，就是低估的模型的性能，模型的能力大大超过了jev所评判的能力标准。统一两者的能力指标对作者来说不太可能的事情。
问题3.提示词，本项目设计了简单的记忆模块和追问功能，目的是追问后confidence提高，至少提高0.05吧，让jev尽可能确定要求模型的能力指标，但是效果不佳。

## Benchmark

当前四维为 `reasoning`、`coding`、`tool_use`、`instruction_following`，已用指令遵循替换 debugging。
启用 4 个有成绩且协议明确的 Benchmark 条目，每项能力各 1 个。前三项来自同一 LLMRouterBench performance-cost 模型群体；不混入另一批小模型的组内分位数。
具体名单见 `data/capabilities.yaml`；未启用的原始成绩仍保留在 `data/benchmarks.yaml`。

- reasoning 使用 LLMRouterBench HLE 的指定协议结果（包含知识因素，属于推理能力代理指标）。
- coding 使用 LLMRouterBench LiveCodeBench 的指定协议结果。
- tool_use 使用已有 LLMRouterBench τ²-Bench 成绩；它衡量模型加执行框架，不能解释成裸模型工具能力。
- instruction_following 使用 [IFBench 作者论文 v3 图 1](https://arxiv.org/pdf/2507.02833v3#page=3) 的单轮 prompt-level loose accuracy。不混用 strict、instruction-level 或 IF-RLVR 训练后结果。

综合 Intelligence Index、SimpleQA、领域知识题等不再作为单项推理能力证据；厂商自报与 simple-evals 复测分开保存。
模型元数据共 64 项，全部保存在 `data/model_metadata.yaml`；当前能构建至少一项能力画像的有 15 项。
其余 49 项在画像文件的 `excluded_models` 中说明原因，不删除原始成绩或补造能力。
当前覆盖：reasoning 13、coding 13、tool_use 12、instruction_following 5；四项齐全的有 3 个模型。要求某项非零能力时，缺该项数据的模型不能成为候选。

### 分数尺度与固定锚点

校准版本为 `router-relative-v1-2026-09-23`。每项 Benchmark 在 YAML 中冻结参考模型、原始分数和插值锚点。
参考样本不少于 10 项时，采用原始 0→0、P10→1、P25→3、P50→5、P75→7、P90→9、参考最高分→10；分位点按 `(n-1)*q` 线性插值。
重复原始锚点省略，端点优先；参考成绩全相同不构建区分尺度。
样本少于 10 项时不估计分位点，使用原始 0→0、固定参考最高分→10 的双锚点初始尺度。
IFBench 的参考群体为论文图 1 左侧 6 个原始模型，固定上锚点为 69.3；其中 5 个在当前模型元数据中，Qwen3-32B 仅用作参考证据。

区间内线性插值、区间外截断；内部统一除以 10，得到 0～1 后由 MeanAggregator 聚合。
Registry 从 raw_value 重新计算，忽略旧 normalized_0_1 缓存；缺失仍是未知。
增加候选模型或按 provider 过滤不会重算锚点。更改尺度须同步更新 Benchmark、能力配置、提示词版本并重建画像。
IFBench 是不同且较小的参考群体，不宣称它的分位数与其他维度等价。模型画像与任务需求之间仍需真实任务校准；目前是相对位置的启发式匹配，不是通过率预测，`success_probability` 返回 null。
IFBench 没有完整披露所有模型的日期快照及推理强度，且不全面测量长上下文和指令优先级，使用时应保留这些限制。


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
JEV_REQUESTION_CONFIDENCE_THRESHOLD=0.5
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

Jev 首轮会预测全部能力且不携带记忆。某项能力的 confidence 低于 `JEV_REQUESTION_CONFIDENCE_THRESHOLD` 时，Predictor 最多再质询两轮；每轮只询问仍低置信度的能力。第二轮会把第一轮响应中该能力对应的单个 `answers[capability]` 作为记忆，第三轮同样只携带第二轮该能力的回答；不会全量发送 Jev 响应，也不会累计更早轮次的记忆。记忆通过 `DEFAULT_MEMORY_INSTRUCTION_TEMPLATE` 注入到本轮 `instructions` 的最前面，随后才是本轮的新提示词。最终逐能力选择 confidence 最高的一轮，并确保 requirement/confidence 来自同一轮，避免低置信度后续结果覆盖更可靠的早期结果。全部能力定义、六级行为标准和三轮逐能力提示词统一配置在 `data/jev_requestion_prompt.yaml`，由 Registry 集中加载校验。三轮依次从交付目标、必要步骤和降档反事实复核，criteria 始终一致。记忆明确是待复核材料，允许修正，不要求提高分数或置信度。

每个能力使用六条具体行为描述，对应 Jev 原始位置 0～5，内部 `score / 5`，展示为 `score / 5 * 10`。这不限制返回小数。
最终 JSON 的 requirements 仍是 0～1；stderr 的 `jev_prediction_summary` 会给出各轮 `requirement_0_10`、confidence 和最终采用轮次。
默认仅在 confidence < 0.5 时重问，等于 0.5 即停止；不保证每次都出现三轮；置信度更高不等于需求判断更准确，变更等级数量后也应重新检查 0.5 阈值。

Windows PowerShell 中保存一次复核记录：

```powershell
.venv/Scripts/python main.py "编写 Python 函数，对整数列表去重并保持顺序；只输出代码。" 1> decision.json 2> jev-review.log
```

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
    "data/model_metadata.yaml",
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
│   ├── jev_requestion_prompt.yaml # 六级标准、公共规则与三轮提示词
│   ├── model_metadata.yaml        # 全部模型元数据（包括暂不可路由者）
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
