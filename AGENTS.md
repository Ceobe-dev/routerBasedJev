# LLM Router 工程约束

- 项目保持“纯路由”：离线构建模型能力画像，在线预测任务需求并选择最低成本的合格模型；程序入口与核心逻辑分离。
- 架构只保留四个模块：`Capability Builder`、`Model Registry`、`Requirement Predictor`、`Router`。
- 只抽象三个可替换接口：`RequirementPredictor`、`CapabilityAggregator`、`RoutingPolicy`；默认实现分别为 `JevPredictor`、`MeanAggregator`、`ShortfallPolicy`，统一通过依赖注入替换。
- Benchmark 原始成绩是事实源，使用公开且可追溯的测试集；能力分数由配置驱动聚合，不覆盖原始成绩。
- 模型、Benchmark、Capability 定义与成本/延迟等元数据保存在 YAML，加载与校验集中在 `registry.py`。
- 默认路由流程：预测需求 → 按 provider/约束过滤 → 按 tier 判断 shortfall → 从合格模型中选择最低成本者。
- tier 阈值默认：`efficiency=0.10`、`balanced=0.05`、`intelligence=0.00`；复杂概率模型不属于 MVP。
- `Router.route()` 只返回 `RouteDecision`，不调用目标模型，也不承担 Agent、Workflow、Tool、RAG、Memory 或 Gateway 职责。
- 已知配置、预测、Registry 与路由异常统一映射为结构化 `RouteError`；`RouteDecision` 始终带 `error` 属性，成功时为 `null`，CLI 输出同一对象。
- Jev 的地址、模型、密钥、超时等运行参数从 `.env` 读取；提交模板和占位值，不提交真实凭据。
- CLI/API/SDK 只能依赖 Router Core 的公开接口；核心模块不得依赖具体入口、UI 或传输协议。
- 项目不保留测试文件或测试依赖；验收时直接运行一次离线构建、Registry 加载、需求预测和 Router 决策的完整链路。

详细设计以 `.doc/LLM_Router_Technical_Architecture.md` 为背景；发生冲突时，以本文件的精简架构约束为准。
