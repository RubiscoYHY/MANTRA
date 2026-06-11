# TradingAgents 修改记录

> 基础版本：v0.2.3（commit: 10c136f，2026-04-07）
> 记录范围：一晴大人本地改造分支的所有非上游修改

---

## 改造 A：双 Provider 配置（2026-04-09）

**目标**：将 `deep_think_llm`（Manager 决策层）和 `quick_think_llm`（Analyst/Researcher/Trader 层）拆分到两个独立的 LLM provider，支持 Claude + Gemini 混合调用，同时新增 HuggingFace Inference API 作为第三方 provider。

### 动机

原始代码中 `deep_client` 和 `quick_client` 共用同一个 `llm_provider` 字段，无法同时使用两个不同的 provider（例如 Claude 做决策层、Gemini 做分析层）。

---

### 修改文件清单

#### `tradingagents/default_config.py`

- 新增 `deep_think_provider`（默认 `"anthropic"`）和 `quick_think_provider`（默认 `"google"`）字段
- 保留 `llm_provider` 作为向后兼容的 fallback
- 更新默认模型：`deep_think_llm = "claude-opus-4-6"`，`quick_think_llm = "gemini-3-flash-preview"`

#### `tradingagents/graph/trading_graph.py`

- `__init__`：`deep_client` 和 `quick_client` 分别读取各自的 provider 字段（`deep_think_provider` / `quick_think_provider`），互不干扰
- 新增 `_get_base_url_for(provider)`：仅对 OpenAI 兼容 provider（openai/ollama/openrouter/xai/huggingface）传入 `backend_url`；Anthropic 和 Google 的 SDK 自管端点，传入本地 URL 会出错，因此返回 `None`
- `_get_provider_kwargs()` → 拆分为 `_get_provider_kwargs_for(role)`：接受 `"deep"` 或 `"quick"` 参数，分别读各自 provider 的特定参数（`anthropic_effort` / `google_thinking_level` / `openai_reasoning_effort`），避免参数串台
- `callbacks` 注入移入 `_get_provider_kwargs_for()` 内部，不再在外层单独处理

#### `tradingagents/llm_clients/factory.py`

- `create_llm_client()`：将 `"huggingface"` 加入 OpenAI-compat 路由分支，使用 `OpenAIClient` 对接 HF Inference API

#### `tradingagents/llm_clients/openai_client.py`

- `_PROVIDER_CONFIG` 新增 `"huggingface"` 条目：默认 endpoint `https://api-inference.huggingface.co/v1/`，API key 读 `HF_TOKEN` 环境变量
- base URL 优先级修改：调用方传入的 `self.base_url`（来自 config 的 `backend_url`）优先于 provider 默认值，支持本地 vLLM 服务覆盖 HF 云端 endpoint

#### `tradingagents/llm_clients/validators.py`

- `validate_model()`：将 `"huggingface"` 加入免校验列表（与 `ollama`、`openrouter` 一样接受任意模型 ID）

#### `tradingagents/llm_clients/model_catalog.py`

- 新增 `"huggingface"` provider 的模型选项（quick/deep 各四个，含 Llama-3.3-70B、Qwen2.5-72B 等支持 tool call 的常用模型）
- `"ollama"` provider 新增 `Gemma4-Quant-31B`、`qwen2.5:32b` 两个本地模型选项

#### `cli/utils.py`

- 新增 `select_analyst_llm_config()` 和 `select_manager_llm_config()` 两个函数，各自完成「选 provider → 选模型 → 选 provider 专属参数」的完整两级交互
- `select_llm_provider()` 新增 HuggingFace 选项
- 原 `select_shallow_thinking_agent()` / `select_deep_thinking_agent()` 保留，供新函数内部调用

#### `cli/main.py`

- Step 6（原 LLM Provider）→ 重命名为 **Step 6: Analyst LLM**，改为调用 `select_analyst_llm_config()`
- Step 7（原 Thinking Agents）→ 重命名为 **Step 7: Manager LLM**，改为调用 `select_manager_llm_config()`
- 原 Step 8（provider 专属思考参数）合并进 Step 6/7，交互流程从 8 步缩为 7 步（provider 参数在选完模型后立即询问）
- `run_analysis()` 中配置注入：分别写入 `deep_think_provider` 和 `quick_think_provider`，并保留 `llm_provider`（向后兼容 fallback）

#### `main.py`

- 示例脚本更新为双 provider 配置写法，展示 Claude + Gemini 混合调用范例

---

### 使用方式变化

**Python API**：
```python
config["deep_think_provider"]  = "anthropic"
config["deep_think_llm"]       = "claude-opus-4-6"
config["quick_think_provider"] = "google"
config["quick_think_llm"]      = "gemini-3-flash-preview"
# 不再需要 config["llm_provider"]（自动 fallback）
```

**CLI**：Step 6 和 Step 7 各自独立选择 provider 和模型，支持任意组合。

---

## 修复：可编辑安装导入冲突（2026-04-10）

**性质**：环境修复（不涉及源代码变更）

### 问题描述

执行 `tradingagents` 命令时报错：
```
ModuleNotFoundError: No module named 'tradingagents.dataflows.interface'
```
但直接在项目目录运行 `python -c "from tradingagents.dataflows.interface import route_to_vendor"` 无报错。

### 根本原因

site-packages 中残留了一个孤立的 `tradingagents/` 命名空间目录（结构：`tradingagents/dataflows/data_cache/AMZN-YFin-data-*.csv`），由此前以非可编辑模式（`pip install tradingagents`）安装时产生的 data cache 文件遗留而来。

切换为可编辑安装（`pip install -e .`）后，pip 移除了 site-packages 中的 Python 文件，但未清除已写入的 data cache 目录。

残留的 `tradingagents/` 目录（无 `__init__.py`）被 Python 的标准 `PathFinder` 识别为命名空间包。`PathFinder` 位于 `sys.meta_path` 前端，先于可编辑安装注册的自定义 `_EditableFinder`（被 append 到末尾）运行，因此 `tradingagents.dataflows` 被解析到 site-packages 路径（该路径下没有 `interface.py`），而非项目源码。

该问题仅在从非项目目录（如 CLI 二进制调用）启动时触发，从项目目录直接 `python -c` 时因 CWD 进入 `sys.path` 而被掩盖。

### 修复操作

```bash
rm -rf "$(python -c "import site; print(site.getsitepackages()[0])")/tradingagents/"
```

删除 site-packages 中的孤立 `tradingagents/` 目录后，可编辑安装的自定义 finder 恢复正常工作，`tradingagents` 命令启动正常。

### 后续影响

- 修复后，新产生的 data cache 文件写入项目目录下的 `tradingagents/dataflows/data_cache/`（`default_config.py` 的 `__file__` 现在指向项目目录），不再写入 site-packages，问题不会复现。
- 唯一会再次触发该问题的操作：执行非可编辑模式的 `pip install tradingagents`（不带 `-e`）。

### 文档更新

- `notes/usage_guide.md`：在「八、开发者模式」章节新增该陷阱的说明、修复命令及验证方法。

---

---

## 修复 + 增强：Ollama 动态模型列举（2026-04-10）

**涉及文件**：`tradingagents/llm_clients/model_catalog.py`、`cli/utils.py`

### 问题描述

CLI 中 Ollama provider 的模型列表是静态硬编码的，默认 tag 为 `Gemma4-Quant-31B`，与本地实际安装的模型名（`Gemma4-31b-tradingagent:latest`）不符，导致启动时报：

```
NotFoundError: Error code: 404 - {'error': {'message': "model 'Gemma4-Quant-31B' not found"}}
```

### 修改内容

#### `tradingagents/llm_clients/model_catalog.py`

- 将 `ollama` provider（quick / deep 两处）的默认 tag 从 `Gemma4-Quant-31B` 修正为本地实际 tag `Gemma4-31b-tradingagent:latest`（应急修复，已被下面的动态方案取代）

#### `cli/utils.py`

**新增 `fetch_ollama_models(base_url)`**
- 调用 Ollama `/api/tags` REST 接口（从 `base_url` 去掉 `/v1` 后缀推导 host）
- 返回本地所有模型 tag 的字符串列表；连不上或超时（5 s）则返回空列表

**新增 `_select_ollama_model(url, label)`**
- 用动态列表展示所有本地模型供选择
- 列表末尾附加「手动输入」选项，供输入列表外的 tag
- 若 `fetch_ollama_models` 返回空（Ollama 未运行）：打印黄色警告并直接 fallback 到手动文本输入

**修改 `select_shallow_thinking_agent(provider, url=None)`**
- 新增 `url` 可选参数
- provider 为 `ollama` 时走 `_select_ollama_model()`；其他 provider 行为不变

**修改 `select_deep_thinking_agent(provider, url=None)`**
- 同上

**修改 `select_analyst_llm_config()` / `select_manager_llm_config()`**
- 将 `select_llm_provider()` 返回的 `url` 传入对应的 `select_*_thinking_agent()` 调用

### 效果

选择 Ollama 作为 provider 后，下一步立即展示从本地实例实时拉取的模型列表，无需手动核对 tag 名称；Ollama 未运行时优雅降级为手动输入，不影响其他 provider 的使用流程。

---

---

## 规划更新：MemPalace 使用策略 + 目录结构（2026-04-10）

**性质**：修改计划文档更新 + 项目目录结构准备（不涉及 tradingagents/ 源代码变更）

### 内容概述

#### 1. MemPalace 使用策略（`modification_plan.md` § 3.11）

新增说明：MemPalace 与 BERT 预处理管道在单日测试和回测场景下的使用策略不同。

- **单日测试**：MemPalace 库为空，查询无意义，跳过以节省启动时间。BERT 预处理管道独立于历史数据，仍全程有效。
- **回测**：MemPalace 随逐日积累收益标注，是核心组件，必须启用。

在 `default_config.py` 规划新增两个控制字段：

```python
"use_sentiment_memory": False,   # 单日测试时 False，回测/生产时 True
"use_bert_preprocessing": True,  # BERT 预处理管道，通常保持开启
```

`SentimentMemoryStore` 的初始化和 ChromaDB 文件创建均延迟到 `use_sentiment_memory=True` 时触发。

#### 2. MemPalace 集成目录结构（`modification_plan.md` § 九）

规划了 MemPalace 作为 git clone 依赖的完整目录结构与许可证处理方案。

### 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `notes/modification_plan.md` | 内容新增 | 新增 § 3.11（使用策略）和 § 九（目录结构） |
| `THIRD_PARTY_LICENSES.md` | **新建** | 根目录，汇总第三方许可证；MemPalace 条目待确认实际 LICENSE 后更新 |
| `third_party/README.md` | **新建** | 指引用户 clone MemPalace 到 `third_party/mempalace/` 并 `pip install -e` |
| `.gitignore` | 追加 | 新增 `third_party/mempalace/`，防止克隆内容进入本仓库 |

### 待办

- [x] 确认 MemPalace 的实际 GitHub 仓库地址：`https://github.com/milla-jovovich/mempalace.git`
- [x] 确认 MemPalace 的实际 LICENSE 类型：MIT（`Copyright (c) 2026 MemPalace Contributors`）

---

---

## 规划更新：改造 F — MemPalace 驱动的风险决策层升级（2026-04-10）

**性质**：修改计划文档更新（不涉及源代码变更）

### 背景

在讨论 MemPalace 时序记忆的应用场景时，评估了一个直觉设计：当向量相似度 > 0.9 时强制将 Aggressive Risk Analyst 降级为 Conservative。该设计被否决（向量相似度 ≠ 市场因果等价；绕过 LLM 推理链；作用层级错误），转而采用"证据注入而非规则强制"的设计原则。

### 核心思路

MemPalace 风险记忆库（Wing=ticker, Room="risk_context"）存储四份分析师报告的摘要文本，并在 Reflector 回调后标注实际收益。在每次风险辩论和最终决策前，查询历史相似情境，作为**结构化证据**注入 prompt，让 LLM 自己权衡。

### 三个注入点

| 注入点 | 文件 | 内容 |
|--------|------|------|
| Conservative Debator | `conservative_debator.py` | 条件性追加回撤警告段落，给保守派更多历史论据 |
| **Portfolio Manager**（核心） | `portfolio_manager.py` | 注入 "Historical Risk Context" 结构化块，含回撤警告和一般历史表现 |
| Reflector 存储 | `reflection.py` | 新增 `reflect_portfolio_manager()`，将市场情境 + 实际收益写入 MemPalace |

### 新增文件

| 文件 | 说明 |
|------|------|
| `tradingagents/agents/utils/risk_memory.py` | `RiskMemoryStore` 类，封装风险层的 MemPalace 操作 |

### 新增 config 字段

```python
"use_risk_memory": False,   # 单日测试时 False，回测/生产时 True
```

### 执行顺序

在原 Step 8 之后追加 Step 9-11（见修改计划第七节）。

---

---

## 规划更新：改造 G（批量回测）+ 改造 H（Budget Manager + 六档评级）（2026-04-10）

**性质**：修改计划文档更新（不涉及源代码变更）

### 改造 G：批量回测功能

| 项目 | 内容 |
|------|------|
| 核心新增 | `TradingAgentsGraph.run_backtest(ticker, start_date, end_date)` |
| 数据预获取 | `prefetch_for_backtest()` 一次性拉取全范围数据，日循环零网络请求 |
| 新增文件 | 无（扩展 `trading_graph.py` 和 `y_finance.py`） |
| 社交媒体限制 | 历史 Reddit/StockTwits 数据不可得，回测社交媒体仍走 Yahoo 新闻，已标注为已知限制 |
| 输出 | 每日记录 CSV + 汇总统计（Max Drawdown、Sharpe Ratio、胜率） |

### 改造 H：Budget Manager + 六档评级

**六档评级参数（已确认）：**

| 评级 | 行为 | 参数 |
|------|------|------|
| ALL IN | 部署全部剩余现金 | 要求 cash > 15% total；LLM 犹豫时选 BUY |
| BUY | `min(cash×40%, total×20%)` | 临界点 cash=50% total |
| OVERWEIGHT | `min(cash×20%, total×10%)` | BUY 的一半 |
| HOLD | 无操作 | — |
| UNDERWEIGHT | 卖出持仓的 50% | 持仓减半 |
| SELL | 清仓 | — |

最小操作门槛：`total × 0.1%`，低于此跳过操作。

**架构原则：**
- BudgetManager 为纯规则引擎（Python 类），无 LLM 调用
- Budget Context 仅注入 Portfolio Manager（辩论层保持纯公司分析）
- `budget_config.enabled=False` 时完全向后兼容现有五档流程

**新增文件：**
- `tradingagents/agents/utils/budget_manager.py`（`BudgetManager` 类）

**执行顺序：** 原 Step 11 后追加 Step 12-15（见修改计划第七节）。

---

## 改造 B 第一阶段：真实社交媒体数据源 + FinBERT 预处理（2026-04-11）

**目标**：为 Social Media Analyst 接入真实的 Reddit 和 StockTwits 散户舆情数据，用 FinBERT 批量处理原始帖子并输出结构化情绪摘要，在不升级模型等级的前提下解决上下文膨胀问题。

### 背景

原始 Social Media Analyst 实际使用的是 `get_news`（Yahoo Finance / Alpha Vantage 新闻聚合），完全没有散户舆情信号。直接将原始帖子传给 LLM 会产生约 12,000 tokens 的上下文压力，超出 27B-30B 本地模型的有效推理范围。FinBERT 聚合将其压缩至约 600 tokens。

### 数据源选型说明

Reddit 官方 API（PRAW）因 2023 年起大幅收紧审批而放弃，改用以下两个无需 key 的公开端点：

- **StockTwits**：官方公开 stream API（`api.stocktwits.com/api/2/streams/symbol/{ticker}.json`），限速 200 次/小时，完全合规
- **Reddit**：`old.reddit.com/r/{sub}/search.json` 未认证 JSON 端点，覆盖 r/wallstreetbets、r/stocks、r/investing，3 秒请求间隔，低频使用风险极低

---

### 新增文件

#### `tradingagents/agents/utils/social_data_tools.py`（新建）

**内部 fetch 函数**：
- `_fetch_stocktwits_raw(ticker, limit)` — 返回结构化 `list[dict]`，字段：`text / source / score / timestamp / native_sentiment`
- `_fetch_reddit_raw(ticker, limit)` — 同结构，覆盖三个子版块，含 3 秒请求间隔

**LangChain 工具（@tool 包装）**：
- `get_stocktwits_stream(ticker, limit)` — 调用 `_fetch_stocktwits_raw`，格式化为字符串输出
- `get_reddit_posts(ticker, limit)` — 调用 `_fetch_reddit_raw`，格式化为字符串输出

**核心处理函数**：
- `get_social_posts_cached(ticker)` — 合并两个来源，带模块级 `dict` 缓存（键为 ticker），避免 ReAct 循环中的重复请求
- `clear_posts_cache()` — 在多轮分析之间清除缓存
- `finbert_aggregate(posts, min_neutral_confidence, top_n)` — FinBERT 批量推理 + 聚合，输出约 600 token 的结构化摘要

**FinBERT 技术细节**：
- 模型：`ProsusAI/finbert`（金融领域 fine-tune，标签：positive / negative / neutral）
- 批量推理：`pipeline(batch_size=32, truncation=True, max_length=512)`
- 过滤逻辑：丢弃 `confidence < 0.65` 的 neutral 帖子（低信号噪音）
- 排序：`upvotes × confidence`，Top-N 进入摘要
- 输出格式：情绪分布统计 + 按类别的 Top-5 帖子列表
- **若 `transformers` 未安装**：自动降级为纯文本列表，系统不崩溃

**依赖**（非默认，需手动安装）：
```bash
pip install transformers torch
```

**模型存储**：HuggingFace 自动缓存至 `~/.cache/huggingface/hub/`（约 440MB），一次性下载，不进入项目目录，无需修改 `.gitignore`。

---

### 修改文件

#### `tradingagents/agents/analysts/social_media_analyst.py`

**架构变化**：社交媒体数据由节点内部直接预取，不再经过 LangGraph ToolNode。

- 节点启动时调用 `get_social_posts_cached(ticker)` + `finbert_aggregate(posts)`，将约 600 token 的 FinBERT 摘要直接嵌入 system prompt
- LLM 可调用的工具缩减为仅 `get_news`（财经新闻，作为补充来源）
- System prompt 更新：明确告知 LLM 社交媒体数据已预处理完毕，指导其综合两个来源生成报告
- 函数签名不变（`create_social_media_analyst(llm)`），上下游无需改动

**上下文压力对比**：

| | 改造前 | 改造后 |
|--|--------|--------|
| 传给 LLM 的 tokens | ~12,000（原始帖子） | ~600（FinBERT 摘要）+ 财经新闻 |
| 数据来源 | Yahoo Finance 新闻 | Reddit + StockTwits + 财经新闻 |
| 所需模型等级 | 受限于上下文 | 27B 本地模型可用 |

---

### 文档更新

#### `notes/modification_plan.md`

- 第 3.5.1 节全面重写：记录实际实现方案（Reddit JSON 端点替代 PRAW）、FinBERT 聚合流程、架构决策原则（FinBERT 为普通函数而非 LangGraph 节点）

#### `notes/usage_guide.md`

- 新增第九节「社交媒体分析依赖（FinBERT）」：包含依赖安装命令、模型存储位置说明、预下载命令、自定义缓存路径配置方法

---

---

## CLI 进度表增强：Media Labeling 行（2026-04-11）

**性质**：交互体验改进（配合改造 B 第一阶段）

### 背景

FinBERT 预处理在 `social_media_analyst_node` 内部同步执行，对 CLI 进度表完全不可见。用户在 Social Analyst 运行期间无法区分「正在抓取/标注帖子」和「LLM 正在生成报告」两个阶段。

### 修改内容

#### `tradingagents/agents/utils/social_data_tools.py`

- 新增模块级 `_finbert_status_callback` 变量（默认 `None`）
- 新增 `set_finbert_status_callback(callback)` — 供外部（CLI）注册状态回调
- 新增 `_notify_finbert_status(status)` — 内部触发辅助函数
- `finbert_aggregate()` 重构为 `try/finally` 结构：进入时调用 `_notify_finbert_status("in_progress")`，退出时（无论正常或异常）调用 `_notify_finbert_status("completed")`
- 实际推理逻辑提取为 `_finbert_aggregate_inner()`，保持可测试性
- 空帖子提前返回路径不触发 callback（无数据时不需要显示进度）

#### `cli/main.py`

- 新增 import：`from tradingagents.agents.utils.social_data_tools import set_finbert_status_callback`
- `MessageBuffer.init_for_analysis()`：当 `"social"` 在 `selected_analysts` 中时，在 `"Social Analyst"` 之前向 `agent_status` 插入 `"Media Labeling": "pending"`
- `update_display()` → `all_teams["Analyst Team"]`：在 `"Social Analyst"` 前加入 `"Media Labeling"`（过滤逻辑 `if a in message_buffer.agent_status` 保证未选 social 时自动隐藏）
- `init_for_analysis()` 调用之后：当 `"social"` 被选时注册 lambda callback，将 FinBERT 状态事件映射到 `message_buffer.update_agent_status("Media Labeling", status)`

### 进度表视觉效果

FinBERT 运行期间：
```
Analyst Team  │ Media Labeling  │ ⠋ in_progress
              │ Social Analyst  │ in_progress
```

FinBERT 完成、LLM 生成报告期间：
```
              │ Media Labeling  │ completed
              │ Social Analyst  │ ⠋ in_progress
```

未选 social analyst 时，`Media Labeling` 行不出现。

*改造 B 第二阶段（MemPalace 记忆系统接入）待讨论后继续实施。*

---

## 改造 B 第二阶段：记忆系统统一重构——BM25 → ChromaDB（2026-04-11）

**目标**：将框架中唯一实际运行的记忆机制（`FinancialSituationMemory` / BM25）替换为基于 `TradingMemoryStore`（ChromaDB + SQLite）的统一持久化记忆，并引入双运行模式（`run_mode`）作为全局记忆开关。

---

### 背景与动机

#### 原系统的根本问题

`FinancialSituationMemory` 使用 BM25（词袋算法）在进程内存中维护 5 个独立实例（bull / bear / trader / invest_judge / portfolio_manager）。这个设计在理论上合理，但在实际使用中**完全失效**，原因如下：

1. **TradingAgents 目前仅支持单日单次运行**：CLI `analyze` 命令运行一次分析后立即结束进程；`main.py` 里 `propagate()` 也只调用一次，`reflect_and_remember()` 甚至是注释掉的状态。BM25 记忆需要先有写入才能被读取，而单日运行里写入发生在 `reflect_and_remember()`（分析结束之后），下一次分析已经是全新进程，内存清空——5 个 BM25 实例**在历史上从未成功积累过任何内容**，所有 `get_memories()` 调用均返回空列表。

2. **BM25 不适合金融叙事文本**：词袋模型对「股价承压」和「估值回调」等同义表达的相似度近乎为 0；4 份报告拼接成的超长查询文本中，高频停用词主导打分，关键信号被稀释。

3. **没有持久化机制**：即使在假设的多日脚本场景下，进程重启即清空，无法跨 run 积累经验。

#### 为什么不是简单修补

在单日模式下修补 BM25（如序列化到磁盘）意义不大：单日分析本身不需要跨日记忆。真正有价值的记忆积累只在**回测模式**（连续多日循环）中才有意义，而回测模式需要一个全局开关来区分「是否真正写入磁盘」。这是从单点修复升级为架构决策的原因。

---

### 核心设计决策

#### 1. `run_mode` 替代多个布尔 flag

将 `use_sentiment_memory: bool` 和 `use_bert_preprocessing: bool` 合并为一个语义更强的枚举字段：

```python
"run_mode": "single"   # 或 "backtest"
```

- `"single"`：`TradingMemoryStore.enabled = False`，所有读写均为 no-op，无磁盘活动，行为与改造前完全一致
- `"backtest"`：`TradingMemoryStore.enabled = True`，ChromaDB + SQLite 正式写入，记忆跨日积累

`run_mode` 是比任何单个布尔 flag 更准确的表达：它不只控制记忆，将来还会控制 BudgetManager 的启用、数据预获取策略等（见改造 G、H）。

#### 2. 用 `reflections_{role}` room 替代 5 个 BM25 实例

每个角色的反思记录存入独立 room（`reflections_bull` 等），同样受 `TradingMemoryStore` 的因果隔离机制保护（`valid_from = T+1`）。检索使用 ChromaDB 向量语义搜索，取代 BM25 词频匹配。

#### 3. 不新增抽象层，直接修改

所有相关文件直接改用 `memory_store` 参数，删除旧 BM25 实例，不保留向后兼容的桥接代码。旧接口调用方式已无使用者（因为记忆从未真正工作），没有保留的必要。

---

### 修改文件清单

#### `tradingagents/default_config.py`

- 删除 `use_sentiment_memory` 和 `use_bert_preprocessing`
- 新增 `"run_mode": "single"`（默认单日模式，磁盘记忆禁用）

#### `tradingagents/agents/utils/memory_store.py`

- `_EXPIRY_DAYS` 新增 5 个 reflection rooms（`reflections_bull` / `reflections_bear` / `reflections_trader` / `reflections_invest_judge` / `reflections_portfolio_manager`），有效期 `None`（永不过期）
- 新增 `store_reflection(ticker, role, situation, recommendation)`：将 LLM 生成的反思文本写入对应 room，`valid_from = T+1`
- 新增 `retrieve_reflections(ticker, role, query, n_results=2)`：语义检索历史反思，因果隔离自动保证

#### `tradingagents/graph/reflection.py`

- `reflect_*` 方法签名由接受各自独立的 BM25 memory 实例改为统一接受 `memory_store`
- 调用 `memory_store.store_reflection()` 替代 `memory.add_situations()`
- 各方法从 `current_state` 中取 `company_of_interest` 作为 ticker

#### `tradingagents/agents/researchers/bull_researcher.py` / `bear_researcher.py`

- `create_*` 函数参数 `memory` → `memory_store`
- `memory.get_memories()` → `memory_store.retrieve_reflections(ticker, role, query)`
- 结果字段 `rec["recommendation"]` → `hit["text"]`

#### `tradingagents/agents/trader/trader.py`

- 同上，role=`"trader"`

#### `tradingagents/agents/managers/research_manager.py`

- 同上，role=`"invest_judge"`

#### `tradingagents/agents/managers/portfolio_manager.py`

- 同上，role=`"portfolio_manager"`

#### `tradingagents/graph/setup.py`

- `GraphSetup.__init__` 删除 `bull_memory` / `bear_memory` / `trader_memory` / `invest_judge_memory` / `portfolio_manager_memory` 五个参数
- 所有 `create_*` 调用统一传入 `self.memory_store`

#### `tradingagents/graph/trading_graph.py`

- 删除 `from tradingagents.agents.utils.memory import FinancialSituationMemory`
- 删除 5 个 `FinancialSituationMemory` 实例化
- `TradingMemoryStore.enabled` 由 `run_mode == "backtest"` 推导，不再读取已删除的 `use_sentiment_memory`
- `GraphSetup(...)` 调用移除 5 个 BM25 参数
- `reflect_and_remember()` 改为向所有 5 个 reflector 方法统一传入 `self.memory_store`

---

### 行为兼容性说明

- **单日模式（默认）**：`run_mode="single"` → `enabled=False` → 所有 `store_*` / `retrieve_*` 均为 no-op → 行为与改造前完全一致（agents 收到空记忆，等同于原来 BM25 也返回空列表）
- **回测模式**：`run_mode="backtest"` → 首次写入时按需初始化 ChromaDB 和 SQLite，记忆跨日积累，语义检索生效
- **`FinancialSituationMemory` 类文件保留**但不再被任何代码引用，可安全忽略

---

*改造 B 全部阶段已完成（2026-04-11）。后续优化方向（金融专用 Embedding / 时序加权检索 / 收益率反馈校准）记录于修改计划「改造 B 第三阶段」。*

---

## 改造 B 运行时 Bug 修复（2026-04-12）

**触发场景**：通过 CLI 启动后，在 Social Analyst 节点产生 `RuntimeError: TradingMemoryStore.set_analysis_date() must be called before any read or write operation.`

### Bug 1：CLI 绕过 `propagate()`，导致 `set_analysis_date()` 从未执行

**原因**：`cli/main.py` 为实现流式输出，直接调用底层 `graph.graph.stream()`，而非 `graph.propagate()`。`set_analysis_date()` 只存在于 `propagate()` 中，CLI 路径下 `_analysis_date` 始终为 `None`。

**修复**：`cli/main.py` 的 `graph.graph.stream()` 调用前插入：

```python
graph.memory_store.set_analysis_date(selections["analysis_date"])
```

### Bug 2：`_require_date()` 在 `enabled=False` 时仍被 Python 提前求值

**原因**：所有 `retrieve_*` 方法以 `as_of=self._require_date()` 的形式传参，Python 在进入 `_search_text` 之前就会先求值参数，因此即使 `_search_text` 内部有 `if not self._enabled: return []` 的保护，`_require_date()` 也会在此之前抛出异常。`store_*` 系列方法同理。

**修复**：在所有 public read/write 方法开头统一加上早返回：

```python
if not self._enabled:
    return []  # 或 return False
```

涉及方法：`store_sentiment_summary`、`store_news_summary`、`store_market_summary`、`store_fundamentals`、`store_lesson`、`store_reflection`（共 6 个写入方法）以及 `retrieve_similar_sentiment`、`retrieve_sector_sentiment`、`retrieve_lessons`、`retrieve_reflections`（共 4 个检索方法）。

**修复后行为**：`run_mode="single"` 时，所有记忆操作在方法入口处立即返回，不触碰日期检查，不产生任何磁盘活动，CLI 可正常运行。

---

## 改造 G.7.2：置信度校准回路 + 回测可视化集成（2026-04-12）

### 概述

实现置信度校准记录，并将回测结束后的校准表格输出和与传统量价策略对比的可视化图表集成进 CLI 回测流程。策略实现修正为与论文一致的多空状态机（long-short state machine）。

---

### `tradingagents/agents/utils/memory_store.py`

**新增方法 `record_calibration_point()`**

每次 `reflect_and_remember()` 调用后，将当日的预测信号、置信度、实际收益率以 JSON Lines 格式追加写入 `memory/calibration.jsonl`：

```python
def record_calibration_point(self, ticker, trade_date, signal, confidence, actual_return):
    # Appends one JSON line: {ticker, date, signal, confidence, actual_return}
```

**新增方法 `load_calibration_records()`**

读回 `calibration.jsonl` 全部记录，支持按 ticker 过滤，返回 `list[dict]`。

---

### `tradingagents/graph/trading_graph.py`

**`propagate()`**：在计算 `signal_dict` 后，缓存到 `self._last_signal_dict`，供后续 `reflect_and_remember()` 使用。

**`_log_state()`**：新增回测模式早返回，避免每日输出 JSON 文件：

```python
if self.config.get("run_mode") == "backtest":
    return
```

**`reflect_and_remember()`**：新增两步调用：

```python
# 1. 记录实际收益率到 SQLite KG
self.memory_store.annotate_return(ticker=ticker, actual_return=float(returns_losses))
# 2. 记录置信度校准数据点到 calibration.jsonl
self.memory_store.record_calibration_point(
    ticker=ticker,
    trade_date=self.memory_store._analysis_date,
    signal=signal_dict.get("signal", "HOLD"),
    confidence=signal_dict.get("confidence", 0.70),
    actual_return=float(returns_losses),
)
```

---

### `cli/main.py`

**新增 `_compute_calibration(results)`**

对回测结果列表按置信度 bucket（0.50–0.60、0.60–0.70、0.70–0.80、0.80–0.90、0.90–1.00）分组，计算每个 bucket 的方向准确率（HOLD/ERROR 排除）。使用 `observed=False` 确保空 bucket 也输出，用字典映射分配 midpoint，避免长度不匹配错误。

**新增 `_display_calibration_table(results)`**

使用 Rich 输出置信度校准表格，列：Confidence Interval、N、Actual Accuracy、Assessment。若方向性信号总数 < 50，显示数据不足警告。过偏/过低置信 bucket（实际准确率与区间中点差值 > 10pp）自动标记。

**新增 `_run_backtest_analysis(all_results, tickers, start_date, end_date, config)`**

在回测结束后，为每个 ticker 调用 `backtest_analyze._build_figure()` 生成与传统策略对比的可视化图表，保存至 `results/{TICKER}/backtest_{TICKER}_{start}_{end}_{timestamp}_analysis.png`。使用 `matplotlib.use("Agg")` 非交互式后端，适用于 CLI 和无 GUI 环境。

**`_run_backtest_mode()` 结尾**：依次调用 `_display_calibration_table()` 和 `_run_backtest_analysis()`，确保一次回测运行同时输出校准表格和对比图表。

---

### `backtest_analyze.py`（新文件，根目录）

独立脚本兼可导入模块，实现以下功能：

**传统策略基线（均为多空 {−1, +1}，B&H 始终 +1）**

| 策略 | 信号逻辑 |
|------|---------|
| Buy & Hold | 始终持有（+1） |
| SMA | 价格 > 50日均线 → +1；否则 −1 |
| MACD | MACD 线上穿信号线 → +1；下穿 → −1 |
| KDJ + RSI | KDJ J线金叉且RSI < 70 → +1；死叉且RSI > 30 → −1 |
| ZMR | 20日对数收益 z-score < −1 → +1（反弹）；> +1 → −1（回调） |

**TradingAgents 三种变体**

| 变体 | 描述 |
|------|-----|
| TA-Signal | 纯状态机：BUY/OW → +1，SELL/UW → −1，HOLD → 维持当前仓位 |
| TA-Filtered | 仅在 confidence ≥ threshold（默认 0.70）时执行信号，否则 HOLD |
| TA-Scaled | 仓位 = base_position × clip((conf − 0.50) × 2, 0, 1)，按置信度缩放 |

**指标计算**（与论文 S1.2 一致）

- CR = (V_end/V_start − 1) × 100
- AR = ((V_end/V_start)^(1/N) − 1) × 100
- SR = (mean_excess / std) × √252（日度超额收益）
- MDD = max((Peak − Trough) / Peak) × 100

**四格图表**

1. 各策略权益曲线对比
2. TA-Signal 仓位柱状图 + TA-Scaled 仓位折线叠加
3. 置信度分布直方图（按 bucket 着色）
4. 置信度校准散点图（实际准确率 vs 区间中点）

**CLI 用法**

```bash
# 传入 CSV 文件（含 date, signal, confidence, actual_return 列）
python backtest_analyze.py results/AAPL/backtest_results.csv

# Demo 模式（自动下载 AAPL 数据演示）
python backtest_analyze.py --demo --ticker AAPL

# 自定义参数
python backtest_analyze.py results.csv --threshold 0.75 --capital 10000 --output chart.png
```

---

## 改造 G.9：回测策略改为 Bounded-Stack 状态机（2026-04-13）

### 动机

改造 G.8 引入的 stacking 策略（每次信号固定金额叠加）存在无界杠杆问题：在所有信号高置信、方向相同的场景下，仓位可无限累积，导致 MDD 高达 50%，远超论文数据（MDD < 3%）。

经确认，论文 Figure 6 中并没有"不等高 cash 线"，只有价格折线和多/空入场箭头。论文策略最可能是有界状态机。

新策略引入**步进式（bounded-stack）状态机**：
- BUY/OW：state = min(state + 1, +1)
- SELL/UW：state = max(state − 1, −1)
- HOLD：不变
- 从做多 +1 转为做空 −1 需要两次 SELL 信号（先平仓至 0，再开空至 −1），反之同理

AMZN 2026-01 至 2026-04 实测对比：

| 策略 | stacking MDD | bounded-stack MDD |
|------|-------------|-------------------|
| TA-Signal | 50.39% | **19.26%** |
| TA-Filtered | 50.39% | **19.26%** |
| TA-Scaled | 14.83% | 14.83%（不变） |

### 修改文件清单

#### `tradingagents/graph/backtest_analyze.py`

**移除**：`_simulate_stacking()`、`_simulate_stacking_filtered()`、`_units_stacking()`

**新增**：
- `_positions_ta_bounded_stack(signals)` — 步进式状态机，BUY +1（上限+1），SELL −1（下限−1）
- `_positions_ta_bounded_stack_filtered(signals, confidences, threshold)` — 同上，仅在 confidence ≥ threshold 时执行

**修改**：
- `_build_figure()`：TA-Signal/Filtered 改用新函数 + `_simulate()`；移除 close_arr/price 重建逻辑；Panel B 恢复 {+1, 0, −1} 柱状图，标题注明 bounded-stack
- `_print_metrics()`：更新策略描述文字
- 模块 docstring：更新 Position logic 说明

---

## 改造 G.8：回测策略对齐论文 + 输出子目录结构（2026-04-13）

### 动机

1. **策略对齐**：原 TA-Signal / TA-Filtered 使用状态机（BUY→+1, SELL→-1），与论文 Fig.6 中 cash 线的不等高峰值行为不符。论文实际采用 **stacking 策略**：每次 BUY/SELL 执行固定金额（initial_capital × 0.8）的买入/卖出，连续相同方向的信号可叠加（允许 2×、3× 做空），导致 cash 线在 N 次连续 SELL 后升至 initial_capital + N × trade_size。
2. **输出结构**：原输出文件散落在 `results/` 根目录，改为每次回测在 `results/{TICKER}-{start_date}-{end_date}/` 下创建独立子目录，包含 `analysis.png` 和 `metrics.csv`，回测原始数据 CSV 也移入同一目录。

### 修改文件清单

#### `tradingagents/graph/backtest_analyze.py`

**移除**
- `_ta_state_machine()`、`_positions_ta_signal()`、`_positions_ta_filtered()`（状态机逻辑，不再用于 TA-Signal/Filtered）

**新增**
- `_simulate_stacking(signals, prices, initial_capital, trade_size_ratio=0.8)` — 论文对齐的 stacking 模拟，每次 BUY/SELL 固定交易 trade_size 金额，无杠杆上限，returns portfolio 时序（长度 n+1）
- `_simulate_stacking_filtered(signals, confidences, prices, initial_capital, threshold, trade_size_ratio=0.8)` — 同上，仅在 confidence ≥ threshold 时执行交易
- `_units_stacking(signals)` — 返回整数仓位单位数（+N=多N单位，-N=空N单位），用于 Panel B 可视化
- `_write_metrics_csv(values, out_path, rf_annual)` — 将性能指标（CR、AR、SR、MDD）写入 CSV

**修改**
- `_build_figure()`：
  - 返回值由 `plt.Figure` 改为 `(plt.Figure, dict)`（图表 + values 字典）
  - OHLCV 缺失时从 `actual_return` 重建价格序列（close_arr，起始价 100）
  - TA-Signal / TA-Filtered 改用 `_simulate_stacking` / `_simulate_stacking_filtered`，不再使用 `_simulate()`
  - TA-Scaled 保留置信度加权状态机（我们自己的扩展）
  - Panel B 改为显示 stacking 单位数（`units_sig`），y 轴标签改为 "Units (× trade_size)"，标题注明 "paper-aligned"
  - Panel A 标题注明两种策略来源的区别
- `main()`：
  - 默认输出路径改为 `results/{TICKER}-{start_date}-{end_date}/analysis.png`（不再 plt.show()）
  - 同时写出 `results/{TICKER}-{start_date}-{end_date}/metrics.csv`
  - `--output FILE` 仍可覆盖为任意路径

#### `cli/main.py`

- `_run_backtest_analysis()`：
  - 导入增加 `_write_metrics_csv`
  - `fig = _build_figure(...)` 改为 `fig, values = _build_figure(...)`
  - 输出路径由 `results_dir/backtest_{ticker}_{...}_analysis.png` 改为 `results_dir/{ticker}-{start_date}-{end_date}/analysis.png`
  - 同时保存 `results_dir/{ticker}-{start_date}-{end_date}/metrics.csv`
- `_run_backtest_mode()`：
  - 回测原始数据 CSV 由统一的 `backtest_{tickers_slug}_{...}.csv`（根目录）改为每个 ticker 单独保存至 `results/{ticker}-{start_date}-{end_date}/backtest_{ticker}_{timestamp}.csv`
