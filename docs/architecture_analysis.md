# TradingAgents 项目架构分析笔记

> 分析日期：2026-04-07
> 版本：0.2.3 | License：Apache 2.0 | Python：3.10+

---

## 一、项目简介

TradingAgents 是一个基于多智能体 LLM 的金融交易分析框架，由多个专职 Agent 协作对股票进行综合分析，最终输出五档交易信号（BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL）。对应 arXiv 论文：2412.20138。

---

## 二、目录结构

```
TradingAgents/
├── tradingagents/              # 主包
│   ├── agents/                 # 所有 Agent 实现
│   │   ├── analysts/           # 分析师（数据采集层）
│   │   ├── researchers/        # 研究员（多空辩论层）
│   │   ├── managers/           # 管理层（裁决决策）
│   │   ├── trader/             # 交易员
│   │   ├── risk_mgmt/          # 风险管理辩论层
│   │   └── utils/              # 工具函数、状态定义、记忆系统
│   ├── graph/                  # LangGraph 编排层
│   ├── dataflows/              # 数据源对接层
│   ├── llm_clients/            # 多 LLM 提供商适配层
│   └── default_config.py       # 默认配置
├── cli/                        # 命令行界面（Typer + Rich）
├── tests/                      # 测试套件
├── main.py                     # Python API 示例入口
├── docker-compose.yml
└── .env.example
```

---

## 三、核心模块详解

### 3.1 LLM 客户端层 (`llm_clients/`)

工厂模式，统一对接多家 LLM 提供商：

| 提供商 | 代表模型 | 环境变量 | 特殊参数 |
|--------|----------|----------|----------|
| OpenAI | GPT-5.4 / mini | `OPENAI_API_KEY` | `reasoning_effort` |
| Anthropic | Claude Sonnet/Opus | `ANTHROPIC_API_KEY` | `anthropic_effort` |
| Google | Gemini | `GOOGLE_API_KEY` | `google_thinking_level` |
| xAI | Grok | `XAI_API_KEY` | — |
| OpenRouter | 多模型代理 | `OPENROUTER_API_KEY` | — |
| Ollama | 本地模型 | 无需 key | — |

- `BaseLLMClient`：抽象基类，做模型名称校验和归一化
- `create_llm_client()`：工厂函数，根据配置返回对应客户端
- 框架区分 `deep_think_llm`（复杂推理）和 `quick_think_llm`（快速任务）

---

### 3.2 数据流层 (`dataflows/`)

策略模式，按类别路由到不同数据供应商：

| 类别 | 可选供应商 | 说明 |
|------|-----------|------|
| `core_stock_apis` | yfinance / alpha_vantage | OHLCV 行情数据 |
| `technical_indicators` | yfinance / alpha_vantage | 技术指标 |
| `fundamental_data` | yfinance / alpha_vantage | 财务报表 |
| `news_data` | yfinance / alpha_vantage | 新闻、宏观数据 |

- `interface.py`：统一路由入口，支持类别级和工具级（`tool_vendors`）双层覆盖
- Alpha Vantage 限速 5次/分钟，有自动 fallback 机制
- 本地缓存目录：`dataflows/data_cache/`

**可用工具函数**：`get_stock_data`, `get_indicators`（SMA/EMA/MACD/RSI/Bollinger/ATR等15+）, `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, `get_income_statement`, `get_news`, `get_global_news`

---

### 3.3 Agent 层 (`agents/`) — 五类角色

#### 分析师（数据采集，并行）

| Agent | 输出 | 工具 |
|-------|------|------|
| Market Analyst | `market_report` | 技术指标（最多8个互补指标） |
| Social Media Analyst | `sentiment_report` | 公司新闻 |
| News Analyst | `news_report` | 全球新闻 + 宏观数据 |
| Fundamentals Analyst | `fundamentals_report` | 财务三表 |

分析师可按需选择，未选的会被跳过。

---

#### 分析师数据来源详解

> 分析日期：2026-04-10 | 源码路径：`tradingagents/agents/analysts/` + `tradingagents/dataflows/`

所有工具均通过统一路由器 `route_to_vendor()`（`dataflows/interface.py`）分发，默认走 **yfinance**，可按分类或按工具粒度切换到 **Alpha Vantage** 作为备用。`AlphaVantageRateLimitError` 触发时自动降级，其他异常直接抛出。

```python
# default_config.py — 默认全部走 yfinance
"data_vendors": {
    "core_stock_apis":       "yfinance",
    "technical_indicators":  "yfinance",
    "fundamental_data":      "yfinance",
    "news_data":             "yfinance",
}
```

---

##### Market Analyst

**绑定工具**（`market_analyst.py`）：`get_stock_data`、`get_indicators`

**`get_stock_data(symbol, start_date, end_date)`**

| 项目 | 内容 |
|------|------|
| 数据来源 | **Yahoo Finance**（`yfinance.Ticker.history()`）/ Alpha Vantage `TIME_SERIES_DAILY_ADJUSTED` |
| 获取手段 | yfinance 库（非官方逆向 Yahoo Finance 接口）|
| 返回内容 | OHLCV 日线数据，CSV 字符串 |
| 限速处理 | `yf_retry()`：指数退避重试，最多 3 次，初始延迟 2 秒 |
| 缓存 | 无 |

**`get_indicators(symbol, indicator, curr_date, look_back_days)`**

| 项目 | 内容 |
|------|------|
| 数据来源 | **Yahoo Finance**（同上）+ `stockstats` 本地计算 / Alpha Vantage 各指标专用端点 |
| 获取手段 | 先拉取 5 年 OHLCV，再用 `stockstats.wrap(df)[indicator]` 在本地计算 |
| 支持指标 | `close_50_sma`、`close_200_sma`、`close_10_ema`、`macd`/`macds`/`macdh`、`rsi`、`boll`/`boll_ub`/`boll_lb`、`atr`、`vwma`（共 12 个） |
| 返回内容 | 指定日期范围内的指标时间序列，CSV 字符串 |
| 缓存 | **5 年 OHLCV 本地缓存**：`data_cache/{symbol}-YFin-data-*.csv`，避免重复拉取 |
| 防未来偏差 | `data[data["Date"] <= curr_date]`：严格过滤，只保留截止日期前的数据 |

**评价**：yfinance + stockstats 的组合合理高效——价格数据从 Yahoo Finance 免费获取，指标全部本地计算（无额外 API 调用）。5 年 OHLCV 缓存设计良好，回测时避免重复网络请求。主要风险是 yfinance 依赖非官方逆向接口，Yahoo 随时可能更改，且历史数据质量（复权处理）偶有问题。Alpha Vantage 备用链路中 `vwma` 不支持，是一个已知缺口。

---

##### Social Media Analyst

**绑定工具**（`social_media_analyst.py`）：`get_news`

**`get_news(ticker, start_date, end_date)`**

| 项目 | 内容 |
|------|------|
| 数据来源 | **Yahoo Finance**（`yfinance.Ticker.get_news(count=20)`）/ Alpha Vantage `NEWS_SENTIMENT` 端点 |
| 获取手段 | yfinance 库拉取与该 ticker 关联的聚合新闻 |
| 返回内容 | 新闻标题、摘要、发布者、链接、发布日期，Markdown 格式 |
| 日期过滤 | 按 `start_date`/`end_date` 过滤，最多 20 条 |
| 缓存 | 无 |
| 实质来源 | Yahoo Finance 新闻聚合器（路透社、MarketWatch、Benzinga 等主流财经媒体） |

**评价**：⚠️ **这是本项目最大的数据质量问题。** "Social Media Analyst"的名称具有严重误导性——它获取的根本不是社交媒体数据，而是经过编辑过滤的主流财经媒体新闻聚合，与 Reddit/StockTwits 的散户情绪信号完全不同。具体缺陷：
1. Yahoo Finance 新闻 **无法体现散户情绪**（WSB 的 FOMO、期权 YOLO 等 Alpha 信号完全缺失）
2. 每次最多拉取 **20 条**，且无法控制来源多样性
3. **无情感标签**，LLM 需要自己从新闻标题推断情绪，准确率低
4. 数据与 News Analyst 的来源高度重叠（同样是 Yahoo Finance 新闻），两个 Agent 存在信息重复
5. 完全没有接入任何真实社交平台（Reddit、Twitter/X、StockTwits）

这正是改造 B（接入 Reddit PRAW + StockTwits API + BERT 预处理 + MemPalace）的核心动机。

---

##### News Analyst

**绑定工具**（`news_analyst.py`）：`get_news`、`get_global_news`

**`get_news`**：同 Social Media Analyst（见上文）

**`get_global_news(curr_date, look_back_days=7, limit=5)`**

| 项目 | 内容 |
|------|------|
| 数据来源 | **Yahoo Finance**（`yfinance.Search()` 多关键词搜索）/ Alpha Vantage `NEWS_SENTIMENT`（主题过滤） |
| 获取手段 | 用 4 个固定宏观关键词并发搜索，结果按日期过滤、去重后合并 |
| 固定搜索词 | `"stock market economy"`、`"Federal Reserve interest rates"`、`"inflation economic outlook"`、`"global markets trading"` |
| 返回内容 | 宏观新闻标题 + 摘要，Markdown 格式，最多 `limit` 条 |
| 去重 | 按标题去重（同一篇文章被多个关键词命中时只保留一条） |
| 防未来偏差 | 跳过发布日期晚于 `curr_date` 的文章 |
| Alpha Vantage 主题过滤 | `financial_markets,economy_macro,economy_monetary` |

**评价**：宏观新闻的覆盖逻辑基本合理，但固定搜索词是硬编码，无法覆盖突发事件（如地缘政治冲击、行业监管新政）。每次只返回最多 5 条全球新闻，信息量偏少。Alpha Vantage 备用链路的主题过滤更精准（有专用分类词），但受限于 5 次/分钟的免费配额。整体而言，对于宏观背景感知任务，这个实现勉强够用，但精度有限。

---

##### Fundamentals Analyst

**绑定工具**（`fundamentals_analyst.py`）：`get_fundamentals`、`get_balance_sheet`、`get_cashflow`、`get_income_statement`

所有工具均从 **Yahoo Finance**（默认）或 **Alpha Vantage** 获取数据，全部有 **防未来偏差** 处理。

| 工具 | yfinance 调用 | AV 端点 | 返回内容 |
|------|-------------|---------|---------|
| `get_fundamentals` | `yf.Ticker.info`（29 个字段） | `OVERVIEW` | PE、EPS、ROE、市值等估值与盈利指标 |
| `get_balance_sheet` | `.quarterly_balance_sheet` / `.balance_sheet` | `BALANCE_SHEET` | 资产负债表（季度/年度） |
| `get_cashflow` | `.quarterly_cashflow` / `.cashflow` | `CASH_FLOW` | 现金流量表（季度/年度） |
| `get_income_statement` | `.quarterly_income_stmt` / `.income_stmt` | `INCOME_STATEMENT` | 利润表（季度/年度） |

**防未来偏差实现**：`filter_financials_by_date(data, curr_date)` 删除所有报告日期晚于 `curr_date` 的列，确保回测时不会使用未来财务数据。

**评价**：这是四个 Analyst 中**数据质量最高、实现最合理**的一个。yfinance 财务数据来自 Yahoo Finance 对 SEC 报告的结构化抓取，覆盖全面（三表 + 29 个关键财务比率），防未来偏差机制也到位。主要缺陷：yfinance 的财务数据偶有延迟（有时比 SEC EDGAR 晚 1-2 天），且非美股（港股、A 股）的覆盖质量参差不齐。

---

##### 数据来源总览与评价

| Analyst | 工具数 | 实际数据源 | 信息类型 | 质量评价 |
|---------|--------|-----------|---------|---------|
| Market | 2 | Yahoo Finance（价格）+ 本地计算（指标） | OHLCV + 12 种技术指标 | ★★★★☆ 免费高效，非官方接口有稳定性风险 |
| Social Media | 1 | Yahoo Finance 新闻聚合 | 财经媒体新闻（**非**社交媒体） | ★★☆☆☆ 名不副实，缺乏真实散户情绪信号 |
| News | 2 | Yahoo Finance 新闻搜索 | 宏观财经新闻 | ★★★☆☆ 覆盖面有限，固定关键词偏窄 |
| Fundamentals | 4 | Yahoo Finance SEC 财务数据 | 三表 + 29 个财务比率 | ★★★★☆ 覆盖全面，防未来偏差到位 |

**共同弱点**：四个 Analyst 的数据来源**高度集中于 Yahoo Finance 单一平台**，存在单点失效风险。yfinance 是非官方逆向接口，历史上已有多次因 Yahoo 接口变更导致的破坏性更新。Alpha Vantage 作为备用链路的免费配额（5 次/分钟）过于有限，实际上难以承担回退的全部负载。

#### 研究员（投资辩论）

- **Bull Researcher**：主张买入，强调成长潜力，通过 BM25 检索历史记忆
- **Bear Researcher**：主张谨慎/卖出，强调风险，同样利用历史记忆
- 多空轮番反驳，轮次可配置（`max_debate_rounds`）

#### 研究经理（裁决）

- `Research Manager`：裁定多空辩论，输出 `investment_plan`（Buy/Sell/Hold + 具体计划）

#### 交易员

- `Trader`：接收 `investment_plan`，输出具体交易建议 `trader_investment_plan`
- 同样利用历史记忆优化决策

#### 风险管理（风险辩论）

| Agent | 立场 |
|-------|------|
| Aggressive Analyst | 高收益高风险 |
| Conservative Analyst | 资本保护、控制下行 |
| Neutral Analyst | 平衡客观 |
| Portfolio Manager | 最终裁决，五档评级输出 |

风险辩论轮次由 `max_risk_discuss_rounds` 控制。

---

### 3.4 图编排层 (`graph/`)

核心使用 **LangGraph StateGraph** 构建有状态的 Agent 工作流：

| 文件 | 职责 |
|------|------|
| `trading_graph.py` | 主编排类 `TradingAgentsGraph` |
| `setup.py` | 构建 StateGraph，动态注册节点和条件边 |
| `propagation.py` | 初始化 Agent 状态（`Propagator`） |
| `conditional_logic.py` | 流程控制：判断是否继续 tool call、辩论是否结束 |
| `reflection.py` | 交易后反思，更新各 Agent 记忆 |
| `signal_processing.py` | 从完整报告中提取核心信号 |

**`TradingAgentsGraph` 主要方法**：
- `propagate(ticker, date)` → 返回 `(final_state, processed_signal)`
- `reflect_and_remember(returns_losses)` → 用交易结果更新记忆
- `process_signal(full_decision)` → 提取 BUY/HOLD/SELL 等

---

### 3.5 记忆系统 (`agents/utils/memory.py`)

- **BM25 检索**（`rank-bm25` 库），纯词法相似度，无需 embedding API，完全离线
- 每个 Agent 维护独立记忆库：Bull / Bear / Trader / Research Manager / Portfolio Manager
- 决策前检索历史相似情境，决策后通过 `Reflector` 写入新记忆
- 无 token 限制，适合长文本存储

---

### 3.6 状态定义 (`agents/utils/agent_states.py`)

三个核心 TypedDict：
- `AgentState`：主状态（包含所有报告、最终决策、消息列表）
- `InvestDebateState`：多空辩论子状态（bull_history / bear_history）
- `RiskDebateState`：风险管理辩论子状态

---

## 四、完整工作流程

```
初始状态（ticker + date）
    ↓
[并行分析师阶段]
├── Market Analyst → market_report
├── Social Media Analyst → sentiment_report
├── News Analyst → news_report
└── Fundamentals Analyst → fundamentals_report
    ↓
[投资辩论阶段]
Bull ↔ Bear（多轮）→ Research Manager 裁决 → investment_plan
    ↓
[交易员决策]
Trader → trader_investment_plan
    ↓
[风险管理辩论阶段]
Aggressive ↔ Conservative ↔ Neutral（多轮）→ Portfolio Manager 裁决
    ↓
[信号处理]
final_trade_decision → 提取 BUY/OVERWEIGHT/HOLD/UNDERWEIGHT/SELL
```

**关键实现细节**：
- Agent 之间通过 LangGraph 的 StateGraph 消息传递
- 每个分析师阶段结束后清除消息（防止 token 膨胀），保留状态字段
- Tool 调用通过 `llm.bind_tools(tools)` 绑定，条件边判断是否还有待执行 tool call

---

## 五、配置参数速查

```python
DEFAULT_CONFIG = {
    "llm_provider": "openai",              # 提供商选择
    "deep_think_llm": "gpt-5.4",           # 深度推理模型
    "quick_think_llm": "gpt-5.4-mini",     # 快速任务模型
    "backend_url": "https://api.openai.com/v1",
    
    "google_thinking_level": None,         # "high" / "minimal"
    "openai_reasoning_effort": None,       # "high" / "medium" / "low"
    "anthropic_effort": None,              # "high" / "medium" / "low"
    
    "output_language": "English",
    
    "max_debate_rounds": 1,                # 投资辩论轮次
    "max_risk_discuss_rounds": 1,          # 风险辩论轮次
    "max_recur_limit": 100,
    
    "data_vendors": {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    },
    "tool_vendors": {},                    # 工具级覆盖（优先级更高）
}
```

---

## 六、启动方式

### CLI 交互界面
```bash
tradingagents
# 或
python -m cli.main
```
基于 Typer + Rich，支持交互式选择 ticker、日期、LLM、分析师组合，实时显示 Agent 进度。

### Python API
```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["deep_think_llm"] = "claude-sonnet-4-6"
config["llm_provider"] = "anthropic"

ta = TradingAgentsGraph(debug=True, config=config)
final_state, decision = ta.propagate("NVDA", "2026-01-15")
print(decision)  # BUY / SELL / HOLD 等
```

### Docker
```bash
cp .env.example .env
docker compose run --rm tradingagents
```

---

## 七、主要依赖

| 库 | 用途 |
|----|------|
| `langchain` / `langgraph` | LLM 框架 + Agent 编排 |
| `langchain-openai/anthropic/google-genai` | 各 LLM 提供商集成 |
| `yfinance` | Yahoo Finance 数据 |
| `rank-bm25` | Agent 记忆检索 |
| `stockstats` | 技术指标计算 |
| `typer` + `rich` | CLI 界面 |
| `backtrader` | 回测框架 |
| `redis` | 可选缓存后端 |
| `pandas` | 数据处理 |

---

## 八、架构设计模式

| 模式 | 应用位置 |
|------|---------|
| 工厂模式 | LLM 客户端创建（`create_llm_client`） |
| 策略模式 | 数据供应商路由（`interface.py`） |
| 状态机 | LangGraph StateGraph 管理 Agent 转换 |
| 装饰器模式 | `@tool` 包装数据检索函数 |
| 仓储模式 | BM25 记忆系统 |
| 观察者模式 | LLM/Tool 执行回调 |

---

## 九、改造方案（详见专项文档）

> 详细实施方案已移至 `notes/modification_plan.md`，包含逐文件、逐行号的完整改造说明。
> 以下为摘要索引。

## 九、改造方案：社交媒体舆情 Alpha + MemPalace 记忆系统

> 讨论日期：2026-04-09

### 9.1 现状问题

- `Social Media Analyst` 的数据源实为 `yfinance`/`alpha_vantage` 的新闻聚合，**不是真实社交媒体数据**
- 现有 BM25 记忆系统无时间维度、无跨标的关联、只存 LLM 反思摘要（损失原始信号）
- BetaFish（GPL-2.0）不适合集成：许可证与 Apache 2.0 冲突，且无明确 Reddit 支持

### 9.2 推荐方案（方案 A：最小侵入式）

**核心思路**：增强现有 `Social Media Analyst` 节点，而非新增节点，`sentiment_report` 字段作为天然接入点。

```
[增强后的 Social Media Analyst]
    ├─ Step 1: 从真实社交媒体 API 采集数据
    │           推荐：Reddit PRAW (r/wallstreetbets, r/stocks)
    │                StockTwits API（cashtag 过滤，专为股票设计）
    │
    ├─ Step 2: 查询 MemPalace（ChromaDB 向量检索 + SQLite 时序知识图谱）
    │           → 检索该 ticker 历史舆情模式
    │           → 检索相关板块传染效应记录（Tunnels 跨标的关联）
    │
    ├─ Step 3: LLM 综合生成 sentiment_report
    │           （当前原始数据 + MemPalace 历史模式）
    │
    └─ Step 4: 将本次原始社交数据 verbatim 写入 MemPalace
               （由 Reflector 回调触发，附加实际收益标注）
```

**下游无需改动**：Bull/Bear/Trader 已在读 `state["sentiment_report"]`，Reflector 也包含它——改善 `sentiment_report` 内容，辩论自动受益。

### 9.3 Token 消耗影响

| 方案 | 增量 token | 占比 |
|------|-----------|------|
| 方案 A（增强现有节点） | ~3,000 tokens | +5-8% |
| 方案 B（新增独立节点，不推荐） | ~10,000-12,000 tokens | +20-30% |

**MemPalace 本身不消耗 LLM tokens**（ChromaDB + SQLite 本地操作），只有注入 prompt 的检索结果才计费。

### 9.4 MemPalace 相比 BM25 的核心优势（针对社交媒体场景）

| 维度 | BM25（现有） | MemPalace |
|------|------------|-----------|
| 时序感知 | 无 | SQLite validity windows，信号有衰减 |
| 原始数据保真 | LLM 二次加工摘要 | Verbatim 存储，原文不经提取 |
| 跨标的关联 | 无 | Tunnels 跨 Wing/Room 交叉引用 |
| 检索语义能力 | 词法匹配（俚语/缩写失效） | 向量检索，R@10: 94.8% |

### 9.5 数据源推荐

- **Reddit PRAW**：Python 官方 API，MIT 许可，结构清晰，r/wallstreetbets 是核心数据源
- **StockTwits API**：专为股票设计，有 cashtag（如 `$NVDA`）原生过滤
- **不推荐 BettaFish**：GPL-2.0 许可证冲突，无明确 Reddit 支持，是独立系统非库

### 9.6 双 LLM 提供商配置方案

> 见第十节

---

## 十、双 Provider 配置：仅使用 Claude + Gemini

### 10.1 现状限制

`trading_graph.py:81-95` 中，`deep_client` 和 `quick_client` 共用同一个 `llm_provider` 字段——原生不支持两个不同 provider。

`_get_provider_kwargs()` 方法也只读单一 provider 的参数。

### 10.2 最小代码改动方案

**在 `default_config.py` 新增两个字段**（`deep_think_provider` / `quick_think_provider`），替代单一的 `llm_provider`：

```python
# default_config.py 新增
"deep_think_provider": "anthropic",     # Claude 做深度推理
"quick_think_provider": "google",       # Gemini 做快速任务
"deep_think_llm": "claude-opus-4-6",
"quick_think_llm": "gemini-2.0-flash",
```

**在 `trading_graph.py` 修改 `__init__`**（约第 81-95 行），将：

```python
deep_client = create_llm_client(
    provider=self.config["llm_provider"],
    model=self.config["deep_think_llm"],
    **llm_kwargs,
)
quick_client = create_llm_client(
    provider=self.config["llm_provider"],
    model=self.config["quick_think_llm"],
    **llm_kwargs,
)
```

改为：

```python
deep_client = create_llm_client(
    provider=self.config.get("deep_think_provider", self.config["llm_provider"]),
    model=self.config["deep_think_llm"],
    base_url=self.config.get("backend_url"),
    **self._get_provider_kwargs_for("deep"),
)
quick_client = create_llm_client(
    provider=self.config.get("quick_think_provider", self.config["llm_provider"]),
    model=self.config["quick_think_llm"],
    base_url=self.config.get("backend_url"),
    **self._get_provider_kwargs_for("quick"),
)
```

同时将 `_get_provider_kwargs()` 拆分为接受 `role` 参数的版本，分别读各自 provider 的特定参数。

### 10.3 所需环境变量（仅两个）

```bash
export ANTHROPIC_API_KEY=...    # Claude（deep_think 用）
export GOOGLE_API_KEY=...       # Gemini（quick_think 用）
```

其余 OpenAI / xAI / OpenRouter / Alpha Vantage 均不需要（除非启用 Alpha Vantage 数据源）。

### 10.4 角色分配建议

| 角色 | LLM | 理由 |
|------|-----|------|
| Research Manager（裁定多空辩论） | Claude Opus | 需要最强推理，这是核心决策节点 |
| Portfolio Manager（最终五档评级） | Claude Opus | 同上 |
| 4 个分析师 | Gemini Flash | 快速、大量 tool call，Flash 延迟低成本低 |
| Bull/Bear Researcher | Gemini Flash | 辩论轮次多，用 Flash 控成本 |
| Trader | Gemini Flash | 执行型决策，不需要最强推理 |
| Reflector / SignalProcessor | Gemini Flash | 结构化提取任务，Flash 足够 |

`factory.py` 已原生支持 `anthropic` 和 `google` 两个 provider，无需新增客户端类。
