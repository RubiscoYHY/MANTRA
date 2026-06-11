# MANTRA GUI Development Log

## 2026-04-22 — Initial GUI Scaffold

### Overview

Created a Flask-based web GUI for MANTRA, mirroring the CLI's full configuration and analysis workflow. The GUI runs on port 5720 and is accessible via the `mantragui` console command.

### Design

- **Background**: `#0C0F0D` (deep ink green)
- **Primary accent**: `#C2A67E` (desaturated amber gold)
- **Font**: monospace stack (SF Mono / Cascadia Code / Fira Code / JetBrains Mono)
- **Layout**: single-page app — config form → live dashboard → results view

### Files Created

| File | Purpose |
|------|---------|
| `gui/__init__.py` | Package init (empty) |
| `gui/app.py` | Flask backend: routes, SSE streaming, analysis job runner |
| `gui/templates/index.html` | SPA template with inline CSS + JS |

### Files Modified

| File | Change |
|------|--------|
| `pyproject.toml` | Added `mantragui = "gui.app:main"` entry point, `flask>=3.1.0` + `python-dotenv>=1.0.0` dependencies, `gui*` in package discovery, `gui = ["templates/*", "static/*"]` in package-data |

### Architecture

```
Browser (localhost:5720)
  │
  ├─ GET /                      → index.html (SPA)
  ├─ GET /api/providers         → list LLM providers
  ├─ GET /api/models/<p>/<m>    → model options for provider+mode
  ├─ POST /api/start            → launch analysis job (returns job_id)
  └─ GET /api/stream/<job_id>   → SSE event stream
        │
        ├─ event: status    → agent pipeline states + stats
        ├─ event: message   → activity log entries
        ├─ event: tool      → tool call notifications
        ├─ event: report    → report section updates
        ├─ event: complete  → final decision + full report
        └─ event: error     → error + traceback
```

### Config Form (mirrors CLI steps 0–7)

0. **Run Mode** — single day / backtest single / backtest multi (radio pills)
1. **Ticker** — text input, label adapts for multi-stock mode
2. **Date** — single date or start/end range, toggled by run mode
3. **Output Language** — dropdown (11 languages + custom)
4. **Analyst Team** — checkboxes: Market, Social, News, Fundamentals
5. **Research Depth** — radio pills: Shallow (1) / Medium (3) / Deep (5)
6. **Analyst LLM** — provider dropdown → dynamic model dropdown; extra config for Google thinking / OpenAI reasoning
7. **Manager LLM** — same pattern; extra config includes Anthropic effort level

### Live Dashboard

- **Agent Pipeline** (left panel): grouped by team, status dots with pulse animation for in-progress
- **Activity Log** (right top): timestamped message stream, auto-scroll
- **Live Report** (right bottom): concatenated report sections as they arrive
- **Stats Bar** (bottom): elapsed time, LLM calls, tool calls, input/output tokens

### Results View

- Decision banner showing signal (BUY/SELL/HOLD/etc.), confidence %, horizon
- Full report in scrollable panel
- "New Analysis" button to return to config form

### Backend Analysis Flow

1. `POST /api/start` receives config JSON, spawns a daemon thread
2. Thread builds `TradingAgentsGraph` with the same config as CLI
3. Streams `graph.graph.stream()` chunks, pushing SSE events to a `Queue`
4. SSE endpoint (`/api/stream/<job_id>`) reads from the queue and yields events
5. Backtest mode supported via `_run_backtest_job()` path

### Entry Point

```bash
mantragui
# → Flask server on 0.0.0.0:5720
# → auto-opens browser to http://127.0.0.1:5720
```

### Status

- [x] Project scaffold and file structure
- [x] Flask backend with SSE streaming
- [x] Config form with all CLI parameters
- [x] Live dashboard with agent status, messages, reports
- [x] Results view with decision banner
- [x] `mantragui` console command registered and verified
- [x] `pip install -e .` successful, import test passed
- [ ] End-to-end test with actual analysis run
- [ ] Error handling edge cases (disconnects, missing API keys)
- [ ] Mobile / responsive layout polish
- [ ] Report markdown rendering (currently plain text)
- [ ] Backtest results visualization (charts)

---

## 2026-04-22 — Agent Pipeline Visualization & Memory Database Panel

### Overview

Enhanced the live dashboard with real-time agent execution mode visualization (parallel vs sequential) and a Memory Database status panel.

### Changes

#### `gui/templates/index.html`

**Agent Pipeline — parallel/serial visualization:**
- Analyst Team group title now shows a `PARALLEL` (green) or `SEQUENTIAL` (amber) badge
- Parallel mode: analysts wrapped in a green left-border bracket, indicating concurrent execution
- Sequential mode: analysts connected by `v` arrow connectors, indicating serial execution
- Research Team: Bull & Bear Researcher always shown in parallel bracket, Research Manager below with arrow
- Risk Management: labeled `DEBATE`, three analysts shown in sequential rotation
- Inter-group connectors (`|`) between team blocks

**Memory Database panel:**
- New card in the right column showing Reads / Writes counters
- Single-day mode: panel greyed out (`opacity: 0.4`, `pointer-events: none`) with italic message "Memory disabled in single-day mode"
- Backtest mode: panel active, counters update in real-time from SSE events

**JS state additions:**
- `parallelAnalysts` (null / true / false) — tracks execution mode from backend
- `currentRunMode` — tracks run mode for DB panel enable/disable logic
- `renderDbStats()` / `updateDbPanel()` — new rendering functions
- SSE `status` handler now processes `parallel_analysts`, `run_mode`, `db_stats` fields

#### `gui/app.py`

- Status events now include `run_mode`, `parallel_analysts`, and `db_stats` fields
- Sequential mode: when an analyst completes, the next analyst in order is automatically marked `in_progress`; when all analysts are done, Bull/Bear Researchers are marked `in_progress`
- Fixed residual `stats_handler.input_tokens` / `output_tokens` bug in the `complete` event (should be `tokens_in` / `tokens_out`)

#### `tradingagents/agents/utils/memory_store.py`

- Added `db_reads` / `db_writes` counters to `TradingMemoryStore.__init__`
- ChromaDB write (`_store_text`): increments `db_writes` after successful upsert
- ChromaDB read (`_search_text`): increments `db_reads` after successful query
- KG writes (`annotate_return`, `store_price`): increment `db_writes`
- KG reads (`get_historical_return`, `get_price`): increment `db_reads`

### Dashboard Layout Update

```
┌──────────────────────────────────────────────────────┐
│  AGENT PIPELINE (left, spans all rows)               │
│                                                      │
│  Analyst Team [PARALLEL] or [SEQUENTIAL]             │
│  ┌─── Market Analyst                                 │
│  │    Social Analyst      (parallel bracket)         │
│  │    News Analyst                                   │
│  └─── Fundamentals Analyst                           │
│    |                                                 │
│  Research Team                                       │
│  ┌─── Bull Researcher                                │
│  └─── Bear Researcher     (parallel bracket)         │
│    v                                                 │
│  Research Manager                                    │
│    |                                                 │
│  Trading Team                                        │
│    Trader                                            │
│    |                                                 │
│  Risk Management [DEBATE]                            │
│    Aggressive Analyst                                │
│    v                                                 │
│    Neutral Analyst                                   │
│    v                                                 │
│    Conservative Analyst                              │
│    |                                                 │
│  Portfolio                                           │
│    Portfolio Manager                                 │
├──────────────────────────────────────────────────────┤
│  ACTIVITY LOG (right, row 1)                         │
├──────────────────────────────────────────────────────┤
│  LIVE REPORT (right, row 2)                          │
├──────────────────────────────────────────────────────┤
│  MEMORY DATABASE (right, row 3)                      │
│    Reads: 0    Writes: 0                             │
│    [greyed out in single-day mode]                   │
├──────────────────────────────────────────────────────┤
│  Stats Bar (full width)                              │
└──────────────────────────────────────────────────────┘
```

---

## 2026-04-22 — Media Labeling 状态修复、Research Team Judge 可视化

### Overview

修复 Media Labeling 状态始终为灰色的 bug，在 Research Team 中新增 DEBATE 标签和 Judge 节点可视化（含辩论轮次显示）。

### Changes

#### `gui/templates/index.html`

**Media Labeling 状态修复：**
- 新增 `lastAgents` 变量缓存最新的完整 agent status map
- SSE `status` 事件处理中增加单 agent 更新逻辑：当收到 `{agent, state}` 格式的消息时（FinBERT 回调），合并到 `lastAgents` 并重新渲染
- 此前 FinBERT 回调发送的单 agent 更新被前端丢弃，导致 Media Labeling 永远停留在 pending（灰色）

**Research Team DEBATE 标签：**
- Research Team 标题旁新增 `DEBATE` 徽章，与 Risk Management 样式一致

**Judge 节点与辩论循环可视化：**
- `AGENT_GROUPS` 中 Research Team 新增 `Judge` 成员
- 渲染逻辑改为：Bull/Bear（并行 bracket）→ v → Judge → Research Manager
- Bull/Bear + Judge 包裹在 `.debate-loop` 容器中，右侧显示 ↻ 循环符号（`&#x21bb;`），表示 Judge 将反馈循环回 Bull/Bear
- Judge 下方显示斜体小字当前辩论轮次（`Round N`），未开始时显示 `Waiting...`
- 新增 `debateMeta` 变量，从 SSE `status` 事件的 `debate_meta` 字段获取轮次数据

**新增 CSS：**
- `.debate-loop`：amber 左边框容器，包裹辩论循环部分
- `.debate-loop-arrow`：绝对定位的循环箭头，半透明 amber
- `.debate-round-label`：斜体小字，显示辩论轮次

#### `gui/app.py`

**Judge agent 状态追踪：**
- `FIXED_AGENTS["Research Team"]` 新增 `"Judge"` 成员
- 新增 `debate_meta` 字典，追踪辩论轮次元数据
- Research Team 状态追踪逻辑中新增 Judge 状态判断：
  - `judge_history` / `judge_critique_bull` / `judge_critique_bear` 非空 → Judge `in_progress`
  - `judge_decision` 非空 → Judge `completed`
- 从 `investment_debate_state["judge_count"]` 读取当前辩论轮次，写入 `debate_meta["research_round"]`
- SSE status 事件中新增 `debate_meta` 字段

**关键变量对应关系：**
- `judge_count`（`InvestDebateState`）：后端控制迭代的实际变量，Judge 每执行一轮 +1
- `judge_iterations`（`conditional_logic.py`）：最大迭代次数，来自 `config["max_debate_rounds"]`
- `debate_meta["research_round"]`：传递给前端的轮次数，直接取自 `judge_count`

### Agent Pipeline 更新

```
Research Team [DEBATE]
┌ ┌─── Bull Researcher
│ └─── Bear Researcher     (parallel bracket)
│   v
│   Judge                                    ↻
│   Round 1                 (debate loop)
│   v
  Research Manager
```

---

## 2026-04-22 — Backtest 模式 `run_backtest` 方法缺失修复

### 问题

GUI 进入回测模式后报错：

```
AttributeError: 'TradingAgentsGraph' object has no attribute 'run_backtest'
```

`gui/app.py:488` 中 `_run_backtest_job` 调用了 `graph.run_backtest(ticker, start_date, end_date)`，但 `TradingAgentsGraph` 类上从未定义过该方法。CLI 的回测（`cli/main.py:_run_backtest_mode`）是在外部手动循环 `propagate()` 实现的，GUI 端却假设图对象自身提供了这个接口。

### 修复

在 `tradingagents/graph/trading_graph.py` 的 `TradingAgentsGraph` 类中新增 `run_backtest` 方法，封装与 CLI 一致的逐日回测逻辑：

1. 用 `yfinance` 获取 `[start_date, end_date]` 区间内的实际交易日列表（fallback 为 `pd.bdate_range`）
2. 逐日调用 `propagate(ticker, trade_date)` 获取信号
3. 从 `BacktestDataCache` 获取次日收益率（`cache.get_next_day_return(trade_date)`）
4. 调用 `reflect_and_remember(actual_return)` 进行反思学习
5. 异常日标记为 `signal="ERROR"`，不中断整体流程
6. 返回 `(results, results)` 元组，匹配 GUI 端 `results, _ = graph.run_backtest(...)` 的解包格式
7. 循环结束后调用 `cache.clear()` 清理缓存

### 修改文件

| File | Change |
|------|--------|
| `tradingagents/graph/trading_graph.py` | 新增 `run_backtest(ticker, start_date, end_date)` 方法 |

### 备注

- GUI 的 `_run_backtest_job` 在调用 `run_backtest` 之前已经通过 `cache.initialize(ticker, start_date, end_date)` 预加载了缓存，`run_backtest` 内部不重复初始化，直接使用已激活的缓存实例
- 返回的 `results` 列表中每个 dict 包含：`ticker`, `date`, `signal`, `confidence`, `horizon`, `actual_return`
