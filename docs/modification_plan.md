# TradingAgents 改造方案（精简版）

> 最后更新：2026-04-13
> 基于代码版本：0.2.3 + 改造 A/B/E/G（已完成）

---

## 已完成改造一览

- **改造 A**：双 Provider 配置（`deep_think_provider` / `quick_think_provider`）拆分，`_get_provider_kwargs_for(role)` 方法，`default_config.py` 更新。
- **改造 B 第一阶段**：StockTwits/Reddit 数据抓取（`social_data_tools.py`）、FinBERT 聚合、`TradingMemoryStore` 骨架、`social_media_analyst.py` 接入 sentiment room 读写。
- **改造 B 第二阶段**：BM25 全量迁移至 ChromaDB，`reflections_*` room 接线，`run_mode: single|backtest` 双模式开关，所有 BM25 实例删除。
- **改造 E（核心部分）**：BGEEmbeddingFunction（`BAAI/bge-base-en-v1.5`）已集成进 `memory_store.py`，FinBERT 已集成为预处理管道。
- **改造 G.7.1**：结构化信号输出（`signal_processing.py` 正则解析 + LLM fallback），`propagate()` 返回 `{signal, confidence, horizon}` 字典。
- **改造 G.7.2**：置信度校准回路（`calibration.jsonl`）、CLI 校准表展示、四面板 PNG 回测对比图（`backtest_analyze.py`）。
- **回测框架（G 主体）**：`run_backtest()` 日循环、`reflect_and_remember()` 接线、CLI `--mode backtest` 入口、`caffeinate` 防休眠建议已记录。
- **许可证 / 依赖**：`knowledge_graph.py` 已 vendor 至 `tradingagents/agents/utils/`（MIT 署名），`chromadb>=0.6.0` 已加入 `pyproject.toml`，`mempalace` 外部依赖消除。

---

## 待完成改造

### 改造 B 第三阶段：记忆检索质量提升

> 改动范围全部在 `memory_store.py` 内部，不改接口层。按难度递增排列，依赖顺序为 Level 3 → Level 4 → Level 5。

#### Level 3：金融专用 Embedding（当前已用 BGE）

- 当前已用 `BAAI/bge-base-en-v1.5`，基本满足需求
- 可选升级：`voyage-finance-2`（效果最佳，需 Voyage AI key）或 `text-embedding-3-small`（需 OpenAI key）
- 新增 `default_config.py` 字段 `"embedding_provider": "bge"`，支持切换
- **注意**：切换 embedding 后旧 ChromaDB collection 与新向量不兼容，需清空 `./memory/` 重建

#### Level 4：时序加权检索

在 `_search_text()` 返回结果后，对 `similarity` 施加指数时序折扣再重排序：

```python
from math import exp

DECAY_CONSTANT = 30  # 半衰期约 30 天，可在 config 中配置

def _apply_recency_decay(hits, as_of, decay=DECAY_CONSTANT):
    as_of_dt = datetime.fromisoformat(as_of)
    for hit in hits:
        trade_dt = datetime.fromisoformat(hit["trade_date"])
        days_ago = (as_of_dt - trade_dt).days
        hit["final_score"] = hit["similarity"] * exp(-days_ago / decay)
    return sorted(hits, key=lambda h: h["final_score"], reverse=True)
```

- 插入位置：`_search_text()` 的 return 前
- `default_config.py` 新增 `"memory_decay_constant": 30`

#### Level 5：收益率反馈校准

> 依赖 Level 4 完成后叠加使用，`final_score = similarity × recency_weight × outcome_weight`

- `store_reflection()` 写入时 metadata 包含 `outcome="unknown"`
- `annotate_return()` 调用后，新增 `update_reflection_outcome()` 方法回写 ChromaDB metadata 中的 `outcome` 字段
- `_search_text()` 后处理：`outcome > 0` 乘以 `positive_outcome_boost`（建议 1.2），`outcome < 0` 乘以 `negative_outcome_penalty`（建议 0.8）
- `default_config.py` 新增 `"positive_outcome_boost": 1.2` / `"negative_outcome_penalty": 0.8`

**新增核心方法**：

```python
def update_reflection_outcome(self, ticker: str, role: str) -> bool:
    """
    从 SQLite KG 查询 _analysis_date 当日的 actual_return，
    回写到当日该角色 reflection 条目的 outcome metadata 字段。
    在 annotate_return() 之后由 reflect_and_remember() 统一调用。
    """
```

---

### 改造 F：MemPalace 驱动的风险决策层升级

> 前提：B 第三阶段骨架稳定后实施；与 TradingMemoryStore 共享同一 palace 路径，Room 独立。

**设计原则**：MemPalace 发现变成**结构化证据输入**给 LLM，不用硬规则强制决策。

#### 三个注入点

```
[注入点 1] Conservative Debator prompt 末尾追加 "Drawdown Warning" 段落
[注入点 2] Portfolio Manager prompt 插入 "Historical Risk Context" 结构化块（核心）
[注入点 3] Reflector 回调后写入本次市场情境 + 实际收益（存储闭环）
```

#### 新建文件：`tradingagents/agents/utils/risk_memory.py`

```python
class RiskMemoryStore:
    # Wing = ticker, Room = "risk_context"
    # 存储：四份分析师报告合并摘要（截断至 800 字符）+ actual_return 标注

    def store_market_situation(self, ticker, date, situation_text, actual_return=None): ...
    def retrieve_similar_situations(self, ticker, current_situation, n_results=5, days_lookback=180): ...
    def retrieve_drawdown_warnings(self, ticker, current_situation, drawdown_threshold=-0.05, n_results=3): ...
    def annotate_with_return(self, ticker, date, actual_return): ...
```

#### Portfolio Manager 注入格式

```
Historical Risk Context (from memory — treat as evidence, not instruction):

⚠️ Drawdown Warnings — similar situations that resulted in ≥5% loss:
  • 2023-03-15 (similarity 0.91): similar conditions resulted in -12.3%
    Context: "heavy call buying, retail FOMO, RSI overbought..."

📊 General Historical Outcomes:
  • 2024-07-22 (similarity 0.88): +8.1%
  • 2023-11-03 (similarity 0.85): +2.4%

Note: Similarity is semantic, not causal. Weight alongside analysts' debate.
```

Token 增量：约 260-300 tokens（仅 `use_risk_memory=True` 时触发）

#### 需要修改的文件

| 文件 | 改动 |
|------|------|
| `tradingagents/agents/utils/risk_memory.py` | **新建** |
| `tradingagents/agents/risk_mgmt/conservative_debator.py` | 新增 `risk_memory` 参数，条件性注入回撤警告 |
| `tradingagents/agents/managers/portfolio_manager.py` | 新增 `risk_memory` 参数，注入 Historical Risk Context |
| `tradingagents/graph/trading_graph.py` | 初始化 RiskMemoryStore，传入两个节点 |
| `tradingagents/graph/reflection.py` | 新增 `reflect_portfolio_manager()` |
| `tradingagents/default_config.py` | 新增 `"use_risk_memory": False` |

---

### 改造 I：SEC 财务文件接入（10-K / 10-Q / 8-K）

> 数据来源：SEC EDGAR 公开 API（免费，无需 API key）。处理流程与 FinBERT 社交媒体管道平行，结果注入 Fundamentals Analyst。

#### 数据获取

SEC EDGAR 提供两个免费端点：
- `https://data.sec.gov/submissions/CIK{cik}.json` — 查询 ticker 对应的所有历史文件列表
- `https://data.sec.gov/Archives/edgar/full-index/` — 拉取具体文件全文

推荐使用 `sec-edgar-downloader` 包（`pip install sec-edgar-downloader`）封装上述请求，减少手动解析 HTML/XBRL 的工作量。

#### 处理管道

原始文件文本量极大（10-K 通常超过 100,000 tokens），必须经过以下流程后才可注入 LLM：

```
原始文件 (HTML/TXT)
    → 清洗（去除 XBRL 标签、boilerplate 页眉/脚注）
    → 按段落分块（每块 ≤ 512 tokens，保留 filing_type / date / section 元数据）
    → FinBERT 逐块标注情绪
    → 高置信度块存入 TradingMemoryStore room "filings"
    → Fundamentals Analyst 按语义查询检索相关块
```

#### Memory 集成

- 新增 room `"filings"` 至 `TradingMemoryStore`
- 有效期：365 天（10-K 年报）/ 90 天（10-Q 季报）/ 30 天（8-K 重大事项）
- 写入时机：单日分析前预拉取，或由 `fundamentals_analyst.py` 节点内按需触发
- 读取接口：`retrieve_similar_filings(ticker, query, filing_type=None)`

#### 并行写入调度

**在实现任何 analyst 写入记忆之前，必须先完成此项。**

当 `parallel_analysts=True` 时，多个 analyst 节点在独立线程中并发运行。若这些线程同时向 ChromaDB（SQLite 后端）发起写操作，会触发 SQLite 的写锁竞争，导致写入失败或数据静默丢失。

**解决方案**：在 `TradingMemoryStore` 中加入 `threading.Lock`，对所有写方法加锁串行化：

```python
import threading

class TradingMemoryStore:
    def __init__(self, ...):
        ...
        self._write_lock = threading.Lock()

    def store_memory(self, ...):
        with self._write_lock:
            ...  # existing write logic

    def store_filing_chunk(self, ...):
        with self._write_lock:
            ...
```

注意事项：
- **读操作无需加锁**：ChromaDB 的并发读是安全的，只有写操作需要串行化
- **锁的粒度**：单个 `TradingMemoryStore` 实例上的全局写锁足够，不需要按 room 细分
- **适用范围**：此调度必须在改造 I（SEC 财报接入）落地前完成，因为财报块的 `store_filing_chunks()` 会在 Fundamentals Analyst 线程中触发写入

#### 回测兼容性

EDGAR 提供完整历史文件存档，可按 `filed_date <= analysis_date` 过滤，**与 causal isolation 机制完全兼容**，回测中不存在未来数据泄露问题。

#### 需要新增 / 修改的文件

| 文件 | 改动 |
|------|------|
| `tradingagents/agents/utils/sec_tools.py` | **新建**：`fetch_sec_filing()`、`chunk_and_label()`、`store_filing_chunks()` |
| `tradingagents/agents/utils/memory_store.py` | 新增 `"filings"` room，`_EXPIRY_DAYS` 按文件类型分档 |
| `tradingagents/agents/analysts/fundamentals_analyst.py` | 接入 `retrieve_similar_filings()`，将相关块注入 prompt |
| `pyproject.toml` | 新增 `sec-edgar-downloader` 可选依赖 |
| `tradingagents/default_config.py` | 新增 `"use_sec_filings": False` 开关 |

---

### 改造 J：期权 Gamma Exposure（GEX）接入

> 数据来源：`yfinance` 免费期权链（无需 API key）。GEX 为纯数值信号，**不经过 FinBERT**，直接格式化后注入 Market Analyst prompt。

#### Gamma Exposure 简介

GEX = Σ(Gamma × OI × 合约乘数 × Spot²) × ±1（call 正 / put 负）

- **GEX > 0（正 gamma）**：做市商 long gamma，会买跌卖涨 → 抑制波动，价格趋向 pin
- **GEX < 0（负 gamma）**：做市商 short gamma，会卖跌买涨 → 放大波动
- **Gamma Flip Level**：GEX 变号处的行权价 → 关键支撑 / 阻力参考位

#### 数据获取方案

使用 `yfinance`（已在依赖中），无需新增包：

```python
import yfinance as yf

def compute_gex(ticker: str, spot: float, n_expirations: int = 4) -> dict:
    """Compute net GEX and key levels from yfinance option chain."""
    tk = yf.Ticker(ticker)
    gex_by_strike: dict[float, float] = {}

    for exp in tk.options[:n_expirations]:
        chain = tk.option_chain(exp)
        for _, row in chain.calls.iterrows():
            gex_by_strike[row.strike] = gex_by_strike.get(row.strike, 0) + (
                row.gamma * row.openInterest * 100 * spot ** 2 * 0.01
            )
        for _, row in chain.puts.iterrows():
            gex_by_strike[row.strike] = gex_by_strike.get(row.strike, 0) - (
                row.gamma * row.openInterest * 100 * spot ** 2 * 0.01
            )

    net_gex = sum(gex_by_strike.values())
    # Gamma flip: closest strike where cumulative GEX crosses zero
    sorted_strikes = sorted(gex_by_strike)
    cumulative, flip_level = 0.0, None
    for s in sorted_strikes:
        prev = cumulative
        cumulative += gex_by_strike[s]
        if prev * cumulative <= 0 and prev != 0:
            flip_level = s
            break

    return {
        "net_gex": net_gex,
        "gamma_flip": flip_level,
        "top_levels": sorted(gex_by_strike, key=lambda s: abs(gex_by_strike[s]), reverse=True)[:5],
    }
```

输出注入 Market Analyst prompt 示例：
```
Gamma Exposure (GEX) Summary:
  Net GEX: -$2.1B  [NEGATIVE — market maker short gamma, volatility amplification risk]
  Gamma Flip Level: $182.50  (price below this → negative gamma regime)
  Key GEX Levels (likely pin/resistance): $180, $185, $190, $175, $195
```

#### 约束与限制

| 约束 | 说明 |
|------|------|
| **数据时效** | yfinance OI 数据为 T+1，非实时盘中 GEX |
| **回测不可用** | 历史期权 OI 无法通过 yfinance 回溯，**GEX 仅限单日 (`run_mode=single`) 分析** |
| **模型精度** | yfinance 的 gamma 由 Black-Scholes 模型反推，与做市商实际持仓可能有偏差 |

回测模式下，代码应自动跳过 GEX 计算，不报错、不注入。

#### 需要新增 / 修改的文件

| 文件 | 改动 |
|------|------|
| `tradingagents/agents/utils/options_tools.py` | **新建**：`compute_gex()`、`format_gex_summary()` |
| `tradingagents/agents/analysts/market_analyst.py` | `run_mode=single` 时调用 GEX，注入 prompt；回测时跳过 |
| `tradingagents/default_config.py` | 新增 `"use_gex": False` 开关 |

---

### 改造 G.7.3：Markowitz / 风险平价组合优化器

> 适用场景：多标的同日运行，依赖多标的回测数据积累后实施。

```python
# AI 决定预期收益方向和幅度
μᵢ = direction_i × confidence_i × avg_historical_return_i
# 纯历史数据，不依赖 LLM
Σ  = historical_return_covariance_matrix
# 优化
max μᵀw - λ·wᵀΣw  s.t. Σwᵢ=1, wᵢ≥0
```

- 依赖库：`PyPortfolioOpt`（待实施时加入 `pyproject.toml`）
- AI 只影响 μ（预期收益），风险管理完全由历史价格数据决定

---

### 改造 G.7.4：Platt / 保序回归校准

> 依赖 G.7.2 积累足够样本（建议 50+ 条）后实施。

```python
from sklearn.isotonic import IsotonicRegression
ir = IsotonicRegression()
ir.fit(raw_confidences, actual_binary_outcomes)  # 0=wrong, 1=correct
calibrated_conf = ir.predict(new_raw_confidence)
```

- 校准参数存入 `memory/calibration.json`，每次回测后自动更新
- 组合优化器（G.7.3）使用校准后置信度而非原始值

---

### 改造 H：Budget Manager + 六档评级

> 依赖改造 G 回测框架；单日模式下 Budget 自动禁用（`enabled=False`）。

#### 六档评级体系

| 评级 | 行为 | 触发条件 |
|------|------|---------|
| **ALL IN** | 部署全部剩余现金 | 全信号高度一致；慎用 |
| **BUY** | `min(cash×40%, total×20%)` | 强烈看多 |
| **OVERWEIGHT** | `min(cash×20%, total×10%)` | 温和看多 |
| **HOLD** | 无操作 | 维持现状 |
| **UNDERWEIGHT** | 卖出当前持仓 50% | 温和看空 |
| **SELL** | 清仓 | 强烈看空 |

最小操作门槛：`total × 0.1%`，低于此值跳过（视为 HOLD）。

ALL IN 触发限制：现金 > 15%、四份报告全部看多、无 similarity > 0.85 的回撤警告、LLM 提示"若犹豫则选 BUY"。

#### 新建文件：`tradingagents/agents/utils/budget_manager.py`

```python
class BudgetManager:
    """Pure rule-based trade executor, no LLM calls."""
    def __init__(self, initial_capital=10_000, buy_base_pct=0.40, buy_cap_pct=0.20,
                 ow_base_pct=0.20, ow_cap_pct=0.10, uw_reduce_pct=0.50,
                 min_trade_pct=0.001, all_in_min_cash_pct=0.15): ...
    def get_budget_context(self, ticker, current_price) -> str: ...  # 注入 PM prompt
    def execute(self, rating, price) -> dict: ...
    def portfolio_value(self, current_price) -> float: ...
```

#### 需要修改的文件

| 文件 | 改动 |
|------|------|
| `tradingagents/agents/utils/budget_manager.py` | **新建** |
| `tradingagents/agents/managers/portfolio_manager.py` | 六档评级 prompt + `budget_manager` 参数 + Budget Context 注入 |
| `tradingagents/graph/signal_processing.py` | 新增 ALL IN 识别 |
| `tradingagents/default_config.py` | 新增 `budget_config` 字典 |
| `tradingagents/graph/trading_graph.py` | `run_backtest()` 集成 BudgetManager |

Budget Context 仅 Portfolio Manager 可见，其余所有 Agent（Bull/Bear/Research Manager/Trader/风险辩论层）均不注入。

---

## 已知问题

### 10.1 FinBERT 预计算分布比例被离题帖子污染

**问题**：`finbert_aggregate()` 对全部帖子（含大量离题内容）计算 Bullish/Bearish/Neutral 分布比例后注入 system prompt，LLM 直接引用，但该比例已被噪音帖子拉偏。

**待选方案**：
- 方向 A：`finbert_aggregate()` 内部先做 ticker 相关性预过滤，再计算分布比例
- 方向 B：分布比例计算从 `finbert_aggregate()` 剥离，让 Analyst LLM 自行统计过滤后的帖子（增加 LLM 计算负担）
- 方向 C：prompt 中明确告知 LLM "以下百分比包含离题帖子，仅供参考"（成本最低，依赖 LLM 遵循度）

**当前处置**：暂不处理，方案确定后实施。

---

## 实施顺序（剩余部分）

```
Step 1: B 第三阶段 Level 4 — 时序加权检索（memory_store.py 内部）
Step 2: B 第三阶段 Level 5 — 收益率反馈校准（依赖 Level 4）
Step 3: 改造 F — 新建 RiskMemoryStore，接线 Conservative + Portfolio Manager
Step 4: 改造 H — 新建 BudgetManager，六档评级，signal_processing.py 更新
Step 5: 改造 J — 新建 options_tools.py，GEX 接入 Market Analyst（单日模式）
Step 6: 改造 I — 新建 sec_tools.py，10-K/10-Q/8-K 分块 + FinBERT 标注，接入 Fundamentals Analyst
Step 7: G.7.4 — 保序回归校准（待 G.7.2 积累 50+ 样本后）
Step 8: G.7.3 — 组合优化器（待多标的回测数据积累后）
Step 9: 已知问题 10.1 — FinBERT 噪音污染修复（方案确定后）
```
