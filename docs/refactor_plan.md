# Mantra 重构计划

> 生成日期：2026-06-11
> **执行状态：Phase 0-4、6 已于 2026-06-11 完成**（Phase 2.5 News Lake 与 Phase 5 10-K RAG 为新功能，另行实施）。
> 全部修复经 26 项测试 + 33 子测试验证，并通过 haiku/sonnet 单日端到端冒烟（AAPL 2026-06-10，信号 HOLD/0.68 正常提取）。
>
> 实施中新发现并一并修复的问题：
> 1. **Chroma 距离空间错配**：collection 默认 L2 距离，而 `similarity = 1 - distance` 的换算假设 cosine——旧代码的相似度数值量纲从一开始就是错的。已显式改为 cosine 空间（`hnsw:space: "cosine"`）。
> 2. `route_to_vendor` 的限速 fallback 链会静默降级到 yfinance 新闻——回测守卫现同时拦截主选与 fallback 两条路径。
> 3. 反思的 LLM 上下文与嵌入键分离：反思 LLM 仍读全文报告（质量），仅嵌入 digest（检索精度），两者由 `build_situation_digest()` 统一。
> 4. 3.11（CLI 改用 propagate 暴露回调）经评估**有意搁置**：现行 stream 路径已含 `set_analysis_date` 修复，重构收益不抵 UI 回归风险。
> 所有 bug 均经过亲自代码验证或实际运行复现，标注了证据位置。子代理报告中被证伪的指控已剔除（见附录 B）。

---

## 0. 本次已完成的清理（无需 Opus 执行）

| 操作 | 内容 |
|---|---|
| 删除 | `build/`、`tradingagents.egg-info/`（构建产物） |
| 删除 | `results/`、`reports/`、`memory/`（运行记录与运行期数据库，原件保留在 `../Mantra/`） |
| 删除 | `experiments/`（实验脚本与数据，原件保留在 `../Mantra/`） |
| 删除 | `third_party/mempalace/`（整个目录运行时零引用，见 1.3 节说明） |
| 删除 | `tradingagents/dataflows/data_cache/`（行情 CSV 缓存） |
| 删除 | `test.py`（yfinance 草稿）、`requirements.txt`（内容只有一个 "."，pyproject.toml 才是真依赖清单） |
| 删除 | 所有 `.DS_Store`、`__pycache__`、LaTeX aux 文件 |
| 移动 | `notes/` 中 9 份设计文档 → `docs/`（进入 git 跟踪）；课程报告/演示/LaTeX 留在 `notes/`（仍被 gitignore） |
| 整理 | `.gitignore` 去重，新增 `experiments/`、`*.rtf` |

清理后验证：核心模块全部正常 import，`pytest tests/` 6 passed + 32 subtests。

**重要说明（关于 mempalace）**：`tradingagents/agents/utils/knowledge_graph.py` 是从 mempalace 完整 vendor 进来的本地副本（文件头有出处声明）；它第 22 行的 `from mempalace...` 只是 docstring 里的用法示例，不是真实 import。全项目对 `third_party/mempalace` 的运行时依赖为零，故整目录删除。`THIRD_PARTY_LICENSES.md` 保留（vendored 文件仍需署名）。

---

## 1. Critical —— 阻断核心目标的 bug

### 1.1 ChromaDB 语义检索在当前环境下 100% 静默失败 ⚠️ 最高优先级

- **位置**：`tradingagents/agents/utils/memory_store.py:65-96`（`_BGEEmbeddingFunction`）、`:343-347`（异常吞没）
- **证据**：实际运行复现。chromadb 1.5.7 在 `validate_embedding_function_conflict_on_get` 中调用 `embedding_function.name()`，自定义 EF 类没有实现该接口 → `AttributeError` → 被 `_search_text()` 的 `except Exception: return []` 吞掉，只留一条 `logger.warning`。
- **后果**：所有 `retrieve_reflections / retrieve_similar_sentiment / retrieve_lessons` 永远返回空列表。**写入成功、读取全失败**——这正是"能存能检索但不确定检索出什么"疑虑的答案。
- **连带影响**：如果四月份的回测实验也在此 chromadb 版本下运行，那么整个回测期间记忆系统从未实际生效，新旧架构对比实验中"记忆"变量可能是无效的。建议重构后重跑一组对照。
- **修复步骤**：
  1. 弃用 chroma 的 embedding_function 挂载机制：写入时自行调用 `self._ef(documents)` 得到向量，用 `col.upsert(embeddings=..., documents=..., ...)` 显式传入；创建 collection 时不传 EF。查询路径本来就用 `query_embeddings`，不受影响。这样彻底与 chroma 的 EF 接口演化解耦。
  2. 把 `_search_text` / `_store_text` 的异常处理从"吞掉返回空"改为：backtest 模式下直接 raise（快速失败），single 模式下 `logger.error`（而非 warning）。
  3. 新增回归测试 `tests/test_memory_store.py`：写入两条 reflection → 检索命中、因果隔离（当日不可见）、跨 ticker 隔离三个断言。本次会话已验证因果隔离和 wing 过滤逻辑本身是正确的，问题只在 chroma 接口层。

### 1.2 yfinance 新闻完全不支持历史日期 —— "提取的不是我想要的日期"的根源

- **位置**：`tradingagents/dataflows/yfinance_news.py:75`（`stock.get_news(count=20)`）、`:113-209`（`yf.Search` 全局新闻）；`tradingagents/default_config.py:60`（`news_data: "yfinance"` 是默认值！）
- **证据**：代码确认。`get_news` 没有任何日期参数，永远返回**最近** 20 条；`start_date/end_date` 只做事后过滤。`yf.Search` 同样只有"现在"。
- **后果**：
  - 回测历史日期：20 条近期新闻全被过滤 → "No news found" → 新闻分析师在整个回测中基本盲跑；
  - 当前日期分析：过滤窗口 `end_dt + 1天` 有一天的边界泄漏（`yfinance_news.py:93`）。
- **修复步骤**：
  1. 把 `default_config.py` 的 `news_data` 默认改为 `alpha_vantage`（其 NEWS_SENTIMENT 端点真正支持 time_from/time_to）；
  2. 在 `route_to_vendor` 层加保护：`run_mode="backtest"` 时若 news vendor 解析为 yfinance，直接 raise 配置错误，禁止静默降级；
  3. `news_analyst.py:21-24` 的 system prompt 增加明确日期指令："你必须以 `{current_date}` 为 end_date、`{current_date}-7d` 为 start_date 调用 get_news"，不要让 LLM 自由发挥（这是日期错乱的第二来源）；
  4. 边界泄漏：把过滤条件改为 `start_dt <= pub < end_dt + 1d` 并统一转 UTC 后比较。

### 1.3 社交数据在回测中存在前视偏差（look-ahead bias）

- **位置**：`tradingagents/agents/utils/social_data_tools.py:82,103,229`
- **证据**：代码确认。`cutoff = datetime.utcnow() - timedelta(days=3)` —— 用**真实当前时间**而非 `trade_date` 过滤帖子。
- **后果**：回测 2026-01-15 时，社交分析师看到的是**今天**的 StockTwits/Reddit 帖子。对一个以日频基本面分析为目的的系统，这让回测中的情绪信号完全失效且方向性污染结果。
- **根本限制**：StockTwits/Reddit 公开 API 本质上拿不到历史帖子。诚实的修复是：
  1. `get_social_posts_cached(ticker)` 增加 `trade_date` 参数；【已确认执行】
  2. 回测模式下：若 `trade_date` 距今超过 `_SOCIAL_RECENCY_DAYS`，**不调用 API**，返回空并在报告中显式声明"无历史社交数据可用"（让下游 LLM 知道这是数据缺失而非中性情绪）；【已确认执行】
  3. 长期方案（每日定时抓取原始帖子并落盘）：【**暂不实施，但保留开关与接口**，2026-06-11 决定：本地机器无大容量存储，默认关闭；未来迁移云端时打开即用】实施要求：
     - 配置项 `archive_raw_social: bool = False`（`default_config.py`）；
     - 定义归档接口 `RawPostArchiver`（抽象方法 `archive(posts: list[dict], source: str, fetched_at: str)`），默认实现 `NullArchiver`（no-op，零成本），另提供 `JsonlDiskArchiver`（按 `{source}/{date}.jsonl.zst` 落盘）作为云端开启时的即用实现；
     - 采集管线**无条件**调用 `archiver.archive(...)`（由配置决定注入哪个实现），业务代码不写 if 判断——将来云端只改一行配置，不动代码。

### 1.4 Alpha Vantage 新闻丢弃 end_date 当天全部新闻

- **位置**：`tradingagents/dataflows/alpha_vantage_common.py:26`
- **证据**：代码确认。`"%Y-%m-%d"` 格式的日期一律格式化为 `T0000`（当天零点）。作为 `time_to` 时，end_date 当天 00:00 之后发布的新闻全部被 API 排除——而分析日当天的新闻恰恰是最重要的。
- **修复**：`format_datetime_for_api` 增加 `is_end: bool = False` 参数，end 侧输出 `T2359`；修改 `alpha_vantage_news.py:24-25` 的两处调用。
- 顺带修复 `_filter_av_feed_by_date`（`backtest_cache.py:549-564`）的字符串比较：当前依赖字典序碰巧正确，应转为数值/日期比较。

---

## 2. Major —— 严重影响质量的问题

### 2.1 FinBERT 无法过滤误标 ticker 的帖子（"乱标 tag"问题）

- **位置**：`social_data_tools.py:179-182`（Reddit 关键词搜索）、`:314-431`（finbert_aggregate）
- **机理确认**：
  - Reddit 搜索就是 `search.json?q={ticker}` 纯关键词——提到即命中，不管主体是谁；
  - FinBERT（ProsusAI/finbert）只做三分类情感，**没有任何"这条帖子是否真的在讨论该 ticker"的判定能力**，且文本截断 512 token；
  - 现有缓解手段只有 `social_media_analyst.py:61-72` 的 prompt 级嘱咐（让 LLM 自己丢弃离题帖），但 LLM 看到的已是 FinBERT 聚合后的统计分布，**聚合数字本身已被污染**，prompt 过滤为时已晚。
- **修复步骤（按性价比排序，三条均已确认执行，2026-06-11）**：
  1. **规则预过滤**（零成本，先做）：在进入 FinBERT 前要求帖子满足任一条件——cashtag 精确匹配（`$AAPL`）、ticker 出现在标题、或正文中目标 ticker 是出现次数最多的 ticker。可消除大部分"提及即命中"的噪声；
  2. **NLI/zero-shot 主体判定**（中成本）：用小型 NLI 模型对剩余帖子做 "This post is primarily about {company}" 的蕴含判定，阈值过滤后再进 FinBERT；
  3. 在 `finbert_aggregate` 输出中附带"过滤前/后帖子数"，让分析师 LLM 知道样本量与噪声水平。

### 2.2 记忆检索无相似度阈值 + Situation/Lesson 混合嵌入

- **位置**：`memory_store.py:318-341`（top-k 无阈值直接返回）、`:680`（`content = "Situation:...\n\nLesson:..."` 整体嵌入）
- **机理确认**：检索 query 是当日四份报告的全文拼接（2000+ token），文档是历史"situation+lesson"全文。BGE 对超长文本只取前 512 token，意味着**实际参与匹配的只有 market report 的开头**；其余内容（包括 lesson 本身）大概率根本没进向量。top-k 无阈值意味着哪怕完全不相关也照样注入 prompt，LLM 会把它当作有效经验。
- **修复步骤**：
  1. 写入时拆分：嵌入向量只编码 **situation 的结构化摘要**（重构时让 Reflector 同时输出一段 ≤200 token 的 situation 摘要），lesson 全文放 document、摘要做 embedding（chroma 支持 `embeddings` 与 `documents` 分离传入，配合 1.1 的修复顺手完成）;
  2. 检索时 query 同样用摘要而非全文拼接（在 manager/researcher 节点先做一次廉价压缩，或直接复用各报告的 markdown 表格部分）；
  3. `_search_text` 增加 `min_similarity` 参数（初始 0.45，回测后校准），低于阈值的命中丢弃；无命中时注入文案改为 "No sufficiently similar past situation found."，禁止注入低质量记忆。

### 2.3 记忆写入不对称 + 死功能

- **证据**：全库 grep 确认。
  - `store_news_summary` / `store_market_summary` / `store_fundamentals` 在 memory_store.py 中定义但**全项目零调用**——只有 social analyst 写入（`social_media_analyst.py:113`）；
  - `store_lesson` / `retrieve_lessons` 零调用（被 reflection 体系取代后残留）；
  - `retrieve_sector_sentiment` 唯一调用点传的是 `related_tickers=[]`（`social_media_analyst.py:35`）——**永远返回空列表**，"行业联动"功能形同虚设。
- **修复步骤**：
  1. 统一写入点：不要在各 analyst 节点内零散写入，改为在 `trading_graph.propagate()` 拿到 final_state 后集中调用四个 store_*（保证四份报告对称持久化，也避免 ReAct 重入导致的重复写）；
  2. sector sentiment 二选一：要么实现 ticker→peers 映射（静态行业表即可），要么删除该方法与调用；
  3. 删除 `store_lesson` / `retrieve_lessons` 或并入 reflection 体系。

### 2.4 backtest_cache 跨 ticker / 跨 run 污染风险

- **位置**：`tradingagents/dataflows/backtest_cache.py:608-614`（模块级单例）、`:48-67`
- **问题**：全局单例 `initialize()` 后 `_active=True`，换 ticker 或从回测切到单日分析时若未显式 `clear()`，旧数据可能被错误命中；prefetch 同时拉 yfinance 和 alpha_vantage 两路新闻但每次 run 只用一路，浪费配额。
- **修复**：cache key 纳入 `(ticker, start, end)` 三元组并在 `initialize()` 时校验不匹配即重建；prefetch 按 `data_vendors` 配置只拉实际使用的源。

---

## 3. Minor —— 冗余、死代码、依赖

| # | 项目 | 位置 | 处理 |
|---|---|---|---|
| 3.1 | `FinancialSituationMemory`（BM25 旧记忆系统）整类死代码 | `agents/utils/memory.py` | 删除文件，从 `agents/__init__.py` 移除 export，依赖 `rank-bm25` 一并移除 |
| 3.2 | `create_bull_researcher` / `create_bear_researcher` 旧架构残留 stub | `bull_researcher.py:70-96`、`bear_researcher.py:70-96` | 删除（Judge 架构下未接入 graph） |
| 3.3 | 未使用依赖 | `pyproject.toml` | 删除 `backtrader`、`parsel`、`pytz`、`tqdm`、`rank-bm25`（已 grep 确认零 import） |
| 3.4 | `cli/announcements.py` 调用原项目作者的 `api.tauric.ai` | `cli/announcements.py`、`cli/main.py:494` | 删除文件及调用（外部硬编码端点，与本项目无关） |
| 3.5 | 双路径分析师执行重复 ReAct 逻辑 | `parallel_analysts.py` vs `setup.py` 顺序路径 + 4 个 msg-clear 节点 | 统一为 parallel 实现，顺序模式改为 max_workers=1，删除 msg-clear 节点 |
| 3.6 | GUI 整段复制 CLI 的消息分类/状态跟踪逻辑 | `gui/app.py:190-400` ≈ `cli/main.py` MessageBuffer | 提取共享模块（如 `tradingagents/streaming.py`），CLI/GUI 共用 |
| 3.7 | `select_analyst_llm_config` / `select_manager_llm_config` 近重复 | `cli/utils.py:437-494` | 合并为 `select_llm_config(role)` |
| 3.8 | 工具异常被转成字符串吞掉 | `parallel_analysts.py:64-66` | 保留字符串返回（ReAct 需要），但同时 `logger.error` 带 traceback |
| 3.9 | 时区不统一（utcnow/本地时间/naive 混用） | yfinance_news.py、social_data_tools.py、alpha_vantage_common.py | 全部统一为 UTC aware → 比较前 normalize |
| 3.10 | `model_catalog.py` 含不存在的型号、effort/thinking 参数无校验 | `llm_clients/` | 校对目录；`validators.py` 增加 effort 枚举校验 |
| 3.11 | CLI 直接 `graph.graph.stream()` 绕过 `propagate()` | `cli/main.py:1505-1507` | changelog 称已修（手动补了 `set_analysis_date`），重构时改为 `propagate()` 暴露 streaming 回调，消除双入口 |
| 3.12 | `memory.py` 删除后 `mempalace_drawers` collection 名等历史命名 | `memory_store.py:209` | 可选：保留命名（迁移成本不值得）|

---

## 4. Feature —— 计划中但未实现（重构后再做）

### 4.1 10-K RAG（Fundamentals Analyst）—— **已于 2026-06-11 实现**

> 实现摘要：`tradingagents/dataflows/filings.py`（EDGAR 免 key 下载，限速 8 req/s，磁盘缓存 `data/filings_cache/`）+ `tradingagents/agents/utils/filing_store.py`（独立 chroma 库 `data/filings_index/`，cosine 空间，与记忆库共享 BGE 单例，`filed_date <= 分析日` 因果隔离）+ `search_10k` 工具（接入 fundamentals analyst，prompt 要求至少调用一次并引用 filing 日期）。实测：AAPL 10-K（filed 2025-10-31）下载→157 chunk 索引→风险因素检索 sim 0.65。
>
> 同日新增**运行观测层** `tradingagents/observability.py`：每次运行将全部中间步骤（各 agent 输入摘要/LLM 轮次与工具名/工具全文/记忆检索/质量指标）原子写入 `runs/{ticker}/{date}.json`；同 (ticker, 日期, 配置摘要) 重跑默认回放缓存（`reuse_cached_run`，Fresh Run 可强制重算）；回测中已回放的日子跳过 reflect。质量指标：social 信噪比+示例帖、新闻条目数、10-K 可用性+示例句。

原计划（已按此执行）：

- 现状：grep 全库无 EDGAR/10-K/filing 相关代码；`store_fundamentals()` 接口已备好但无人调用；设计文档在 `docs/concurrency_and_10k_design.md` 与 `docs/modification_plan.md`（改造 I）。
- 实施轮廓（依赖 1.1 与 2.2 先行修复）：
  1. EDGAR 下载器（公司 CIK → 最新 10-K/10-Q，纯 requests + SEC 官方 JSON API，注意 UA 要求与限速；数据源细节见 `docs/data_source_options.md` 第 2 节，分节抽取用 edgartools）；
  2. 分节切块（Item 1A Risk Factors、Item 7 MD&A 优先），每块 ≤512 token，`valid_from = filed_date`（天然因果隔离）；
  3. fundamentals analyst 增加 `search_10k(query)` 工具，ReAct 循环中按需检索；
  4. 回测缓存：filing 按 (ticker, accession_no) 落盘，避免重复下载。
- **存储架构决策（已定，2026-06-11）：10-K 语料与反思记忆分库存储、共用封装。**
  - 理由：① 生命周期不同——反思是实验数据（换实验需清空重建），filings 是参考语料（构建成本高、跨实验复用），共库会导致"重置实验 = 炸掉昂贵索引"或"舍不得删 = 实验间记忆污染"；② 体量悬殊（filings chunk 与 reflection 数量比约 100:1），共用 collection 导致 HNSW 索引被 filings 主导；③ 嵌入/切块策略将来会分化，一个 collection 锁死一个嵌入模型；④ 检索语义不同（top-2 情境匹配 vs 大 k 章节检索 + 按 accession 聚合）。
  - 实施：抽取 `BaseVectorStore` 基类（chroma client + 因果隔离 `_store/_search` + 读写计数），`TradingMemoryStore`（实验记忆，存 `memory/`，含 KG 与 calibration）与新建 `FilingStore`（参考语料，存 `data/filings_index/`，metadata 含 ticker/accession_no/form_type/item_section/filed_date/chunk_index）共同继承。
  - **关键约束：`_BGEEmbeddingFunction` 实例必须由外部构造后注入两个 store**（依赖注入），禁止各自实例化——否则 BGE 模型（约 400MB）加载两份。

### 4.2 本地模型支持

- 现状：`ollama` provider 路径存在且 `parallel_analysts` 会自动降为顺序执行；未实测。重构 Phase 6 中用一个小模型冒烟验证即可，不是 bug。

---

## 5. 给 Opus 的分阶段执行顺序

> 每个 Phase 结束跑 `pytest tests/ -q` + 指定冒烟脚本，绿了才进下一阶段。模型分工：分析师=claude-haiku-4-5，manager=claude-sonnet-4-6（API key 从环境变量 `ANTHROPIC_API_KEY` 读取，勿写入任何文件）。

- **Phase 0 — 测试脚手架**：为 memory_store、新闻日期过滤、social 日期过滤补单元测试（先写测试锁定 bug 行为，再修）。
- **Phase 1 — 记忆层**（1.1 → 2.2 → 2.3）：chroma EF 解耦修复 → 阈值与拆分嵌入 → 统一写入点。这是优先级最高的链条。
- **Phase 2 — 新闻数据层**（1.2 → 1.4 → 2.4）：默认 vendor 切换、AV end_date 修复、prompt 日期指令、cache 防护。
- **Phase 2.5 — News Lake 采集层**：新数据源接入与统一存储，方案见 `docs/data_source_options.md` 第 4 节。
- **Phase 3 — 社交/FinBERT**（1.3 → 2.1）：trade_date 贯穿、回测降级声明、规则预过滤。
- **Phase 4 — 结构去重**（3.x 全部）：死代码、依赖、双路径合并、GUI/CLI 共享模块。
- **Phase 5 — 10-K RAG**（4.1）。
- **Phase 6 — 端到端验证**：单日分析（haiku/sonnet）跑通 → 5 个交易日迷你回测确认记忆读写计数（`db_reads/db_writes`）均 >0 且检索命中非空 → 与旧结果对比。

---

## 附录 A：建议重构后的目录形态（当前已基本达成）

```
Mantra2/
├── main.py                 # Python API 入口
├── pyproject.toml          # 唯一依赖清单
├── cli/                    # 终端入口（mantra）
├── gui/                    # Flask 入口（mantragui）
├── tradingagents/          # 核心包
│   ├── agents/             # analysts / researchers / managers / risk / trader / utils
│   ├── dataflows/          # 数据源与缓存
│   ├── graph/              # LangGraph 编排、reflection、backtest
│   └── llm_clients/        # 多 provider 抽象
├── tests/                  # pytest（待扩充）
├── docs/                   # 设计文档 + 本计划（git 跟踪）
└── notes/                  # 课程材料（gitignore，不属于代码库）
```

运行期目录 `memory/`、`results/`、`reports/` 由代码按需创建，保持 gitignore。

## 附录 B：排查中被证伪的指控（不必处理）

1. "ChromaDB collection 创建时未传 embedding_function 导致 768/384 维度不匹配" —— 证伪：`memory_store.py:208-211` 明确传入了 EF，且实测现存数据库 collection 维度统一为 768。真正的问题是 1.1 的接口不兼容。
2. "valid_from 元数据 string/int 大面积混杂导致过滤失效" —— 实测旧数据库 433 条中仅 1 条早期遗留为 string（aapl/sentiment/2026-01-12），当前代码写入路径已统一为 int。旧库已随本次清理删除，无需迁移。
3. "risk_debate_state['judge_decision'] 未初始化会 KeyError" —— 证伪：`propagation.py:32,51` 中初始化为空字符串。
4. "signal_processing 置信度正则有 bug" —— 证伪：`(0\.\d+|1\.0+)` 模式行为正确。

## 附录 C：安全提醒

本次会话中 Anthropic API key 以明文出现在对话与终端历史中。重构完成后建议到 console.anthropic.com 轮换该 key，并今后通过 `.env`（已 gitignore）传递。
