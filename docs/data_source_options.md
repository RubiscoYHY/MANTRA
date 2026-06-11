# 信息渠道扩展方案（舆情 / 10-K / 主流媒体新闻）

> 调研日期：2026-06-11，所有 API 状态均经实际请求或官网核实，非凭记忆。
> 与 `refactor_plan.md` 配套：本文件回答"接什么数据源"，refactor_plan 回答"怎么修现有管线"。

---

## 1. 舆情渠道（带 ticker + 时间戳）

### 推荐接入（按优先级）

| 渠道 | 成本 | ticker 标注 | 历史深度 | 时间戳 | 备注 |
|---|---|---|---|---|---|
| **Polygon.io News + insights**（公司更名 Massive 中） | 免费 5 req/min | **原生，ticker 级情绪 + 推理文本** | 多年（情绪字段 2024 起） | 秒级 | 免费层质量最高的新闻情绪源；`GET /v2/reference/news?ticker=AAPL`，响应含 `insights[].sentiment` + `sentiment_reasoning` |
| **Alpha Vantage NEWS_SENTIMENT** | 免费 25 req/天 | **原生**，每篇带 `ticker_sentiment_score` 和 `relevance_score` | **回溯至 2022-03**，支持 time_from/to 分页 | 秒级 | 每次最多 1000 篇；25/天够日更 watchlist，回填历史需排批数周或 $49.99/月 |
| **Reddit 历史切片（Arctic Shift 网页导出/API）** | 免费 | 需自提（cashtag/正则，与现有管线一致） | **2005-06 → 2026-04**，按 subreddit+时间窗定向导出 | 秒级 `created_utc` | 回测王牌。**注意（2026-06-11 约束）：本地存储有限，不下载 TB 级全量种子**——用 Arctic Shift 的定向导出只取 watchlist 相关 subreddit × 回测窗口的切片（百 MB 量级），过 FinBERT 后只落盘聚合结果，原始帖子用完即删 |
| **Tiingo News** | 免费（~1000 req/天） | 原生 | 免费层仅近 3 个月 | 秒级 | 无情绪分，正好喂现有 FinBERT；定位是"从今天开始积累自有历史库" |
| **ApeWisdom** | 免费无 key | 原生（提及计数） | 无历史端点，须自建 cron 落盘 | 小时级 | `GET apewisdom.io/api/v1.0/filter/wallstreetbets`；半小时即可接入，做 WSB 热度信号 |
| **GDELT**（DOC 2.0 API / BigQuery GKG） | 免费无 key | **无**，需公司名→ticker 映射（歧义风险高） | DOC API 约 3-12 个月；BigQuery 2015 至今 | 15 分钟粒度 | 文章级 tone 而非实体级；适合宏观/大盘情绪维度，不适合个股精确归因 |

### 明确不推荐

- **Finnhub 社交情绪**：免费层 403（事实上 premium-only），免费只剩 company-news 可用。
- **NewsAPI.org / GNews / NewsData.io**：无 ticker 标注、无历史或仅 1 个月、ToS 限制，性价比低于上表。
- **Bluesky**：无 cashtag 文化、财经密度低、禁止批量历史抓取；暂缓。
- **Quiver Quantitative**（$30/月）：WSB 数据集据 QuantConnect 文档已于 2025-02 停更，确认恢复前勿付费。
- **X/Twitter**：API 价格远超预算，不列入。

---

## 2. 10-K 全文获取：SEC EDGAR（免费、无需 API key）

**结论：存在这个"通用网站"，就是 SEC 官方 EDGAR，而且连 key 都不用**——唯一要求是 HTTP 头带 `User-Agent: 姓名 邮箱`（缺失返回 403），限速 10 req/sec。已实测全链路可用（2026-06-11）：

```
ticker → CIK:    https://www.sec.gov/files/company_tickers.json
filing 列表:     https://data.sec.gov/submissions/CIK{cik:010d}.json
全文搜索:        https://efts.sec.gov/LATEST/search-index?q=...&forms=10-K
下载原文:        https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no}/{doc}
```

**配套库：`edgartools`**（v5.36.0，2026-06-09 仍在活跃发版）——`Company("AAPL").latest("10-K").obj()` 后可直接 `tenk["Item 1A"]`、`tenk["Item 7"]` 按章节取干净文本。**不要自己写分节正则**（各公司 HTML 千差万别），也不要下载整包 `.txt`（含 exhibits 可达 50MB），用 `primaryDocument` 定位主文档（1-5MB），并剔除 `<ix:header>` 的 iXBRL 元数据块。

第三方（sec-api.io $49/月、FMP、Polygon filings $29/月）数据源头都是 EDGAR，免费场景没有理由绕道。

→ 直接支撑 refactor_plan **Phase 5（10-K RAG）**：EDGAR 下载 + edgartools 分节 + 按 `filed_date` 设 `valid_from` 入库。

---

## 2.5 Earnings Call Transcripts（已实现，2026-06-11）

**$0 成本组合**（SEC EDGAR 不托管 transcripts，8-K Ex-99.1 只是 press release）：

| 用途 | 来源 | 说明 |
|---|---|---|
| 每季增量 | **API Ninjas** `earningscalltranscript` | 免费 ~10k req/月，全文 JSON（含分发言人），最新季度免费；2025 年前历史需 premium。key 放 `API_NINJAS_KEY` |
| 历史回填 | **kurry/sp500_earnings_transcripts**（HF，MIT，2005-2025，33k 份）或 **defeatbeta-api** | `scripts/backfill_transcripts.py AAPL MSFT ...`，只保留向量索引不存原始数据 |
| 排除 | FMP（transcripts 在 $99/月 Ultimate 档）、Finnhub（机构级付费）、Seeking Alpha/Motley Fool（付费墙/反爬，ToS 风险） | |

实现：与 10-K 同库（`data/filings_index/`，form="EARNINGS_CALL"），**独立有效期 365 天**（指引每季被取代，但同比叙事有价值；命中带 call 日期由 LLM 自行权衡）；10-K 检索与 transcript 检索通过 form 过滤互不污染。因果隔离：按真实 call 日期；API 未返回日期时保守取"季末+45 天"（只会晚见、不会早见）。工具 `search_earnings_call` 已接入 fundamentals analyst。

**配套决策（2026-06-11）**：10-K 有效期 **3 年**（`filing_validity_days: 1095`）；RAG 以 **10-K 为唯一主体**——10-Q 的 Item 1A/footnotes 远不如 10-K 详尽，且其结构化数据已由 `get_balance_sheet` 等工具覆盖，10-Q 仅作为"无 10-K 公司（新 IPO）"的fallback。所有日期/有效期约束均为 chroma `where` **搜索内前置过滤**，非取出后丢弃。

## 3. FT / Reuters / Google News

**先说硬约束：没有任何免费且合法的途径能拿到 FT 或 Reuters 的正文全文。** 能免费拿到的是"标题 + 来源 + 秒级时间戳（FT 另有一句话导语）"。对日频分析而言，标题级情绪/事件信号在学术文献中是常规做法，损失可控。

| 来源 | 可行路径（已实测） | 拿到什么 |
|---|---|---|
| **FT** | 官方 RSS：`ft.com/rss/home`、`ft.com/markets?format=rss`、`/companies?format=rss` 等 | 标题 + 一句话导语 + 秒级 pubDate + 直链；三者中质量最好的免费源 |
| **Reuters** | 官方 RSS 2020 年已停、无公开 API（LSEG 企业级）。替代：Google News RSS `q=Apple source:Reuters`（实测有效） | Reuters 署名标题 + 时间戳 |
| **Google News** | RSS：`news.google.com/rss/search?q=<公司名>+stock&hl=en-US&gl=US&ceid=US:en` | 标题（含来源名）+ 秒级 pubDate；**无摘要** |

Google News 两个坑：① 链接是加密跳转（`CBMi...`），批量解码会被 429 限流——**不要依赖解码原文 URL**，只用标题层；② feed 声明 personal/non-commercial，学术项目温和限速（查询间隔数秒 + 缓存）即可。查询用**公司名**而非纯 ticker 效果更好。Python 包用 `gnews`（pygooglenews 已停更）。补充选项：SerpAPI 免费 250 次/月，返回干净直链，量小时可用。

---

## 4. 架构建议：News Lake + FinBERT 评分 + 仅对长文档用 RAG

**"是否需要 RAG 管理新闻？"——对新闻本身：不需要向量 RAG；需要的是一个带元数据索引的新闻库（News Lake）。** 理由：日频分析的检索模式是确定性的 `(ticker, 日期窗口)` 过滤，SQL 一行就能精确解决；向量检索反而引入"语义相似但日期/主体错误"的噪声——这正是 refactor_plan 1.2/2.2 里那些 bug 的根源。向量 RAG 只该用在两处：**10-K 等长文档**（必须语义检索）和**已有的跨日记忆系统**（reflections）。

**"是否用 FinBERT 处理这些新闻？"——用，但分工要清楚：**
- 标题/短摘要 → FinBERT 逐条评分（本地、零成本、512 token 对标题绰绰有余）；
- Polygon / Alpha Vantage 自带 ticker 级情绪分的 → 直接采用原生分数，FinBERT 作交叉校验（双源一致性本身就是质量信号）；
- 主体相关性过滤（refactor_plan 2.1 的"乱标 tag"问题）仍须在 FinBERT **之前**做规则预过滤——FinBERT 只会打分，不会判断"这篇是不是在说这家公司"。原生 `relevance_score`（AV）和 `insights`（Polygon）能大幅替代自建判定。

### 数据流

```
[采集层 · cron 每日]
  Polygon news ─┐
  AV NEWS_SENTIMENT ─┤   统一 schema:
  Tiingo news ─┤──→  {source, tickers[], ts_utc, title, snippet,
  FT RSS ─┤           url, native_sentiment?, relevance?}
  Google News RSS (含 source:Reuters) ─┤
  ApeWisdom 快照 ─┘
        │  去重（URL/标题 simhash）→ 主体规则过滤 → FinBERT 补分
        ▼
[存储层]  news_lake.sqlite（按 ticker+ts 索引；回测天然无前视——只查 ts <= trade_date）
        ▼
[消费层]
  News/Social Analyst：SELECT by (ticker, date window) → top-N 按 relevance 排序 → LLM 摘要
  10-K：EDGAR + edgartools 分节 → chunk → ChromaDB room "filings"（真正的 RAG）
  回测：直接查 lake，彻底取代 yfinance 实时新闻（解决 refactor_plan 1.2/1.3 的根本限制）
```

**存储约束（2026-06-11 决定）**：项目在本地机器运行，refactor_plan 1.3 的"原始社媒帖子持续落盘"**默认关闭，但保留开关与接口**（`archive_raw_social` + `RawPostArchiver`，详见 refactor_plan 1.3 第 3 条）——未来迁移云端时改一行配置即可启用。本地默认行为：
- lake 只存**标题级元数据 + 情绪分**（每条数百字节，全年全 watchlist 约几十 MB），**不存正文、不存原始社媒帖子**；
- Reddit 历史只取定向切片（见上表），FinBERT 聚合后丢弃原始数据（同样经由 archiver 接口，云端开启后自动保留）；
- 历史回填以 Alpha Vantage（2022-至今，本身就是摘要级）为主。
在此约束下，"自有可回测语料"的目标不变，本地磁盘占用控制在百 MB 量级。

### 落地顺序（并入 refactor_plan 执行序列）

1. **Phase 2.5（新增）**：News Lake schema + 采集 cron（ApeWisdom、Polygon、AV、Tiingo、FT RSS、Google News RSS）——半天工作量；
2. **Phase 3** 的 FinBERT 预过滤改造直接复用 lake 的统一 schema；
3. **Phase 5**：10-K RAG 按第 2 节方案实施；
4. 回填任务（AV 2022-至今 + Arctic Shift dump 处理）作为后台长任务独立运行，不阻塞重构。
