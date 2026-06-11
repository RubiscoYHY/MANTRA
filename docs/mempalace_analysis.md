# MemPalace 工作原理深度分析

> 分析日期：2026-04-10
> 分析对象：`third_party/mempalace/`（来源：https://github.com/milla-jovovich/mempalace.git）
> 文档目的：学习 MemPalace 的架构与调用方式，为改造 B（SentimentMemoryStore）做准备

---

## 目录

1. [整体架构](#一整体架构)
2. [记忆的层级结构：宫殿比喻](#二记忆的层级结构宫殿比喻)
3. [三种调用方式](#三三种调用方式)
4. [写入流程：你提供什么，它怎么处理](#四写入流程你提供什么它怎么处理)
5. [检索流程：四层按需召回](#五检索流程四层按需召回)
6. [时序机制：知识图谱与有效期窗口](#六时序机制知识图谱与有效期窗口)
7. [跨标的关联：Tunnels](#七跨标的关联tunnels)
8. [底层存储实现](#八底层存储实现)
9. [Embedding 模型](#九embedding-模型)
10. [对 TradingAgents 改造的启示](#十对-tradingagents-改造的启示)

---

## 一、整体架构

### 目录结构

```
mempalace/
├── palace.py           # ChromaDB 连接单例，集合管理
├── layers.py           # 四层记忆栈（L0-L3）核心实现
├── miner.py            # 项目文件扫描、分块、写入 ChromaDB
├── convo_miner.py      # 对话文本导入和处理
├── searcher.py         # 语义搜索统一接口
├── knowledge_graph.py  # SQLite 时序知识图谱（三元组 + 有效期窗口）
├── palace_graph.py     # 图遍历层（Tunnel 检测与跨 Wing 查询）
├── mcp_server.py       # MCP 协议服务器（19 个工具）
├── cli.py              # 命令行接口
├── config.py           # 配置管理、名称校验
├── entity_detector.py  # 自动检测人物和项目名
├── general_extractor.py# 五类记忆类型提取
└── dialect.py          # AAAK 压缩方言（token 压缩）
```

### 核心模块关系

```
调用者（Claude / Python 代码 / CLI）
        │
        ▼
┌─────────────────────┐
│  三种入口            │
│  CLI / MCP / Python │
└──────────┬──────────┘
           │
     ┌─────▼─────┐
     │ layers.py │  ← 四层记忆栈（决定从哪里取）
     └─────┬─────┘
           │
    ┌──────┴──────────────────┐
    │                         │
┌───▼────────┐      ┌─────────▼──────┐
│ ChromaDB   │      │ SQLite         │
│ palace.py  │      │ knowledge_     │
│ miner.py   │      │ graph.py       │
│ searcher.py│      │                │
│ (向量检索)  │      │ (时序知识图谱)  │
└───────────-┘      └────────────────┘
         └──────────────────┘
                  │
          palace_graph.py
          (Tunnel：跨 Wing 图遍历)
```

---

## 二、记忆的层级结构：宫殿比喻

MemPalace 用"记忆宫殿"的比喻组织所有记忆，层级如下：

```
PALACE（整个记忆系统）
├── Wing（翼）= 一个独立的项目或人物
│   ├── Hall（厅）= 全局统一的记忆类型（五类，跨所有 Wing 一致）
│   │   ├── hall_facts       — 锁定的决策、不变的事实
│   │   ├── hall_events      — 会话记录、里程碑
│   │   ├── hall_discoveries — 突破性见解、新发现
│   │   ├── hall_preferences — 习惯、偏好
│   │   └── hall_advice      — 建议、解决方案
│   │
│   └── Room（房间）= 该项目内的某个具体话题
│       └── Drawer（抽屉）= 一条具体的原文本记录
│
└── Tunnel（隧道）= 自动检测的跨 Wing 连接
    （同一个 Room 名称出现在多个 Wing 中时自动形成）
```

### 各层含义

**Wing（翼）**：命名空间的最顶层，类比"项目"或"人"。
- 例如：`wing_nvda`（英伟达相关记忆）、`wing_aapl`（苹果）
- 所有 Drawer 都属于某个 Wing

**Room（房间）**：Wing 内的话题分区，类比"子主题"。
- 例如：`wing_nvda` 下可以有 `room_sentiment`、`room_earnings`、`room_technicals`
- Room 名称是关键：**同名 Room 出现在多个 Wing 时，自动形成 Tunnel**

**Hall（厅）**：记忆类型标签，与 Room 并列，作为 metadata 字段存在。
- Hall 是全局统一的五类分类，不受 Wing 限制
- 一条 Drawer 可以同时有 Room 和 Hall 标签

**Drawer（抽屉）**：最小存储单元，存储一段原文本（800 字符左右），加上 metadata。
- 这是实际存进 ChromaDB 的一条记录
- 内容是 **verbatim（原文）**，不经过 LLM 摘要

**Tunnel（隧道）**：自动形成的跨 Wing 关联。
- 例：`wing_nvda/room_competition` 和 `wing_amd/room_competition` 自动连接
- 不需要手动声明，由 `palace_graph.py` 从 metadata 自动推断

---

## 三、三种调用方式

### A. CLI 模式（运维 / 数据导入）

适合批量导入项目文件或对话记录。

```bash
# 初始化（检测目录结构，生成 mempalace.yaml）
mempalace init ~/projects/myapp

# 扫描项目文件，分块写入 ChromaDB
mempalace mine ~/projects/myapp

# 导入对话记录
mempalace mine --mode convos --dir ~/convos --wing myapp

# 语义搜索
mempalace search "auth migration decision"

# 输出 L0+L1（系统提示词注入用）
mempalace wake-up
```

**CLI 入口**：`mempalace/cli.py` → `cmd_*()` 函数

### B. MCP 模式（Claude / ChatGPT / Cursor 集成，推荐）

MemPalace 作为 MCP 服务器运行，AI 助手自动调用其工具。

```bash
# 注册为 MCP server
claude mcp add mempalace -- python -m mempalace.mcp_server
```

注册后，Claude 在对话中会自动决定何时调用 MemPalace 的 19 个工具，无需用户干预。

**19 个 MCP 工具（按功能分类）**：

| 类别 | 工具名 | 作用 |
|------|--------|------|
| 读取 | `mempalace_status` | 查看宫殿概览 |
| 读取 | `mempalace_search` | 语义搜索（L3） |
| 读取 | `mempalace_list_wings` | 列出所有 Wing |
| 读取 | `mempalace_kg_query` | 查询知识图谱实体 |
| 读取 | `mempalace_kg_timeline` | 时间线视图 |
| 写入 | `mempalace_add_drawer` | 手动写入一条记忆 |
| 写入 | `mempalace_kg_add` | 向知识图谱添加三元组 |
| 写入 | `mempalace_kg_invalidate` | 标记三元组过期 |
| 写入 | `mempalace_diary_write` | 写日记（带时间戳） |
| 图遍历 | `mempalace_traverse` | 从某 Room 出发遍历关联 |
| 图遍历 | `mempalace_find_tunnels` | 找两个 Wing 之间的隧道 |
| 图遍历 | `mempalace_graph_stats` | 图统计 |

**MCP 服务器入口**：`mempalace/mcp_server.py` → `tool_*()` 函数

### C. Python API 模式（嵌入到 Python 工程）

这是我们改造 B 将使用的方式。

```python
from mempalace.layers import MemoryStack
from mempalace.searcher import search_memories
from mempalace.knowledge_graph import KnowledgeGraph

# ─── 初始化 ───
stack = MemoryStack(palace_path="./sentiment_palace")

# ─── 读取 L0+L1（固定上下文，系统提示词用）───
wake_text = stack.wake_up(wing="nvda")   # ~170 tokens

# ─── L2：按 Wing/Room 过滤召回 ───
room_contents = stack.retrieve(wing="nvda", room="sentiment", n_results=10)

# ─── L3：深度语义搜索 ───
results = stack.search("bullish retail sentiment spike", wing="nvda", n_results=5)

# ─── 写入一条 Drawer ───
from mempalace.miner import add_drawer
col = stack.collection  # ChromaDB 集合对象
add_drawer(col, wing="nvda", room="sentiment",
           content="Reddit WSB: heavy call buying, 2.3k upvotes...",
           source_file="reddit_2026-04-10", chunk_index=0, agent="tradingagents")

# ─── 知识图谱操作 ───
kg = KnowledgeGraph(db_path="./sentiment_palace/kg.sqlite3")
kg.add_triple("nvda_sentiment_2026-04-10", "showed", "retail_fomo",
              valid_from="2026-04-10", valid_to=None)
kg.query_entity("nvda_sentiment_2026-04-10", as_of="2026-04-10")
```

---

## 四、写入流程：你提供什么，它怎么处理

### 你需要提供的数据

写入一条 Drawer 所需的最少信息：

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `wing` | str | 所属 Wing（命名空间） | `"nvda"` |
| `room` | str | 话题分区 | `"sentiment"` |
| `content` | str | 要存入的原文本 | Reddit 帖子内容 |
| `source_file` | str | 来源标识 | `"reddit_2026-04-10"` |
| `chunk_index` | int | 若内容已分块，当前块序号 | `0` |
| `agent` | str | 写入者标识 | `"tradingagents"` |

### 完整写入管道

```
你的原始文本（50-100 条 Reddit/StockTwits 帖子）
        │
        ▼
Step 1: 名称校验（config.sanitize_name）
        检查 wing/room 字符合法性（只允许字母数字下划线等）
        防止路径穿越攻击
        │
        ▼
Step 2: 分块（miner.chunk_text）
        CHUNK_SIZE = 800 字符
        CHUNK_OVERLAP = 100 字符
        优先在段落边界（\n\n）或行边界（\n）处切割
        过短（< 50 字符）的块被丢弃
        │
        ▼
Step 3: 生成 Drawer ID
        drawer_id = f"drawer_{wing}_{room}_{content_hash[:24]}"
        相同内容的 upsert 不会重复存储
        │
        ▼
Step 4: 写入 ChromaDB（collection.upsert）
        {
          "id": "drawer_nvda_sentiment_abc123...",
          "documents": ["原文内容"],
          "metadatas": [{
            "wing": "nvda",
            "room": "sentiment",
            "hall": "hall_events",     # 可选
            "source_file": "reddit_2026-04-10",
            "chunk_index": 0,
            "added_by": "tradingagents",
            "filed_at": "2026-04-10T15:30:45",
            "source_mtime": 1744290000.0
          }]
        }
        │
        ▼
Step 5: ChromaDB 自动计算 embedding（对调用者透明）
        默认使用 sentence-transformers（all-MiniLM-L6-v2）
        │
        ▼
Step 6（可选）：写入 SQLite 知识图谱
        适合存储结构化事实（如"本次情感为 Bullish，实际收益 +3.2%"）
        带时间有效期窗口
```

### 关于去重

写入前可先调用重复检测：
```python
# mcp_server.py: tool_check_duplicate
# 在 ChromaDB 中查询 top-5 相似，相似度 >= 0.9 则提示重复
```
对于 TradingAgents 的使用场景，每日采集的帖子内容基本不重复，可以跳过此步骤。

---

## 五、检索流程：四层按需召回

这是 MemPalace 最核心的设计，四层对应四种不同的"召回需求"。

### L0：身份层（约 100 tokens，固定）

```python
# layers.py: Layer0.render()
# 读取 ~/.mempalace/identity.txt（用户手写的世界观/角色设定）
# 内容示例：
# "I am TradingAgents' sentiment memory. I store retail sentiment signals.
#  My wings correspond to stock tickers. Each room is a topic category."
```

**特点**：
- 每次都全量加载，不做检索
- 内容由用户手写，100% 可控
- 用于注入系统提示词，让 LLM 知道当前记忆系统的背景

### L1：核心故事层（约 120 tokens）

```python
# layers.py: Layer1.generate()
# 从 ChromaDB 拉取 top-15 重要 Drawer
# 按 room 分组，每条压缩到 200 字符
# 总上限 3200 字符
```

**特点**：
- 按重要性/情感权重排序，不是按时间
- 这是"永远在场"的背景知识，LLM 每次对话都能看到
- 适合放置最关键的长期记忆

### L2：按需过滤层（约 200-500 tokens per query）

```python
# layers.py: Layer2.retrieve(wing, room, n_results)
# 构建 ChromaDB where 过滤：
#   {"$and": [{"wing": "nvda"}, {"room": "sentiment"}]}
# col.get(where=..., limit=n_results)  ← 精确过滤，不做语义排序
```

**特点**：
- 按 Wing/Room 精确过滤，返回该分区的所有内容
- 不做语义排序（按存储顺序返回）
- 适合"给我看 NVDA 的所有 sentiment 记录"这类查询

### L3：深度语义搜索（无限制）

```python
# searcher.py: search_memories(query, wing, room, n_results)
# 或 layers.py: Layer3.search(query, wing, room)
#
# 内部调用：
# col.query(
#     query_texts=["bullish retail sentiment spike"],
#     n_results=5,
#     where={"$and": [{"wing": "nvda"}]}  # 可选
# )
# 返回按余弦相似度排序的结果
```

**特点**：
- 真正的向量语义检索
- 可以加 Wing/Room 过滤缩小搜索范围（速度提升约 34%）
- 返回相似度分数（1 - cosine_distance）

### 检索返回格式

```python
# search_memories() 的返回值：
{
    "query": "bullish retail sentiment spike",
    "filters": {"wing": "nvda"},
    "results": [
        {
            "text": "Reddit WSB: 大量 call 期权买入，帖子 2.3k 赞...",
            "wing": "nvda",
            "room": "sentiment",
            "source_file": "reddit_2026-04-10",
            "similarity": 0.912,    # 越高越相关
            "filed_at": "2026-04-10T15:30:45"
        },
        ...
    ]
}
```

**重要设计原则：返回的永远是原文本（verbatim），不是摘要。**

### 四层对比

| 层 | 触发时机 | token 消耗 | 排序方式 | 适合场景 |
|----|---------|-----------|---------|---------|
| L0 | 每次对话 | ~100，固定 | 无（全量） | 系统背景/角色设定 |
| L1 | 每次对话 | ~120，动态 | 按重要性 | 最关键的长期记忆 |
| L2 | 显式召回 Wing/Room | 200-500 | 按存储顺序 | 浏览某主题全量内容 |
| L3 | 有语义查询时 | 按需，无上限 | 按余弦相似度 | 找历史相似情境 |

**对 TradingAgents 的意义**：我们只需要使用 L3（`search_memories()`），在 Social Media Analyst 生成报告前查询历史相似舆情。L0/L1 是为 AI 助手"常驻记忆"设计的，不适合我们的批量分析场景。

---

## 六、时序机制：知识图谱与有效期窗口

### 两套存储的不同时序能力

| 组件 | 时序能力 | 实现方式 |
|------|---------|---------|
| ChromaDB（Drawer） | **无内置时序**，只有 `filed_at` metadata | 需自己在 where 过滤中比较字符串日期 |
| SQLite（知识图谱） | **完整有效期窗口** | `valid_from / valid_to` 列 + SQL 过滤 |

### 知识图谱：三元组模型

```
实体A ──[谓词]──> 实体B
 │                    │
valid_from          valid_to
（开始日期）         （结束日期，NULL=至今有效）
```

**写入三元组**（`knowledge_graph.py: KnowledgeGraph.add_triple()`）：
```python
kg.add_triple(
    subject="nvda_sentiment",          # 主语
    predicate="showed",                # 谓词
    obj="retail_fomo",                 # 宾语
    valid_from="2026-04-10",           # 开始有效
    valid_to=None,                     # 至今有效（None = 仍然为真）
    confidence=0.87,                   # 置信度 0-1
    source_closet="drawer_nvda_..."    # 指向对应 Drawer（可选）
)
```

**ID 格式**：`t_{subject_id}_{predicate}_{object_id}_{hash[:12]}`

**时间过滤查询**（`knowledge_graph.py: KnowledgeGraph.query_entity()`）：
```python
kg.query_entity("nvda_sentiment", as_of="2026-04-08")
# SQL 中的时间过滤条件：
# (valid_from IS NULL OR valid_from <= '2026-04-08')
# AND
# (valid_to IS NULL OR valid_to >= '2026-04-08')
```

**标记过期**（`knowledge_graph.py: KnowledgeGraph.invalidate()`）：
```python
kg.invalidate("nvda_sentiment", "showed", "retail_fomo", ended="2026-04-09")
# 将 valid_to 设为 2026-04-09，该事实不再有效
```

### 对 TradingAgents 的应用方式

```python
# Reflector 回调时，将实际收益写入知识图谱：
kg.add_triple(
    subject="nvda_2026-04-10",         # ticker + 日期
    predicate="actual_return",
    obj="+3.2%",
    valid_from="2026-04-10",
    valid_to="2026-04-10",             # 只在这一天有效
    confidence=1.0,
    source_closet="drawer_nvda_sentiment_..."
)

# 回测时，查询某日的历史数据：
kg.query_entity("nvda_2026-03-15", as_of="2026-03-15")
# → 返回该日已标注的实际收益
```

---

## 七、跨标的关联：Tunnels

### 自动形成机制

Tunnel 不需要手动声明。**当同一个 Room 名称出现在多个 Wing 中时，Tunnel 自动形成。**

```
wing_nvda/room_competition  ──Tunnel──  wing_amd/room_competition
wing_nvda/room_sentiment    ──Tunnel──  wing_tsla/room_sentiment
```

**检测代码**（`palace_graph.py: build_graph()`）：
```python
# 遍历所有 Drawer 的 metadata，按 room → set(wings) 分组
# 如果某 room 出现在 2+ 个 wing 中 → 这是一个 Tunnel
room_data = {
    "sentiment": {
        "wings": ["wing_nvda", "wing_amd", "wing_tsla"],  # 3 个 Wing 共享这个 Room
        "count": 127   # 该 Room 的总 Drawer 数量
    },
    ...
}
```

### 跨 Wing 查询

```python
# 找 NVDA 和 AMD 之间的所有隧道（共同话题）：
from mempalace.palace_graph import find_tunnels
tunnels = find_tunnels(wing_a="wing_nvda", wing_b="wing_amd")
# 返回：[{"room": "sentiment", "count": 47}, {"room": "competition", "count": 23}]

# 从某个 Room 出发，沿 Tunnel 遍历关联 Room（最多 2 跳）：
from mempalace.palace_graph import traverse
graph = traverse(start_room="sentiment", max_hops=2)
# 返回：所有通过共享 Wing 连接到 "sentiment" 的 Room 名称
```

### 在 TradingAgents 中的应用

```python
# Social Media Analyst 处理 NVDA 时，自动拉取同板块的 AMD、INTC 的 sentiment 历史：
tunnels = find_tunnels(wing_a="wing_nvda", wing_b="wing_amd")
# 如果 "sentiment" 在隧道列表中，则额外搜索 AMD 的 sentiment Room
sector_results = search_memories(
    query=current_nvda_summary,
    wing="wing_amd",
    room="sentiment",
    n_results=2
)
```

---

## 八、底层存储实现

### ChromaDB（向量数据库）

- **位置**：`palace_path/`（例：`./sentiment_palace/`）
- **集合名**：`mempalace_drawers`（固定，只有一个集合）
- **写入**：`collection.upsert(documents, ids, metadatas)`
- **读取**：`collection.get(where=...)` 或 `collection.query(query_texts=...)`
- **去重**：基于 content hash 生成的 ID，相同内容 upsert 不重复

**一条 Drawer 的完整结构**：
```python
{
    "id":        "drawer_nvda_sentiment_abc123def456789012345678",
    "document":  "Reddit WSB: 大量 call 期权买入，评论区充斥 NVDA 看涨情绪...",
    "metadata": {
        "wing":         "nvda",
        "room":         "sentiment",
        "hall":         "hall_events",          # 可选，五类之一
        "source_file":  "reddit_2026-04-10",
        "chunk_index":  0,
        "added_by":     "tradingagents",
        "filed_at":     "2026-04-10T15:30:45",
        "source_mtime": 1744290000.0            # 文件修改时间（重新挖矿检测用）
    }
}
```

### SQLite（知识图谱）

- **位置**：`palace_path/knowledge_graph.sqlite3`
- **表结构**：

```sql
CREATE TABLE entities (
    id          TEXT PRIMARY KEY,   -- slug 化后的名称
    name        TEXT NOT NULL,      -- 显示名
    type        TEXT,               -- person / project / event / concept
    properties  TEXT,               -- JSON 额外属性
    created_at  TEXT
);

CREATE TABLE triples (
    id           TEXT PRIMARY KEY,  -- t_{subj}_{pred}_{obj}_{hash12}
    subject      TEXT NOT NULL,     -- 主语 entity.id
    predicate    TEXT NOT NULL,     -- 关系谓词
    object       TEXT NOT NULL,     -- 宾语 entity.id
    valid_from   TEXT,              -- ISO 日期，NULL=无始
    valid_to     TEXT,              -- ISO 日期，NULL=至今有效
    confidence   REAL DEFAULT 1.0,
    source_closet TEXT,             -- 关联的 drawer_id
    source_file   TEXT,
    extracted_at  TEXT
);

CREATE INDEX idx_triples_valid ON triples(valid_from, valid_to);
```

### 其他持久化文件

| 文件 | 内容 |
|------|------|
| `~/.mempalace/identity.txt` | L0 身份文本（用户手写） |
| `~/.mempalace/config.json` | 全局配置 |
| `~/.mempalace/wal/write_log.jsonl` | WAL 审计日志（每次写操作追加） |
| `mempalace.yaml` | 项目级配置（Wing/Room 定义） |

---

## 九、Embedding 模型

### 现状

MemPalace **完全委托给 ChromaDB** 处理 embedding，自身不调用任何 embedding 模型。

```python
# palace.py: get_collection()
client = chromadb.PersistentClient(path=palace_path)
col = client.get_or_create_collection("mempalace_drawers")
# ↑ 没有传入 embedding_function 参数 → 使用 ChromaDB 默认
```

ChromaDB 的默认 embedding 是 `all-MiniLM-L6-v2`（384 维，约 80MB）。

### 能否替换？

**不能直接通过 MemPalace API 替换**，这是一个设计限制。

如果要使用更好的 embedding（如 `BAAI/bge-base-en-v1.5`），需要：

1. **方案 A（推荐）**：在我们的 `SentimentMemoryStore` 中，自己创建 ChromaDB 集合并传入自定义 `embedding_function`，绕过 MemPalace 的 `get_collection()` 方法。

2. **方案 B**：分开维护两个 ChromaDB 路径——MemPalace 用默认路径（标准 MCP 功能），我们自己的 `SentimentMemoryStore` 用单独路径（BGE embedding）。

**对改造 B 的影响**：由于我们的 `SentimentMemoryStore` 是自己封装的类，完全可以选择不依赖 MemPalace 的 `palace.py`，而是直接使用 ChromaDB + 自定义 embedding，只是借鉴 MemPalace 的数据模型（Wing/Room/Drawer 结构和时序知识图谱 schema）。

---

## 十、对 TradingAgents 改造的启示

### 我们实际需要用到哪些部分

| MemPalace 模块 | 是否直接使用 | 说明 |
|---------------|------------|------|
| `palace.py` | **可选** | 可以直接用 chromadb，不必经过此封装 |
| `layers.py` | **不使用** | L0/L1 是为 AI 助手常驻记忆设计的，不适合批量分析 |
| `miner.py:add_drawer` | **直接使用** | 写入 Drawer 的核心函数 |
| `searcher.py:search_memories` | **直接使用** | L3 语义检索 |
| `knowledge_graph.py:KnowledgeGraph` | **直接使用** | 存储实际收益标注，带时间窗口 |
| `palace_graph.py:find_tunnels` | **间接使用** | 板块传染效应（NVDA→AMD→INTC 联动） |
| `mcp_server.py` | **不使用** | 我们通过 Python API 直接调用 |
| `cli.py` | **不使用**（调试除外） | 生产代码走 Python API |

### SentimentMemoryStore 的实现思路

基于以上分析，`SentimentMemoryStore` 的核心操作可以这样映射：

```python
from mempalace.miner import add_drawer
from mempalace.searcher import search_memories
from mempalace.knowledge_graph import KnowledgeGraph
from mempalace.palace_graph import find_tunnels

class SentimentMemoryStore:
    """
    Wing  = ticker（如 "nvda"）
    Room  = "sentiment"（固定）
    Hall  = "hall_events"（固定，社交媒体事件）
    """

    def store_raw_posts(self, ticker, posts, trade_date, actual_return=None):
        # → 调用 add_drawer()，写入 ChromaDB
        # → 如果 actual_return 已知，调用 kg.add_triple() 写入 SQLite

    def retrieve_similar_patterns(self, ticker, current_summary, n_results=3):
        # → 调用 search_memories(query=current_summary, wing=ticker, room="sentiment")

    def retrieve_sector_patterns(self, related_tickers, current_summary, n_results=2):
        # → 先调用 find_tunnels() 确认 Tunnel 存在
        # → 对每个 related_ticker 调用 search_memories()

    def annotate_with_return(self, ticker, trade_date, actual_return):
        # → 调用 kg.add_triple(subject=f"{ticker}_{trade_date}",
        #                       predicate="actual_return", obj=str(actual_return),
        #                       valid_from=trade_date, valid_to=trade_date)
```

### 数据流全景（对 TradingAgents 改造后）

```
每日采集（50-100 条 Reddit/StockTwits 帖子）
        │
        ▼
BERT 预处理管道（改造 E，与 MemPalace 无关）
  ├─ 相关性过滤（BGE embedding 相似度）
  ├─ 情感预评分（twitter-roberta 分类）
  ├─ 去重聚类（DBSCAN）
  └─ 排序，取 top-10
        │
        ▼ top-10 帖子 + 情感分布统计
┌──────────────────────────────────┐
│ SentimentMemoryStore             │
│                                  │
│ Step 1: 查询历史（L3 语义检索）    │  ← search_memories()
│   → 该 ticker 近 90 天相似舆情    │
│   → 相关板块传染效应记录           │
│                                  │
│ Step 2: 注入 LLM prompt          │  ← 约 500-800 tokens
│   → 历史模式 top-3               │
│   → 板块联动 top-2               │
│                                  │
│ Step 3: LLM 生成 sentiment_report│
│                                  │
│ Step 4: 写入当日帖子              │  ← add_drawer()
│   （actual_return=None，待标注）  │
└──────────────────────────────────┘
        │
        ▼ sentiment_report → 下游 Bull/Bear/Trader...
        ▼ 交易执行，返回实际收益
        │
Reflector 回调
        │
        ▼
SentimentMemoryStore.annotate_with_return()
  → kg.add_triple(actual_return="+3.2%", valid_from=trade_date)
  → 闭环完成，下次检索时历史记录带有收益标注
```

---

## 十一、TradingMemoryStore 设计评审（2026-04-11）

本节记录对 MemPalace 接入方案的架构评审结论，以及最终 `TradingMemoryStore` 的设计决策。

### 评审结论汇总

#### L2 + L3 层级划分

**正确。** L0/L1 是为 AI 助手常驻记忆设计的，批量自动化分析场景无需使用。L2（精确过滤）+ L3（语义检索）分工合理：
- L2：`col.get(where=...)` 精确过滤，用于"给我 NVDA 今天的 sentiment 全量"
- L3：`col.query(query_texts=...)` 语义向量检索，用于"找历史最相似的舆情模式"

**补充**：L2 的 `col.get()` 不按时间排序（按存储顺序）。跨 Wing 查询同一天的 sentiment 需要在 `where` 中加 `trade_date` 字段过滤。该字段必须在写入时主动添加到 metadata，MemPalace 原生的 `add_drawer()` 没有它。

#### 股价数据的存储后端

**Close price 不应放 ChromaDB。** ChromaDB 存 text embedding，"close price = 150.23 on 2024-01-15" 字符串的向量没有任何语义价值，L3 检索对数值无意义。正确存储位置是 **SQLite 知识图谱**：

```python
kg.add_triple("aapl_2024-01-15", "close_price", "150.23",
              valid_from="2024-01-15", valid_to=None)
kg.query_entity("aapl_2024-01-15", as_of="2024-01-15")
```

SQLite KG 支持完整的时间窗口查询，且对结构化数值数据更快、更准确。

#### ChromaDB 没有内置的 `valid_to` / `expires_at`

**原生 `add_drawer()` 的 metadata 只有**：`wing / room / source_file / chunk_index / added_by / filed_at / source_mtime`，没有有效期字段。

要实现"媒体消息 3 天过期"，必须：
1. 绕过 `add_drawer()`，直接调用 `collection.upsert()` 并传入自定义 metadata 字段
2. 写入时添加 `expires_at`（ISO 日期字符串，永不过期用哨兵值 `"9999-12-31"`）
3. 读取时在 `where` 过滤中加 `{"expires_at": {"$gte": as_of}}`

ChromaDB 支持字符串的 `$gte` / `$lte` 比较，ISO 格式日期字符串的字典序与时间序一致，可以正确比较。

#### 因果隔离：教训的 off-by-one 问题

若分析 T 日产生教训并设 `valid_from = T`，下次回测 T 日时查询 `valid_from <= T` 会召回该教训，造成**同日自我强化污染**。

**正确做法**：教训的 `valid_from = trade_date + 1 day`：

```python
# 分析 2024-03-01 产生的教训，次日起才生效
valid_from = "2024-03-02"   # +1 day，严格排除当日
```

其他数据（sentiment、news、market、fundamentals）的 `valid_from = trade_date`（当日即可用，因为分析时这些数据已存在）。

#### 多 Writer 必须集中管控

三个 Writer 的数据类型、有效期规则、room 命名各不相同：

| Writer | Room | 有效期 | 后端 |
|--------|------|--------|------|
| Social Analyst | `sentiment` | +3 天 | ChromaDB |
| News Analyst | `news` | +3 天 | ChromaDB |
| Market Analyst | `market` | +7 天 | ChromaDB |
| Fundamentals Analyst | `fundamentals` | +365 天 | ChromaDB |
| Portfolio Manager | `lessons` | 永不，`valid_from +1d` | ChromaDB |
| Reflector | ——（KG triple） | 单日事实 | SQLite KG |
| 数据管道 | ——（KG triple） | 永久历史事实 | SQLite KG |

若各 Agent 各自调用底层 API，room 命名会不一致，schema 会发散，有效期逻辑会散落各处。**所有写入和读取必须经过统一的 `TradingMemoryStore` 类路由。**

---

### TradingMemoryStore 设计概览

**文件位置**：`tradingagents/agents/utils/memory_store.py`

**类型**：一个 Python 类，在 `trading_graph.py` 初始化时实例化，注入给各 Agent 节点。

**核心设计原则**：
- 方法名即写入者身份，调用方无需知道后端是 ChromaDB 还是 SQLite
- 有效期、`valid_from`、room 名称、metadata schema 全部在类内部管理
- `enabled=False` 时所有方法为空操作，单日测试零开销
- 懒初始化：ChromaDB client 和 SQLite KG 在第一次调用时才创建

**Metadata schema（ChromaDB Drawer）**：

```python
{
    "wing":        "aapl",              # ticker.lower()
    "room":        "sentiment",         # 主题分区
    "trade_date":  "2024-01-15",        # 关联的分析日（用于 L2 精确过滤）
    "valid_from":  "2024-01-15",        # 因果隔离门控（lessons 为 +1 day）
    "expires_at":  "2024-01-18",        # 有效期（永不过期用 "9999-12-31"）
    "recorded_at": "2024-01-15T16:32:00",  # 实际写入时间
    "added_by":    "social_analyst",    # 写入者标识
    "source_file": "aapl_sentiment_2024-01-15",  # Drawer ID 的 hash 基础
    "chunk_index": 0,
    "filed_at":    "2024-01-15T16:32:00",  # MemPalace 兼容字段
}
```

**读取过滤逻辑（所有 L3 查询统一）**：

```python
where = {
    "$and": [
        {"wing":       ticker},
        {"room":       room},
        {"valid_from": {"$lte": as_of}},   # 因果隔离
        {"expires_at": {"$gte": as_of}},   # 有效期过滤
    ]
}
```

**公开写入 API**：
- `store_sentiment_summary(ticker, trade_date, summary)` → Social Analyst
- `store_news_summary(ticker, trade_date, summary)` → News Analyst
- `store_market_summary(ticker, trade_date, summary)` → Market Analyst
- `store_fundamentals(ticker, trade_date, report)` → Fundamentals Analyst
- `store_lesson(ticker, trade_date, lesson, decision, outcome)` → Portfolio Manager
- `annotate_return(ticker, trade_date, actual_return)` → Reflector → SQLite KG
- `store_price(ticker, trade_date, close_price)` → 数据管道 → SQLite KG

**公开读取 API**：
- `retrieve_similar_sentiment(ticker, query, as_of, n_results)` → L3
- `retrieve_sector_sentiment(related_tickers, query, as_of, n_results)` → 跨 Wing L3
- `retrieve_lessons(ticker, query, as_of, n_results)` → L3
- `get_historical_return(ticker, trade_date)` → SQLite KG 精确查找
- `get_price(ticker, trade_date)` → SQLite KG 精确查找

实现文件：`tradingagents/agents/utils/memory_store.py`

---

*本文档基于对 `third_party/mempalace/` 源代码的直接阅读，关键引用已注明文件和功能。具体行号可能随版本更新而变化，请以实际源码为准。*
