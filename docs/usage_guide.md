# TradingAgents 改版使用指南

> 文档版本：基于 commit 10c136f + 改造 A（双 Provider 配置）+ CLI Step 6/7 重构（2026-04-10）
> 适用人群：项目小组成员，包括本地部署、Google Colab、纯 API 调用等多种场景

---

## 目录

1. [架构概述](#一架构概述)
2. [支持的 LLM Provider 一览](#二支持的-llm-provider-一览)
3. [环境变量配置](#三环境变量配置)
4. [Python API 使用方法](#四python-api-使用方法)
5. [CLI 使用方法](#五cli-使用方法)
6. [各场景配置示例](#六各场景配置示例)
7. [Provider 注意事项](#七provider-注意事项)
8. [记忆系统与运行模式](#八记忆系统与运行模式)

---

## 一、架构概述

TradingAgents 使用双 LLM 配置，两个角色可以独立使用不同的 provider 和模型：

| 角色 | 配置字段 | 负责的 Agent |
|------|---------|-------------|
| **深度推理（Deep）** | `deep_think_provider` + `deep_think_llm` | Research Manager、Portfolio Manager（最终决策层） |
| **快速任务（Quick）** | `quick_think_provider` + `quick_think_llm` | 4 个 Analyst、Bull/Bear Researcher、Trader、Risk 辩论团队、Reflector |

**核心原则**：决策层（Manager）使用最强的推理模型；数据采集和辩论层用高性价比的快速模型。

---

## 二、支持的 LLM Provider 一览

| Provider 名称 | 字符串值 | 所需 API Key | 默认 Endpoint |
|-------------|---------|------------|--------------|
| OpenAI | `"openai"` | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| Anthropic | `"anthropic"` | `ANTHROPIC_API_KEY` | SDK 自动管理 |
| Google | `"google"` | `GOOGLE_API_KEY` | SDK 自动管理 |
| xAI | `"xai"` | `XAI_API_KEY` | `https://api.x.ai/v1` |
| OpenRouter | `"openrouter"` | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` |
| **HuggingFace** | `"huggingface"` | `HF_TOKEN` | `https://api-inference.huggingface.co/v1/` |
| Ollama（本地） | `"ollama"` | 不需要 | `http://localhost:11434/v1` |

### HuggingFace 两种接入方式

**方式 A：HuggingFace Inference API（云端）**
- 无需本地 GPU
- 设置 `HF_TOKEN` 环境变量即可
- 适合 Google Colab 用户或不想在本地运行大模型的用户
- 支持 70B+ 模型（需有 PRO 账号或已申请权限的 gated 模型）

**方式 B：本地 vLLM 服务（本地 GPU / Colab A100）**
- 在 Colab 或本地用 vLLM 启动一个 OpenAI 兼容服务器
- 设置 `backend_url` 指向 vLLM 地址（如 `http://localhost:8000/v1`）
- 此时 `quick_think_provider` 仍填 `"huggingface"`，但实际流量走 vLLM

---

## 三、环境变量配置

在项目根目录创建 `.env` 文件（参考 `.env.example`）：

```bash
# 按实际使用的 provider 填写，不使用的可以留空或删除

# Anthropic（用于 Claude Opus 4.6 manager 层）
ANTHROPIC_API_KEY=sk-ant-...

# Google（用于 Gemini Flash 快速层）
GOOGLE_API_KEY=AIza...

# OpenAI（用于 GPT manager 层）
OPENAI_API_KEY=sk-...

# HuggingFace（用于 HF Inference API 快速层）
HF_TOKEN=hf_...

# 其他（按需填写）
XAI_API_KEY=...
OPENROUTER_API_KEY=...
```

---

## 四、Python API 使用方法

### 基础用法

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from dotenv import load_dotenv

load_dotenv()  # Load .env file

config = DEFAULT_CONFIG.copy()
# Override providers and models here (see examples below)

ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate("NVDA", "2024-05-10")
print(decision)  # BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL
```

### 配置字段说明

```python
config = {
    # --- Provider 配置 ---
    "deep_think_provider":  str,   # manager 层 provider，见上表
    "deep_think_llm":       str,   # manager 层模型名
    "quick_think_provider": str,   # analyst/researcher 层 provider
    "quick_think_llm":      str,   # analyst/researcher 层模型名

    # Ollama / vLLM / OpenRouter 等 OpenAI 兼容端点的 URL
    # cloud provider（anthropic/google）填 None 即可
    "backend_url":          str | None,

    # --- Provider 专属思考配置（可选）---
    "anthropic_effort":         None | "low" | "medium" | "high",
    "google_thinking_level":    None | "minimal" | "low" | "medium" | "high",
    "openai_reasoning_effort":  None | "low" | "medium" | "high",

    # --- 并行控制 ---
    # 仅控制4个 analyst 层（market/social/news/fundamentals）的并行
    # Risk analyst（aggressive/conservative/neutral）始终保持串行（即时反驳语义）
    # None  → 自动检测：cloud provider 默认并行，ollama/huggingface 默认串行
    # True  → 强制并行（本地模型慎用）
    # False → 强制串行
    "parallel_analysts":      None | bool,

    # --- 其他 ---
    "max_debate_rounds":      int,  # Bull/Bear 辩论轮次（默认 1）
    "max_risk_discuss_rounds": int, # 风险辩论轮次（默认 1）
    "output_language":        str,  # 输出语言，默认 "English"
}
```

---

## 五、CLI 使用方法

### 安装

从 GitHub clone 仓库后，执行：

```bash
pip install -e .
```

安装完成后即可直接使用 `tradingagents` 命令。

### 启动方式

```bash
# 方式一：通过已安装的命令（推荐）
tradingagents

# 方式二：直接运行模块
python -m cli.main
```

---

<!-- ============================================================
     [DEV ONLY — 发布前删除此节]

     本地开发调试安装方式：
     直接执行 `tradingagents` 会调用 pip 安装的旧版本，不会反映本地修改。
     需用可编辑模式安装，使命令指向本地源码目录：

         cd /path/to/TradingAgents
         pip install -e .

     验证是否指向本地目录（而非 site-packages）：

         pip show tradingagents | grep Location

     安装一次后，后续对代码的任何修改均立即生效，无需重新安装。
     ============================================================ -->

CLI 交互流程（共 7 步）：

1. **Ticker Symbol** — 输入股票代码（如 `NVDA`、`0700.HK`）
2. **Analysis Date** — 输入分析日期（`YYYY-MM-DD`）
3. **Output Language** — 选择报告语言
4. **Analysts Team** — 勾选启用的分析师（最多 4 个）
5. **Research Depth** — 选择辩论深度（Shallow / Medium / Deep）
6. **Analyst LLM** — 为 Analyst / Researcher / Trader 层选 provider，再选模型
7. **Manager LLM** — 为 Research Manager / Portfolio Manager 层选 provider，再选模型

### CLI 选择流程说明

Steps 6 & 7 均采用相同的两级交互：

```
? Select your LLM Provider:   ← 第一级：选 provider（OpenAI / Google / Anthropic / Ollama 等）
? Select Your [Quick/Deep-Thinking LLM Engine]:  ← 第二级：选该 provider 下的具体模型
```

选完 provider 后，若该 provider 支持额外参数，会进一步询问：
- **Anthropic** → Effort Level（high / medium / low）
- **Google** → Thinking Mode（Enable / Minimal）
- **OpenAI** → Reasoning Effort（high / medium / low）

两步选择完全独立，支持任意组合，例如 Ollama analyst + Anthropic manager，或 Google analyst + OpenAI manager。

### 常用组合参考

| Analyst LLM | Manager LLM | 需要的 Key |
|-------------|-------------|-----------|
| Ollama（本地 Gemma4） | Anthropic Claude Opus 4.6 | `ANTHROPIC_API_KEY` |
| Google Gemini Flash | Anthropic Claude Opus 4.6 | `GOOGLE_API_KEY` + `ANTHROPIC_API_KEY` |
| Ollama（本地模型） | Ollama（本地模型） | 不需要 |
| HuggingFace Llama | OpenAI GPT | `HF_TOKEN` + `OPENAI_API_KEY` |
| Google Gemini Flash | Google Gemini Pro | `GOOGLE_API_KEY` |

---

## 六、各场景配置示例

### 场景 A：推荐配置（Claude + Gemini）

```python
# 需要：ANTHROPIC_API_KEY, GOOGLE_API_KEY
config = DEFAULT_CONFIG.copy()
config["deep_think_provider"]  = "anthropic"
config["deep_think_llm"]       = "claude-opus-4-6"
config["quick_think_provider"] = "google"
config["quick_think_llm"]      = "gemini-3-flash-preview"
# backend_url 留 None，两个 SDK 自动读各自的 API key
```

### 场景 B：混合配置（Claude 决策层 + 本地 Ollama 分析层）

```python
# 需要：ANTHROPIC_API_KEY + 本地运行 Ollama
config = DEFAULT_CONFIG.copy()
config["deep_think_provider"]  = "anthropic"
config["deep_think_llm"]       = "claude-opus-4-6"
config["quick_think_provider"] = "ollama"
config["quick_think_llm"]      = "gemma4:latest"       # 替换为你的 Ollama 实际 tag
config["backend_url"]          = "http://localhost:11434/v1"
```

> **并行说明**：`quick_think_provider = "ollama"` 时，`parallel_analysts` 默认为 `False`（串行），
> 避免多线程同时向 Ollama 发请求造成排队延迟。如需并行可显式设置 `config["parallel_analysts"] = True`，
> 此时 Ollama 会将请求自动串行排队，实际效果取决于你的硬件。
> 决策层（deep_think）走云端 Claude，不占本地 VRAM。

### 场景 C：Colab / HuggingFace（OpenAI 决策层 + HF 分析层）

```python
# 需要：OPENAI_API_KEY, HF_TOKEN
# 适合：Google Colab 用户，使用 HuggingFace Inference API（云端）

config = DEFAULT_CONFIG.copy()
config["deep_think_provider"]  = "openai"
config["deep_think_llm"]       = "gpt-5.4"              # 或 "gpt-4.1" 等
config["quick_think_provider"] = "huggingface"
config["quick_think_llm"]      = "meta-llama/Llama-3.3-70B-Instruct"
config["backend_url"]          = None                   # 使用 HF 默认 API 端点
```

### 场景 D：Colab + 本地 vLLM（Colab A100 自托管）

```python
# 需要：OPENAI_API_KEY
# 在 Colab 中先用 vLLM 启动本地服务，再运行 TradingAgents

# Step 1（在 Colab 中先执行）：
# !pip install vllm
# !python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Llama-3.3-70B-Instruct \
#     --port 8000 &

# Step 2：配置 TradingAgents
config = DEFAULT_CONFIG.copy()
config["deep_think_provider"]  = "openai"
config["deep_think_llm"]       = "gpt-5.4"
config["quick_think_provider"] = "huggingface"          # provider 仍填 huggingface
config["quick_think_llm"]      = "meta-llama/Llama-3.3-70B-Instruct"
config["backend_url"]          = "http://localhost:8000/v1"  # 指向本地 vLLM
# backend_url 覆盖 HF 默认端点，流量走本地 vLLM
```

### 场景 E：完全本地（Ollama，无 API key）

```python
# 需要：本地运行 Ollama
config = DEFAULT_CONFIG.copy()
config["deep_think_provider"]  = "ollama"
config["deep_think_llm"]       = "gemma4:latest"       # 替换为你的 Ollama 实际 tag
config["quick_think_provider"] = "ollama"
config["quick_think_llm"]      = "gemma4:latest"
config["backend_url"]          = "http://localhost:11434/v1"
# 注意：Manager 层也使用本地模型，决策质量低于云端 Claude/GPT
# Ollama tag 必须与 `ollama list` 输出完全一致（含大小写和冒号）
```

### 场景 F：纯 OpenAI（简单上手）

```python
# 需要：OPENAI_API_KEY
config = DEFAULT_CONFIG.copy()
config["deep_think_provider"]  = "openai"
config["deep_think_llm"]       = "gpt-5.4"
config["quick_think_provider"] = "openai"
config["quick_think_llm"]      = "gpt-5.4-mini"
```

---

## 七、Provider 注意事项

### HuggingFace Inference API

- **Tool calling 支持**：并非所有模型都支持 Function Calling / Tool Use。TradingAgents 中 Analyst 层需要 tool call 能力。
  - **推荐模型**（已验证支持 tool call）：
    - `meta-llama/Llama-3.3-70B-Instruct`
    - `Qwen/Qwen2.5-72B-Instruct`
  - 使用不支持 tool call 的模型会导致 Analyst 阶段报错。
- **Speed Tier**：HF Inference API 免费版有速率限制；PRO 账号（$9/月）限制更宽松。
- **Gated 模型**：Llama 系列需在 HuggingFace 上申请访问权限后才能通过 `HF_TOKEN` 调用。

### Ollama（本地）

- **并发排队**：多个 Agent 同时调用时，Ollama 会自动串行排队，不会加载多个模型实例。
- **模型 tag**：`config["quick_think_llm"]` 必须与 `ollama list` 中显示的 tag 完全一致，包括大小写和冒号。
- **内存限制**：31B 量化模型约需 20GB VRAM。**不要**同时将 `deep_think_provider` 和 `quick_think_provider` 都设为 `ollama` 且使用不同模型——会触发模型切换，速度极慢。同一模型则无问题。

### Anthropic（Claude）

- `anthropic_effort`：控制 Claude 4.5+/4.6 的推理深度（`"high"` / `"medium"` / `"low"`）。设为 `None` 时由 Claude 自动决定。
- `backend_url` 对 Anthropic 无效（SDK 自管端点），传 `None` 即可。

### Google（Gemini）

- `google_thinking_level`：控制 Gemini 的思考模式。Gemini 3 Flash 默认不需要额外配置。
- `backend_url` 对 Google 无效（SDK 自管端点），传 `None` 即可。

---

## 八、记忆系统与运行模式

### 两种运行模式

TradingAgents 支持两种运行模式，由 config 中的 `run_mode` 字段控制：

| 模式 | 值 | 记忆行为 | 适用场景 |
|------|---|---------|---------|
| **单日分析** | `"single"`（默认） | 记忆禁用，无磁盘写入 | CLI 交互分析、单次测试 |
| **回测模式** | `"backtest"` | ChromaDB + SQLite 持久化，跨日积累 | `run_backtest()` 多日循环 |

```python
# 单日分析（默认，无需显式设置）
config["run_mode"] = "single"

# 回测模式（改造 G 实现后使用）
config["run_mode"] = "backtest"
```

### 记忆系统架构

所有记忆读写统一经由 `TradingMemoryStore`（`tradingagents/agents/utils/memory_store.py`）路由，调用方（各 agent 节点）不直接接触存储后端。

**后端分工：**
- **ChromaDB**（向量数据库，`./memory/` 目录）：存储文本类记忆，支持语义检索
- **SQLite KG**（`./memory/kg.sqlite3`）：存储结构化数值（实际收益率、收盘价），精确查询

**记忆 Room 一览：**

| Room | 写入者 | 有效期 | valid_from |
|------|--------|--------|-----------|
| `sentiment` | Social Analyst | 3 天 | 当天 |
| `news` | News Analyst（待接线） | 3 天 | 当天 |
| `market` | Market Analyst（待接线） | 7 天 | 当天 |
| `fundamentals` | Fundamentals Analyst（待接线） | 365 天 | 当天 |
| `lessons` | Portfolio Manager（待接线） | 永不过期 | 当天 +1 日 |
| `reflections_bull` | Reflector（Bull） | 永不过期 | 当天 +1 日 |
| `reflections_bear` | Reflector（Bear） | 永不过期 | 当天 +1 日 |
| `reflections_trader` | Reflector（Trader） | 永不过期 | 当天 +1 日 |
| `reflections_invest_judge` | Reflector（Research Manager） | 永不过期 | 当天 +1 日 |
| `reflections_portfolio_manager` | Reflector（Portfolio Manager） | 永不过期 | 当天 +1 日 |

### 因果隔离机制

记忆系统内建防未来信息泄露机制：

1. `trading_graph.py` 在每天分析开始前调用 `memory_store.set_analysis_date(trade_date)`，统一设定当日日期
2. 所有读操作隐式过滤 `valid_from <= analysis_date`，确保只能看到当天及之前的记忆
3. 反思类记忆（`reflections_*`）和 lessons 的 `valid_from = T+1`，不会在写入当天被读取

### 回测模式下的记忆流程

```
每天开始:  memory_store.set_analysis_date(date)
分析阶段:  agents 读取 retrieve_reflections() → 语义检索历史反思
分析结束:  reflect_and_remember(actual_return) 被调用:
             ├── 5 个角色各写入 store_reflection() → ChromaDB
             └── annotate_return() → SQLite KG
次日分析:  昨天的反思（valid_from = today）开始可被检索
```

### `memory_palace_path` 自定义

ChromaDB 和 SQLite 的存储目录默认为 `./memory/`，可通过以下方式覆盖：

```python
# Python API
config["memory_palace_path"] = "/path/to/custom/memory"

# 环境变量（优先级更高）
export TRADINGAGENTS_MEMORY_PATH=/path/to/custom/memory
```

---

## 九、开发者模式（本地修改版启动指南）

> 本节适用于在本地对 TradingAgents 源码进行修改、调试并运行最新版本的开发者。

### 一次性安装（可编辑模式）

首次设置时，在项目根目录执行：

```bash
cd /path/to/TradingAgents
pip install -e .
```

安装后，`tradingagents` 命令将直接指向本地源码目录。此后对源码的任何修改均**立即生效**，无需重新安装。

### 日常开发工作流

```
修改代码 → 直接运行 tradingagents（无需任何额外操作）
```

依赖包未变动时，**不需要**重新 `pip install`，也不需要重启任何服务。

若你在 `pyproject.toml` 或 `setup.py` 中新增了依赖包，则需要重新执行一次 `pip install -e .`。

### 已知陷阱：`ModuleNotFoundError: No module named 'tradingagents.dataflows.interface'`

**症状**：运行 `tradingagents` 时报上述错误，但 `python -c "from tradingagents.dataflows.interface import route_to_vendor"` 却正常。

**原因**：site-packages 中残留了一个孤立的 `tradingagents/` 命名空间目录（通常是之前 `pip install tradingagents` 非可编辑安装后产生的缓存文件夹）。Python 的标准 `PathFinder` 在可编辑安装的自定义 finder 之前运行，会优先找到 site-packages 中的残留目录，导致 `tradingagents.dataflows` 解析到 site-packages（该路径下没有 `interface.py`），而非项目源码。

**修复方法**（一次性操作）：

```bash
# 查找并删除 site-packages 中的残留目录
rm -rf "$(python -c "import site; print(site.getsitepackages()[0])")/tradingagents/"

# 验证已清除
ls "$(python -c "import site; print(site.getsitepackages()[0])")/tradingagents/" 2>&1
# 应输出：No such file or directory

# 可选：重新确认可编辑安装仍然有效
tradingagents --help
```

**预防**：该残留目录通常由 data_cache 文件写入 site-packages 触发（`default_config.py` 将 `data_cache_dir` 设置在 `tradingagents/dataflows/data_cache/` 下，若包曾以非可编辑模式安装过，缓存文件会写入 site-packages 内部）。修复一次后，只要不再执行 `pip install tradingagents`（非 `-e` 模式），即不会复现。

### 验证当前安装状态

```bash
# 确认 tradingagents 指向本地项目目录（而非 site-packages）
pip show tradingagents | grep Location
# 应显示：Editable project location: /path/to/TradingAgents

# 确认 tradingagents.dataflows 解析到本地源码
python -c "import tradingagents.dataflows; print(tradingagents.dataflows.__file__)"
# 应显示：/path/to/TradingAgents/tradingagents/dataflows/__init__.py
# 若显示 None 或 site-packages 路径，则按上方「修复方法」处理
```

---

## 十、社交媒体分析依赖（FinBERT）

Social Media Analyst 使用 `ProsusAI/finbert` 对 Reddit 和 StockTwits 的帖子进行批量情绪分析，在传给 LLM 之前将原始帖子压缩为结构化摘要。

### FinBERT 不包含在默认依赖中

`transformers` 和 `torch` 体积较大，未列入 `pyproject.toml` 的默认依赖。需手动安装：

```bash
pip install transformers torch
```

> 若未安装，系统不会崩溃——`finbert_aggregate()` 会自动降级为纯文本列表输出。但 Social Media Analyst 的报告质量会下降（LLM 接收的是未过滤、未压缩的原始帖子）。

### 模型存储位置

**不需要在项目文件夹内建专用目录。** HuggingFace 自动将模型缓存到：

```
~/.cache/huggingface/hub/models--ProsusAI--finbert/   (~440MB, 一次性下载)
```

上传 GitHub 时无需任何额外操作——模型权重不在项目目录内，不会被 git 追踪。

### 预下载（可选，避免首次分析时等待）

```bash
python -c "
from transformers import pipeline
print('Downloading FinBERT...')
pipeline('text-classification', model='ProsusAI/finbert')
print('Done. Model cached at ~/.cache/huggingface/')
"
```

### 使用自定义缓存路径（可选）

如需将模型缓存放在指定位置（例如外接硬盘或团队共享目录），在 `.env` 中设置：

```bash
HF_HOME=/your/custom/path
```

HuggingFace 的所有模型缓存（包括 FinBERT）都会写入该路径。

---

---

## 十一、回测模式：多空策略与可视化输出

### 策略说明

CLI 回测模式（`--mode backtest`）的 TA-Signal / TA-Filtered 使用与论文（TradingAgents, 2024）Fig.6 一致的 **stacking（叠加）策略**；TA-Scaled 使用置信度加权状态机（本项目扩展）。

#### Bounded-Stack 策略（TA-Signal / TA-Filtered）

状态机步进式过渡，每次信号将仓位移动一步，但上下限为 ±1：

```
信号         状态变化
BUY / OW  →  state = min(state + 1, +1)
SELL / UW →  state = max(state − 1, −1)
HOLD      →  state 不变

state ∈ {−1, 0, +1}
position = state  （直接用于日收益乘法）
```

- 从做多 +1 切换到做空 −1 需要**两次 SELL** 信号（+1 → 0 → −1）
- 从做空 −1 切换到做多 +1 需要**两次 BUY** 信号（−1 → 0 → +1）
- 避免一个 SELL 信号直接产生过激的方向翻转，仓位始终有界

#### TA-Scaled（置信度加权状态机，本项目扩展）

```
weight    = clip((confidence − 0.50) × 2, 0, 1)
position  = ±1 × weight   （方向由 BUY/SELL/HOLD 状态机决定）
portfolio = _simulate(position, daily_returns, initial_capital)
```

#### TradingAgents 三种变体

| 变体名称 | 策略类型 | 说明 |
|---------|---------|------|
| **TA-Signal** | Stacking（论文对齐） | 每次信号执行固定金额交易，无过滤 |
| **TA-Filtered** | Stacking（论文对齐） | 仅在 confidence ≥ threshold（默认 0.65）时执行 |
| **TA-Scaled** | 置信度加权状态机 | 仓位大小随置信度线性缩放，本项目扩展 |

### 传统策略基线

与论文 Table 1 一致，所有基线均使用多空 {−1, +1} 状态机（B&H 恒为 +1）：

| 基线策略 | 信号逻辑概述 |
|---------|------------|
| Buy & Hold | 始终做多 |
| SMA | 价格 vs 5/20日均线金叉/死叉 |
| MACD | 12/26/9 MACD 金叉/死叉 |
| KDJ + RSI | KDJ K线金叉 + RSI 超买超卖过滤 |
| ZMR | 20日对数收益 z-score 均值回归 |

### 回测流程与自动输出

启动回测后（`tradingagents --mode backtest`），CLI 自动依次完成：

1. **逐日分析**：调用 TradingAgents 完成每日信号生成
2. **置信度校准表格**（控制台输出）：按 0.10 宽度的置信度 bucket 汇总方向准确率

   ```
   Confidence Interval │  N  │ Actual Accuracy │ Assessment
   ────────────────────┼─────┼─────────────────┼─────────────────
   0.50–0.60           │  12 │      58.3%      │ OK
   ...
   ```

3. **输出子目录**：每个 ticker 自动在 `results/` 下创建独立子目录：
   ```
   results/{TICKER}-{start_date}-{end_date}/
   ├── backtest_{TICKER}_{timestamp}.csv   # 逐日信号 + 收益率
   ├── analysis.png                        # 策略对比图表（四格）
   └── metrics.csv                         # 各策略性能指标表格
   ```

   图表包含四格：
   - 各策略权益曲线对比
   - TA-Signal stacking 单位数柱状图 + TA-Scaled 叠加（paper-aligned 注释）
   - 置信度分布直方图
   - 置信度校准散点图

### 单独运行可视化（Python API）

```python
import matplotlib
matplotlib.use("Agg")
from tradingagents.graph.backtest_analyze import _build_figure, _download_ohlcv, _write_metrics_csv
import pandas as pd

# 加载已有的回测结果（含 signal, confidence, actual_return 列）
df = pd.read_csv("results/AAPL-2024-01-01-2024-03-29/backtest_AAPL_20240101_120000.csv")

# 下载 OHLCV 数据
prices = _download_ohlcv("AAPL", "2024-01-02", "2024-04-08")

# 生成图表（返回 figure + values 字典）
fig, values = _build_figure(df=df, ticker="AAPL", ohlcv=prices,
                            initial_capital=10000.0, threshold=0.65, rf_annual=0.05)
fig.savefig("aapl_analysis.png", dpi=150, bbox_inches="tight")

# 保存指标 CSV
_write_metrics_csv(values, "aapl_metrics.csv", rf_annual=0.05)
```

### Demo 模式（独立测试）

无需真实回测数据，使用合成数据生成演示图表（自动输出至 `results/DEMO-*/`）：

```bash
python -m tradingagents.graph.backtest_analyze --demo
```

---

---

## 十二、致谢与许可证声明

本项目基于以下两个开源项目修改而来，均已在项目根目录保留原始许可证全文。

### TradingAgents

| 项目 | TradingAgents |
|------|---------------|
| **原作者** | Yijia Xiao et al. (TauricResearch) |
| **原始仓库** | https://github.com/TauricResearch/TradingAgents |
| **许可证** | Apache License 2.0（见 `LICENSE`） |
| **本项目改动** | 双 Provider LLM 架构、回测框架、记忆系统、CLI 重构等（详见 `notes/modification_plan.md`） |

### MemPalace

| 项目 | MemPalace |
|------|-----------|
| **原作者** | milla-jovovich |
| **原始仓库** | https://github.com/milla-jovovich/mempalace |
| **许可证** | MIT License（见 `THIRD_PARTY_LICENSES.md`） |
| **使用方式** | `knowledge_graph.py` 以 vendor 形式并入 `tradingagents/agents/utils/knowledge_graph.py`，用于回测模式下的 SQLite 时序知识图谱 |

> Apache 2.0 与 MIT 均允许在保留原始版权声明的前提下自由修改与再发布，两者兼容。

---

*本文档对应改造 A（双 Provider 配置）+ CLI Step 6/7 重构 + 改造 B 第一阶段（社交媒体数据源 + FinBERT）+ 改造 G.7.2（置信度校准回路 + 回测可视化集成，2026-04-12）。*
