# Modification Plan: Judge-Mediated Debate Architecture

## Background & Motivation

原有辩论流程为 Bull → Bear 的顺序单轨交互，存在两层结构性偏差：

1. **后手优势（targeted-rebuttal asymmetry）**：Bear 能读到 Bull 的完整论点后再作答；Bull 第一轮只能基于空白状态发言，处于信息劣势。
2. **新近效应（recency bias）**：Research Manager 接收的上下文中，最后发言方的逻辑最为显著，系统性地偏向 Bear 的立场。

实证依据（MU 回测，20260414_222905 vs 20260414_224253）：1轮单轨输出 **Sell / Underweight**，3轮单轨输出 **Hold / Hold**，差异完全来源于上述结构偏差。

更深层的问题：即使消除顺序偏差，双方仍可能对相同事实作出相反解读而不被追究，或选择性忽视对方的证据。这是**论证质量**问题，不只是顺序问题。

**解决方案**：引入 Judge 角色，双方独立（不互视）进行分析，Judge 专门检测逻辑矛盾与证据缺口，迭代后交由 Research Manager 汇总。

---

## 当前架构（已实现）

### 辩论流程

```
[Analyst layer]
      │
      ▼
Researcher Round          ← Bull 和 Bear 并行独立分析，互不可见
      │
      ▼ (judge_count < judge_iterations)
Judge Researcher          ← 检测逻辑矛盾，向双方各自出具批评指示
      │
      ▼ (static edge，永远回到 Researcher Round)
Researcher Round          ← 双方各自回应 Judge 的指示
      │
      ▼ (judge_count >= judge_iterations)
Research Manager          ← 接收完整对话历史，作出最终决策
```

### API 调用次数

公式：`2 + judge_iterations × 3`（Bull+Bear 每轮并行计为 2 次，Judge 计为 1 次）

| `judge_iterations` | API 调用次数 | 等效时间槽（wall-clock） |
|--------------------|-------------|------------------------|
| 1                  | 5           | 3                      |
| 2                  | 8           | 5                      |
| 3                  | 11          | 7                      |

配置键：`default_config.py` 中 `"judge_iterations": 1`（默认值）。

### 核心设计原则

- **信息隔离**：每轮 Researcher Round 中，Bull 只看自己的历史和 Judge 对 Bull 的指示；Bear 只看自己的历史和 Judge 对 Bear 的指示。双方不交叉阅读。
- **Judge 不作判断**：Judge 仅输出结构化批评，不得给出买卖建议。
- **输出格式强制校验**：Judge 输出必须包含 `<bull_directive>` 和 `<bear_directive>` 两个 XML 块，否则最多重试 3 次；回测模式可配置降级返回空字符串。

---

## 待完成事项

### 1. `fallback_on_failure` 回测模式配置（优先级：低）

当前 `judge_researcher.py` 中 `fallback_on_failure=False`（硬编码在 `setup.py` 的节点创建处）。

建议将其提升为配置键，在回测模式下自动设为 `True`，避免单次格式失败中断整个回测循环：

```python
# default_config.py（待添加）
"judge_fallback_on_failure": None  # None = auto (True in backtest, False in single)
```

---

### 2. Risk Analyst 职能扩充（优先级：中，延后至 Portfolio 阶段）

与 Portfolio Manager 改造一并进行。核心方向：
- 扩充 Risk Analyst 可接触的信息（当前只有四份报告 + 交易提案，缺少 Research Manager 的综合推理和置信度信号）。
- 三档风险偏好（Aggressive / Neutral / Conservative）与 Portfolio Manager 的五档量表对齐，确保 Overweight / Underweight 的颗粒度在风险校准层产生，而不是在 Research Manager 层提前锁定。

---

## 已完成文件变更记录

| 文件 | 状态 | 说明 |
|------|------|------|
| `agents/utils/agent_states.py` | ✅ 已完成 | 新增 `judge_history`、`judge_critique_bull`、`judge_critique_bear`、`judge_count` |
| `agents/utils/judge_parser.py` | ✅ 已完成 | 新建：XML 解析 + 重试逻辑 |
| `agents/researchers/judge_researcher.py` | ✅ 已完成 | 新建：Judge 节点，正式提示词（含迭代条件、3条上限、无问题转达对手论点） |
| `agents/researchers/researcher_round.py` | ✅ 已完成 | 新建：Bull+Bear 并行执行节点 |
| `agents/researchers/bull_researcher.py` | ✅ 已完成 | 提取 `_build_bull_argument()`，正式提示词（数据溯源规则、两级行为约束） |
| `agents/researchers/bear_researcher.py` | ✅ 已完成 | 提取 `_build_bear_argument()`，正式提示词（数据溯源规则、两级行为约束） |
| `graph/conditional_logic.py` | ✅ 已完成 | `max_debate_rounds` → `judge_iterations`，路由方法替换 |
| `graph/propagation.py` | ✅ 已完成 | 新字段初始值补充 |
| `graph/setup.py` | ✅ 已完成 | 新图拓扑：Researcher Round → Judge → Researcher Round 循环 |
| `graph/trading_graph.py` | ✅ 已完成 | config 键更新，`_log_state` 补充 judge 字段 |
| `default_config.py` | ✅ 已完成 | `max_debate_rounds` → `judge_iterations: 1` |
| `agents/__init__.py` | ✅ 已完成 | 导出新 factory 函数 |
| `agents/managers/research_manager.py` | ✅ 已完成 | 正式提示词（身份重定义、数据溯源规则、Judge 四步阅读框架、结构化输出） |
| `notes/agent_prompts.md` | ✅ 已完成 | 新增 Judge 为第 7 号角色，更新 Bull/Bear/Research Manager 章节，更新 LLM 层对应表 |

---

## Changelog

### 2026-04-18 — Research Phase Prompt 全面修订（本次会话）

**Judge Researcher**
- 替换占位符为正式提示词：Task 1（一致性检查，仅第 1 轮激活）、Task 2（交互检查：对立解读 + 未回应新论点）、Task 3（逻辑有效性）。
- 指令措辞约束：只能以"请解释/请说明来源/请回应对方"的形式发出，禁止直接断言"你是错的"。
- 新增每轮每侧最多 3 条指令上限，优先级 Task 2 > Task 3 > Task 1。
- 无问题时不再输出空指令，改为转达对手最强论点要求 rebuttal。

**Bull / Bear Researcher**
- 新增数据溯源绝对规则：所有主张必须来自四份分析师报告。
- 将对 Judge 的行为约束拆分为两级：事实层（可修正，无来源必须撤回）/ 立场层（受保护，方向性结论不因 Judge 指令而动摇）。
- 删除旧提示词中的 Bull/Bear Counterpoints 结构，改为直接回应 Judge 指令的框架。

**Research Manager**
- 身份重定义：从"debate facilitator + portfolio manager"改为"研究阶段最终裁定者"。
- 新增数据溯源规则：综合判断只能基于辩论内容和历史记忆，不得引入辩论外的分析。
- 新增 Judge 四步阅读框架：剔除撤回主张→权衡交互检查回应质量→标记未解决歧义→基于存活论点综合。
- 输出格式改为三段式结构：Recommendation / Reasoning / Investment Plan。

**延后事项**
- Risk Analyst 职能扩充（接触信息范围 + 与 Portfolio Manager 五档量表对齐）延后至 Portfolio 阶段改造时一并完成。

---

### 2026-04-18 — Judge-Mediated Architecture 骨架实现

**废弃内容（原 Symmetric Debate Architecture 方案）**

- 废弃双轨并行辩论方案（dual-track，Track A / Track B）：该方案仅消除顺序偏差，不解决论证质量问题，且工程复杂度高。
- 废弃 `investment_debate_state_b`、`bull_closing_statement` 字段计划。
- 废弃 `parallel_debate.py`、`bull_closing.py` 新建计划。
- 废弃 `parallel_debate` 配置键。
- 废弃 Config A / B / C 三档配置体系，统一替换为 `judge_iterations`。

**新增内容**

- 确立 Judge-Mediated 架构：Bull 和 Bear 独立分析，不互视；Judge 检测逻辑矛盾与证据缺口；迭代完成后交 Research Manager 汇总。
- 新增 `judge_iterations` 配置键（默认值 1），对应 API 调用次数 5 / 8 / 11。
- 新建 `judge_parser.py`：`<bull_directive>` / `<bear_directive>` XML 解析，支持最多 3 次重试，可配置降级模式。
- 新建 `judge_researcher.py`：Judge 节点，提示词为占位符，待下一步设计。
- 新建 `researcher_round.py`：Bull+Bear 并行执行节点，基于 `ThreadPoolExecutor`。
- 重构 `bull_researcher.py` / `bear_researcher.py`：提取核心逻辑为模块级函数，改读 `judge_critique_bull` / `judge_critique_bear`，原 LangGraph 节点包装器保留作兼容接口。
- 更新图拓扑：`Researcher Round → Judge Researcher → Researcher Round`（循环）→ `Research Manager`。
- `_log_state` 新增记录 judge 相关字段。
