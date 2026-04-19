"""
tradingagents/agents/researchers/judge_researcher.py

Judge Researcher node for the investment debate.

The Judge sits between each pair of parallel Bull/Bear rounds. It receives
the full accumulated histories of both analysts plus its own prior critiques,
identifies logical inconsistencies and evidentiary gaps, then issues targeted
directives to each side via structured XML output.

The Judge does NOT render a final verdict — that is the Research Manager's
role. It only deepens the quality of argumentation before passing everything
to the Manager.

Prompt enforces: (1) individual consistency checks against analyst reports,
(2) cross-examination of conflicting interpretations and unaddressed new points,
(3) logical validity review. Directives are phrased as questions/requests only —
never as direct assertions that a claim is wrong.
"""

import logging

from tradingagents.agents.utils.judge_parser import parse_judge_output_with_retry

logger = logging.getLogger(__name__)


def create_judge_researcher(llm, fallback_on_failure: bool = False):
    """
    Create the Judge Researcher LangGraph node.

    Args:
        llm:                  LLM instance — should be the deep-thinking model
                              (same tier as Research Manager).
        fallback_on_failure:  If True, format failures after all retries return
                              empty directives instead of raising RuntimeError.
                              Set True in backtest mode to avoid aborting a run.

    Returns:
        A LangGraph-compatible node function.
    """

    def judge_node(state: dict) -> dict:
        debate_state = state["investment_debate_state"]

        bull_history   = debate_state.get("bull_history",      "")
        bear_history   = debate_state.get("bear_history",      "")
        judge_history  = debate_state.get("judge_history",     "")
        judge_count    = debate_state.get("judge_count",        0)
        full_history   = debate_state.get("history",           "")

        # Task 1 (consistency check) is only active on the first iteration.
        if judge_count == 0:
            consistency_task_block = (
                "TASK 1 — INDIVIDUAL CONSISTENCY CHECK [ACTIVE: first iteration only]\n"
                "For each analyst, examine whether every factual claim, statistic, and market observation they make "
                "can be traced back to the four analyst reports available to them (market research report, sentiment "
                "report, news report, fundamentals report). If a claim appears invented, extrapolated beyond what the "
                "reports explicitly state, or significantly overstated, issue a directive asking the analyst to cite "
                "the specific report and section that supports that claim.\n"
            )
        else:
            consistency_task_block = (
                "TASK 1 — INDIVIDUAL CONSISTENCY CHECK [INACTIVE: only performed in iteration 1. "
                "Do NOT perform this task in this iteration.]\n"
            )

        prompt = f"""You are an impartial Debate Judge overseeing an investment analysis debate between a Bull Analyst and a Bear Analyst. Your role is strictly methodological: you evaluate the logical quality and evidentiary grounding of arguments. You do NOT form any view on whether to buy, sell, or hold the stock. You have no emotional stake in either side.

Your work consists of THREE tasks:

---

{consistency_task_block}
TASK 2 — CROSS-EXAMINATION
A. Conflicting interpretations: Identify cases where BOTH analysts cite the same fact, phenomenon, or market event but reach opposite conclusions. When found, relay the opponent's interpretation to each side and require each analyst to provide a deeper, evidence-grounded analysis explaining why their interpretation is more logically consistent with the full body of evidence.

B. Unaddressed new points: Identify cases where one analyst raises a new, substantive point which the other analyst has not addressed. Relay that point to the analyst who has not addressed it and require a substantive rebuttal grounded in evidence from the reports.

TASK 3 — LOGICAL VALIDITY
Flag any logical fallacies, unsupported inferential leaps, or circular reasoning in either analyst's argument. Ask the responsible analyst to provide direct supporting evidence from the reports or to clarify the logical connection.

---

HARD CONSTRAINTS — observe all of these without exception:

1. EMOTIONAL NEUTRALITY: Your directives must contain zero sentiment about the investment outcome. Do not use language that implies one side is winning, stronger, or more credible.

2. DIRECTIVE PHRASING — you must NEVER tell an analyst their claim is "wrong," "incorrect," "mistaken," or "flawed." Frame all directives as requests to explain, source, or respond.

   CORRECT phrasing:
   - "Please identify which section of the analyst reports supports your claim that [X]."
   - "The Bear Analyst has argued [Y]. Please provide a more detailed analysis of why your interpretation of [shared phenomenon] is more consistent with the available evidence."
   - "Your argument relies on the premise that [Z]. Please elaborate on the evidentiary basis for this premise."
   - "Please clarify the logical connection between [evidence cited] and [conclusion drawn]."

   NEVER use:
   - "Your claim about X is wrong / incorrect / unsupported."
   - "This is a weak / poor / flawed argument."
   - "The Bear is right / wrong to suggest..."
   - Any language expressing your own view on the investment merits.

3. NO SUMMARIES: Do not summarize or paraphrase the debate. Issue only targeted directives tied to specific claims.

4. NO VERDICTS: You do not decide, hint at, or imply a final investment recommendation.

5. DIRECTIVE LIMIT: Issue at most 3 directives per analyst per round. Select only the most critical issues. When deciding what to include, prioritize Task 2 (cross-examination) over Tasks 1 and 3.

6. IF NO ISSUES FOUND: If after careful examination you find no logical or evidentiary issues for one side, do NOT leave the directive empty. Instead, identify the single strongest argument made by the OPPOSING analyst that this side has not yet addressed, relay it clearly and in full, and ask this side to provide a substantive rebuttal grounded in evidence from the reports.

---

=== Bull Analyst — full argument history ===
{bull_history if bull_history else "(no arguments yet)"}

=== Bear Analyst — full argument history ===
{bear_history if bear_history else "(no arguments yet)"}

=== Your previous critiques (Judge history) ===
{judge_history if judge_history else "(none — this is your first critique)"}

=== Iteration ===
This is Judge iteration {judge_count + 1}.
{"Task 1 (consistency check) is ACTIVE this iteration." if judge_count == 0 else "Task 1 (consistency check) is INACTIVE this iteration — proceed directly to Tasks 2 and 3."}

You MUST end your response with EXACTLY the following XML structure.
Do not add any text after </bear_directive>.

<bull_directive>
[Up to 3 directives addressed to the Bull Analyst only. Reference specific claims from their arguments. Follow the required phrasing constraints.]
</bull_directive>

<bear_directive>
[Up to 3 directives addressed to the Bear Analyst only. Reference specific claims from their arguments. Follow the required phrasing constraints.]
</bear_directive>"""

        bull_directive, bear_directive = parse_judge_output_with_retry(
            llm,
            prompt,
            max_retries=3,
            fallback_on_failure=fallback_on_failure,
        )

        judge_turn = (
            f"\n--- Judge Critique (Iteration {judge_count + 1}) ---\n"
            f"[To Bull Analyst]: {bull_directive}\n"
            f"[To Bear Analyst]: {bear_directive}\n"
        )

        new_debate_state = {
            "bull_history":      bull_history,
            "bear_history":      bear_history,
            "history":           full_history + "\n" + judge_turn,
            "judge_history":     judge_history + "\n" + judge_turn,
            "judge_critique_bull": bull_directive,
            "judge_critique_bear": bear_directive,
            "judge_count":       judge_count + 1,
            # Carry-forward fields (unchanged by Judge)
            "current_response":  debate_state.get("current_response", ""),
            "judge_decision":    debate_state.get("judge_decision",   ""),
            "count":             debate_state.get("count",             0),
        }

        return {"investment_debate_state": new_debate_state}

    return judge_node
