"""
tradingagents/agents/utils/judge_parser.py

Parser and retry wrapper for Judge Researcher output.

The Judge is required to terminate every response with two XML blocks:

    <bull_directive>
    ...critique for Bull Analyst...
    </bull_directive>

    <bear_directive>
    ...critique for Bear Analyst...
    </bear_directive>

parse_judge_output() extracts both blocks; returns None on any format failure.
parse_judge_output_with_retry() wraps LLM invocation with up to max_retries
attempts before raising RuntimeError.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

_BULL_PATTERN = re.compile(
    r"<bull_directive>(.*?)</bull_directive>", re.DOTALL | re.IGNORECASE
)
_BEAR_PATTERN = re.compile(
    r"<bear_directive>(.*?)</bear_directive>", re.DOTALL | re.IGNORECASE
)


def parse_judge_output(text: str) -> Optional[tuple[str, str]]:
    """
    Extract <bull_directive> and <bear_directive> blocks from Judge output.

    Returns (bull_directive, bear_directive) on success.
    Returns None if either block is missing; logs which tags are absent.
    Leading/trailing whitespace is stripped from each block's content.
    """
    bull_match = _BULL_PATTERN.search(text)
    bear_match = _BEAR_PATTERN.search(text)

    if not bull_match or not bear_match:
        missing = []
        if not bull_match:
            missing.append("<bull_directive>")
        if not bear_match:
            missing.append("<bear_directive>")
        logger.warning(
            "Judge output missing required XML tags: %s", ", ".join(missing)
        )
        return None

    return bull_match.group(1).strip(), bear_match.group(1).strip()


def parse_judge_output_with_retry(
    llm,
    prompt: str,
    max_retries: int = 3,
    fallback_on_failure: bool = False,
) -> tuple[str, str]:
    """
    Invoke LLM with the Judge prompt, retrying up to max_retries times if
    the required XML tags are absent from the response.

    Args:
        llm:                LangChain LLM instance (must support .invoke()).
        prompt:             Full Judge prompt string.
        max_retries:        Maximum number of invocation attempts (default 3).
        fallback_on_failure: If True, return ("", "") after exhausting retries
                            instead of raising RuntimeError. Use in backtest
                            mode to avoid aborting the full run on a format
                            failure.

    Returns:
        (bull_directive, bear_directive) — stripped text from each XML block.

    Raises:
        RuntimeError: If all attempts fail and fallback_on_failure is False.
    """
    for attempt in range(1, max_retries + 1):
        response = llm.invoke(prompt)
        result = parse_judge_output(response.content)
        if result is not None:
            if attempt > 1:
                logger.info(
                    "Judge output parsed successfully on attempt %d/%d.",
                    attempt,
                    max_retries,
                )
            return result
        logger.warning(
            "Judge format check failed (attempt %d/%d). Retrying...",
            attempt,
            max_retries,
        )

    if fallback_on_failure:
        logger.error(
            "Judge failed to produce valid XML output after %d attempts. "
            "Returning empty directives and continuing (fallback mode).",
            max_retries,
        )
        return ("", "")

    raise RuntimeError(
        f"Judge failed to produce valid output after {max_retries} attempts. "
        "Ensure the Judge prompt ends with a hard format requirement for "
        "<bull_directive> and <bear_directive> tags."
    )
