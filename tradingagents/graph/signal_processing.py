import re
from typing import Any


_VALID_SIGNALS = frozenset({"BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL"})


class SignalProcessor:
    """Processes Portfolio Manager output to extract a structured trading signal."""

    def __init__(self, quick_thinking_llm: Any):
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> dict:
        """
        Parse the STRUCTURED SIGNAL block from Portfolio Manager output.

        Returns a dict with keys:
            signal     — one of BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL
            confidence — float in [0.50, 1.00]
            horizon    — one of "1-5d" / "5-20d" / "20d+"

        Strategy:
          1. Try regex extraction of the structured block (fast, no LLM call).
          2. Fall back to LLM extraction for signal only if the block is absent;
             confidence defaults to 0.70 and horizon to "1-5d" in that case.
        """
        structured = self._parse_structured_block(full_signal)
        if structured:
            return structured
        signal = self._llm_extract_signal(full_signal)
        return {"signal": signal, "confidence": 0.70, "horizon": "1-5d"}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_structured_block(self, text: str) -> dict | None:
        """
        Extract signal, confidence, and horizon from the STRUCTURED SIGNAL block.
        Returns None if the signal field cannot be found.
        """
        signal_match = re.search(
            r"Signal:\s*(BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL)",
            text,
            re.IGNORECASE,
        )
        if not signal_match:
            return None

        conf_match = re.search(
            r"Confidence:\s*(0\.\d+|1\.0+)",
            text,
            re.IGNORECASE,
        )
        horizon_match = re.search(
            r"Horizon:\s*(1-5d|5-20d|20d\+)",
            text,
            re.IGNORECASE,
        )

        signal = signal_match.group(1).upper()
        confidence = float(conf_match.group(1)) if conf_match else 0.70
        horizon = horizon_match.group(1) if horizon_match else "1-5d"
        confidence = max(0.50, min(1.00, confidence))  # clamp to valid range

        return {"signal": signal, "confidence": confidence, "horizon": horizon}

    def _llm_extract_signal(self, full_signal: str) -> str:
        """
        Legacy fallback: ask the LLM to extract a single signal word.
        Returns "HOLD" if the LLM output is not a recognised signal.
        """
        messages = [
            (
                "system",
                "You are an efficient assistant that extracts the trading decision "
                "from analyst reports. Extract the rating as exactly one of: "
                "BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, SELL. "
                "Output only the single rating word, nothing else.",
            ),
            ("human", full_signal),
        ]
        result = self.quick_thinking_llm.invoke(messages).content.strip().upper()
        return result if result in _VALID_SIGNALS else "HOLD"
