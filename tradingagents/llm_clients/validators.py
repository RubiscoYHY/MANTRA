"""Model name validators for each provider."""

from .model_catalog import get_known_models


VALID_MODELS = {
    provider: models
    for provider, models in get_known_models().items()
    if provider not in ("ollama", "openrouter")
}


def validate_model(provider: str, model: str) -> bool:
    """Check if model name is valid for the given provider.

    For ollama, openrouter, huggingface - any model ID is accepted.
    """
    provider_lower = provider.lower()

    if provider_lower in ("ollama", "openrouter", "huggingface"):
        return True

    if provider_lower not in VALID_MODELS:
        return True

    return model in VALID_MODELS[provider_lower]


# Valid values for provider-specific reasoning-control parameters. A typo
# (e.g. "hgh") previously passed client init silently and only failed at the
# first invoke, deep inside a run.
_EFFORT_PARAM_VALUES = {
    "anthropic_effort":        {"low", "medium", "high"},
    "openai_reasoning_effort": {"minimal", "low", "medium", "high"},
    "google_thinking_level":   {"minimal", "low", "medium", "high"},
}


def validate_effort_param(name: str, value) -> None:
    """Raise ValueError for a known effort/thinking param with an invalid value.

    None (unset) is always accepted; unknown param names are ignored.
    """
    if value is None:
        return
    allowed = _EFFORT_PARAM_VALUES.get(name)
    if allowed is not None and value not in allowed:
        raise ValueError(
            f"Invalid value {value!r} for {name}; expected one of {sorted(allowed)}"
        )
