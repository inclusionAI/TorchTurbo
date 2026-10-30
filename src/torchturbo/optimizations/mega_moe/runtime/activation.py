"""SwiGLU semantic modes shared by execution owners and task serialization."""
MODES = ("none", "pre_activation", "post_activation")


def resolve_mode(mode: str | None, limit: float | None) -> str:
    """Normalize the semantic contract without changing model mathematics."""
    if mode is None:
        return "none" if limit is None else "post_activation"
    if mode not in MODES:
        raise ValueError(f"swiglu_mode must be one of {MODES}, got {mode!r}")
    if (mode == "none") != (limit is None):
        raise ValueError("Unclamped SwiGLU requires no limit; clipped modes require a positive limit.")
    return mode
