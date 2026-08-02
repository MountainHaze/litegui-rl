"""Cheap screenshot state-change signals used instead of an external PRM."""

from __future__ import annotations

import base64
from io import BytesIO


def screen_change_score(previous_b64: str | None, current_b64: str | None) -> float:
    """Return mean normalized pixel difference in ``[0, 1]``.

    The images are converted to grayscale and resized before comparison, so the
    score is cheap and insensitive to screenshot resolution. Invalid or missing
    images receive zero reward rather than an optimistic fallback.
    """
    if not previous_b64 or not current_b64:
        return 0.0
    try:
        import numpy as np
        from PIL import Image

        def decode(value: str):
            raw = base64.b64decode(value)
            return np.asarray(
                Image.open(BytesIO(raw)).convert("L").resize((128, 128)), dtype=np.float32
            )

        previous = decode(previous_b64)
        current = decode(current_b64)
        return float(min(1.0, max(0.0, abs(previous - current).mean() / 255.0)))
    except (ValueError, OSError, TypeError):
        return 0.0

