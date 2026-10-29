"""Lossless JSON for report payloads: every field and array entry is kept."""
from dataclasses import asdict, is_dataclass
import json

import numpy as np


def readable(value):
    """Keep all fields/array entries; represent nonfinite raw values explicitly."""
    if is_dataclass(value):
        return readable(asdict(value))
    if isinstance(value, np.ndarray):
        return readable(value.tolist())
    if isinstance(value, np.generic):
        return readable(value.item())
    if isinstance(value, dict):
        return {str(k): readable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [readable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "__dict__"):
        return {"__class__": type(value).__name__, **readable(vars(value))}
    return repr(value)


def number(value):
    try:
        val = float(value)
        return val if np.isfinite(val) else None
    except (TypeError, ValueError):
        return None


def json_text(value):
    return json.dumps(readable(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def embed_json(value):
    # JSON remains valid while strings cannot close the containing script tag.
    return json_text(value).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
