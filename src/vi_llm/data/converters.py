"""Dataset-specific converters that normalise any HF dataset row → plain text.

Each converter is a function ``(row: dict) -> str | None``.
Return ``None`` to skip the row entirely.

Register a new converter with the ``@register("org/dataset-id")`` decorator.
The loader resolves converters in this priority order:
  1. Registered converter for the dataset ID
  2. ``text_columns`` from configs/datasets.yaml  (simple column extraction)
  3. ValueError — you need one or the other.
"""

from __future__ import annotations

from typing import Callable

Row = dict
Converter = Callable[[Row], str | None]

_REGISTRY: dict[str, Converter] = {}


def register(dataset_id: str) -> Callable[[Converter], Converter]:
    """Decorator: register a row-to-text converter for a dataset."""
    def decorator(fn: Converter) -> Converter:
        _REGISTRY[dataset_id] = fn
        return fn
    return decorator


def get(dataset_id: str) -> Converter | None:
    """Return the registered converter, or None if none exists."""
    return _REGISTRY.get(dataset_id)


# ── Built-in converters ───────────────────────────────────────────────────────

@register("th1nhng0/vietnamese-legal-documents")
def _(row: Row) -> str | None:
    return row.get("content_html") or None


@register("vietgpt/wikipedia_vi")
def _(row: Row) -> str | None:
    title = (row.get("title") or "").strip()
    text = (row.get("text") or "").strip()
    if not text:
        return None
    return f"# {title}\n\n{text}" if title else text


# ── Examples for common dataset patterns ─────────────────────────────────────
# Uncomment and adapt when adding a new dataset.

# Instruction-following → pre-train prose
# @register("tatsu-lab/alpaca")
# def _(row: Row) -> str | None:
#     inst = row.get("instruction", "").strip()
#     out = row.get("output", "").strip()
#     return f"{inst}\n\n{out}" if inst or out else None

# Multi-turn chat → joined turns
# @register("HuggingFaceH4/ultrachat_200k")
# def _(row: Row) -> str | None:
#     turns = row.get("messages") or []
#     lines = [f"{t['role']}: {t['content']}" for t in turns if t.get("content")]
#     return "\n".join(lines) or None

# Multiple text columns with custom separator
# @register("wikimedia/wikipedia")
# def _(row: Row) -> str | None:
#     title = row.get("title", "").strip()
#     body = row.get("text", "").strip()
#     return f"# {title}\n\n{body}" if body else None
