"""Step-level action dedup from SWE-Verifier §3.2 and Appendix A.

Whole trajectories are not near-duplicate filtered. These rules apply to
action candidates sampled at one state.
"""
from __future__ import annotations

import re

_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def token_set(text):
    if not isinstance(text, str):
        return set()
    return set(_TOKEN.findall(text.lower()))


def jaccard(left, right):
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def drop_text_near_duplicates(commands, threshold=0.8):
    """Keep the earlier action when pairwise token-set Jaccard is at least 0.8."""
    if not 0 <= threshold <= 1:
        raise ValueError("Jaccard threshold must be between 0 and 1")
    kept = []
    kept_tokens = []
    for index, command in enumerate(commands):
        tokens = token_set(command)
        if any(jaccard(tokens, previous) >= threshold for previous in kept_tokens):
            continue
        kept.append(index)
        kept_tokens.append(tokens)
    return kept


def drop_same_observations(outputs):
    """Drop a later candidate whose non-empty observation matches one already kept."""
    kept = []
    seen = set()
    for index, output in enumerate(outputs):
        text = output.strip() if isinstance(output, str) else ""
        if text and text in seen:
            continue
        if text:
            seen.add(text)
        kept.append(index)
    return kept


def should_verify(unique_count):
    """Inference only compares candidates when at least two unique ones remain."""
    return isinstance(unique_count, int) and not isinstance(unique_count, bool) and unique_count >= 2
