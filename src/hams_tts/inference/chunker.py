"""Streaming text chunker — the lever that buys us a low **time-to-first-audio**.

A non-autoregressive VITS synthesises a whole input in one pass, so the way to make
TTFA small is to synthesise a *short first phrase* immediately, stream it, and continue
with the rest while the listener is already hearing audio.  We therefore split incoming
text into chunks at natural prosodic boundaries, with a deliberately **small first
chunk**, and let the engine pipeline chunk N+1's synthesis under chunk N's playout.

Boundaries are chosen to avoid cutting inside a word or, ideally, inside a clause, so
prosody stays natural and code-switch spans are not split mid-token.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# Arabic ؟ ، ؛ are normalised to ? , ; upstream, but we keep them here too for safety.
_HARD = set(".!?…؟")
_SOFT = set(",;:،—)")


@dataclass
class ChunkerConfig:
    first_chunk_max_chars: int = 36  # keep the first phrase short -> low TTFA
    max_chars: int = 160
    min_chars: int = 8  # avoid emitting tiny fragments after the first chunk


def _split_words(text: str) -> List[str]:
    return text.split()


def chunk_text(text: str, cfg: ChunkerConfig | None = None) -> List[str]:
    """Greedily split ``text`` into streamable chunks at prosodic boundaries."""
    cfg = cfg or ChunkerConfig()
    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def budget() -> int:
        # first chunk is intentionally smaller
        return cfg.first_chunk_max_chars if not chunks else cfg.max_chars

    def flush() -> None:
        nonlocal buf, buf_len
        if buf:
            chunks.append(" ".join(buf).strip())
            buf, buf_len = [], 0

    for word in _split_words(text):
        buf.append(word)
        buf_len += len(word) + 1
        ends_hard = word and word[-1] in _HARD
        ends_soft = word and word[-1] in _SOFT
        over_budget = buf_len >= budget()

        if ends_hard and buf_len >= cfg.min_chars:
            flush()
        elif over_budget and (ends_soft or buf_len >= cfg.max_chars):
            flush()
        elif over_budget and not chunks:  # force-flush the very first chunk for TTFA
            flush()

    flush()
    # merge a trailing micro-chunk into its predecessor (avoids a clipped final word)
    if len(chunks) >= 2 and len(chunks[-1]) < cfg.min_chars:
        chunks[-2] = chunks[-2] + " " + chunks[-1]
        chunks.pop()
    return chunks


if __name__ == "__main__":
    demo = (
        "مرحبا، كيف حالك اليوم؟ بدي أحجز flight من بيروت to London بكرا الساعة 9، "
        "and please confirm the booking by email. شكرا!"
    )
    for i, c in enumerate(chunk_text(demo)):
        print(f"chunk {i} ({len(c):3d} chars): {c}")
