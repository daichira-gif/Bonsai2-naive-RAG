from __future__ import annotations

from ..types import RetrievedChunk


def retrieval_metrics(
    ranked: list[RetrievedChunk],
    gold_source: str,
    hit_k: int,
) -> dict:
    matched = next(
        (item for item in ranked if item.chunk.source == gold_source),
        None,
    )

    if matched is None:
        return {
            "gold_rank": None,
            "gold_score": None,
            "hit_at_1": False,
            f"hit_at_{hit_k}": False,
            "reciprocal_rank": 0.0,
        }

    rank = matched.rank
    return {
        "gold_rank": rank,
        "gold_score": round(matched.score, 6),
        "hit_at_1": rank <= 1,
        f"hit_at_{hit_k}": rank <= hit_k,
        "reciprocal_rank": 1.0 / rank,
    }
