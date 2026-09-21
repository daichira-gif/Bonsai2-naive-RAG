from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    source: str
    chunk_id: int
    text: str

    @property
    def citation(self) -> str:
        return f"{self.source}#chunk-{self.chunk_id}"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int
