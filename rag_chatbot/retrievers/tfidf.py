from __future__ import annotations

from collections.abc import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..types import Chunk, RetrievedChunk


class TfidfRetriever:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        ngram_min: int = 2,
        ngram_max: int = 4,
    ) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("chunks must not be empty")

        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(ngram_min, ngram_max),
            lowercase=False,
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])

    def rank(self, question: str, limit: int | None = None) -> list[RetrievedChunk]:
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.matrix).ravel()

        indices = sorted(
            range(len(self.chunks)),
            key=lambda i: (
                -float(scores[i]),
                self.chunks[i].source,
                self.chunks[i].chunk_id,
            ),
        )

        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be >= 1")
            indices = indices[:limit]

        return [
            RetrievedChunk(
                chunk=self.chunks[i],
                score=float(scores[i]),
                rank=rank,
            )
            for rank, i in enumerate(indices, start=1)
        ]

    def retrieve(self, question: str, top_k: int = 3) -> list[RetrievedChunk]:
        return self.rank(question, limit=min(top_k, len(self.chunks)))
