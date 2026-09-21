from __future__ import annotations

from collections.abc import Iterator

from .chunking import load_text_chunks
from .config import AppConfig
from .generators.bonsai2 import Bonsai2Generator
from .prompts.naive import build_naive_messages
from .retrievers.tfidf import TfidfRetriever
from .types import RetrievedChunk


class RAGPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        r = config.retrieval

        chunks = load_text_chunks(
            knowledge_dir=r.knowledge_dir,
            chunk_size=r.chunk_size,
            chunk_overlap=r.chunk_overlap,
        )
        if not chunks:
            raise RuntimeError(f"No documents found in {r.knowledge_dir}")

        self.retriever = TfidfRetriever(
            chunks,
            ngram_min=r.ngram_min,
            ngram_max=r.ngram_max,
        )
        self.generator = Bonsai2Generator(config.generator)

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        return self.retriever.retrieve(
            question,
            top_k=top_k or self.config.retrieval.top_k,
        )

    def rank_all(self, question: str) -> list[RetrievedChunk]:
        return self.retriever.rank(question)

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, list[RetrievedChunk]]:
        evidence = self.retrieve(question, top_k=top_k)
        messages = build_naive_messages(question, evidence)
        answer = self.generator.generate(messages, max_tokens=max_tokens)
        return answer, evidence

    def stream_answer(
        self,
        question: str,
        top_k: int | None = None,
        max_tokens: int | None = None,
    ) -> tuple[Iterator[str], list[RetrievedChunk]]:
        evidence = self.retrieve(question, top_k=top_k)
        messages = build_naive_messages(question, evidence)
        return self.generator.stream(messages, max_tokens=max_tokens), evidence
