from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_BASE_URL = os.getenv("BONSAI_BASE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_HEALTH_URL = os.getenv("BONSAI_HEALTH_URL", "http://127.0.0.1:8000/health")
DEFAULT_MODEL = os.getenv(
    "BONSAI_MODEL",
    "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",
)


@dataclass(frozen=True)
class Chunk:
    source: str
    chunk_id: int
    text: str

    @property
    def citation(self) -> str:
        return f"{self.source}#chunk-{self.chunk_id}"


class NaiveRAG:
    """
    Minimal text-only RAG:
      files -> character chunks -> char n-gram TF-IDF -> top-k retrieval -> Bonsai 2
    """

    def __init__(
        self,
        knowledge_dir: str | Path = "knowledge",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.chunks = self._load_and_chunk()
        if not self.chunks:
            raise RuntimeError(f"No .txt/.md documents found in {self.knowledge_dir}")

        # Japanese-friendly naive lexical baseline:
        # char n-grams avoid dependence on a Japanese morphological analyzer.
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            lowercase=False,
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])

        self.client = OpenAI(
            base_url=DEFAULT_BASE_URL,
            api_key="dummy",
        )

    def _iter_documents(self) -> Iterable[Path]:
        for pattern in ("*.txt", "*.md"):
            yield from sorted(self.knowledge_dir.rglob(pattern))

    def _load_and_chunk(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        for path in self._iter_documents():
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue

            rel = str(path.relative_to(self.knowledge_dir))
            start = 0
            chunk_id = 0

            while start < len(text):
                end = min(len(text), start + self.chunk_size)
                piece = text[start:end].strip()
                if piece:
                    chunks.append(Chunk(rel, chunk_id, piece))

                if end >= len(text):
                    break

                start = max(0, end - self.chunk_overlap)
                chunk_id += 1

        return chunks

    def retrieve(self, question: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.matrix).ravel()
        ranked = scores.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked]

    def answer(
        self,
        question: str,
        top_k: int = 3,
        stream: bool = False,
        enable_thinking: bool = False,
        thinking_budget: int = 1024,
        max_tokens: int = 512,
    ):
        retrieved = self.retrieve(question, top_k=top_k)

        context = "\n\n".join(
            f"[{chunk.citation}]\n{chunk.text}"
            for chunk, _score in retrieved
        )

        system_prompt = """あなたはRAGの回答エンジンです。
与えられた「検索根拠」だけを根拠として回答してください。
根拠にない事実を補わないでください。
十分な根拠がない場合は「根拠資料からは確認できません。」と答えてください。
回答中には、使用した根拠を [ファイル名#chunk-N] の形式で明示してください。
"""

        user_prompt = f"""# 質問
{question}

# 検索根拠
{context}
"""

        extra_body = {
            "enable_thinking": enable_thinking,
        }
        if enable_thinking:
            extra_body["thinking_budget"] = thinking_budget

        response = self.client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
            stream=stream,
            extra_body=extra_body,
        )

        if stream:
            return response, retrieved

        return response.choices[0].message.content or "", retrieved


def health_check() -> str:
    with urllib.request.urlopen(DEFAULT_HEALTH_URL, timeout=3) as response:
        return response.read().decode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--knowledge-dir", default="knowledge")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--thinking-budget", type=int, default=1024)
    parser.add_argument("--show-retrieval", action="store_true")
    parser.add_argument("--health", action="store_true")
    args = parser.parse_args()

    if args.health:
        try:
            print(health_check())
        except Exception as exc:
            print(f"Health check failed: {exc}", file=sys.stderr)
            raise SystemExit(1)

    if not args.question:
        if args.health:
            return
        parser.error("question is required unless --health is used")

    rag = NaiveRAG(args.knowledge_dir)

    result, retrieved = rag.answer(
        args.question,
        top_k=args.top_k,
        stream=args.stream,
        enable_thinking=args.thinking,
        thinking_budget=args.thinking_budget,
    )

    if args.show_retrieval:
        print("\n=== RETRIEVAL ===", file=sys.stderr)
        for chunk, score in retrieved:
            print(
                f"{score:.4f}\t{chunk.citation}\t{chunk.text[:120].replace(chr(10), ' ')}",
                file=sys.stderr,
            )
        print("=== ANSWER ===", file=sys.stderr)

    if args.stream:
        for event in result:
            if not event.choices:
                continue
            delta = event.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
        print()
    else:
        print(result)


if __name__ == "__main__":
    main()
