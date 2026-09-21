from __future__ import annotations

import argparse
import json
import sys

from rag_chatbot.config import load_config
from rag_chatbot.pipeline import RAGPipeline


def show_retrieval(evidence) -> None:
    print("\n=== RETRIEVAL ===", file=sys.stderr)
    for item in evidence:
        preview = item.chunk.text[:120].replace("\n", " ")
        print(
            f"rank={item.rank}\tscore={item.score:.4f}\t"
            f"{item.chunk.citation}\t{preview}",
            file=sys.stderr,
        )
    print("=== ANSWER ===", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--config", default="config/baseline-v2.json")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--show-retrieval", action="store_true")
    parser.add_argument("--health", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = RAGPipeline(config)

    if args.health:
        print(json.dumps(pipeline.generator.health(), ensure_ascii=False))
        if not args.question:
            return

    if not args.question:
        parser.error("question is required unless --health is used")

    if args.stream:
        stream, evidence = pipeline.stream_answer(
            args.question,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
        )
        if args.show_retrieval:
            show_retrieval(evidence)
        for text in stream:
            print(text, end="", flush=True)
        print()
    else:
        answer, evidence = pipeline.answer(
            args.question,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
        )
        if args.show_retrieval:
            show_retrieval(evidence)
        print(answer)


if __name__ == "__main__":
    main()
