from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

from rag import NaiveRAG


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    # Character-level metric that does not require Japanese tokenization.
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)


def char_f1(prediction: str, reference: str) -> float:
    pred = list(normalize(prediction))
    ref = list(normalize(reference))
    if not pred or not ref:
        return float(pred == ref)

    # Multiset overlap.
    from collections import Counter

    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/qa.jsonl")
    parser.add_argument("--knowledge-dir", default="knowledge")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", default="eval/results.jsonl")
    args = parser.parse_args()

    rag = NaiveRAG(args.knowledge_dir)

    rows = [
        json.loads(line)
        for line in Path(args.dataset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    out = []
    for row in rows:
        answer, retrieved = rag.answer(
            row["question"],
            top_k=args.top_k,
            stream=False,
            enable_thinking=False,
            max_tokens=256,
        )

        sources = [chunk.source for chunk, _score in retrieved]
        reference = row["reference_answer"]
        gold_source = row.get("gold_source")

        result = {
            "id": row["id"],
            "question": row["question"],
            "reference_answer": reference,
            "prediction": answer,
            "char_f1": round(char_f1(answer, reference), 4),
            "reference_substring_match": normalize(reference) in normalize(answer),
            "retrieved_sources": sources,
            "retrieval_hit": (gold_source in sources) if gold_source else None,
        }
        out.append(result)
        print(json.dumps(result, ensure_ascii=False))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n",
        encoding="utf-8",
    )

    if out:
        mean_f1 = sum(x["char_f1"] for x in out) / len(out)
        substring_acc = sum(x["reference_substring_match"] for x in out) / len(out)
        retrieval_rows = [x for x in out if x["retrieval_hit"] is not None]
        retrieval_acc = (
            sum(x["retrieval_hit"] for x in retrieval_rows) / len(retrieval_rows)
            if retrieval_rows
            else None
        )

        print("\n=== SUMMARY ===")
        print(f"n={len(out)}")
        print(f"mean_char_f1={mean_f1:.4f}")
        print(f"reference_substring_accuracy={substring_acc:.4f}")
        if retrieval_acc is not None:
            print(f"retrieval_hit_rate={retrieval_acc:.4f}")
        print(f"results={output_path}")


if __name__ == "__main__":
    main()
