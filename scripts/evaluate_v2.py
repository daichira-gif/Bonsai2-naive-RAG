from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_chatbot.config import load_config
from rag_chatbot.evaluation.answer import char_f1, reference_substring_match
from rag_chatbot.evaluation.retrieval import retrieval_metrics
from rag_chatbot.pipeline import RAGPipeline
from rag_chatbot.prompts.naive import build_naive_messages


def mean_bool(rows: list[dict], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/baseline-v2.json")
    parser.add_argument("--dataset", default="eval/qa.jsonl")
    parser.add_argument("--output", default="results/v2-results.jsonl")
    parser.add_argument("--retrieval-k", type=int, default=3)
    parser.add_argument("--evidence-k", type=int)
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = RAGPipeline(config)

    rows = [
        json.loads(line)
        for line in Path(args.dataset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    output_rows: list[dict] = []
    for row in rows:
        question = row["question"]
        gold_source = row.get("gold_source")
        ranked = pipeline.rank_all(question)

        retrieval = (
            retrieval_metrics(ranked, gold_source, args.retrieval_k)
            if gold_source
            else {}
        )

        evidence_k = args.evidence_k or config.retrieval.top_k
        evidence = ranked[: min(evidence_k, len(ranked))]

        result = {
            "id": row["id"],
            "question": question,
            "gold_source": gold_source,
            **retrieval,
            "retrieved": [
                {
                    "rank": item.rank,
                    "source": item.chunk.source,
                    "citation": item.chunk.citation,
                    "score": round(item.score, 6),
                }
                for item in evidence
            ],
        }

        if not args.retrieval_only:
            reference = row["reference_answer"]
            messages = build_naive_messages(question, evidence)
            prediction = pipeline.generator.generate(messages, max_tokens=256)

            result.update(
                {
                    "reference_answer": reference,
                    "prediction": prediction,
                    "char_f1_diagnostic": round(char_f1(prediction, reference), 4),
                    "reference_substring_match_diagnostic": (
                        reference_substring_match(prediction, reference)
                    ),
                }
            )

        output_rows.append(result)
        print(json.dumps(result, ensure_ascii=False))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in output_rows) + "\n",
        encoding="utf-8",
    )

    if not output_rows:
        return

    metric_rows = [r for r in output_rows if r.get("gold_source")]
    print("\n=== RETRIEVAL SUMMARY ===")
    print(f"n={len(output_rows)}")

    if metric_rows:
        hit_k_key = f"hit_at_{args.retrieval_k}"
        hit1 = mean_bool(metric_rows, "hit_at_1")
        hitk = mean_bool(metric_rows, hit_k_key)
        mrr = sum(r["reciprocal_rank"] for r in metric_rows) / len(metric_rows)

        print(f"hit_at_1={hit1:.4f}")
        print(f"{hit_k_key}={hitk:.4f}")
        print(f"mrr={mrr:.4f}")

    if not args.retrieval_only:
        f1 = sum(r["char_f1_diagnostic"] for r in output_rows) / len(output_rows)
        substring = mean_bool(output_rows, "reference_substring_match_diagnostic")
        print("\n=== ANSWER DIAGNOSTICS ===")
        print(f"mean_char_f1={f1:.4f}")
        print(f"reference_substring_accuracy={substring:.4f}")
        print("NOTE=answer metrics above are diagnostic, not authoritative.")

    print(f"results={output_path}")


if __name__ == "__main__":
    main()
