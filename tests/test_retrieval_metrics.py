from __future__ import annotations

import unittest

from rag_chatbot.evaluation.retrieval import retrieval_metrics
from rag_chatbot.types import Chunk, RetrievedChunk


class RetrievalMetricTests(unittest.TestCase):
    def test_gold_rank_hit_and_mrr(self) -> None:
        ranked = [
            RetrievedChunk(Chunk("wrong.txt", 0, "x"), 0.9, 1),
            RetrievedChunk(Chunk("gold.txt", 0, "y"), 0.5, 2),
        ]
        metrics = retrieval_metrics(ranked, "gold.txt", hit_k=3)

        self.assertEqual(metrics["gold_rank"], 2)
        self.assertFalse(metrics["hit_at_1"])
        self.assertTrue(metrics["hit_at_3"])
        self.assertEqual(metrics["reciprocal_rank"], 0.5)

    def test_missing_gold(self) -> None:
        ranked = [RetrievedChunk(Chunk("wrong.txt", 0, "x"), 0.9, 1)]
        metrics = retrieval_metrics(ranked, "gold.txt", hit_k=3)

        self.assertIsNone(metrics["gold_rank"])
        self.assertFalse(metrics["hit_at_1"])
        self.assertFalse(metrics["hit_at_3"])
        self.assertEqual(metrics["reciprocal_rank"], 0.0)


if __name__ == "__main__":
    unittest.main()
