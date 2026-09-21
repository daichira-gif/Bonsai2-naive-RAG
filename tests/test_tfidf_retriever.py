from __future__ import annotations

import unittest

from rag_chatbot.retrievers.tfidf import TfidfRetriever
from rag_chatbot.types import Chunk


class TfidfRetrieverTests(unittest.TestCase):
    def test_expected_document_ranks_first(self) -> None:
        chunks = [
            Chunk("password.txt", 0, "パスワードを5回間違えると15分ロックされる。"),
            Chunk("vpn.txt", 0, "VPN E42ではWi-Fiを確認する。"),
        ]
        retriever = TfidfRetriever(chunks)
        ranked = retriever.rank("パスワードを何回間違えるとロックされますか？")

        self.assertEqual(ranked[0].chunk.source, "password.txt")
        self.assertEqual(ranked[0].rank, 1)


if __name__ == "__main__":
    unittest.main()
