from __future__ import annotations

from pathlib import Path

from .types import Chunk


def load_text_chunks(
    knowledge_dir: str | Path,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    root = Path(knowledge_dir)
    paths = sorted(
        [*root.rglob("*.txt"), *root.rglob("*.md")],
        key=lambda p: str(p.relative_to(root)),
    )

    chunks: list[Chunk] = []
    for path in paths:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        source = str(path.relative_to(root))
        start = 0
        chunk_id = 0

        while start < len(text):
            end = min(len(text), start + chunk_size)
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(source=source, chunk_id=chunk_id, text=piece))

            if end >= len(text):
                break

            start = end - chunk_overlap
            chunk_id += 1

    return chunks
