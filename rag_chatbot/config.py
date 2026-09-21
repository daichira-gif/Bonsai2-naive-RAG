from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .generators.bonsai2 import Bonsai2Config


@dataclass(frozen=True)
class RetrievalConfig:
    knowledge_dir: str = "knowledge"
    chunk_size: int = 500
    chunk_overlap: int = 100
    ngram_min: int = 2
    ngram_max: int = 4
    top_k: int = 3


@dataclass(frozen=True)
class AppConfig:
    retrieval: RetrievalConfig
    generator: Bonsai2Config


def load_config(path: str | Path) -> AppConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return AppConfig(
        retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        generator=Bonsai2Config(**raw.get("generator", {})),
    )
