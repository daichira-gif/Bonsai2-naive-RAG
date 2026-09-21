# V2 research refactor

V2 is additive. The verified V0.1 files `rag.py` and `evaluate.py` remain unchanged.

## Design boundary

```text
Question
  -> Retriever
  -> Evidence surface
  -> Prompt builder
  -> Bonsai2Generator
  -> Answer

QA dataset
  -> full retrieval ranking
  -> Hit@1 / Hit@K / gold rank / MRR
  -> fixed evidence surface
  -> Bonsai2Generator
  -> answer diagnostics
```

## Run

```bash
python -m unittest discover -s tests -v

python -m scripts.run_rag --health

python -m scripts.run_rag \
  "パスワードを何回間違えるとロックされますか？" \
  --show-retrieval

python -m scripts.run_rag \
  "VPNのE42エラーが出た場合の手順を教えてください。" \
  --stream \
  --show-retrieval

python -m scripts.evaluate_v2

python -m scripts.evaluate_v2 --retrieval-only
```

## Principles

- Keep Bonsai2Generator independent of retrieval logic.
- Keep generation settings explicit and fixed in config/baseline-v2.json.
- Rank the full corpus for retrieval metrics.
- Use only configured top-k evidence for generation.
- Treat answer string metrics as diagnostics, not authoritative semantic scores.
- Keep V0.1 intact until V2 regression checks pass.
