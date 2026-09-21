# Bonsai2 Naive RAG baseline v0.1

Git baseline:

- tag: `bonsai2-naive-rag-v0.1`
- commit: `6e64e7d899970b9823c5604d961f9c1a7d66eb57`

## Verified pipeline

- Bonsai 2 OpenAI-compatible API: PASS
- `rag.py` health check: PASS
- char 2-4 gram TF-IDF retrieval: PASS
- Q001 correct document ranked #1: PASS
- Q002 correct document ranked #1: PASS
- grounded Bonsai 2 answer generation: PASS
- `stream=True`: PASS
- batch evaluation: PASS
- JSONL result persistence: PASS

## Observed retrieval

Q001:

- `helpdesk_password.txt#chunk-0`: 0.4142
- `helpdesk_vpn.txt#chunk-0`: 0.0045
- gold rank: 1

Q002:

- `helpdesk_vpn.txt#chunk-0`: 0.2213
- `helpdesk_password.txt#chunk-0`: 0.0402
- gold rank: 1

## Existing evaluation output

- n: 2
- mean char F1: 0.4475
- reference substring accuracy: 0.0000
- retrieval hit rate: 1.0000

The current character-F1 and substring metrics are connection-test metrics,
not authoritative answer-quality metrics. Semantically correct generated answers
receive low scores because the model adds explanatory text and citations.

The retrieval metric also needs improvement because the toy corpus contains
only two documents while the current default top-k exceeds the corpus size.

## Next evaluation changes

Add:

- Hit@1
- Hit@k
- gold rank
- MRR
- retrieval score
- answer/evidence separation

Keep the generator configuration fixed while comparing retrieval methods.
