# Bonsai 2 Naive RAG

`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` を `http://127.0.0.1:8000/v1`
の OpenAI 互換APIとして利用する、最小の text-only Naive RAG サンプルです。

## 構成

```text
knowledge/*.txt, *.md
        |
        v
文字数ベースのchunking
        |
        v
char 2-4 gram TF-IDF
        |
        v
cosine similarity top-k
        |
        v
検索根拠をpromptへ挿入
        |
        v
Bonsai 2
        |
        v
回答 + source citation
```

日本語の最小baselineなので、形態素解析を追加せず character n-gram を使います。

## 1. Bonsai 2 API確認

```bash
curl http://127.0.0.1:8000/health
```

必要なら、既存の Bonsai 2 起動方法でAPI serverを起動してください。

## 2. VS Codeでこのフォルダを開く

```bash
cd bonsai2_naive_rag
code .
```

## 3. Python環境

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt
```

この `.venv` はプロジェクト配下に置きます。
`~/.lmstudio/models` 配下には置かないでください。

## 4. health check

```bash
python rag.py --health
```

## 5. 1問推論

```bash
python rag.py \
  "パスワードを何回間違えるとロックされますか？" \
  --show-retrieval
```

## 6. streaming

```bash
python rag.py \
  "VPNのE42エラーが出た場合の手順を教えてください。" \
  --stream \
  --show-retrieval
```

## 7. thinkingを有効化

```bash
python rag.py \
  "資料に基づいてVPN E42の対処手順を順番に整理してください。" \
  --thinking \
  --thinking-budget 1024
```

RAGの比較実験では、回答モデル条件を統制するため、通常は thinking の有無を条件間で固定してください。

## 8. 簡易評価

```bash
python evaluate.py
```

出力:

```text
eval/results.jsonl
```

最低限の指標:

- `char_f1`: 日本語形態素解析なしの文字単位F1
- `reference_substring_match`: 参照回答が生成回答に含まれるか
- `retrieval_hit`: gold source がtop-kに入ったか

これは研究用の最終評価ではありません。
Naive RAGの実装・接続確認用baselineです。

## 9. 自分の文書へ置換

`knowledge/` 内の `.txt` / `.md` を置き換えるだけです。
起動時にインデックスを再構築します。

## 10. 環境変数

デフォルト:

```text
BONSAI_BASE_URL=http://127.0.0.1:8000/v1
BONSAI_HEALTH_URL=http://127.0.0.1:8000/health
BONSAI_MODEL=prism-ml/Ternary-Bonsai-2-27B-mlx-2bit
```

例:

```bash
export BONSAI_BASE_URL=http://127.0.0.1:8000/v1
```

## 11. 研究パイプラインへ発展させる場合

この最小版の境界を保ったまま、後で以下を差し替えられます。

```text
Document loader
Chunker
Retriever
Reranker
Prompt builder
Generator (Bonsai 2)
Evaluator
```

そのため、A0/A1/A2/A3等の比較では
「Bonsai 2を回答モデルとして固定し、Retriever部分だけを差し替える」
構成へ発展させやすくなっています。
