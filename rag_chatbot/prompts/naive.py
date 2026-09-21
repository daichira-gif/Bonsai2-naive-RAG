from __future__ import annotations

from ..types import RetrievedChunk


SYSTEM_PROMPT = """あなたはRAGの回答エンジンです。
与えられた「検索根拠」だけを根拠として回答してください。
根拠にない事実を補わないでください。
十分な根拠がない場合は「根拠資料からは確認できません。」と答えてください。
回答中には、使用した根拠を [ファイル名#chunk-N] の形式で明示してください。
"""


def build_naive_messages(
    question: str,
    evidence: list[RetrievedChunk],
) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"[{item.chunk.citation}]\n{item.chunk.text}"
        for item in evidence
    )

    user_prompt = f"""# 質問
{question}

# 検索根拠
{context}
"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
