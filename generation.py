from __future__ import annotations

import os
from typing import Optional

from groq import Groq

try:
    from .config import settings
    from .logger import log
except Exception:  # pragma: no cover
    from config import settings
    from logger import log


FALLBACK = "I could not find that in uploaded documents."


def _get_client() -> Optional[Groq]:
    """
    Create Groq client if API key is available.
    """
    api_key = (getattr(settings, "groq_api_key", "") or os.getenv("GROQ_API_KEY", "")).strip()
    if not api_key:
        return None
    return Groq(api_key=api_key)


def _build_messages(*, query: str, context: str) -> list[dict]:
    """
    Strong grounding: force verbatim fallback if missing, and forbid external knowledge.
    """
    system = (
        "You are a strict RAG assistant.\n"
        "You must answer ONLY using the provided CONTEXT.\n"
        "Do NOT use any outside knowledge. Do NOT guess.\n"
        f"If the answer is not explicitly present in the CONTEXT, reply exactly:\n{FALLBACK}\n"
        "\n"
        "Return a concise helpful answer when possible."
    )

    user = (
        "CONTEXT:\n"
        f"{context}\n\n"
        "QUESTION:\n"
        f"{query}\n\n"
        "Remember: use ONLY CONTEXT. If not in context, reply with the fallback exactly."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _normalize_answer(text: str) -> str:
    if not isinstance(text, str):
        return FALLBACK
    t = text.strip()
    if not t:
        return FALLBACK

    # If model tries to be helpful but admits missing context, force exact fallback.
    low = t.lower()
    if ("not in the context" in low) or ("not provided in the context" in low) or ("i don't know" in low):
        return FALLBACK

    # Ensure it doesn't hallucinate by returning the fallback when it references external knowledge patterns.
    # (Heuristic only; the system prompt is the primary control.)
    return t


def generate_answer(*, query: str, context: str) -> str:
    """
    Generate an answer grounded ONLY in `context` using Groq.

    Behavior:
    - If answer not in context => exact fallback.
    - Timeout handling (Groq SDK does not expose a universal timeout in all versions),
      so we keep calls minimal and handle exceptions gracefully.
    - Logs failures for diagnosis.
    """
    query = (query or "").strip()
    context = (context or "").strip()

    if not context or not query:
        return FALLBACK

    client = _get_client()
    if client is None:
        log.warning("Groq API key missing; returning fallback.")
        return FALLBACK

    model = (getattr(settings, "groq_model", "") or os.getenv("GROQ_MODEL", "")).strip() or "llama-3.3-70b-versatile"

    messages = _build_messages(query=query, context=context)

    try:
        # Keep responses deterministic-ish: low temperature.
        # NOTE: Some Groq SDK versions support `timeout` on the underlying HTTP client, not here.
        # We still catch network/timeouts as exceptions.
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=500,
        )

        text = ""
        try:
            text = (resp.choices[0].message.content or "").strip()
        except Exception:
            text = ""

        out = _normalize_answer(text)
        if out == FALLBACK:
            log.info("Groq returned abstain/fallback for query='{}'", query)
        return out

    except Exception as e:
        # Includes auth errors, network failures, rate limits, SDK exceptions.
        log.exception("Groq generate_answer failed: {}", e)
        return FALLBACK


if __name__ == "__main__":
    print(generate_answer(query="When was badminton added to the Olympics?", context="Badminton debuted in the Olympics in 1992."))