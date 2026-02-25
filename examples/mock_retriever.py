"""
mock_retriever.py
-----------------
A deterministic fake retriever + generator for running the example
golden set without any live RAG system, API keys, or vector database.

Used by: python rag_eval.py --golden examples/golden_set.json
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Fake knowledge base  — mirrors the source_ids in golden_set.json
# ─────────────────────────────────────────────────────────────────────────────

KNOWLEDGE_BASE: dict[str, str] = {
    "billing-api-limits-v2": (
        "Vanta Billing API rate limits: Free tier — 100 requests per minute. "
        "Growth tier — 1 000 requests per minute. Enterprise — custom limits. "
        "Requests exceeding the limit receive HTTP 429 and should be retried "
        "with exponential back-off."
    ),
    "billing-api-auth-guide": (
        "Authentication: All requests must include an Authorization header with "
        "a Bearer token. Tokens are generated from the API Keys section of the "
        "Vanta dashboard. Each token expires after 90 days. Revoke tokens from "
        "the same dashboard section."
    ),
    "reporting-api-auth-guide": (
        "Reporting API authentication: API keys are scoped per product. "
        "Generate a Reporting API key from Settings > Reporting > API Keys. "
        "Do not reuse Billing API keys — they will be rejected with HTTP 403."
    ),
    "billing-retry-policy": (
        "Failed payment retry policy: Vanta retries failed invoice payments "
        "up to three times over seven days. After the third failure the "
        "subscription transitions to 'past_due' and a webhook event "
        "'invoice.payment_failed' is dispatched."
    ),
    "billing-subscription-states": (
        "Subscription states: active, trialing, past_due, cancelled. "
        "A subscription in past_due remains accessible for 14 days. If payment "
        "is not resolved within 14 days the subscription moves to cancelled and "
        "a final invoice is issued. Cancelled subscriptions cannot be reactivated; "
        "a new subscription must be created."
    ),
    "billing-api-subscriptions": (
        "Creating subscriptions: POST /subscriptions. Required fields: customer_id, "
        "plan_id, start_date. The start_date field is set at creation and is "
        "immutable — it cannot be changed via the API after the subscription is created."
    ),
    "billing-support-escalation": (
        "Support escalation: For issues that cannot be resolved via the API, "
        "contact support at support@vanta.example.com and include your account ID, "
        "a description of the issue, and any relevant timestamps."
    ),
    "billing-api-invoices": (
        "Invoices are issued in the currency configured on the customer object. "
        "The Vanta Billing API does not perform foreign exchange conversion. "
        "Supported currencies: USD, EUR, GBP, AUD, CAD. Set the currency when "
        "creating the customer; it cannot be changed after the first invoice."
    ),
    "billing-api-customers": (
        "Customer objects: POST /customers. Fields include name, email, currency "
        "(ISO 4217 code), and metadata. Each customer is tied to one currency. "
        "For multi-currency billing, create separate customer objects per currency "
        "and manage exchange rates in your own system."
    ),
    "billing-api-trials": (
        "Trial management: Subscriptions can include a trial_end date. By default "
        "billing begins automatically at trial_end. To charge before the trial ends "
        "call POST /subscriptions/{id}/convert with the body "
        '{"charge_immediately": true}. The customer must have a valid payment method '
        "on file. Omitting charge_immediately (or setting it to false) schedules "
        "billing for the original trial_end date."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Retriever
# Returns the most relevant chunk(s) for each question based on keyword overlap.
# Deterministic — same question always returns the same results.
# ─────────────────────────────────────────────────────────────────────────────

def _score_chunk(question: str, chunk_text: str) -> int:
    """Simple keyword overlap score (case-insensitive)."""
    q_words  = set(question.lower().split())
    c_words  = set(chunk_text.lower().split())
    return len(q_words & c_words)


def mock_retriever_fn(question: str, k: int = 3) -> list[dict]:
    """
    Returns the top-k chunks from the mock knowledge base.
    Replaces this with your real retriever:

        def my_retriever(question: str) -> list[dict]:
            results = vectorstore.similarity_search(question, k=3)
            return [{"text": r.page_content, "source_id": r.metadata["source"]} for r in results]
    """
    scored = [
        {"text": text, "source_id": sid, "_score": _score_chunk(question, text)}
        for sid, text in KNOWLEDGE_BASE.items()
    ]
    scored.sort(key=lambda x: x["_score"], reverse=True)

    # Return top-k, dropping the internal _score key
    return [{"text": d["text"], "source_id": d["source_id"]} for d in scored[:k]]


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# Returns a canned answer for each question ID for fully deterministic output.
# ─────────────────────────────────────────────────────────────────────────────

CANNED_ANSWERS: dict[str, str] = {
    "q001": "The Vanta Billing API free tier allows 100 requests per minute.",
    "q002": (
        "Authentication uses Bearer tokens passed in the Authorization header. "
        "Tokens are generated from the API Keys section of the dashboard and expire "
        "after 90 days."
    ),
    "q003": (
        "API keys are scoped per product. You need a separate key for the Billing "
        "API and a separate key for the Reporting API."
    ),
    "q004": (
        "After three failed payment attempts the subscription enters a past_due state. "
        "A webhook is fired. If payment is not resolved within 14 days the subscription "
        "is cancelled and a final invoice is issued."
    ),
    "q005": (
        "Subscription start dates cannot be backdated via the API — the start_date "
        "field is immutable after creation. For compliance backdating, contact support "
        "with your account ID and the required date."
    ),
    "q006": (
        "The Vanta Billing API does not automatically handle FX conversion. Invoices "
        "are issued in the currency set on the customer object. For multi-currency "
        "billing you must create separate customer objects per currency."
    ),
    "q007": (
        "I could not find information about refund policies for accidentally provisioned "
        "seats in the knowledge base. Please contact support at support@vanta.example.com."
    ),
    "q008": (
        "Yes. Call POST /subscriptions/{id}/convert with charge_immediately: true "
        "and ensure the customer has a valid payment method on file. Without that flag "
        "billing starts at the trial end date."
    ),
}


def mock_generator_fn(question: str, context: list[str]) -> str:
    """
    Returns a canned answer based on the question content.
    Replaces this with your real generator:

        def my_generator(question: str, context: list[str]) -> str:
            ctx_text = "\\n".join(context)
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"Answer using only this context:\\n{ctx_text}"},
                    {"role": "user",   "content": question},
                ],
            )
            return response.choices[0].message.content
    """
    # Map question to a canned answer by matching question IDs via content
    for qid, answer in CANNED_ANSWERS.items():
        # Simple heuristic: check if key question words match
        pass

    # Fallback: answer based on the retrieved context (naive extraction)
    if context:
        # Return the first sentence of the top context chunk as a simple fallback
        first_chunk = context[0].split(". ")[0] + "."
        return first_chunk

    return "The answer could not be found in the knowledge base."


def _make_question_to_id_map() -> dict[str, str]:
    """Build a map from golden set questions to their IDs at runtime."""
    import json
    from pathlib import Path
    golden_path = Path(__file__).parent / "golden_set.json"
    if not golden_path.exists():
        return {}
    with open(golden_path) as f:
        entries = json.load(f)
    return {e["question"]: e["id"] for e in entries}


_Q_TO_ID: dict[str, str] = {}


def mock_generator_fn(question: str, context: list[str]) -> str:  # noqa: F811
    """
    Deterministic generator — returns canned answers matched by question text.
    Replaces this with your real generator.
    """
    global _Q_TO_ID
    if not _Q_TO_ID:
        _Q_TO_ID = _make_question_to_id_map()

    qid = _Q_TO_ID.get(question)
    if qid and qid in CANNED_ANSWERS:
        return CANNED_ANSWERS[qid]

    # Fallback: return first sentence of top context chunk
    if context:
        return context[0].split(". ")[0] + "."
    return "The answer could not be found in the knowledge base."
