"""
rag-evaluation-starter
======================
Evaluate any RAG pipeline against a golden set in under 30 minutes.
No SaaS. No signup. Pure Python.

Usage (bundled demo):
    python rag_eval.py --golden examples/golden_set.json --no-embeddings
    python rag_eval.py --golden examples/golden_set.json --k 5 --llm-judge
    python rag_eval.py --golden examples/golden_set.json --csv --output results/run1

The CLI runs only the bundled demo: the mock retriever + generator in
examples/mock_retriever.py against examples/golden_set.json. Any other
--golden file is rejected (exit code 2), because scoring it against the
demo knowledge base would produce meaningless results.

To evaluate your own RAG system and golden set, import rag_eval and call
evaluate() with your own retriever and generator (README: "How to plug in
your own system").
"""
from __future__ import annotations

# ─────────────────────────────────────────────
# CONFIGURABLE THRESHOLDS  (edit these)
# ─────────────────────────────────────────────
THRESHOLDS = {
    "retrieval_recall":   {"warn": 0.5,  "fail": 0.3},
    "answer_faithfulness": {"warn": 0.4, "fail": 0.2},
    "answer_relevance":   {"warn": 0.6,  "fail": 0.4},
    "exact_match":        {"warn": 0.4,  "fail": 0.2},
    "token_f1":           {"warn": 0.4,  "fail": 0.2},
}

# Cost per 1k tokens (USD) — used for LLM-judge cost estimate
LLM_COST_PER_1K = {
    "gpt-4o-mini":      {"input": 0.00015, "output": 0.0006},
    "claude-haiku-4-5-20251001": {"input": 0.001,   "output": 0.005},
}

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import argparse
import csv
import json
import os
import string
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# Optional heavy deps — imported lazily so the script runs without them
# in the mock-only mode.
_sentence_transformers = None
_tiktoken = None
_rich_available = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from rich.text import Text
    _rich_available = True
    console = Console()
except ImportError:
    console = None  # type: ignore


# ─────────────────────────────────────────────
# TYPE ALIASES
# ─────────────────────────────────────────────
GoldenEntry = dict        # one row from golden_set.json
RetrievedDoc = dict       # {text: str, source_id: str}
EvalResult   = dict       # per-question result dict


# ─────────────────────────────────────────────
# 1. GOLDEN SET LOADER
# ─────────────────────────────────────────────
REQUIRED_FIELDS = {"id", "question", "expected_answer", "expected_source_ids"}

def load_golden_set(path: str) -> list[GoldenEntry]:
    """Load and validate a golden set JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Golden set not found: {path}")

    with open(p, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in golden set: {e}") from e

    if not isinstance(data, list):
        raise ValueError("Golden set must be a JSON array of objects.")

    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"Entry #{i} must be an object.")
        missing = REQUIRED_FIELDS - set(entry.keys())
        if missing:
            raise ValueError(
                f"Entry #{i} (id={entry.get('id', '?')}) is missing required fields: {missing}\n"
                f"Required fields: {REQUIRED_FIELDS}"
            )
        sources = entry["expected_source_ids"]
        if not isinstance(sources, list):
            raise ValueError(
                f"Entry #{i}: expected_source_ids must be a list, got "
                f"{type(sources).__name__}"
            )
        answerable = entry.get("answerable", True)
        if not isinstance(answerable, bool):
            raise ValueError(f"Entry #{i}: answerable must be true or false.")
        if any(not isinstance(s, str) or not s.strip() for s in sources):
            raise ValueError(f"Entry #{i}: expected_source_ids must contain non-empty strings.")
        if len(set(sources)) != len(sources):
            raise ValueError(f"Entry #{i}: expected_source_ids must not contain duplicates.")
        if answerable and not sources:
            raise ValueError(
                f"Entry #{i}: answerable question requires expected_source_ids; "
                "set answerable=false for an intentionally unanswerable question."
            )
        if not answerable and sources:
            raise ValueError(
                f"Entry #{i}: unanswerable question must have empty expected_source_ids."
            )
    return data


# ─────────────────────────────────────────────
# 2. EMBEDDER  (lazy-loaded, optional)
# ─────────────────────────────────────────────
_embedder_model = None

def _get_embedder():
    global _sentence_transformers, _embedder_model
    if _embedder_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for embedding-based metrics.\n"
                "Install with: pip install sentence-transformers\n"
                "Or use --no-embeddings to skip cosine similarity metrics."
            ) from e
    return _embedder_model


def default_embedder_fn(texts: list[str]) -> np.ndarray:
    """Default embedder using sentence-transformers all-MiniLM-L6-v2."""
    model = _get_embedder()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors (assumes pre-normalised)."""
    return float(np.dot(a, b))


# ─────────────────────────────────────────────
# 3. TOKENIZER  (lazy-loaded)
# ─────────────────────────────────────────────
def _tokenize(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokenizer.
    Falls back to tiktoken cl100k if available (for cost counting only).
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def _tiktoken_count(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens using tiktoken, with graceful fallback."""
    global _tiktoken
    if _tiktoken is None:
        try:
            import tiktoken as _tk
            _tiktoken = _tk
        except ImportError:
            return len(text.split())   # fallback: word count
    enc = _tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


# ─────────────────────────────────────────────
# 4. METRICS
# ─────────────────────────────────────────────

def metric_retrieval_recall(
    retrieved_docs: list[RetrievedDoc],
    expected_source_ids: list[str],
    k: int,
) -> Optional[float]:
    """
    Retrieval Recall@K for questions with expected sources.
    Returns None (not applicable) when there are no expected sources.
    Score = (# expected sources found in top-K) / (# expected sources)
    """
    if not expected_source_ids:
        return None
    top_k_ids = {d["source_id"] for d in retrieved_docs[:k]}
    hits = sum(1 for sid in expected_source_ids if sid in top_k_ids)
    return hits / len(expected_source_ids)


def metric_token_f1(predicted: str, expected: str) -> tuple[float, float, float]:
    """
    Token-level F1 (SQuAD-style).
    Returns (f1, precision, recall).
    """
    pred_tokens  = _tokenize(predicted)
    gold_tokens  = _tokenize(expected)

    if not pred_tokens and not gold_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0

    pred_counts  = defaultdict(int)
    gold_counts  = defaultdict(int)
    for t in pred_tokens: pred_counts[t]  += 1
    for t in gold_tokens:  gold_counts[t] += 1

    common = sum(min(pred_counts[t], gold_counts[t]) for t in pred_counts)

    precision = common / len(pred_tokens)
    recall    = common / len(gold_tokens)
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


def metric_exact_match(predicted: str, expected: str) -> float:
    """1.0 if normalised strings match exactly, 0.0 otherwise."""
    def _norm(s: str) -> str:
        s = s.lower().strip()
        s = s.translate(str.maketrans("", "", string.punctuation))
        return " ".join(s.split())
    return 1.0 if _norm(predicted) == _norm(expected) else 0.0


def metric_faithfulness(answer: str, context_chunks: list[str]) -> float:
    """
    Token-overlap faithfulness.
    What fraction of answer tokens appear in the retrieved context?
    No API key required. Optional LLM judge available via --llm-judge.
    """
    if not context_chunks:
        return 0.0
    answer_tokens  = set(_tokenize(answer))
    context_tokens = set(_tokenize(" ".join(context_chunks)))
    if not answer_tokens:
        return 1.0
    overlap = answer_tokens & context_tokens
    return len(overlap) / len(answer_tokens)


def metric_answer_relevance(
    question: str,
    answer: str,
    embedder_fn: Callable,
) -> float:
    """
    Cosine similarity between question embedding and answer embedding.
    Higher = answer is more relevant to the question.
    """
    embeddings = embedder_fn([question, answer])
    return cosine_similarity(embeddings[0], embeddings[1])


# ─────────────────────────────────────────────
# 5. OPTIONAL LLM-AS-JUDGE
# ─────────────────────────────────────────────
JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator of RAG system outputs.
Score each dimension on a scale of 1–5, then give a one-sentence reason.
Respond ONLY with valid JSON in this exact format:
{
  "faithfulness": {"score": <int 1-5>, "reason": "<one sentence>"},
  "relevance":    {"score": <int 1-5>, "reason": "<one sentence>"}
}
Faithfulness: Is the answer grounded in the provided context? (5 = fully grounded, 1 = hallucinated)
Relevance: Does the answer actually address the question? (5 = directly answers, 1 = off-topic)"""

JUDGE_USER_TEMPLATE = """Question: {question}

Context:
{context}

Answer: {answer}"""


# Supported judge models -> (provider, required API key env var).
# The requested model is always the model called: no provider/model fallback.
JUDGE_MODELS = {
    "gpt-4o-mini":               ("openai",    "OPENAI_API_KEY"),
    "claude-haiku-4-5-20251001": ("anthropic", "ANTHROPIC_API_KEY"),
}


class JudgeConfigError(RuntimeError):
    """LLM judge requested, but the model is unsupported or its API key is not set."""


def _resolve_judge(model: str) -> str:
    """Return the provider for `model`, or raise JudgeConfigError. Never substitutes a model."""
    if model not in JUDGE_MODELS:
        raise JudgeConfigError(
            f"Unsupported judge model {model!r}. Supported: {', '.join(JUDGE_MODELS)}."
        )
    provider, key_var = JUDGE_MODELS[model]
    if not os.environ.get(key_var):
        raise JudgeConfigError(
            f"LLM judge model {model!r} requires {key_var}, which is not set (no usable API key). "
            "Set it, choose another --judge-model, or run without --llm-judge."
        )
    return provider


def new_judge_tracker() -> dict:
    """
    Judge usage accounting. Estimated and provider-reported usage are kept apart:
      calls_succeeded / calls_failed : provider request outcome (usage may be billed on success)
      calls_scored                   : request succeeded AND the reply passed validation
      invalid_responses              : request succeeded but the reply was unusable
      estimated_prompt_tokens        : local tiktoken estimate (None if the tokenizer failed)
      input_tokens / output_tokens   : provider-reported usage, successful requests only
    """
    return {
        "model_used": None,
        "calls_attempted": 0,
        "calls_succeeded": 0,
        "calls_failed": 0,
        "calls_scored": 0,
        "invalid_responses": 0,
        "estimated_prompt_tokens": 0,
        "estimate_error": None,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def _validate_judge_response(parsed) -> dict:
    """Return the expected judge structure (extra keys dropped), or raise ValueError."""
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    out = {}
    for dim in ("faithfulness", "relevance"):
        item = parsed.get(dim)
        if not isinstance(item, dict):
            raise ValueError(f"missing or non-object {dim!r}")
        score, reason = item.get("score"), item.get("reason")
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"{dim}.score must be an integer 1-5, got {score!r}")
        if not isinstance(reason, str):
            raise ValueError(f"{dim}.reason must be a string, got {type(reason).__name__}")
        out[dim] = {"score": score, "reason": reason}
    return out


def llm_judge(
    question: str,
    answer: str,
    context_chunks: list[str],
    model: str,
    token_tracker: dict,
) -> dict:
    """
    Call OpenAI or Anthropic to score faithfulness and relevance (1–5).
    Returns dict with scores and reasons, or error info.
    token_tracker must come from new_judge_tracker().
    """
    try:
        provider = _resolve_judge(model)
    except JudgeConfigError as e:
        return {"error": str(e)}

    context_text = "\n---\n".join(context_chunks[:5])  # cap context length
    user_msg = JUDGE_USER_TEMPLATE.format(
        question=question,
        context=context_text,
        answer=answer,
    )
    token_tracker["model_used"] = model
    token_tracker["calls_attempted"] += 1

    # The estimate is informational only: a tokenizer failure must not stop the run.
    if token_tracker["estimate_error"] is None:
        try:
            token_tracker["estimated_prompt_tokens"] += _tiktoken_count(JUDGE_SYSTEM_PROMPT + user_msg)
        except Exception as e:
            token_tracker["estimate_error"] = f"{type(e).__name__}: {e}"
            token_tracker["estimated_prompt_tokens"] = None
            _warn(
                f"judge prompt-token estimate unavailable ({token_tracker['estimate_error'].splitlines()[0]}); "
                "continuing without it. Provider-reported usage is unaffected."
            )

    try:
        if provider == "openai":
            import openai
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            in_tokens  = response.usage.prompt_tokens
            out_tokens = response.usage.completion_tokens
        else:
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            response = client.messages.create(
                model=model,
                max_tokens=200,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text
            in_tokens  = response.usage.input_tokens
            out_tokens = response.usage.output_tokens
    except Exception as e:
        token_tracker["calls_failed"] += 1
        return {"error": f"{type(e).__name__}: {e}"}

    # The request succeeded, so its usage counts even if the reply is unusable;
    # only a validated reply counts as scored.
    token_tracker["calls_succeeded"] += 1
    token_tracker["input_tokens"]  += in_tokens
    token_tracker["output_tokens"] += out_tokens
    try:
        scores = _validate_judge_response(json.loads(raw))
    except (ValueError, TypeError) as e:
        token_tracker["invalid_responses"] += 1
        return {"error": f"Invalid judge response: {e}"}
    token_tracker["calls_scored"] += 1
    return scores


# ─────────────────────────────────────────────
# 6. LATENCY + COST TRACKING
# ─────────────────────────────────────────────

def estimate_cost(token_tracker: dict, model: str) -> Optional[float]:
    """
    Estimate USD cost from token counts at list price.
    This is an estimate, not a bill. Returns None if the model has no known price.
    """
    rates = LLM_COST_PER_1K.get(model)
    if rates is None:
        return None
    input_cost  = token_tracker.get("input_tokens",  0) / 1000 * rates["input"]
    output_cost = token_tracker.get("output_tokens", 0) / 1000 * rates["output"]
    return input_cost + output_cost


# ─────────────────────────────────────────────
# 7. VERDICT HELPER
# ─────────────────────────────────────────────

def verdict(metric_name: str, score: float) -> str:
    """Return PASS / WARN / FAIL based on configured thresholds."""
    t = THRESHOLDS.get(metric_name, {"warn": 0.5, "fail": 0.3})
    if score >= t["warn"]:
        return "PASS"
    if score >= t["fail"]:
        return "WARN"
    return "FAIL"


def _relevance_metric(score: Optional[float], error: Optional[str]) -> dict:
    """Relevance result: ERROR (embedder failed) is kept distinct from SKIP (not requested)."""
    if error is not None:
        return {"score": None, "method": "error", "verdict": "ERROR", "error": error}
    if score is None:
        return {"score": None, "method": "skipped", "verdict": "SKIP"}
    return {
        "score":   round(score, 4),
        "method":  "cosine_similarity",
        "verdict": verdict("answer_relevance", score),
    }


# ─────────────────────────────────────────────
# 8. CORE EVALUATE FUNCTION
# ─────────────────────────────────────────────

def evaluate(
    golden_set: list[GoldenEntry],
    retriever_fn: Callable[[str], list[RetrievedDoc]],
    generator_fn: Callable[[str, list[str]], str],
    embedder_fn:  Callable[[list[str]], np.ndarray] = default_embedder_fn,
    k: int = 3,
    use_llm_judge: bool = False,
    llm_judge_model: str = "gpt-4o-mini",
    no_embeddings: bool = False,
) -> list[EvalResult]:
    """
    Run the full evaluation loop.

    Parameters
    ----------
    golden_set       : list of golden set entries (from load_golden_set)
    retriever_fn     : fn(question) -> [{text, source_id}, ...]
    generator_fn     : fn(question, context_texts) -> answer_str
    embedder_fn      : fn(texts) -> np.ndarray  (default: MiniLM)
    k                : top-K for retrieval recall
    use_llm_judge    : enable LLM-as-judge for faithfulness + relevance
    llm_judge_model  : which model to use for judging
    no_embeddings    : skip cosine-similarity relevance (saves ~1 GB RAM)

    Returns
    -------
    list of per-question result dicts

    Raises
    ------
    JudgeConfigError if use_llm_judge is set but no usable API key is configured.
    """
    if use_llm_judge:
        _resolve_judge(llm_judge_model)   # fail before running the pipeline

    results = []
    token_tracker = new_judge_tracker()
    # After the first embedder failure, relevance is disabled for the rest of the
    # run (no re-initialising a broken model per question) and reported as ERROR.
    embedding_error: Optional[str] = None

    for entry in golden_set:
        qid      = entry["id"]
        question = entry["question"]
        expected = entry["expected_answer"]
        exp_srcs = entry["expected_source_ids"]

        # ── Retrieval ──────────────────────────────────────────
        t0 = time.perf_counter()
        retrieved = retriever_fn(question)
        retrieval_latency = time.perf_counter() - t0

        context_texts = [d["text"] for d in retrieved]

        # ── Generation ────────────────────────────────────────
        t1 = time.perf_counter()
        answer = generator_fn(question, context_texts)
        generation_latency = time.perf_counter() - t1

        # ── Metrics ───────────────────────────────────────────
        rec_at_k = metric_retrieval_recall(retrieved, exp_srcs, k)
        faith    = metric_faithfulness(answer, context_texts)
        f1, prec, rec = metric_token_f1(answer, expected)
        em       = metric_exact_match(answer, expected)

        relevance = None
        if not no_embeddings and embedding_error is None:
            try:
                relevance = metric_answer_relevance(question, answer, embedder_fn)
            except Exception as e:
                embedding_error = f"{type(e).__name__}: {e}"
                _warn(
                    f"answer relevance failed on question {qid!r}: "
                    f"{embedding_error.splitlines()[0]}\n"
                    "  Relevance is disabled for the rest of this run and reported as ERROR "
                    "(not SKIP). Fix the embedder, or use --no-embeddings to skip it intentionally."
                )

        # ── LLM judge (optional) ──────────────────────────────
        judge_scores = None
        judge_row_cost = None
        judge_status = None
        if use_llm_judge:
            before = dict(token_tracker)
            judge_scores = llm_judge(
                question, answer, context_texts, llm_judge_model, token_tracker
            )
            judge_row_cost = estimate_cost(
                {t: token_tracker[t] - before[t] for t in ("input_tokens", "output_tokens")},
                llm_judge_model,
            )
            if token_tracker["calls_scored"] > before["calls_scored"]:
                judge_status = "scored"
            elif token_tracker["invalid_responses"] > before["invalid_responses"]:
                judge_status = "invalid_response"
            elif token_tracker["calls_failed"] > before["calls_failed"]:
                judge_status = "request_failed"
            else:
                judge_status = "not_attempted"

        # ── Assemble result ───────────────────────────────────
        result: EvalResult = {
            "id":            qid,
            "category":      entry.get("category", "—"),
            "difficulty":    entry.get("difficulty", "—"),
            "question":      question,
            "expected_answer": expected,
            "generated_answer": answer,
            "retrieved_chunks": retrieved,
            "metrics": {
                "retrieval_recall_at_k": {
                    "score":   round(rec_at_k, 4) if rec_at_k is not None else None,
                    "k":       k,
                    "verdict": verdict("retrieval_recall", rec_at_k) if rec_at_k is not None else "SKIP",
                },
                "answer_faithfulness": {
                    "score":   round(faith, 4),
                    "method":  "token_overlap",
                    "verdict": verdict("answer_faithfulness", faith),
                },
                "answer_relevance": _relevance_metric(relevance, embedding_error),
                "exact_match": {
                    "score":   round(em, 4),
                    "verdict": verdict("exact_match", em),
                },
                "token_f1": {
                    "score":     round(f1, 4),
                    "precision": round(prec, 4),
                    "recall":    round(rec, 4),
                    "verdict":   verdict("token_f1", f1),
                },
            },
            "latency": {
                "retrieval_s":   round(retrieval_latency,  3),
                "generation_s":  round(generation_latency, 3),
                "total_s":       round(retrieval_latency + generation_latency, 3),
            },
        }

        if judge_scores:
            result["llm_judge"] = judge_scores
            result["llm_judge_status"] = judge_status
            result["_judge_estimated_cost_usd"] = (
                round(judge_row_cost, 6) if judge_row_cost is not None else None
            )

        results.append(result)

    # ── Attach usage + cost estimate if LLM judge was used ────
    if use_llm_judge:
        cost = estimate_cost(token_tracker, llm_judge_model)
        results.append({
            "_summary": True,
            "llm_judge_model": llm_judge_model,
            "llm_judge_calls": {
                "attempted":         token_tracker["calls_attempted"],
                "request_succeeded": token_tracker["calls_succeeded"],
                "request_failed":    token_tracker["calls_failed"],
                "scored":            token_tracker["calls_scored"],
                "invalid_response":  token_tracker["invalid_responses"],
            },
            # Provider-reported usage from successful requests only.
            "llm_judge_usage": {
                "input_tokens":  token_tracker["input_tokens"],
                "output_tokens": token_tracker["output_tokens"],
            },
            "llm_judge_estimated_prompt_tokens": token_tracker["estimated_prompt_tokens"],
            "llm_judge_estimate_error": token_tracker["estimate_error"],
            # List price x provider-reported usage. An estimate, not an invoice.
            "llm_judge_estimated_cost_usd": round(cost, 6) if cost is not None else None,
        })

    return results


# ─────────────────────────────────────────────
# 9. SUMMARY STATISTICS
# ─────────────────────────────────────────────

def summarise(results: list[EvalResult]) -> dict:
    """Compute mean scores and overall verdict counts across all questions."""
    rows = [r for r in results if not r.get("_summary")]
    if not rows:
        return {}

    metric_keys = list(rows[0]["metrics"].keys())
    totals: dict[str, list[float]] = defaultdict(list)

    for r in rows:
        for mk in metric_keys:
            s = r["metrics"][mk]["score"]
            if s is not None:
                totals[mk].append(s)

    means = {mk: round(sum(vs) / len(vs), 4) for mk, vs in totals.items() if vs}

    verdicts: dict[str, dict] = {}
    for mk in metric_keys:
        counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0, "ERROR": 0}
        for r in rows:
            v = r["metrics"][mk]["verdict"]
            counts[v] = counts.get(v, 0) + 1
        verdicts[mk] = counts

    avg_latency = round(sum(r["latency"]["total_s"] for r in rows) / len(rows), 3)

    # Anything not computed for every row it should have been. Empty == complete.
    # status "partial": some rows have a result; "error": none do.
    incomplete: dict[str, dict] = {}
    rel_errors = verdicts["answer_relevance"]["ERROR"]
    if rel_errors:
        computed = len(totals["answer_relevance"])
        incomplete["answer_relevance"] = {
            "status":   "partial" if computed else "error",
            "computed": computed,
            "error":    rel_errors,
        }
    statuses = [r["llm_judge_status"] for r in rows if "llm_judge_status" in r]
    scored = statuses.count("scored")
    if scored < len(statuses):
        incomplete["llm_judge"] = {
            "status": "partial" if scored else "error",
            "scored": scored,
            **{s: statuses.count(s) for s in ("request_failed", "invalid_response", "not_attempted")
               if s in statuses},
        }

    out = {
        "n_questions":   len(rows),
        "mean_scores":   means,
        "metric_counts": {mk: len(totals[mk]) for mk in metric_keys},
        "verdicts":      verdicts,
        "avg_latency_s": avg_latency,
        "incomplete":    incomplete,
    }
    judge = next((r for r in results if r.get("_summary")), None)
    if judge:
        out["llm_judge"] = {k: v for k, v in judge.items() if k != "_summary"}
    return out


# ─────────────────────────────────────────────
# 10. OUTPUT — RICH TABLE
# ─────────────────────────────────────────────

VERDICT_STYLE = {
    "PASS": "bold green", "WARN": "bold yellow", "FAIL": "bold red",
    "SKIP": "dim", "ERROR": "bold magenta", "PARTIAL": "bold magenta",
}

def _verdict_cell(v: str, score) -> "Text":
    from rich.text import Text
    score_str = f"{score:.2f}" if isinstance(score, float) else "N/A"
    label = f"{score_str} {v}"
    return Text(label, style=VERDICT_STYLE.get(v, ""))


def _mean_cell(metric_name: str, score, status: Optional[str] = None) -> "Text":
    """status comes from summary["incomplete"]: "partial" or "error" (None == complete)."""
    if status == "error":
        return _verdict_cell("ERROR", None)
    if status == "partial":
        return _verdict_cell("PARTIAL", score)
    if score is None:
        return _verdict_cell("SKIP", score)
    return _verdict_cell(verdict(metric_name, score), score)


def _incomplete_lines(summary: dict) -> list[str]:
    """Human-readable notes for every metric in summary["incomplete"]."""
    lines = []
    n = summary.get("n_questions", 0)
    rel = summary.get("incomplete", {}).get("answer_relevance")
    if rel and rel["status"] == "partial":
        lines.append(
            f"answer_relevance: PARTIAL — computed for {rel['computed']}/{n} rows, ERROR on "
            f"{rel['error']}; the mean covers only the computed rows."
        )
    elif rel:
        lines.append(f"answer_relevance: ERROR — not computed for any row (0/{n}); no mean.")
    judge = summary.get("incomplete", {}).get("llm_judge")
    if judge:
        detail = ", ".join(f"{k}={v}" for k, v in judge.items() if k not in ("status", "scored"))
        label = "PARTIAL" if judge["status"] == "partial" else "ERROR"
        lines.append(f"llm_judge: {label} — scored {judge['scored']}/{n} rows ({detail}).")
    return lines


def _judge_summary_text(j: dict) -> str:
    calls = j["llm_judge_calls"]
    cost = j["llm_judge_estimated_cost_usd"]
    est = j["llm_judge_estimated_prompt_tokens"]
    cost_str = f"~${cost:.4f}" if cost is not None else "unknown (no list price for model)"
    return (
        f"model={j['llm_judge_model']}  scored={calls['scored']}/{calls['attempted']}  "
        f"requests_succeeded={calls['request_succeeded']}/{calls['attempted']}  "
        f"reported_tokens={j['llm_judge_usage']}  "
        f"estimated_prompt_tokens={est if est is not None else 'unavailable'}  "
        f"estimated_cost={cost_str} (list price × provider-reported usage; not a bill)"
    )


def print_rich_table(results: list[EvalResult], summary: dict) -> None:
    """Print a colour-coded Rich table to the terminal."""
    if not _rich_available:
        _print_plain_table(results, summary)
        return

    table = Table(
        title="RAG Evaluation Results",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("ID",         style="dim", width=8)
    table.add_column("Cat",        style="dim", width=10)
    table.add_column("Diff",       style="dim", width=6)
    table.add_column("Recall@K",   justify="center", width=12)
    table.add_column("Faithful",   justify="center", width=12)
    table.add_column("Relevance",  justify="center", width=12)
    table.add_column("Token F1",   justify="center", width=12)
    table.add_column("EM",         justify="center", width=8)
    table.add_column("Latency",    justify="right",  width=9)

    rows = [r for r in results if not r.get("_summary")]

    for r in rows:
        m = r["metrics"]
        rel = m["answer_relevance"]
        table.add_row(
            str(r["id"]),
            str(r.get("category", "—"))[:10],
            str(r.get("difficulty", "—"))[:5],
            _verdict_cell(m["retrieval_recall_at_k"]["verdict"], m["retrieval_recall_at_k"]["score"]),
            _verdict_cell(m["answer_faithfulness"]["verdict"],   m["answer_faithfulness"]["score"]),
            _verdict_cell(rel["verdict"], 0.0 if rel["verdict"] == "SKIP" else rel["score"]),
            _verdict_cell(m["token_f1"]["verdict"],              m["token_f1"]["score"]),
            _verdict_cell(m["exact_match"]["verdict"],           m["exact_match"]["score"]),
            f"{r['latency']['total_s']:.2f}s",
        )

    # Summary row
    if summary:
        ms = summary["mean_scores"]
        table.add_section()
        table.add_row(
            "[bold]MEAN[/bold]", "", "",
            _mean_cell("retrieval_recall",   ms.get("retrieval_recall_at_k")),
            _mean_cell("answer_faithfulness", ms.get("answer_faithfulness")),
            _mean_cell("answer_relevance",   ms.get("answer_relevance"),
                       summary.get("incomplete", {}).get("answer_relevance", {}).get("status")),
            _mean_cell("token_f1",           ms.get("token_f1")),
            _mean_cell("exact_match",        ms.get("exact_match")),
            f"{summary['avg_latency_s']:.2f}s",
        )

    console.print(table)

    # LLM judge usage + estimated cost
    if summary and "llm_judge" in summary:
        console.print(Text.assemble(("\nLLM Judge ", "bold cyan"), _judge_summary_text(summary["llm_judge"])))

    # Incomplete metrics
    notes = _incomplete_lines(summary) if summary else []
    if notes:
        console.print(Text("\nINCOMPLETE RESULTS", style="bold magenta"))
        for line in notes:
            console.print(Text(f"  {line}", style="magenta"))


def _format_optional_score(score, width: int) -> str:
    return f"{score:>{width}.2f}" if score is not None else f"{'N/A':>{width}}"


def _print_plain_table(results: list[EvalResult], summary: dict) -> None:
    """Fallback plain-text table when Rich is not installed."""
    rows = [r for r in results if not r.get("_summary")]
    header = f"{'ID':<10} {'Recall@K':>9} {'Faithful':>9} {'Relevance':>10} {'Token F1':>9} {'EM':>6} {'Latency':>8}"
    print("\nRAG Evaluation Results")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        m  = r["metrics"]
        ra = m["retrieval_recall_at_k"]["score"]
        fa = m["answer_faithfulness"]["score"]
        re = m["answer_relevance"]["score"]
        f1 = m["token_f1"]["score"]
        em = m["exact_match"]["score"]
        la = r["latency"]["total_s"]
        print(
            f"{str(r['id']):<10} {_format_optional_score(ra, 9)} "
            f"{_format_optional_score(fa, 9)} {_format_optional_score(re, 10)} "
            f"{_format_optional_score(f1, 9)} {_format_optional_score(em, 6)} "
            f"{la:>7.2f}s"
        )
    if summary:
        ms = summary["mean_scores"]
        print("-" * len(header))
        print(
            f"{'MEAN':<10} "
            f"{_format_optional_score(ms.get('retrieval_recall_at_k'), 9)} "
            f"{_format_optional_score(ms.get('answer_faithfulness'), 9)} "
            f"{_format_optional_score(ms.get('answer_relevance'), 10)} "
            f"{_format_optional_score(ms.get('token_f1'), 9)} "
            f"{_format_optional_score(ms.get('exact_match'), 6)} "
            f"{summary['avg_latency_s']:>7.2f}s"
        )
        if "llm_judge" in summary:
            print(f"\nLLM Judge {_judge_summary_text(summary['llm_judge'])}")
        notes = _incomplete_lines(summary)
        if notes:
            print("\nINCOMPLETE RESULTS")
            for line in notes:
                print(f"  {line}")
    print()


# ─────────────────────────────────────────────
# 11. OUTPUT — JSON
# ─────────────────────────────────────────────

def save_json(results: list[EvalResult], summary: dict, output_path: str) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "results": [r for r in results if not r.get("_summary")],
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    _log(f"Results saved → {p}")


# ─────────────────────────────────────────────
# 12. OUTPUT — CSV
# ─────────────────────────────────────────────

def save_csv(results: list[EvalResult], output_path: str) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for r in results if not r.get("_summary")]
    if not rows:
        return

    fieldnames = [
        "id", "category", "difficulty", "question",
        "retrieval_recall_at_k", "answer_faithfulness", "answer_relevance",
        "exact_match", "token_f1", "token_precision", "token_recall",
        "retrieval_latency_s", "generation_latency_s", "total_latency_s",
        "generated_answer",
        # Distinguishes a blank relevance cell: cosine_similarity / skipped / error.
        "answer_relevance_method",
    ]
    judged = any("llm_judge_status" in r for r in rows)
    if judged:
        fieldnames += ["llm_judge_status", "judge_estimated_cost_usd"]

    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            m = r["metrics"]
            writer.writerow({
                "id":                    r["id"],
                "category":              r.get("category", ""),
                "difficulty":            r.get("difficulty", ""),
                "question":              r["question"],
                "retrieval_recall_at_k": m["retrieval_recall_at_k"]["score"],
                "answer_faithfulness":   m["answer_faithfulness"]["score"],
                "answer_relevance":      m["answer_relevance"]["score"],
                "exact_match":           m["exact_match"]["score"],
                "token_f1":              m["token_f1"]["score"],
                "token_precision":       m["token_f1"]["precision"],
                "token_recall":          m["token_f1"]["recall"],
                "retrieval_latency_s":   r["latency"]["retrieval_s"],
                "generation_latency_s":  r["latency"]["generation_s"],
                "total_latency_s":       r["latency"]["total_s"],
                "generated_answer":      r["generated_answer"],
                "answer_relevance_method": m["answer_relevance"]["method"],
                "llm_judge_status":      r.get("llm_judge_status", ""),
                "judge_estimated_cost_usd": r.get("_judge_estimated_cost_usd", ""),
            })
    _log(f"CSV saved → {p}")


# ─────────────────────────────────────────────
# 13. LOGGING HELPER
# ─────────────────────────────────────────────

def _log(msg: str) -> None:
    if _rich_available:
        console.print(f"[dim]{msg}[/dim]")
    else:
        print(msg)


def _warn(msg: str) -> None:
    print(f"Warning: {msg}", file=sys.stderr)


# ─────────────────────────────────────────────
# 14. CLI
# ─────────────────────────────────────────────

BUNDLED_GOLDEN = Path(__file__).resolve().parent / "examples" / "golden_set.json"


def _is_bundled_golden(path: str) -> bool:
    """True if `path` is the bundled example file (by file identity, not path text)."""
    try:
        return os.path.samefile(path, BUNDLED_GOLDEN)
    except OSError:
        return False


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate the bundled demo RAG pipeline against its golden set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--golden",       required=True,
                   help="Path to the bundled examples/golden_set.json (the CLI runs only the demo "
                        "pipeline; use the Python API for your own golden set)")
    p.add_argument("--k",            type=int, default=3,    help="Top-K for retrieval recall (default: 3)")
    p.add_argument("--output",       default="results/results",
                   help="Output path prefix (default: results/results -> results/results.json)")
    p.add_argument("--csv",          action="store_true",    help="Also export results as CSV")
    p.add_argument("--llm-judge",    action="store_true",    help="Enable LLM-as-judge (requires API key)")
    p.add_argument("--judge-model",  default="gpt-4o-mini",  help="Model for LLM judge (default: gpt-4o-mini)")
    p.add_argument("--no-embeddings",action="store_true",    help="Skip cosine-similarity relevance metric")
    p.add_argument("--no-json",      action="store_true",    help="Skip saving results.json")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # The CLI is demo-only: refuse to score another golden set against the mock
    # knowledge base. A missing path falls through to load_golden_set's error.
    if Path(args.golden).exists() and not _is_bundled_golden(args.golden):
        print(
            f"Error: --golden {args.golden} is not the bundled examples/golden_set.json.\n"
            "  This CLI runs only the bundled demo pipeline (mock retriever + generator in\n"
            "  examples/mock_retriever.py), which can only answer the bundled example questions.\n"
            "  Scoring another golden set against it would give meaningless results, so nothing\n"
            "  was evaluated or written.\n"
            "  To evaluate your own golden set, call rag_eval.evaluate() from Python with your own\n"
            "  retriever and generator: see \"How to plug in your own system\" in README.md.",
            file=sys.stderr,
        )
        sys.exit(2)

    _log(f"Loading golden set: {args.golden}")
    golden = load_golden_set(args.golden)
    _log(f"  {len(golden)} questions loaded")

    # ── Import the user-defined functions ──────────────────────
    # By default we use the mock retriever + generator from examples/.
    # Replace these with your own functions.
    try:
        from examples.mock_retriever import mock_retriever_fn, mock_generator_fn  # type: ignore
        retriever_fn = mock_retriever_fn
        generator_fn = mock_generator_fn
        _log("Using mock retriever + generator (examples/mock_retriever.py)")
    except ImportError:
        _log("[yellow]Warning: examples/mock_retriever.py not found. "
             "Plug in your own retriever_fn / generator_fn.[/yellow]")
        return

    _log("Running evaluation…\n")
    try:
        results = evaluate(
            golden_set     = golden,
            retriever_fn   = retriever_fn,
            generator_fn   = generator_fn,
            k              = args.k,
            use_llm_judge  = args.llm_judge,
            llm_judge_model= args.judge_model,
            no_embeddings  = args.no_embeddings,
        )
    except JudgeConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    summary = summarise(results)
    print_rich_table(results, summary)

    json_path = args.output + ".json"
    if not args.no_json:
        save_json(results, summary, json_path)

    if args.csv:
        save_csv(results, args.output + ".csv")

    # Results are still written above; exit 1 flags that they are incomplete.
    if summary and summary["incomplete"]:
        for line in _incomplete_lines(summary):
            _warn(line)
        _warn("results are incomplete (see summary.incomplete in the JSON); exiting with status 1.")
        sys.exit(1)


if __name__ == "__main__":
    main()
