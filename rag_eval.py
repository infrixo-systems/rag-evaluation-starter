"""
rag-evaluation-starter
======================
Evaluate any RAG pipeline against a golden set in under 30 minutes.
No SaaS. No signup. Pure Python.

Usage:
    python rag_eval.py --golden examples/golden_set.json
    python rag_eval.py --golden examples/golden_set.json --k 5 --llm-judge
    python rag_eval.py --golden examples/golden_set.json --csv --output results/run1

Plug in your own retriever and generator at the bottom of this file,
or import and call evaluate() directly from your own code.
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
    "claude-haiku-4-5-20251001": {"input": 0.00025, "output": 0.00125},
}

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import argparse
import csv
import json
import os
import string
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
        missing = REQUIRED_FIELDS - set(entry.keys())
        if missing:
            raise ValueError(
                f"Entry #{i} (id={entry.get('id', '?')}) is missing required fields: {missing}\n"
                f"Required fields: {REQUIRED_FIELDS}"
            )
        if not isinstance(entry["expected_source_ids"], list):
            raise ValueError(
                f"Entry #{i}: expected_source_ids must be a list, got "
                f"{type(entry['expected_source_ids']).__name__}"
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
) -> float:
    """
    Retrieval Recall@K
    Did the right source(s) appear in the top-K results?
    Score = (# expected sources found in top-K) / (# expected sources)
    """
    if not expected_source_ids:
        return 1.0
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
    """
    context_text = "\n---\n".join(context_chunks[:5])  # cap context length
    user_msg = JUDGE_USER_TEMPLATE.format(
        question=question,
        context=context_text,
        answer=answer,
    )
    prompt_tokens  = _tiktoken_count(JUDGE_SYSTEM_PROMPT + user_msg)
    token_tracker["input_tokens"] += prompt_tokens

    # ── OpenAI path ──────────────────────────────────────────
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key and model in ("gpt-4o-mini",):
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
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
            out_tokens = response.usage.completion_tokens
            token_tracker["output_tokens"] += out_tokens
            return json.loads(raw)
        except Exception as e:
            return {"error": str(e)}

    # ── Anthropic path ────────────────────────────────────────
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text
            out_tokens = response.usage.output_tokens
            token_tracker["output_tokens"] += out_tokens
            return json.loads(raw)
        except Exception as e:
            return {"error": str(e)}

    return {"error": "No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY."}


# ─────────────────────────────────────────────
# 6. LATENCY + COST TRACKING
# ─────────────────────────────────────────────

def estimate_cost(token_tracker: dict, model: str) -> float:
    """Estimate USD cost from token counts."""
    rates = LLM_COST_PER_1K.get(model, {"input": 0.0, "output": 0.0})
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
    """
    results = []
    token_tracker = {"input_tokens": 0, "output_tokens": 0}

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
        if not no_embeddings:
            try:
                relevance = metric_answer_relevance(question, answer, embedder_fn)
            except Exception:
                relevance = None   # if sentence-transformers not installed

        # ── LLM judge (optional) ──────────────────────────────
        judge_scores = None
        if use_llm_judge:
            judge_scores = llm_judge(
                question, answer, context_texts, llm_judge_model, token_tracker
            )

        # ── Token count for cost tracking ─────────────────────
        total_text = question + answer + " ".join(context_texts)
        token_tracker["input_tokens"] += _tiktoken_count(total_text)

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
                    "score":   round(rec_at_k, 4),
                    "k":       k,
                    "verdict": verdict("retrieval_recall", rec_at_k),
                },
                "answer_faithfulness": {
                    "score":   round(faith, 4),
                    "method":  "token_overlap",
                    "verdict": verdict("answer_faithfulness", faith),
                },
                "answer_relevance": {
                    "score":   round(relevance, 4) if relevance is not None else None,
                    "method":  "cosine_similarity" if relevance is not None else "skipped",
                    "verdict": verdict("answer_relevance", relevance) if relevance is not None else "SKIP",
                },
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

        results.append(result)

    # ── Attach cost estimate if LLM judge was used ────────────
    if use_llm_judge:
        cost = estimate_cost(token_tracker, llm_judge_model)
        for r in results:
            r["_judge_cost_usd"] = round(cost / len(results), 6)
        results.append({
            "_summary": True,
            "llm_judge_total_cost_usd": round(cost, 4),
            "llm_judge_model": llm_judge_model,
            "total_tokens": token_tracker,
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
        counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}
        for r in rows:
            v = r["metrics"][mk]["verdict"]
            counts[v] = counts.get(v, 0) + 1
        verdicts[mk] = counts

    avg_latency = round(sum(r["latency"]["total_s"] for r in rows) / len(rows), 3)

    return {
        "n_questions":   len(rows),
        "mean_scores":   means,
        "verdicts":      verdicts,
        "avg_latency_s": avg_latency,
    }


# ─────────────────────────────────────────────
# 10. OUTPUT — RICH TABLE
# ─────────────────────────────────────────────

VERDICT_STYLE = {"PASS": "bold green", "WARN": "bold yellow", "FAIL": "bold red", "SKIP": "dim"}

def _verdict_cell(v: str, score) -> "Text":
    from rich.text import Text
    score_str = f"{score:.2f}" if isinstance(score, float) else "—"
    label = f"{score_str} {v}"
    return Text(label, style=VERDICT_STYLE.get(v, ""))


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
        table.add_row(
            str(r["id"]),
            str(r.get("category", "—"))[:10],
            str(r.get("difficulty", "—"))[:5],
            _verdict_cell(m["retrieval_recall_at_k"]["verdict"], m["retrieval_recall_at_k"]["score"]),
            _verdict_cell(m["answer_faithfulness"]["verdict"],   m["answer_faithfulness"]["score"]),
            _verdict_cell(m["answer_relevance"]["verdict"],      m["answer_relevance"]["score"] or 0.0),
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
            _verdict_cell(verdict("retrieval_recall",   ms.get("retrieval_recall_at_k", 0)),   ms.get("retrieval_recall_at_k",   0.0)),
            _verdict_cell(verdict("answer_faithfulness", ms.get("answer_faithfulness",  0)),   ms.get("answer_faithfulness",     0.0)),
            _verdict_cell(verdict("answer_relevance",   ms.get("answer_relevance",      0)),   ms.get("answer_relevance",        0.0)),
            _verdict_cell(verdict("token_f1",           ms.get("token_f1",              0)),   ms.get("token_f1",                0.0)),
            _verdict_cell(verdict("exact_match",        ms.get("exact_match",           0)),   ms.get("exact_match",             0.0)),
            f"{summary['avg_latency_s']:.2f}s",
        )

    console.print(table)

    # LLM judge cost
    cost_rows = [r for r in results if r.get("_summary")]
    if cost_rows:
        cr = cost_rows[0]
        console.print(
            f"\n[bold cyan]LLM Judge[/bold cyan] model={cr['llm_judge_model']}  "
            f"total_cost=${cr['llm_judge_total_cost_usd']:.4f}  "
            f"tokens={cr['total_tokens']}"
        )


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
        re = m["answer_relevance"]["score"] or 0.0
        f1 = m["token_f1"]["score"]
        em = m["exact_match"]["score"]
        la = r["latency"]["total_s"]
        print(f"{str(r['id']):<10} {ra:>9.2f} {fa:>9.2f} {re:>10.2f} {f1:>9.2f} {em:>6.2f} {la:>7.2f}s")
    if summary:
        ms = summary["mean_scores"]
        print("-" * len(header))
        print(
            f"{'MEAN':<10} "
            f"{ms.get('retrieval_recall_at_k', 0):>9.2f} "
            f"{ms.get('answer_faithfulness', 0):>9.2f} "
            f"{ms.get('answer_relevance', 0):>10.2f} "
            f"{ms.get('token_f1', 0):>9.2f} "
            f"{ms.get('exact_match', 0):>6.2f} "
            f"{summary['avg_latency_s']:>7.2f}s"
        )
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
    ]

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


# ─────────────────────────────────────────────
# 14. CLI
# ─────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a RAG pipeline against a golden set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--golden",       required=True,          help="Path to golden_set.json")
    p.add_argument("--k",            type=int, default=3,    help="Top-K for retrieval recall (default: 3)")
    p.add_argument("--output",       default="results/results",      help="Output path prefix (default: results)")
    p.add_argument("--csv",          action="store_true",    help="Also export results as CSV")
    p.add_argument("--llm-judge",    action="store_true",    help="Enable LLM-as-judge (requires API key)")
    p.add_argument("--judge-model",  default="gpt-4o-mini",  help="Model for LLM judge (default: gpt-4o-mini)")
    p.add_argument("--no-embeddings",action="store_true",    help="Skip cosine-similarity relevance metric")
    p.add_argument("--no-json",      action="store_true",    help="Skip saving results.json")
    return p


def main() -> None:
    args = _build_parser().parse_args()

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
    results = evaluate(
        golden_set     = golden,
        retriever_fn   = retriever_fn,
        generator_fn   = generator_fn,
        k              = args.k,
        use_llm_judge  = args.llm_judge,
        llm_judge_model= args.judge_model,
        no_embeddings  = args.no_embeddings,
    )

    summary = summarise(results)
    print_rich_table(results, summary)

    json_path = args.output + ".json"
    if not args.no_json:
        save_json(results, summary, json_path)

    if args.csv:
        save_csv(results, args.output + ".csv")


if __name__ == "__main__":
    main()
