# Understanding the evaluation metrics

This document describes the starter evaluator's available metrics and how to interpret the sample output. The default metrics are lightweight signals, **not** substitutes for a human review or calibrated LLM-based evaluation.

## At a glance

| Metric | What it measures | Main limitation |
|---|---|---|
| Retrieval Recall@K | Fraction of known relevant source IDs found among the top-K retrieved chunks | Requires annotated expected source IDs; does not measure ranking quality or irrelevant extra chunks |
| Answer Faithfulness | Token-overlap-based grounding proxy comparing the answer with retrieved context | Lexical overlap does not prove factual support or detect all contradictions |
| Answer Relevance | Embedding similarity between the question and generated answer | Semantic similarity is not answer correctness; skipped with `--no-embeddings`; `ERROR` if the embedder fails |
| Token F1 | Token overlap between generated and expected answers | Penalizes good paraphrases; may reward shared words in incorrect answers |
| Exact Match | Exact normalized comparison of generated and reference answers | Strict for many correct alternative phrasings |
| Latency | Time recorded for parts of the evaluation | Depends on local environment, mock versus real integrations, and model setup |

## Retrieval Recall@K

For an **answerable** question, let `E` be its set of expected source IDs, and `R_K` the IDs of the first `K` retrieved chunks:

```
Recall@K = |E ∩ R_K| / |E|
```

If two sources are expected and only one appears in the top three retrieved chunks, **Recall@3 = 0.5**. The default `K` is `3` unless you change it using `--k`.

**Important:** The reported mean is an average of *per-question recall fractions*. A mean of `0.6` does **not** necessarily mean exactly 40% of questions failed retrieval; questions with multiple expected sources may contribute partial recall.

### Unanswerable cases: N/A is not perfect recall

For an entry with no expected source IDs, set:

```json
{
  "answerable": false,
  "expected_source_ids": []
}
```

Retrieval recall is undefined for this case, because there is no expected source to retrieve. The evaluator records:

- Per-row JSON score: `null`
- Per-row verdict: `SKIP`
- Terminal display: `N/A SKIP`
- Aggregate recall: **excludes** this row

The sample's `q007` is one such case. Its other answer-quality metrics may still be evaluated. **The tool does not independently grade whether a model abstained correctly**; adding that requires a distinct abstention metric and test policy.

**Breaking change for older golden sets:** An empty `expected_source_ids` list is no longer accepted for a question that is answerable (including an entry where `answerable` is omitted and therefore defaults to `true`). Update the source annotations or declare the case explicitly unanswerable.

### Read denominators before comparing averages

The JSON summary exposes `metric_counts` for each metric. For the bundled eight-question example, seven answerable questions contribute to Retrieval Recall@K. Different metrics can have **different denominators**, especially when embeddings are disabled or all cases are unanswerable.

If every case is unanswerable, the retrieval-recall mean is **not present** in `mean_scores` (rather than an invented value of `0.0` or `1.0`).

## Answer Faithfulness: a lexical proxy

The default faithfulness score uses token overlap between the answer and retrieved material. It can help flag answers that introduce substantial unsupported wording, but it is not claim-level entailment, does not reliably spot subtle factual errors, and can reward an incorrect claim that shares tokens with the source.

Use it as a **triage indicator**. For high-stakes production cases, review answer claims and the cited context directly or add a separately calibrated judging approach.

## Answer Relevance: embedding similarity

This compares the question and answer representations using embeddings. It addresses topical alignment, **not** whether the answer follows the retrieved evidence or is factually correct.

When invoked with `--no-embeddings`, the metric is **not computed** and has a `SKIP` verdict. A placeholder `0.00` in the per-question terminal display is **not** a meaningful relevance value; consult the verdict and the summary `metric_counts`.

If relevance was requested but the embedder fails (model download blocked, package missing, an exception in your `embedder_fn`), the row is **not** `SKIP`:

- Per-row JSON: `{"score": null, "method": "error", "verdict": "ERROR", "error": "<ExceptionType>: <message>"}`
- One warning is printed to stderr. Relevance is then disabled for the rest of the run: the embedder is not reloaded for every question, and the remaining rows are also `ERROR`.
- The CSV `answer_relevance_method` column reads `error`, `skipped`, or `cosine_similarity`, so a blank relevance cell can be told apart.
- Mean row: `N/A ERROR` if no row was computed; `<mean> PARTIAL` if only some rows were computed. A `PARTIAL` mean covers only the computed rows; `metric_counts.answer_relevance` gives its denominator.

## Token F1 and Exact Match

These compare the model output against a reference answer. They are useful for concise factual outputs (an API limit, identifier, specific date), but may give low scores to reasonable paraphrases or long responses. Review mismatches before assuming the retrieval or generator is defective.

## PASS, WARN, FAIL, SKIP, ERROR, and PARTIAL

- **PASS / WARN / FAIL** are interpretations against configurable thresholds, not universal standards or guarantees of quality.
- **SKIP** means the score is not applicable (for example, recall for an unanswerable question) or was intentionally not computed (`--no-embeddings`). Do not treat it as a failure, success, or numeric zero.
- **ERROR** (per row) means the metric was requested but its computation failed. The result is missing, not zero, and the run is incomplete. Summary `verdicts` count `ERROR` alongside the other verdicts.
- **PARTIAL** appears only on the mean row. It means some rows were computed and others are `ERROR`, so the mean covers only the computed rows.

### Incomplete runs and exit codes

The JSON `summary.incomplete` lists every metric that was not computed for all of the rows it applies to. It is `{}` for a complete run:

```json
"incomplete": {
  "answer_relevance": {"status": "partial", "computed": 5, "error": 3},
  "llm_judge":        {"status": "error", "scored": 0, "request_failed": 6, "invalid_response": 2}
}
```

`status` is `"partial"` when at least one row has a result, and `"error"` when none do. The terminal also prints an `INCOMPLETE RESULTS` block, and the CLI exits with:

| Exit code | Meaning |
|---|---|
| `0` | Complete run |
| `1` | Results were written, but are incomplete (any relevance `ERROR` or any unscored LLM-judge row) |
| `2` | Configuration error before evaluation (unsupported `--judge-model`, or its API key is not set); no results written |

## Optional LLM judge: usage and estimated cost

`--judge-model` selects exactly one model, and that model is the one called. There is no fallback to another provider or model:

| `--judge-model` | Required key |
|---|---|
| `gpt-4o-mini` (default) | `OPENAI_API_KEY` |
| `claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` |

Each row records `llm_judge_status`:

- `scored`: the request succeeded and the reply passed validation (two integer scores from 1 to 5, each with a string reason).
- `invalid_response`: the request succeeded (usage may have been billed), but the reply was not valid judge output. The row has **no** judge score.
- `request_failed`: the provider request raised an error, and no usage was reported.

The JSON `summary.llm_judge` reports:

- `llm_judge_calls`: `attempted`, `request_succeeded`, `request_failed`, `scored`, and `invalid_response`.
- `llm_judge_usage`: provider-reported input/output tokens, from successful requests only.
- `llm_judge_estimated_prompt_tokens`: a local tokenizer estimate, kept separate from provider usage. It is `null` if the tokenizer could not be loaded; `llm_judge_estimate_error` says why, and judging continues.
- `llm_judge_estimated_cost_usd`: list price × provider-reported usage. This is **an estimate, not a bill**. It includes `invalid_response` requests, because those were billed.

Per row, `_judge_estimated_cost_usd` is that row's own estimate. The CSV adds `llm_judge_status` and `judge_estimated_cost_usd` columns when the judge is used.

## CSV export

The CSV keeps every score column (blank when there is no score) and then appends, after all other columns, one verdict per metric plus the retrieval `k`:

`retrieval_recall_at_k_verdict`, `retrieval_k`, `answer_faithfulness_verdict`, `answer_relevance_verdict`, `exact_match_verdict`, `token_f1_verdict`

Each verdict is the row's JSON verdict: `PASS`, `WARN`, `FAIL`, `SKIP`, or `ERROR`. A blank score cell is therefore never ambiguous; for example, `q007` has a blank `retrieval_recall_at_k` and a `SKIP` verdict. `PARTIAL` applies only to a mean, so it never appears in the per-row CSV; a partial run shows computed rows alongside `ERROR` rows.

## Breaking output changes (P2-A)

These changes replace fields whose old values were misleading:

- **Removed:** the in-memory `_summary` pseudo-row returned by `evaluate(..., use_llm_judge=True)` no longer has `llm_judge_total_cost_usd` or `total_tokens`. The old values counted non-judge text and calls that never happened. Use `llm_judge_estimated_cost_usd`, `llm_judge_usage`, and `llm_judge_calls`, which are also in `summary.llm_judge` in the JSON.
- **Removed:** per-row `_judge_cost_usd`, which split the total evenly across rows. Use `_judge_estimated_cost_usd`.
- **Judge routing:** a judge model is no longer silently swapped for another. For example, the default `gpt-4o-mini` with only `ANTHROPIC_API_KEY` set used to call Claude Haiku. To use Anthropic, pass `--judge-model claude-haiku-4-5-20251001`.
- **Missing key or unsupported model:** `evaluate(..., use_llm_judge=True)` raises `JudgeConfigError` before running the pipeline, and the CLI exits with `2`. Previously, every row got an `error` and the run exited `0`.
- **Exit status:** incomplete runs now exit with `1` (see above).
- **Python API:** `estimate_cost()` returns `None`, instead of `0.0`, for a model with no list price.
- **Additive changes:**
  - `"ERROR"` key in every `summary.verdicts` entry;
  - `summary.incomplete` and `summary.llm_judge`;
  - per-row `llm_judge_status`;
  - relevance `method: "error"` with an `error` message;
  - CSV columns `answer_relevance_method` (always), and `llm_judge_status` and `judge_estimated_cost_usd` (judge runs only), appended after the existing columns.

  Existing fields keep their names and meaning. A `SKIP` relevance row is unchanged.

A sensible debugging order is: **read the question and reference → check retrieved source IDs → inspect retrieved text → inspect answer → investigate individual metric signals**. Avoid deciding solely from the mean row.

## What to evaluate next

To assess operational readiness beyond this starter, consider separately tracking: abstention/refusal correctness, citation attribution, evidence sufficiency, retrieved-document freshness, access-control filtering, prompt-injection resilience, latency under production load, and recurring regressions. Those are **not included** merely because the default score table is green.
