# Understanding the evaluation metrics

This document describes the starter evaluator's available metrics and how to interpret the sample output. The default metrics are lightweight signals, **not** substitutes for a human review or calibrated LLM-based evaluation.

## At a glance

| Metric | What it measures | Main limitation |
|---|---|---|
| Retrieval Recall@K | Fraction of known relevant source IDs found among the top-K retrieved chunks | Requires annotated expected source IDs; does not measure ranking quality or irrelevant extra chunks |
| Answer Faithfulness | Token-overlap-based grounding proxy comparing the answer with retrieved context | Lexical overlap does not prove factual support or detect all contradictions |
| Answer Relevance | Embedding similarity between the question and generated answer | Semantic similarity is not answer correctness; skipped with `--no-embeddings` |
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

## Token F1 and Exact Match

These compare the model output against a reference answer. They are useful for concise factual outputs (an API limit, identifier, specific date), but may give low scores to reasonable paraphrases or long responses. Review mismatches before assuming the retrieval or generator is defective.

## PASS, WARN, FAIL, and SKIP

- **PASS / WARN / FAIL** are interpretations against configurable thresholds, not universal standards or guarantees of quality.
- **SKIP** means the score is not applicable or was not computed for that case or metric. Do not treat it as a failure, success, or numeric zero.

A sensible debugging order is: **read the question and reference → check retrieved source IDs → inspect retrieved text → inspect answer → investigate individual metric signals**. Avoid deciding solely from the mean row.

## What to evaluate next

To assess operational readiness beyond this starter, consider separately tracking: abstention/refusal correctness, citation attribution, evidence sufficiency, retrieved-document freshness, access-control filtering, prompt-injection resilience, latency under production load, and recurring regressions. Those are **not included** merely because the default score table is green.
