# rag-evaluation-starter

Most RAG systems look fine in demos. This tells you where yours breaks — before your users do.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/infrixo-systems/rag-evaluation-starter/blob/main/notebook/rag_eval_walkthrough.ipynb)
[![Last Commit](https://img.shields.io/github/last-commit/infrixo-systems/rag-evaluation-starter)](https://github.com/infrixo-systems/rag-evaluation-starter/commits/main)

---

## What you get

Five evaluation metrics per question, with PASS, WARN, FAIL, SKIP or ERROR verdicts as applicable, plus latency. No SaaS, signup or mandatory API key.

**Retrieval Recall@K** — Did the right document chunk actually come back in the top results? This is the most common failure point, and the one teams check last.

**Answer Faithfulness** — Is the answer grounded in what was retrieved, or is the model making things up? Measured by token overlap — no LLM judge required.

**Answer Relevance** — Does the generated answer actually address the question? A system can be grounded and still off-topic. Measured by cosine similarity between question and answer embeddings from a local model (downloaded on first use); `--no-embeddings` skips it.

**Exact Match** and **Token F1** — How closely does the answer match the reference? Useful for factual questions where the answer is a specific value: a price, a date, a policy rule.

**Latency** — How long do retrieval and generation take for each query? Catches performance regressions before they reach production. Latency is recorded, not scored.

The only part that costs money is the [optional LLM judge](#optional-llm-as-judge), which calls a paid API and prints its estimated cost.

---

## Quickstart

New here? Follow the [step-by-step getting started guide](GETTING_STARTED.md). For score interpretation and limitations, see the [metrics guide](METRICS.md).


macOS / Linux (Windows: see [GETTING_STARTED.md](GETTING_STARTED.md)):

```bash
git clone https://github.com/infrixo-systems/rag-evaluation-starter.git
cd rag-evaluation-starter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python rag_eval.py --golden examples/golden_set.json --no-embeddings
```

This runs the bundled demo (a mock retriever and generator over a fictional billing API) with no API key. You'll see a colour-coded table in your terminal, and the results are written to `results/results.json`. [`examples/sample_output.json`](examples/sample_output.json) shows the expected JSON.

The CLI only runs this bundled demo, and it rejects any other `--golden` file. To evaluate your own system and golden set, use the Python API ([How to plug in your own system](#how-to-plug-in-your-own-system)).

To include the cosine-similarity relevance metric (requires ~400 MB model download on first run):

```bash
python rag_eval.py --golden examples/golden_set.json
```

---

## How to plug in your own system

Supply two functions. That's it.

`rag_eval.py` is a standalone script, not an installable Python package. Install its dependencies using `requirements.txt`, then make `rag_eval.py` importable through `PYTHONPATH` or by running your script from the repository root.

This complete example runs as-is, with no API key and no model download. Save the golden set as `my_golden_set.json`:

```json
[
  {
    "id": "s1",
    "question": "How long does standard shipping take?",
    "expected_answer": "Standard shipping takes 3-5 business days.",
    "expected_source_ids": ["shipping-policy"],
    "category": "shipping",
    "difficulty": "easy"
  },
  {
    "id": "s2",
    "question": "Do you ship to Antarctica?",
    "expected_answer": "I don't know.",
    "expected_source_ids": [],
    "answerable": false,
    "category": "shipping",
    "difficulty": "adversarial"
  }
]
```

Save the script next to it as `my_eval.py`:

```python
from rag_eval import evaluate, load_golden_set, print_rich_table, save_csv, save_json, summarise

# Stand-ins for your knowledge base, retriever and LLM. Replace them with your own.
DOCS = {
    "shipping-policy": "Standard shipping takes 3-5 business days.",
    "returns-policy": "Items can be returned within 30 days of delivery.",
}


def my_retriever(question: str) -> list[dict]:
    """Return retrieved chunks, best match first, as {"text": str, "source_id": str}."""
    words = set(question.lower().split())
    ranked = sorted(DOCS.items(), key=lambda kv: -len(words & set(kv[1].lower().split())))
    return [{"text": text, "source_id": source_id} for source_id, text in ranked]


def my_generator(question: str, context: list[str]) -> str:
    """Return the answer to `question`, given the retrieved chunk texts."""
    words = set(question.lower().split())
    if context and words & set(context[0].lower().split()):
        return context[0]
    return "I don't know."


golden = load_golden_set("my_golden_set.json")   # validates the file
results = evaluate(golden, my_retriever, my_generator, k=3, no_embeddings=True)
summary = summarise(results)
print_rich_table(results, summary)
save_json(results, summary, "results/my_eval.json")
save_csv(results, "results/my_eval.csv")
```

Run it from the directory containing both files:

```bash
# macOS / Linux
PYTHONPATH=/path/to/rag-evaluation-starter python my_eval.py

# Windows PowerShell
$env:PYTHONPATH = "C:\path\to\rag-evaluation-starter"; python my_eval.py
```

**Function contracts** (what `evaluate()` expects):

- **Retriever** — `retriever_fn(question: str) -> list[dict]`. Called once per question. Each chunk must have `"text"` and `"source_id"` keys. Order matters: Recall@K counts only the first `k` chunks, but every returned chunk is passed to the generator and used for faithfulness. The retriever is not told `k`, so return at least `k` chunks.
- **Generator** — `generator_fn(question: str, context: list[str]) -> str`. `context` is the `text` of every retrieved chunk, in retrieval order.
- **Errors** — an exception raised by your retriever or generator is not caught; it stops the evaluation.
- **Embedder (optional)** — `embedder_fn(texts: list[str]) -> numpy.ndarray`, one vector per text. Vectors must be L2-normalised, because relevance is their dot product. The default is `sentence-transformers/all-MiniLM-L6-v2`, which downloads on first use; pass `no_embeddings=True` to skip relevance, or your own function for consistency with your production embeddings.

With a real stack, the two functions might look like this (illustrative; needs your own vector store and client):

```python
def my_retriever(question: str) -> list[dict]:
    results = my_vectorstore.similarity_search(question, k=3)
    return [{"text": r.page_content, "source_id": r.metadata["source"]} for r in results]


def my_generator(question: str, context: list[str]) -> str:
    ctx = "\n\n".join(context)
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Answer using only this context:\n{ctx}"},
            {"role": "user",   "content": question},
        ],
    )
    return response.choices[0].message.content
```

---

## How to build your golden set


This is the section most teams skip — and why most RAG evals are useless.

**Unanswerable cases:** For a question whose answer is absent from the knowledge base,
set `"answerable": false` and `"expected_source_ids": []`. Other questions default
to `answerable: true` and must list at least one expected source ID.

Retrieval Recall@K is **N/A / SKIP** for unanswerable questions and excluded from
the retrieval-recall mean; they remain included in other evaluation metrics.
The output JSON summary includes `metric_counts` to show each metric's denominator.
Abstention behavior is not independently graded by this change.

In the bundled sample, `q007` is such a case. Its recall is `N/A SKIP` by design, but its
abstention answer is still scored by the normal rules against its reference answer:
faithfulness `WARN` (0.38, as the answer uses words that are not in the retrieved context),
exact match `FAIL` (0.00) and token F1 `PASS` (0.58). These verdicts do not say whether
abstaining was the right behavior.

**Breaking change:** Existing golden sets containing an empty `expected_source_ids` list
for an answerable question now fail validation. For intentionally unanswerable cases,
set `"answerable": false`; otherwise, provide at least one expected source ID.

**Start with your support tickets.** The questions users ask when things go wrong are exactly the queries your system needs to handle. Look at the last 3 months. Pull out 30–50 questions that were hard to answer or answered incorrectly.

**Group by type.** A good golden set has a mix:

- **Easy (factual)** — single-chunk lookup, unambiguous phrasing. Establishes a baseline.
- **Medium (multi-chunk)** — the answer requires combining information from two sources. Tests whether retrieval and synthesis both work.
- **Hard (reasoning)** — ambiguous question, answer requires judgment over context. Reveals where the model starts guessing.
- **Adversarial** — the answer is *not* in the knowledge base. Tests whether the system says "I don't know" or fabricates an answer. Include at least one.
- **Edge cases** — questions at the boundary of what your system knows. These are the first ones to break when you update your docs.

**Aim for 30–50 entries** before you trust the scores. Eight is enough to get started; fewer than that and a single bad answer swings your metrics by 12 percentage points.

**Keep expected answers short and specific.** "The rate limit is 100 requests per minute" is a better expected answer than a paragraph — Token F1 scoring rewards precision.

**Update the golden set when your docs change.** A stale golden set gives you false confidence. Add new entries every time you add a major feature to your product.

The `examples/golden_set.json` file in this repo uses a fictional billing API as the domain. Use it as a template for your own entries.

---

## Interpreting results

### Retrieval Recall@K

| Score | Verdict | Signal | Action |
|---|---|---|---|
| ≥ 0.5 | PASS | Most expected sources come back in the top K | Focus on generation quality |
| 0.3 – < 0.5 | WARN | Expected sources are often missed | Adjust chunk size or embedding model |
| < 0.3 | FAIL | Retrieval misses most expected sources | Fix retrieval before touching generation |

These are the default `THRESHOLDS`. The same cut-offs grade each question and the mean row.

A mean Recall@3 of 0.6 means the average per-question retrieval recall is 60%, not necessarily that 40% of questions failed retrieval. See [METRICS.md](METRICS.md) for interpretation.

### Faithfulness vs Relevance

Low faithfulness + high recall → the model is hallucinating even when given the right context. Tighten the system prompt.

Low relevance + high faithfulness → the model is accurately summarising the wrong thing. Your retrieval is returning off-topic chunks.

### When to WARN vs FAIL

The thresholds at the top of `rag_eval.py` are starting points, not ground truth. For a customer-facing support bot you might set FAIL at 0.7 for faithfulness. For an internal search tool you might tolerate 0.4. Edit the `THRESHOLDS` dict to match your requirements.

The defaults, applied to each question's score and to each mean:

| Metric | PASS | WARN | FAIL |
|---|---|---|---|
| Retrieval Recall@K | ≥ 0.5 | 0.3 – < 0.5 | < 0.3 |
| Answer Faithfulness | ≥ 0.4 | 0.2 – < 0.4 | < 0.2 |
| Answer Relevance | ≥ 0.6 | 0.4 – < 0.6 | < 0.4 |
| Exact Match | ≥ 0.4 | 0.2 – < 0.4 | < 0.2 |
| Token F1 | ≥ 0.4 | 0.2 – < 0.4 | < 0.2 |

Exact Match is always 0 or 1 per question, so a question either passes or fails it; only the mean can be WARN.

If your scores are consistently low and you're not sure which lever to pull, this is the kind of diagnostic work we cover in a [Foundation Check](https://infrixo.com/start?utm_source=github&utm_medium=readme&utm_campaign=rag-eval).

---

## What to do when scores are bad

**Retrieval Recall is low:**
Chunk your documents smaller (try 256 tokens with 64-token overlap). Switch to a retrieval-focused embedding model like `BAAI/bge-large-en-v1.5`. Add a CrossEncoder reranker on top of your ANN results. Read more: [Why RAG Systems Fail in Production](https://infrixo.com/insights/why-rag-systems-fail?utm_source=github&utm_medium=readme&utm_campaign=rag-eval).

**Faithfulness is low:**
Strengthen the system prompt with explicit grounding instructions. Reduce `max_tokens` to limit padding. Consider filtering retrieved chunks to a minimum relevance threshold before passing them to the generator.

**Relevance is low:**
Your retriever is returning chunks that are topically related but not directly answering the question. Add a reranker or try hybrid search (keyword + dense vector).

**Token F1 is low on easy questions:**
Check whether your expected answers are phrased differently from how your documents are written. Update `expected_answer` fields to match what a well-behaved system would actually produce.

---

## Optional: LLM-as-judge

The default metrics require no API key. For more nuanced faithfulness and relevance scoring, enable LLM-as-judge:

```bash
# Using OpenAI
OPENAI_API_KEY=sk-... python rag_eval.py --golden examples/golden_set.json --llm-judge

# Using Anthropic
ANTHROPIC_API_KEY=sk-ant-... python rag_eval.py --golden examples/golden_set.json --llm-judge --judge-model claude-haiku-4-5-20251001
```

The judge scores faithfulness and relevance on a 1–5 scale with a one-sentence reason per score. Its scores do not replace the default metrics: the terminal table, mean row, and CSV still show the token-overlap and embedding scores. The judge's scores and reasons are stored separately, in each row's `llm_judge` field in the JSON output, next to `llm_judge_status`; the CSV records only `llm_judge_status` and `judge_estimated_cost_usd`. A usage and estimated-cost line is printed at the end of each run. See [METRICS.md](METRICS.md#optional-llm-judge-usage-and-estimated-cost) for details.

---

## Limitations

**LLM judge bias.** GPT-4o-mini and Claude Haiku have positional and length biases. Use judge scores as a signal, not a ground truth. The token-overlap metrics are noisier but unbiased.

**Golden sets go stale.** When you update your documentation, your golden set needs updating too. A test that always passes is not testing anything useful.

**This is diagnostic, not continuous eval.** This tool is designed for point-in-time evaluation against a fixed set. It will tell you whether your current system passes a threshold; it won't tell you when something breaks in production. For continuous monitoring, look at [Ragas](https://docs.ragas.io), [LangSmith](https://smith.langchain.com), or [TruLens](https://www.trulens.org).

**Token F1 penalises valid paraphrasing.** A generated answer of "authentication requires a Bearer token" and an expected answer of "use Bearer token authentication" will score below 1.0 despite being semantically equivalent. This is a known limitation of string-matching metrics.

---

## When you've outgrown this script

This tool intentionally covers the minimum viable eval for a team that has nothing. When you need more:

- **[Ragas](https://docs.ragas.io)** — the most comprehensive open-source RAG metrics framework. Use this when you need LLM-based metrics at scale.
- **[LangSmith](https://smith.langchain.com)** — evaluation, tracing, and dataset management integrated with LangChain.
- **[TruLens](https://www.trulens.org)** — feedback functions and a dashboard for LLM apps.
- **[DeepEval](https://docs.confident-ai.com)** — pytest-style unit tests for LLM outputs.

---

## CLI reference

```
python rag_eval.py --golden PATH [options]

Required:
  --golden PATH         Path to the bundled examples/golden_set.json

Options:
  --k INT               Top-K for retrieval recall (default: 3)
  --output PATH         Output path prefix (default: results/results -> results/results.json)
  --csv                 Also export results as CSV
  --llm-judge           Enable LLM-as-judge scoring
  --judge-model MODEL   Model for LLM judge (default: gpt-4o-mini)
  --no-embeddings       Skip cosine-similarity relevance metric
  --no-json             Skip saving results.json
```

The CLI runs only the bundled demo pipeline (`examples/mock_retriever.py`). Any `--golden` file other than the bundled `examples/golden_set.json`, including an edited copy, is rejected with exit code 2, and no results are written. For your own data, use the [Python API](#how-to-plug-in-your-own-system).

---

## Requirements

```
sentence-transformers>=2.0
tiktoken>=0.7
rich>=13.0
numpy>=1.24
```

Optional (for LLM judge): `openai>=1.0` or `anthropic>=0.20`

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). New metrics and example golden sets are especially welcome.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Going Beyond Evaluation — Production Guardrails

Evaluation tells you where your RAG is broken.
Guardrails stop those breaks from reaching your users.

Once your scores are green, the next layer is a production safety 
pipeline:

| Layer | What it catches |
|---|---|
| Faithfulness gate | LLM answers that distort or ignore retrieved context |
| Input guardrails | Prompt injection via user queries and retrieved documents |
| Output guardrails | PII leaks, toxic responses, off-topic drift |
| Citation attribution | Unverifiable claims with no traceable source |
| Uncertainty abstention | Low-confidence answers that should never reach users |

We built a complete implementation of this stack — claim-level 
LLM-as-judge, Presidio PII redaction, consistency-based abstention, 
and Prometheus monitoring — documented across a technical playbook 
and 7 executed Jupyter notebooks.

**[Get the Production RAG Guardrails Playbook →](https://infrixo.gumroad.com/l/rag-guardrails)**

The Core Playbook is free. The full implementation kit (Labs A–G) 
is $39.

*Built by 
[Infrixo Systems](https://infrixo.com/?utm_source=github&utm_medium=readme&utm_campaign=rag-eval)
 · [hello@infrixo.com](mailto:hello@infrixo.com)*
