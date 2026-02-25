# rag-evaluation-starter

Most RAG systems look fine in demos. This tells you where yours breaks — before your users do.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/infrixo-systems/rag-evaluation-starter/blob/main/notebooks/rag_eval_walkthrough.ipynb)
[![Last Commit](https://img.shields.io/github/last-commit/infrixo-systems/rag-evaluation-starter)](https://github.com/infrixo-systems/rag-evaluation-starter/commits/main)

---

## What you get

Five metrics in a single run. No SaaS. No signup. No mandatory API key.

**Retrieval Recall@K** — Did the right document chunk actually come back in the top results? This is the most common failure point, and the one teams check last.

**Answer Faithfulness** — Is the answer grounded in what was retrieved, or is the model making things up? Measured by token overlap — no LLM judge required.

**Answer Relevance** — Does the generated answer actually address the question? A system can be grounded and still off-topic. Measured by cosine similarity between question and answer embeddings.

**Exact Match / Token F1** — How closely does the answer match the reference? Useful for factual questions where the answer is a specific value: a price, a date, a policy rule.

**Latency + Cost** — How long does each query take? How much will the LLM judge cost? Catches performance regressions before they reach production.

---

## Quickstart

```bash
git clone https://github.com/infrixo-systems/rag-evaluation-starter.git
cd rag-evaluation-starter
pip install -r requirements.txt
python rag_eval.py --golden examples/golden_set.json --no-embeddings
```

You'll see a colour-coded table in your terminal within seconds. No configuration needed.

To include the cosine-similarity relevance metric (requires ~400 MB model download on first run):

```bash
python rag_eval.py --golden examples/golden_set.json
```

---

## How to plug in your own system

Supply two functions. That's it.

```python
from rag_eval import load_golden_set, evaluate, summarise, print_rich_table

# 1. Your retriever — returns [{text, source_id}, ...]
def my_retriever(question: str) -> list[dict]:
    results = my_vectorstore.similarity_search(question, k=3)
    return [{"text": r.page_content, "source_id": r.metadata["source"]} for r in results]

# 2. Your generator — returns an answer string
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

# 3. Run
golden  = load_golden_set("examples/golden_set.json")
results = evaluate(golden, my_retriever, my_generator, k=3)
summary = summarise(results)
print_rich_table(results, summary)
```

The `embedder_fn` parameter is optional — it defaults to `sentence-transformers/all-MiniLM-L6-v2`. Pass your own if you want consistency with your production embeddings.

---

## How to build your golden set

This is the section most teams skip — and why most RAG evals are useless.

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

| Score | Signal | Action |
|---|---|---|
| 0.8 – 1.0 | Retrieval is working | Focus on generation quality |
| 0.5 – 0.8 | Some queries miss the right chunk | Adjust chunk size or embedding model |
| < 0.5 | Retrieval is broken for many queries | Fix retrieval before touching generation |

A Recall@3 of 0.6 means 40% of questions didn't surface the right source. No prompt engineering will fix that — your generator is being asked to answer from wrong context.

### Faithfulness vs Relevance

Low faithfulness + high recall → the model is hallucinating even when given the right context. Tighten the system prompt.

Low relevance + high faithfulness → the model is accurately summarising the wrong thing. Your retrieval is returning off-topic chunks.

### When to WARN vs FAIL

The thresholds at the top of `rag_eval.py` are starting points, not ground truth. For a customer-facing support bot you might set FAIL at 0.7 for faithfulness. For an internal search tool you might tolerate 0.4. Edit the `THRESHOLDS` dict to match your requirements.

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

The judge scores faithfulness and relevance on a 1–5 scale with a one-sentence reason per score. Results are clearly labelled `[LLM-judged]` in the output. A cost estimate is printed at the end of each run.

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
  --golden PATH         Path to golden_set.json

Options:
  --k INT               Top-K for retrieval recall (default: 3)
  --output PATH         Output path prefix (default: results)
  --csv                 Also export results as CSV
  --llm-judge           Enable LLM-as-judge scoring
  --judge-model MODEL   Model for LLM judge (default: gpt-4o-mini)
  --no-embeddings       Skip cosine-similarity relevance metric
  --no-json             Skip saving results.json
```

---

## Requirements

```
sentence-transformers>=2.0
tiktoken>=0.5
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

*Built by [Infrixo Systems](https://infrixo.com) · [hello@infrixo.com](mailto:hello@infrixo.com)*
