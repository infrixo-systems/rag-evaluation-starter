# Getting started with the RAG Evaluation Starter

Run the bundled fictional billing-API example before connecting your own retriever or generator. The first run uses a **mock** pipeline, does **not** call an LLM, and needs **no API key**.

## 1. Set up Python

Use Python 3.9 or newer. A project virtual environment keeps dependencies separate from other projects.

**macOS / Linux**

```bash
git clone https://github.com/infrixo-systems/rag-evaluation-starter.git
cd rag-evaluation-starter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Windows PowerShell**

```powershell
git clone https://github.com/infrixo-systems/rag-evaluation-starter.git
cd rag-evaluation-starter
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

> The dependency installation can take some time, particularly on a fresh machine. `--no-embeddings` skips loading an embedding model *during evaluation*; it does not remove packages from `requirements.txt`.

## 2. Run the sample without embeddings

```bash
mkdir -p results
python rag_eval.py --golden examples/golden_set.json --no-embeddings --csv --output results/quickstart
```

The CLI uses `examples/mock_retriever.py` to retrieve sample contexts and generate sample answers. It evaluates **eight** questions and writes:

- `results/quickstart.json` — full per-question results plus an aggregate summary
- `results/quickstart.csv` — a flat, spreadsheet-friendly results export

There are two distinct kinds of `SKIP` in this run:

- **`q007` / Retrieval Recall@K: `N/A SKIP`.** This is an intentionally unanswerable question; no ground-truth source is expected. The evaluator does *not* give it a perfect recall score or include it in the recall mean.
- **Answer Relevance: `SKIP`.** You requested `--no-embeddings`, so the question–answer embedding similarity is not computed. A displayed `0.00 SKIP` must not be interpreted as an evaluated relevance score of zero.

The current example golden set contains seven answerable questions, so the retrieval-recall aggregate uses **seven**, not eight, observations. See [METRICS.md](METRICS.md) for the meaning and limitations of each metric.

## 3. Inspect the machine-readable result

```bash
python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("results/quickstart.json").read_text(encoding="utf-8"))
summary = report["summary"]
row = next(r for r in report["results"] if r["id"] == "q007")

print("Questions:", summary["n_questions"])
print("Recall observations:", summary["metric_counts"]["retrieval_recall_at_k"])
print("Mean Recall@K:", summary["mean_scores"].get("retrieval_recall_at_k"))
print("q007 recall:", row["metrics"]["retrieval_recall_at_k"])

assert summary["n_questions"] == 8
assert summary["metric_counts"]["retrieval_recall_at_k"] == 7
assert row["metrics"]["retrieval_recall_at_k"]["score"] is None
assert row["metrics"]["retrieval_recall_at_k"]["verdict"] == "SKIP"
PY
```

If all assertions pass, you have confirmed the example handles an intentionally unanswerable question correctly. Those counts are properties of the **bundled example**, not fixed requirements for your own golden set.

## 4. Run regression tests

```bash
python -m unittest discover -s tests -v
```

No LLM credentials are needed for the P0 regression suite.

## 5. Evaluate your own RAG system

See [How to plug in your own system](README.md#how-to-plug-in-your-own-system) for the two Python functions you provide:

- **Retriever:** accepts a question and returns chunks with `text` and `source_id` fields.
- **Generator:** accepts the question plus retrieved contexts and returns an answer string.

Create a golden-set JSON array based on `examples/golden_set.json`. Each entry needs a question, a reference answer, and expected source IDs. Intentionally unanswerable questions must set **`"answerable": false`** and **`"expected_source_ids": []`**. An answerable entry must have at least one expected source ID. The validator rejects ambiguous empty-source entries instead of silently assigning perfect recall.

## 6. Optional: run answer relevance with embeddings

```bash
python rag_eval.py --golden examples/golden_set.json --csv --output results/with-embeddings
```

This computes question–answer embedding similarity in addition to the metrics available in the first run. The first model use may download additional model files. **Embedding relevance is semantic similarity, not proof that the answer is correct.**

## Troubleshooting

| Symptom | What to check |
|---|---|
| `ModuleNotFoundError` (for example `numpy`) | Activate `.venv` and run `python -m pip install -r requirements.txt` in that environment. |
| An answerable case with `expected_source_ids: []` is rejected | Supply the expected sources, or explicitly set `"answerable": false` if the answer is absent from the knowledge base. |
| Relevance says `SKIP` | Remove `--no-embeddings` if you want that metric computed. |
| A JSON/CSV output path cannot be written | Ensure the parent directory exists (`mkdir -p results` on macOS/Linux). |
| Scores look surprisingly good | Remember the sample uses a mock pipeline and a small, fictional dataset. Evaluate your own pipeline on a representative golden set. |

**What this tool is not:** It is a diagnostic starter, not a production-grade fact-checker, claim-level groundedness evaluator, or independently scored abstention benchmark. Treat the scores as signals for investigation rather than correctness guarantees.
