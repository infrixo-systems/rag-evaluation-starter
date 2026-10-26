"""P2-C usability regression tests (F-05, F-07, F-09, F-10, F-11, F-12, F-13).

Run with: python -m unittest discover -s tests -v
No test needs an API key or network access.
"""
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import rag_eval
from examples.mock_retriever import mock_generator_fn, mock_retriever_fn
from rag_eval import THRESHOLDS, evaluate, load_golden_set, save_csv, save_json, summarise

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "examples" / "golden_set.json"
SAMPLE = REPO / "examples" / "sample_output.json"
README = (REPO / "README.md").read_text(encoding="utf-8")
KEY_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
# Temp files stay inside the repo, never the system temp dir; removed after the run.
TEST_TMP = REPO / ".test-tmp"

# CSV columns before P2-C, in order. These must never move or change.
EXISTING_COLUMNS = [
    "id", "category", "difficulty", "question",
    "retrieval_recall_at_k", "answer_faithfulness", "answer_relevance",
    "exact_match", "token_f1", "token_precision", "token_recall",
    "retrieval_latency_s", "generation_latency_s", "total_latency_s",
    "generated_answer", "answer_relevance_method",
]
JUDGE_COLUMNS = ["llm_judge_status", "judge_estimated_cost_usd"]
NEW_COLUMNS = [
    "retrieval_recall_at_k_verdict", "retrieval_k",
    "answer_faithfulness_verdict", "answer_relevance_verdict",
    "exact_match_verdict", "token_f1_verdict",
]
METRICS = ["retrieval_recall_at_k", "answer_faithfulness", "answer_relevance",
           "exact_match", "token_f1"]


def _tempdir():
    TEST_TMP.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=TEST_TMP)


def tearDownModule():
    if TEST_TMP.is_dir() and not any(TEST_TMP.iterdir()):
        TEST_TMP.rmdir()


def _entries(n):
    return [{"id": f"q{i}", "question": f"question {i}", "expected_answer": "alpha beta",
             "expected_source_ids": ["s1"]} for i in range(n)]


def _retriever(question):
    return [{"text": "alpha beta gamma", "source_id": "s1"}]


def _generator(question, context):
    return "alpha beta"


def _csv(results):
    """save_csv() the results; return (header, rows) as read back from disk."""
    with _tempdir() as temp, redirect_stdout(StringIO()):
        path = Path(temp) / "out.csv"
        save_csv(results, str(path))
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return reader.fieldnames, list(reader)


def _section(text, heading):
    """Body of the markdown `## heading` section, up to the next `## ` heading."""
    start = text.index(f"\n## {heading}\n")
    end = text.find("\n## ", start + 1)
    return text[start:end if end != -1 else None]


def _fenced(text, lang):
    """All ```lang fenced blocks in `text`, in order."""
    return re.findall(rf"^```{lang}\n(.*?)^```$", text, re.M | re.S)


def _offline_env(temp: Path, pythonpath=None):
    env = {k: v for k, v in os.environ.items() if k not in KEY_VARS + ("PYTHONPATH",)}
    env.update(
        TIKTOKEN_CACHE_DIR=str(temp / "tiktoken-cache"),
        HF_HOME=str(temp / "hf-home"),
        XDG_CACHE_HOME=str(temp / "xdg-cache"),
        TMPDIR=str(temp),
        HF_HUB_OFFLINE="1",
        HTTPS_PROXY="http://127.0.0.1:9",
        HTTP_PROXY="http://127.0.0.1:9",
        PYTHONDONTWRITEBYTECODE="1",
        COLUMNS="200",
    )
    if pythonpath is not None:
        env["PYTHONPATH"] = pythonpath
    return env


class TestF05CsvVerdicts(unittest.TestCase):
    def _assert_row_matches_json(self, row, result):
        for mk in METRICS:
            metric = result["metrics"][mk]
            self.assertEqual(row[f"{mk}_verdict"], metric["verdict"], (result["id"], mk))
            if metric["score"] is None:
                self.assertEqual(row[mk], "", (result["id"], mk))
            else:
                self.assertEqual(float(row[mk]), metric["score"], (result["id"], mk))
        self.assertEqual(int(row["retrieval_k"]), result["metrics"]["retrieval_recall_at_k"]["k"])
        self.assertEqual(row["answer_relevance_method"], result["metrics"]["answer_relevance"]["method"])

    def test_header_appends_new_columns_without_moving_existing_ones(self):
        header, _ = _csv(evaluate(_entries(1), _retriever, _generator, no_embeddings=True))
        self.assertEqual(header, EXISTING_COLUMNS + NEW_COLUMNS)

    def test_judge_header_keeps_judge_columns_before_new_ones(self):
        judged = {"faithfulness": {"score": 4, "reason": "r"}, "relevance": {"score": 5, "reason": "r"}}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-not-real"}), \
                patch.object(rag_eval, "llm_judge", return_value=judged):
            results = evaluate(_entries(1), _retriever, _generator, no_embeddings=True,
                               use_llm_judge=True)
        header, rows = _csv(results)
        self.assertEqual(header, EXISTING_COLUMNS + JUDGE_COLUMNS + NEW_COLUMNS)
        self._assert_row_matches_json(rows[0], results[0])

    def test_ordinary_scores_keep_numbers_and_record_verdicts_and_k(self):
        results = evaluate(load_golden_set(str(GOLDEN)), mock_retriever_fn, mock_generator_fn,
                           k=5, no_embeddings=True)
        _, rows = _csv(results)
        self.assertEqual([r["id"] for r in rows], [r["id"] for r in results])
        for row, result in zip(rows, results):
            self._assert_row_matches_json(row, result)
            self.assertEqual(row["retrieval_k"], "5")
        seen = {row[f"{mk}_verdict"] for row in rows for mk in METRICS}
        self.assertTrue({"PASS", "WARN", "FAIL", "SKIP"} <= seen, seen)

    def test_computed_relevance_has_numeric_score_and_verdict(self):
        embedder = Mock(return_value=np.array([[1.0, 0.0], [1.0, 0.0]]))
        results = evaluate(_entries(1), _retriever, _generator, embedder_fn=embedder)
        _, rows = _csv(results)
        self.assertEqual(float(rows[0]["answer_relevance"]), 1.0)
        self.assertEqual(rows[0]["answer_relevance_verdict"], "PASS")
        self.assertEqual(rows[0]["answer_relevance_method"], "cosine_similarity")

    def test_intentional_skip_for_unanswerable_q007_and_no_embeddings(self):
        results = evaluate(load_golden_set(str(GOLDEN)), mock_retriever_fn, mock_generator_fn,
                           no_embeddings=True)
        _, rows = _csv(results)
        q007 = next(r for r in rows if r["id"] == "q007")
        self.assertEqual(q007["retrieval_recall_at_k"], "")
        self.assertEqual(q007["retrieval_recall_at_k_verdict"], "SKIP")
        self.assertEqual(q007["retrieval_k"], "3")
        self.assertEqual(q007["difficulty"], "adversarial")
        # Its answer is still scored under the normal rules.
        self.assertEqual(q007["answer_faithfulness_verdict"], "WARN")
        self.assertEqual(q007["exact_match_verdict"], "FAIL")
        self.assertEqual(q007["token_f1_verdict"], "PASS")
        for row in rows:
            self.assertEqual((row["answer_relevance"], row["answer_relevance_verdict"],
                              row["answer_relevance_method"]), ("", "SKIP", "skipped"))
        answerable = [r for r in rows if r["id"] != "q007"]
        self.assertTrue(all(r["retrieval_recall_at_k_verdict"] != "SKIP" for r in answerable))
        self.assertTrue(all(r["retrieval_recall_at_k"] != "" for r in answerable))

    def test_relevance_error_is_error_not_skip(self):
        with redirect_stderr(StringIO()):
            results = evaluate(_entries(2), _retriever, _generator,
                               embedder_fn=Mock(side_effect=OSError("model unavailable")))
        _, rows = _csv(results)
        for row in rows:
            self.assertEqual((row["answer_relevance"], row["answer_relevance_verdict"],
                              row["answer_relevance_method"]), ("", "ERROR", "error"))
            self.assertEqual(row["token_f1_verdict"], "PASS")   # other metrics unaffected

    def test_partial_relevance_mixes_computed_and_error_rows(self):
        embedder = Mock(side_effect=[np.array([[1.0, 0.0], [1.0, 0.0]]), OSError("model gone")])
        with redirect_stderr(StringIO()):
            results = evaluate(_entries(3), _retriever, _generator, embedder_fn=embedder)
        self.assertEqual(summarise(results)["incomplete"]["answer_relevance"],
                         {"status": "partial", "computed": 1, "error": 2})
        _, rows = _csv(results)
        self.assertEqual([r["answer_relevance_verdict"] for r in rows], ["PASS", "ERROR", "ERROR"])
        self.assertEqual([r["answer_relevance_method"] for r in rows],
                         ["cosine_similarity", "error", "error"])
        self.assertEqual(float(rows[0]["answer_relevance"]), 1.0)
        self.assertEqual([r["answer_relevance"] for r in rows[1:]], ["", ""])
        for row, result in zip(rows, results):
            self._assert_row_matches_json(row, result)
        self.assertNotIn("PARTIAL", {row[c] for row in rows for c in NEW_COLUMNS})


class TestF07ReadmeIntegrationExample(unittest.TestCase):
    def test_readme_example_runs_from_outside_the_repo_via_pythonpath(self):
        section = _section(README, "How to plug in your own system")
        golden_json = _fenced(section, "json")[0]
        script = _fenced(section, "python")[0]
        self.assertIn("PYTHONPATH=/path/to/rag-evaluation-starter python my_eval.py", section)

        with _tempdir() as temp:
            temp = Path(temp)
            work = temp / "my-project"
            work.mkdir()
            (work / "my_golden_set.json").write_text(golden_json, encoding="utf-8")
            (work / "my_eval.py").write_text(script, encoding="utf-8")

            # Without PYTHONPATH, rag_eval is not importable: it is not a package.
            proc = subprocess.run([sys.executable, "my_eval.py"], cwd=work, env=_offline_env(temp),
                                  capture_output=True, text=True, timeout=120)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("No module named 'rag_eval'", proc.stderr)

            proc = subprocess.run([sys.executable, "my_eval.py"], cwd=work,
                                  env=_offline_env(temp, pythonpath=str(REPO)),
                                  capture_output=True, text=True, timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            report = json.loads((work / "results" / "my_eval.json").read_text(encoding="utf-8"))
            self.assertTrue((work / "results" / "my_eval.csv").is_file())

        rows = {r["id"]: r for r in report["results"]}
        self.assertEqual(report["summary"]["n_questions"], 2)
        self.assertEqual(rows["s1"]["metrics"]["retrieval_recall_at_k"]["score"], 1.0)
        self.assertEqual(rows["s2"]["metrics"]["retrieval_recall_at_k"],
                         {"score": None, "k": 3, "verdict": "SKIP"})
        self.assertEqual(rows["s2"]["generated_answer"], "I don't know.")
        self.assertEqual(report["summary"]["incomplete"], {})

    def test_readme_documents_callable_contracts(self):
        section = _section(README, "How to plug in your own system")
        for text in ("not an installable Python package",
                     "retriever_fn(question: str) -> list[dict]",
                     "generator_fn(question: str, context: list[str]) -> str",
                     "embedder_fn(texts: list[str]) -> numpy.ndarray",
                     "Recall@K counts only the first `k` chunks"):
            self.assertIn(text, section)


class TestF10TiktokenMinimum(unittest.TestCase):
    MINIMUM = (0, 7)   # first release mapping gpt-4o / gpt-4o-* to o200k_base

    def test_requirement_is_not_below_supported_minimum(self):
        reqs = (REPO / "requirements.txt").read_text(encoding="utf-8").split()
        spec = [r for r in reqs if r.startswith("tiktoken")]
        self.assertEqual(len(spec), 1, reqs)
        match = re.fullmatch(r"tiktoken>=(\d+)\.(\d+)(?:\.\d+)?", spec[0])
        self.assertIsNotNone(match, spec[0])
        self.assertGreaterEqual(tuple(map(int, match.groups())), self.MINIMUM)

    def test_readme_requirements_match_requirements_txt(self):
        block = _fenced(_section(README, "Requirements"), "")[0]
        self.assertEqual(block.split(), (REPO / "requirements.txt").read_text(encoding="utf-8").split())

    @unittest.skipUnless(importlib.util.find_spec("tiktoken"), "tiktoken not installed")
    def test_installed_tiktoken_maps_judge_model_without_loading_a_tokenizer(self):
        import tiktoken.model
        # Metadata-only lookup; fail loudly if anything tries to load (download) an encoding.
        with patch.object(tiktoken.model, "get_encoding",
                          side_effect=AssertionError("tokenizer load attempted")):
            self.assertEqual(tiktoken.model.encoding_name_for_model("gpt-4o-mini"), "o200k_base")


class TestF11Notebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = json.loads((REPO / "notebook" / "rag_eval_walkthrough.ipynb").read_text(encoding="utf-8"))
        cls.cells = {c.get("id"): "".join(c["source"]) for c in cls.nb["cells"]}

    def test_mean_recall_is_explained_as_average_of_fractions(self):
        text = self.cells["interpret-header"]
        self.assertNotIn("40% of questions didn't find", text)
        self.assertIn("average of the per-question recall fractions", text)

    def test_field_reference_documents_answerable(self):
        text = self.cells["golden-set-schema"]
        self.assertIn("| `answerable` | Optional |", text)
        self.assertIn("expected_source_ids: []", text)
        self.assertIn("SKIP", text)

    def test_recall_bands_match_thresholds(self):
        t = THRESHOLDS["retrieval_recall"]
        text = self.cells["interpret-header"]
        self.assertIn(f"| ≥ {t['warn']} | PASS |", text)
        self.assertIn(f"| {t['fail']} – < {t['warn']} | WARN |", text)
        self.assertIn(f"| < {t['fail']} | FAIL |", text)

    def test_code_cells_have_no_outputs(self):
        for cell in self.nb["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [], cell.get("id"))
                self.assertIsNone(cell["execution_count"], cell.get("id"))


class TestF12Contributing(unittest.TestCase):
    def test_bug_report_link_and_csv_checklist(self):
        text = (REPO / "CONTRIBUTING.md").read_text(encoding="utf-8")
        self.assertNotIn("ISSUE_TEMPLATE", text)
        self.assertIn("https://github.com/infrixo-systems/rag-evaluation-starter/issues/new", text)
        self.assertIn("`save_csv()`", text)


@unittest.skipUnless(rag_eval._rich_available, "rich not installed")
class TestF13Presentation(unittest.TestCase):
    def test_difficulty_column_is_not_truncated(self):
        from rich.console import Console
        results = evaluate(load_golden_set(str(GOLDEN)), mock_retriever_fn, mock_generator_fn,
                           no_embeddings=True)
        out = StringIO()
        with patch.object(rag_eval, "console", Console(file=out, width=120, color_system=None)):
            rag_eval.print_rich_table(results, summarise(results))
        lines = out.getvalue().splitlines()
        self.assertIn("adversarial", next(l for l in lines if "q007" in l))
        self.assertIn("medium", next(l for l in lines if "q003" in l))


class TestF13F09ReadmeConsistency(unittest.TestCase):
    LABELS = {
        "retrieval_recall": "Retrieval Recall@K", "answer_faithfulness": "Answer Faithfulness",
        "answer_relevance": "Answer Relevance", "exact_match": "Exact Match", "token_f1": "Token F1",
    }

    def test_threshold_tables_match_thresholds(self):
        section = _section(README, "Interpreting results")
        self.assertNotIn("0.8 – 1.0", section)
        t = THRESHOLDS["retrieval_recall"]
        self.assertIn(f"| ≥ {t['warn']} | PASS |", section)
        self.assertIn(f"| {t['fail']} – < {t['warn']} | WARN |", section)
        self.assertIn(f"| < {t['fail']} | FAIL |", section)
        self.assertEqual(set(self.LABELS), set(THRESHOLDS))
        for key, label in self.LABELS.items():
            w, f = THRESHOLDS[key]["warn"], THRESHOLDS[key]["fail"]
            self.assertIn(f"| {label} | ≥ {w} | {f} – < {w} | < {f} |", section)

    def test_cli_reference_matches_parser_options(self):
        block = _fenced(_section(README, "CLI reference"), "")[0]
        documented = set(re.findall(r"^\s+(--[a-z-]+)", block, re.M))
        actual = {o for a in rag_eval._build_parser()._actions for o in a.option_strings
                  if o.startswith("--") and o != "--help"}
        self.assertEqual(documented, actual)

    def test_metric_overview_separates_scored_metrics_from_judge_cost(self):
        section = _section(README, "What you get")
        self.assertIn("Five evaluation metrics per question, with PASS, WARN, FAIL, SKIP or ERROR "
                      "verdicts as applicable, plus latency.", section)
        self.assertNotIn("Five scored metrics", section)
        self.assertNotIn("Latency + Cost", section)
        for name in ("Retrieval Recall@K", "Answer Faithfulness", "Answer Relevance",
                     "Exact Match", "Token F1"):
            self.assertIn(f"**{name}**", section)

    def test_q007_explanation_matches_sample_output(self):
        sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
        m = next(r for r in sample["results"] if r["id"] == "q007")["metrics"]
        text = " ".join(README.split())
        self.assertEqual(m["retrieval_recall_at_k"]["verdict"], "SKIP")
        self.assertIn("`q007` is such a case. Its recall is `N/A SKIP` by design", text)
        for label, key in (("faithfulness", "answer_faithfulness"), ("exact match", "exact_match"),
                           ("token F1", "token_f1")):
            claim = f"{label} `{m[key]['verdict']}` ({m[key]['score']:.2f}"
            self.assertIn(claim, text)

    def test_no_llm_judged_label_claim(self):
        self.assertNotIn("[LLM-judged]", README)
        self.assertNotIn("LLM-judged", (REPO / "rag_eval.py").read_text(encoding="utf-8"))
        self.assertIn("each row's `llm_judge` field", _section(README, "Optional: LLM-as-judge"))


if __name__ == "__main__":
    unittest.main()
