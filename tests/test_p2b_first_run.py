"""P2-B first-run usability regression tests (F-03, F-06).

Run with: python -m unittest discover -s tests -v
No test needs an API key or network access.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from examples.mock_retriever import mock_generator_fn, mock_retriever_fn
from rag_eval import evaluate, load_golden_set, save_json, summarise

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "rag_eval.py"
GOLDEN = REPO / "examples" / "golden_set.json"
SAMPLE = REPO / "examples" / "sample_output.json"
KEY_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
# Temp files stay inside the repo, never the system temp dir; removed after the run.
TEST_TMP = REPO / ".test-tmp"


def _tempdir():
    TEST_TMP.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=TEST_TMP)


def tearDownModule():
    if TEST_TMP.is_dir() and not any(TEST_TMP.iterdir()):
        TEST_TMP.rmdir()


def _run_cli(temp: Path, *args, cwd=REPO):
    """Run the CLI offline with no API keys and all caches under `temp`."""
    env = {k: v for k, v in os.environ.items() if k not in KEY_VARS}
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
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=300,
    )


class TestF03DemoOnlyCli(unittest.TestCase):
    def test_bundled_golden_path_forms_are_accepted(self):
        with _tempdir() as temp:
            temp = Path(temp)
            link = temp / "golden-link.json"
            link.symlink_to(GOLDEN)
            elsewhere = temp / "cwd"
            elsewhere.mkdir()
            cases = [
                ("relative", "examples/golden_set.json", REPO),
                ("dot-relative", "./examples/golden_set.json", REPO),
                ("dot-dot", "examples/../examples/golden_set.json", REPO),
                ("absolute", str(GOLDEN), REPO),
                ("symlink", str(link), REPO),
                ("from another cwd", os.path.relpath(GOLDEN, elsewhere), elsewhere),
            ]
            for name, golden, cwd in cases:
                with self.subTest(name):
                    out = temp / f"out-{name.replace(' ', '-')}"
                    proc = _run_cli(temp, "--golden", golden, "--no-embeddings",
                                    "--output", str(out), cwd=cwd)
                    self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                    data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
                    self.assertEqual(data["summary"]["n_questions"], 8)
                    self.assertEqual(data["summary"]["metric_counts"]["retrieval_recall_at_k"], 7)

    def _assert_rejected(self, temp: Path, golden: Path):
        out_dir = temp / "results-should-not-exist"
        out = out_dir / "run"
        proc = _run_cli(temp, "--golden", str(golden), "--no-embeddings", "--csv",
                        "--output", str(out))
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn(f"--golden {golden} is not the bundled examples/golden_set.json", proc.stderr)
        self.assertIn("runs only the bundled demo pipeline", proc.stderr)
        self.assertIn("nothing\n  was evaluated or written", proc.stderr)
        self.assertIn("rag_eval.evaluate()", proc.stderr)
        self.assertIn("How to plug in your own system", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        # Rejected before loading or evaluating: no table, no outputs.
        self.assertEqual(proc.stdout, "")
        self.assertFalse(out_dir.exists())

    def test_custom_golden_set_is_rejected_before_evaluation(self):
        with _tempdir() as temp:
            temp = Path(temp)
            custom = temp / "my_golden.json"
            custom.write_text(json.dumps([{
                "id": "s1",
                "question": "How long does standard shipping take?",
                "expected_answer": "3-5 business days",
                "expected_source_ids": ["shipping-policy"],
            }]), encoding="utf-8")
            self._assert_rejected(temp, custom)

    def test_unedited_copy_of_bundled_golden_is_also_rejected(self):
        # Identity, not content: a copy is where users start editing their own set.
        with _tempdir() as temp:
            temp = Path(temp)
            copy = temp / "golden_set.json"
            shutil.copyfile(GOLDEN, copy)
            self._assert_rejected(temp, copy)

    def test_missing_golden_path_keeps_not_found_error(self):
        with _tempdir() as temp:
            temp = Path(temp)
            proc = _run_cli(temp, "--golden", "examples/golden_set.jsn", "--no-embeddings",
                            "--output", str(temp / "out"))
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Golden set not found: examples/golden_set.jsn", proc.stderr)
        self.assertNotIn("not the bundled", proc.stderr)

    def test_help_documents_demo_only_cli_and_output_default(self):
        with _tempdir() as temp:
            proc = _run_cli(Path(temp), "--help")
        self.assertEqual(proc.returncode, 0)
        help_text = " ".join(proc.stdout.split())
        self.assertIn("The CLI runs only the bundled demo", help_text)
        self.assertIn("call evaluate() with your own retriever and generator", help_text)
        self.assertIn("default: results/results -> results/results.json", help_text)
        self.assertNotIn("at the bottom of this file", help_text)

    def test_python_api_still_accepts_a_custom_golden_set(self):
        with _tempdir() as temp:
            path = Path(temp) / "my_golden.json"
            path.write_text(json.dumps([{
                "id": "s1", "question": "Shipping time?", "expected_answer": "3-5 days",
                "expected_source_ids": ["shipping-policy"],
            }]), encoding="utf-8")
            golden = load_golden_set(str(path))
        results = evaluate(
            golden,
            lambda q: [{"source_id": "shipping-policy", "text": "Shipping takes 3-5 days"}],
            lambda q, ctx: "3-5 days",
            no_embeddings=True,
        )
        self.assertEqual(results[0]["metrics"]["retrieval_recall_at_k"]["score"], 1.0)


class TestF06SampleOutput(unittest.TestCase):
    @staticmethod
    def _strip_latency(report: dict) -> dict:
        """Drop only measured latency values; everything else must match exactly."""
        report = json.loads(json.dumps(report))
        report["summary"].pop("avg_latency_s")
        for row in report["results"]:
            row.pop("latency")
        return report

    def _regenerate(self) -> dict:
        """Same steps as `rag_eval.py --golden examples/golden_set.json --no-embeddings`."""
        golden = load_golden_set(str(GOLDEN))
        results = evaluate(golden, mock_retriever_fn, mock_generator_fn, no_embeddings=True)
        with _tempdir() as temp, redirect_stdout(StringIO()):
            path = Path(temp) / "regenerated.json"
            save_json(results, summarise(results), str(path))
            return json.loads(path.read_text(encoding="utf-8"))

    def test_committed_sample_matches_current_evaluator(self):
        committed = json.loads(SAMPLE.read_text(encoding="utf-8"))
        regenerated = self._regenerate()

        # Latency is measured, so only its shape is checked.
        for report in (committed, regenerated):
            self.assertIsInstance(report["summary"]["avg_latency_s"], (int, float))
            for row in report["results"]:
                self.assertEqual(set(row["latency"]), {"retrieval_s", "generation_s", "total_s"})
                self.assertTrue(all(isinstance(v, (int, float)) for v in row["latency"].values()))

        self.assertEqual(self._strip_latency(committed), self._strip_latency(regenerated),
                         "examples/sample_output.json is stale; regenerate it with: python "
                         "rag_eval.py --golden examples/golden_set.json --no-embeddings "
                         "--output examples/sample_output")

    def test_committed_sample_has_current_semantics(self):
        sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
        summary = sample["summary"]
        q007 = next(r for r in sample["results"] if r["id"] == "q007")

        self.assertEqual(summary["n_questions"], 8)
        self.assertEqual(len(sample["results"]), 8)
        self.assertEqual(q007["metrics"]["retrieval_recall_at_k"],
                         {"score": None, "k": 3, "verdict": "SKIP"})
        self.assertAlmostEqual(summary["mean_scores"]["retrieval_recall_at_k"], 0.7143, places=4)
        self.assertEqual(summary["metric_counts"], {
            "retrieval_recall_at_k": 7, "answer_faithfulness": 8, "answer_relevance": 0,
            "exact_match": 8, "token_f1": 8,
        })
        self.assertEqual(summary["verdicts"]["retrieval_recall_at_k"],
                         {"PASS": 6, "WARN": 0, "FAIL": 1, "SKIP": 1, "ERROR": 0})
        self.assertEqual(summary["verdicts"]["answer_relevance"]["SKIP"], 8)
        self.assertNotIn("answer_relevance", summary["mean_scores"])
        self.assertEqual(summary["incomplete"], {})
        for row in sample["results"]:
            self.assertEqual(row["metrics"]["answer_relevance"],
                             {"score": None, "method": "skipped", "verdict": "SKIP"})

        # No LLM judge ran, so there is no judge usage or cost anywhere.
        self.assertNotIn("llm_judge", summary)
        self.assertNotIn("_summary", json.dumps(sample))
        for row in sample["results"]:
            self.assertFalse([k for k in row if "judge" in k], row["id"])


if __name__ == "__main__":
    unittest.main()
