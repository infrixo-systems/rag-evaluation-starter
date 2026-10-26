"""P0 regression tests: execute with python -m unittest discover -s tests -v."""
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from rag_eval import (
    _print_plain_table,
    evaluate,
    load_golden_set,
    metric_retrieval_recall,
    print_rich_table,
    save_csv,
    save_json,
    summarise,
)


class TestRecallP0(unittest.TestCase):
    def _entry(self, **overrides):
        data = {
            "id": "case",
            "question": "What is the API limit?",
            "expected_answer": "100 per minute",
            "expected_source_ids": ["source-a"],
        }
        data.update(overrides)
        return data

    def _load(self, entries):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "golden.json"
            path.write_text(json.dumps(entries), encoding="utf-8")
            return load_golden_set(str(path))

    def _evaluate(self, entries):
        return evaluate(
            entries,
            lambda question: [{"source_id": "source-a", "text": "100 per minute"}],
            lambda question, contexts: "100 per minute",
            no_embeddings=True,
        )

    def test_recall_none_for_empty_expected_sources(self):
        self.assertIsNone(metric_retrieval_recall([], [], 3))

    def test_regular_recall_top_k_and_partial_match(self):
        docs = [{"source_id": "source-a"}, {"source_id": "source-b"}]
        self.assertEqual(metric_retrieval_recall(docs, ["source-a", "source-c"], 2), 0.5)
        self.assertEqual(metric_retrieval_recall(docs, ["source-a", "source-c"], 1), 0.5)
        self.assertEqual(metric_retrieval_recall(docs, ["source-b"], 1), 0.0)

    def test_valid_explicit_unanswerable_entry(self):
        entry = self._entry(answerable=False, expected_source_ids=[])
        self.assertEqual(self._load([entry]), [entry])

    def test_empty_sources_for_answerable_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires expected_source_ids"):
            self._load([self._entry(expected_source_ids=[])])

    def test_missing_source_field_rejected(self):
        entry = self._entry()
        del entry["expected_source_ids"]
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            self._load([entry])

    def test_non_boolean_answerable_rejected(self):
        with self.assertRaisesRegex(ValueError, "answerable must be"):
            self._load([self._entry(answerable="false")])

    def test_unanswerable_with_sources_rejected(self):
        with self.assertRaisesRegex(ValueError, "must have empty"):
            self._load([self._entry(answerable=False)])

    def test_malformed_source_ids_rejected(self):
        for sources in ([""], ["source-a", "source-a"], [42]):
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                self._load([self._entry(expected_source_ids=sources)])

    def test_non_object_entry_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be an object"):
            self._load(["bad entry"])

    def test_unanswerable_score_and_summary(self):
        entries = [self._entry(), self._entry(
            id="abstain", question="Unknown question", expected_answer="I don't know",
            expected_source_ids=[], answerable=False,
        )]
        results = self._evaluate(entries)
        self.assertEqual(results[0]["metrics"]["retrieval_recall_at_k"]["score"], 1.0)
        self.assertIsNone(results[1]["metrics"]["retrieval_recall_at_k"]["score"])
        self.assertEqual(results[1]["metrics"]["retrieval_recall_at_k"]["verdict"], "SKIP")
        summary = summarise(results)
        self.assertEqual(summary["n_questions"], 2)
        self.assertEqual(summary["mean_scores"]["retrieval_recall_at_k"], 1.0)
        self.assertEqual(summary["metric_counts"]["retrieval_recall_at_k"], 1)
        self.assertEqual(summary["verdicts"]["retrieval_recall_at_k"]["SKIP"], 1)
        self.assertEqual(summary["metric_counts"]["exact_match"], 2)
        with redirect_stdout(StringIO()) as out:
            _print_plain_table(results, summary)
        self.assertIn("N/A", out.getvalue())
        with tempfile.TemporaryDirectory() as temp:
            save_json(results, summary, str(Path(temp) / "out.json"))
            save_csv(results, str(Path(temp) / "out.csv"))
            saved = json.loads((Path(temp) / "out.json").read_text())
            self.assertIsNone(saved["results"][1]["metrics"]["retrieval_recall_at_k"]["score"])
            self.assertIn("abstain", (Path(temp) / "out.csv").read_text())

    def test_all_unanswerable_gives_no_mean_and_renders(self):
        entry = self._entry(answerable=False, expected_source_ids=[])
        results = self._evaluate([entry])
        summary = summarise(results)
        self.assertNotIn("retrieval_recall_at_k", summary["mean_scores"])
        self.assertEqual(summary["metric_counts"]["retrieval_recall_at_k"], 0)
        with redirect_stdout(StringIO()) as out:
            _print_plain_table(results, summary)
        self.assertIn("N/A", out.getvalue())
        # Rich enabled by default; verify it does not crash on missing mean.
        print_rich_table(results, summary)

    def test_repo_example_q007_is_explicitly_unanswerable(self):
        golden_path = (
            Path(__file__).resolve().parent.parent
            / "examples"
            / "golden_set.json"
        )
        entries = load_golden_set(str(golden_path))
        q007 = next(row for row in entries if row["id"] == "q007")
        self.assertFalse(q007["answerable"])
        self.assertEqual(q007["expected_source_ids"], [])


if __name__ == "__main__":
    unittest.main()
