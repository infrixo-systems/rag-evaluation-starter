"""P2-A reliability regression tests (F-01, F-02, F-09 and follow-up corrections).

Run with: python -m unittest discover -s tests -v
No test makes a real network or paid LLM call: the CLI tests block sockets,
and the judge tests inject fake openai / anthropic / tiktoken modules.
"""
import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import rag_eval
from rag_eval import JudgeConfigError, estimate_cost, evaluate, save_csv, save_json, summarise

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "examples" / "golden_set.json"
KEY_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
HAIKU = "claude-haiku-4-5-20251001"
# Temp files stay inside the repo, never the system temp dir; removed after the run.
TEST_TMP = REPO / ".test-tmp"


def _tempdir():
    TEST_TMP.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=TEST_TMP)


def tearDownModule():
    if TEST_TMP.is_dir() and not any(TEST_TMP.iterdir()):
        TEST_TMP.rmdir()


# Runs the CLI with every outbound connection blocked, so any download fails.
OFFLINE_RUNNER = textwrap.dedent("""
    import runpy, socket, sys
    def _blocked(*args, **kwargs):
        raise OSError("network access blocked by test")
    socket.socket.connect = _blocked
    socket.socket.connect_ex = _blocked
    socket.create_connection = _blocked
    socket.getaddrinfo = _blocked
    sys.argv = ["rag_eval.py"] + sys.argv[1:]
    runpy.run_path("rag_eval.py", run_name="__main__")
""")

# Stub modules for CLI subprocess tests (put on PYTHONPATH ahead of site-packages).
STUB_TIKTOKEN = textwrap.dedent("""
    class _Enc:
        def encode(self, text):
            return text.split()
    def encoding_for_model(model):
        return _Enc()
""")
# FAIL_FIRST=n: the first n requests raise; later ones return a valid judge reply.
STUB_OPENAI = textwrap.dedent("""
    import json, os
    from types import SimpleNamespace as NS
    _calls = [0]
    REPLY = json.dumps({"faithfulness": {"score": 4, "reason": "ok"},
                        "relevance": {"score": 5, "reason": "ok"}})
    class _C:
        def create(self, **kwargs):
            _calls[0] += 1
            if _calls[0] <= int(os.environ["FAIL_FIRST"]):
                raise ConnectionError("fake API unreachable")
            return NS(choices=[NS(message=NS(content=REPLY))],
                      usage=NS(prompt_tokens=100, completion_tokens=20))
    class OpenAI:
        def __init__(self, api_key):
            self.chat = NS(completions=_C())
""")


def _env_without_keys(**extra):
    env = {k: v for k, v in os.environ.items() if k not in KEY_VARS}
    env.update(extra)
    return env


def _run_cli_offline(temp: Path, *args, extra_env=None):
    """Run the CLI with no network, empty tokenizer + HF caches, and no API keys."""
    tk_cache = temp / "tiktoken-cache"
    tk_cache.mkdir(exist_ok=True)
    env = _env_without_keys(
        TIKTOKEN_CACHE_DIR=str(tk_cache),
        DATA_GYM_CACHE_DIR=str(tk_cache),
        HF_HOME=str(temp / "hf-home"),
        XDG_CACHE_HOME=str(temp / "xdg-cache"),
        TMPDIR=str(temp),
        HF_HUB_OFFLINE="1",
        HTTPS_PROXY="http://127.0.0.1:9",
        HTTP_PROXY="http://127.0.0.1:9",
        https_proxy="http://127.0.0.1:9",
        http_proxy="http://127.0.0.1:9",
        PYTHONDONTWRITEBYTECODE="1",
        COLUMNS="200",
    )
    env.update(extra_env or {})
    proc = subprocess.run(
        [sys.executable, "-c", OFFLINE_RUNNER, *args],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=300,
    )
    return proc, tk_cache


def _stub_dir(temp: Path, **modules) -> Path:
    stubs = temp / "stubs"
    stubs.mkdir(exist_ok=True)
    for name, source in modules.items():
        (stubs / f"{name}.py").write_text(source, encoding="utf-8")
    return stubs


def _entries(n):
    return [
        {
            "id": f"q{i}",
            "question": "What is the API limit?",
            "expected_answer": "100 per minute",
            "expected_source_ids": ["source-a"],
        }
        for i in range(n)
    ]


def _retriever(question):
    return [{"source_id": "source-a", "text": "100 per minute"}]


def _generator(question, contexts):
    return "100 per minute"


def _rich_output(results, summary):
    from rich.console import Console
    buf = StringIO()
    with patch.object(rag_eval, "console", Console(file=buf, width=300)):
        rag_eval.print_rich_table(results, summary)
    return buf.getvalue()


class TestF01NoTokenizerWithoutJudge(unittest.TestCase):
    def test_evaluate_without_judge_never_counts_tokens(self):
        with patch.object(rag_eval, "_tiktoken_count",
                          side_effect=AssertionError("tokenizer used without judge")):
            results = evaluate(_entries(2), _retriever, _generator, no_embeddings=True)
        self.assertEqual(len(results), 2)
        self.assertFalse(any(r.get("_summary") for r in results))

    def test_cli_no_key_quickstart_runs_offline_with_empty_tokenizer_cache(self):
        with _tempdir() as temp:
            temp = Path(temp)
            out = temp / "offline"
            proc, tk_cache = _run_cli_offline(
                temp, "--golden", str(GOLDEN), "--no-embeddings", "--csv", "--output", str(out),
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertEqual(list(tk_cache.iterdir()), [], "tokenizer data was downloaded/cached")
            data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
            with open(out.with_suffix(".csv"), newline="", encoding="utf-8") as f:
                csv_rows = list(csv.DictReader(f))

        # P0 semantics are unchanged on this path.
        summary = data["summary"]
        q007 = next(r for r in data["results"] if r["id"] == "q007")
        self.assertEqual(summary["n_questions"], 8)
        self.assertEqual(summary["metric_counts"]["retrieval_recall_at_k"], 7)
        self.assertEqual(summary["mean_scores"]["retrieval_recall_at_k"], 0.7143)
        self.assertIsNone(q007["metrics"]["retrieval_recall_at_k"]["score"])
        self.assertEqual(q007["metrics"]["retrieval_recall_at_k"]["verdict"], "SKIP")
        self.assertEqual(summary["incomplete"], {})
        self.assertNotIn("llm_judge", summary)
        for r in data["results"]:
            self.assertEqual(r["metrics"]["answer_relevance"]["method"], "skipped")
            self.assertEqual(r["metrics"]["answer_relevance"]["verdict"], "SKIP")
        self.assertNotIn("INCOMPLETE", proc.stdout)
        # CSV: skipped relevance is labelled; no judge columns without --llm-judge.
        self.assertTrue(all(r["answer_relevance_method"] == "skipped" for r in csv_rows))
        self.assertNotIn("llm_judge_status", csv_rows[0])
        q007_csv = next(r for r in csv_rows if r["id"] == "q007")
        self.assertEqual(q007_csv["retrieval_recall_at_k"], "")


class TestF02EmbeddingFailureIsNotSkip(unittest.TestCase):
    def test_injected_embedder_failure_reports_error_once(self):
        calls = []

        def broken_embedder(texts):
            calls.append(texts)
            raise OSError("cannot download all-MiniLM-L6-v2\nlong traceback detail")

        with redirect_stderr(StringIO()) as err:
            results = evaluate(_entries(3), _retriever, _generator, embedder_fn=broken_embedder)

        self.assertEqual(len(calls), 1, "failed embedder was retried for every question")
        for r in results:
            rel = r["metrics"]["answer_relevance"]
            self.assertIsNone(rel["score"])
            self.assertEqual(rel["method"], "error")
            self.assertEqual(rel["verdict"], "ERROR")
            self.assertIn("OSError: cannot download all-MiniLM-L6-v2", rel["error"])
            # Other metrics are still computed.
            self.assertEqual(r["metrics"]["retrieval_recall_at_k"]["score"], 1.0)

        warning = err.getvalue()
        self.assertEqual(warning.count("Warning:"), 1)
        self.assertIn("cannot download all-MiniLM-L6-v2", warning)
        self.assertIn("--no-embeddings", warning)

        summary = summarise(results)
        self.assertEqual(summary["verdicts"]["answer_relevance"]["ERROR"], 3)
        self.assertEqual(summary["verdicts"]["answer_relevance"]["SKIP"], 0)
        self.assertEqual(summary["metric_counts"]["answer_relevance"], 0)
        # Complete failure is ERROR, not PARTIAL.
        self.assertEqual(summary["incomplete"]["answer_relevance"],
                         {"status": "error", "computed": 0, "error": 3})
        printed = _rich_output(results, summary)
        self.assertIn("N/A ERROR", printed)
        self.assertNotIn("PARTIAL", printed)
        self.assertIn("answer_relevance: ERROR — not computed for any row (0/3)", printed)

    def test_partial_relevance_is_partial_not_complete_or_error(self):
        vectors = iter([[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]])

        def flaky_embedder(texts):
            try:
                return next(vectors)
            except StopIteration:
                raise RuntimeError("CUDA out of memory") from None

        with redirect_stderr(StringIO()):
            results = evaluate(_entries(5), _retriever, _generator, embedder_fn=flaky_embedder)
        summary = summarise(results)
        self.assertEqual([r["metrics"]["answer_relevance"]["verdict"] for r in results],
                         ["PASS", "FAIL", "ERROR", "ERROR", "ERROR"])
        self.assertEqual(summary["incomplete"]["answer_relevance"],
                         {"status": "partial", "computed": 2, "error": 3})
        self.assertEqual(summary["mean_scores"]["answer_relevance"], 0.5)
        self.assertEqual(summary["metric_counts"]["answer_relevance"], 2)

        printed = _rich_output(results, summary)
        self.assertIn("0.50 PARTIAL", printed)
        self.assertIn("answer_relevance: PARTIAL — computed for 2/5 rows, ERROR on 3", printed)
        with redirect_stdout(StringIO()) as out:
            rag_eval._print_plain_table(results, summary)
        self.assertIn("INCOMPLETE RESULTS", out.getvalue())
        self.assertIn("answer_relevance: PARTIAL", out.getvalue())

    def test_default_embedder_init_failure_is_not_retried(self):
        with patch.object(rag_eval, "_get_embedder", side_effect=OSError("offline")) as get, \
                redirect_stderr(StringIO()):
            results = evaluate(_entries(4), _retriever, _generator)
        get.assert_called_once()
        self.assertTrue(all(r["metrics"]["answer_relevance"]["verdict"] == "ERROR" for r in results))

    def test_intentional_no_embeddings_is_still_skip_and_silent(self):
        embedder = Mock()
        with redirect_stderr(StringIO()) as err:
            results = evaluate(_entries(2), _retriever, _generator,
                               embedder_fn=embedder, no_embeddings=True)
        embedder.assert_not_called()
        self.assertEqual(err.getvalue(), "")
        for r in results:
            self.assertEqual(r["metrics"]["answer_relevance"],
                             {"score": None, "method": "skipped", "verdict": "SKIP"})
        summary = summarise(results)
        self.assertEqual(summary["verdicts"]["answer_relevance"]["ERROR"], 0)
        self.assertEqual(summary["incomplete"], {})

    def test_csv_distinguishes_error_from_skip(self):
        with redirect_stderr(StringIO()):
            failed = evaluate(_entries(1), _retriever, _generator,
                              embedder_fn=Mock(side_effect=OSError("x")))
        skipped = evaluate(_entries(1), _retriever, _generator, no_embeddings=True)
        with _tempdir() as temp, redirect_stdout(StringIO()):
            methods = []
            for name, results in (("failed", failed), ("skipped", skipped)):
                path = Path(temp) / f"{name}.csv"
                save_csv(results, str(path))
                with open(path, newline="", encoding="utf-8") as f:
                    row = next(csv.DictReader(f))
                self.assertEqual(row["answer_relevance"], "")
                methods.append(row["answer_relevance_method"])
        self.assertEqual(methods, ["error", "skipped"])

    @unittest.skipUnless(importlib.util.find_spec("sentence_transformers"),
                         "sentence-transformers not installed")
    def test_cli_offline_model_load_failure_exits_nonzero_with_error(self):
        with _tempdir() as temp:
            temp = Path(temp)
            out = temp / "emb"
            proc, _ = _run_cli_offline(temp, "--golden", str(GOLDEN), "--output", str(out))
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            self.assertIn("answer relevance failed", proc.stderr)
            self.assertIn("results are incomplete", proc.stderr)
            self.assertIn("INCOMPLETE RESULTS", proc.stdout)
            data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(data["summary"]["verdicts"]["answer_relevance"]["ERROR"], 8)
        self.assertEqual(data["summary"]["incomplete"]["answer_relevance"]["status"], "error")
        for r in data["results"]:
            self.assertEqual(r["metrics"]["answer_relevance"]["method"], "error")
            self.assertTrue(r["metrics"]["answer_relevance"]["error"])
        # Recall P0 semantics unaffected by the embedding failure.
        self.assertEqual(data["summary"]["metric_counts"]["retrieval_recall_at_k"], 7)


def _fake_openai(responses):
    """Fake openai module; each item is an Exception or (content, prompt_tokens, completion_tokens)."""
    queue = list(responses)
    created = []

    class _Completions:
        def create(self, **kwargs):
            created.append(kwargs)
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            content, prompt_tokens, completion_tokens = item
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(prompt_tokens=prompt_tokens,
                                      completion_tokens=completion_tokens),
            )

    class OpenAI:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(completions=_Completions())

    mod = types.ModuleType("openai")
    mod.OpenAI = OpenAI
    return mod, created


def _fake_anthropic(content, input_tokens, output_tokens):
    created = []

    class _Messages:
        def create(self, **kwargs):
            created.append(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(text=content)],
                usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
            )

    class Anthropic:
        def __init__(self, api_key):
            self.messages = _Messages()

    mod = types.ModuleType("anthropic")
    mod.Anthropic = Anthropic
    return mod, created


def _judge_json(faith=5, rel=4, **extra):
    data = {
        "faithfulness": {"score": faith, "reason": "grounded"},
        "relevance": {"score": rel, "reason": "on topic"},
    }
    data.update(extra)
    return json.dumps(data)


def _run_judge(responses, n=None, tiktoken=None):
    """evaluate() with a fake OpenAI judge; returns (rows, summary_row, created)."""
    fake, created = _fake_openai(responses)
    env = _env_without_keys(OPENAI_API_KEY="test-not-real")
    tk = tiktoken or patch.object(rag_eval, "_tiktoken_count", return_value=111)
    with patch.dict(sys.modules, {"openai": fake}), patch.dict(os.environ, env, clear=True), tk:
        results = evaluate(_entries(n or len(responses)), _retriever, _generator,
                           use_llm_judge=True, no_embeddings=True)
    return results[:-1], results[-1], created, results


class TestF09JudgeCost(unittest.TestCase):
    def test_no_key_raises_before_running_pipeline(self):
        retriever = Mock(side_effect=_retriever)
        with patch.dict(os.environ, _env_without_keys(), clear=True), \
                patch.object(rag_eval, "_tiktoken_count", side_effect=AssertionError("counted")):
            with self.assertRaisesRegex(JudgeConfigError, "no usable API key"):
                evaluate(_entries(2), retriever, _generator,
                         use_llm_judge=True, no_embeddings=True)
        retriever.assert_not_called()

    def test_cli_no_key_exits_2_and_reports_no_cost(self):
        with _tempdir() as temp:
            temp = Path(temp)
            out = temp / "judge"
            proc, tk_cache = _run_cli_offline(
                temp, "--golden", str(GOLDEN), "--no-embeddings", "--llm-judge",
                "--output", str(out),
            )
            self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
            self.assertIn("no usable API key", proc.stderr)
            self.assertNotIn("cost", proc.stdout.lower())
            self.assertFalse(out.with_suffix(".json").exists())
            self.assertEqual(list(tk_cache.iterdir()), [])

    def test_cost_only_from_successful_requests_and_kept_apart_from_estimate(self):
        rows, summary_row, created, results = _run_judge([
            (_judge_json(), 1000, 50),        # scored
            ConnectionError("boom"),          # request failed: no usage
            ("not json", 400, 10),            # request ok, reply unusable: usage counts
        ])
        self.assertEqual(len(created), 3)
        self.assertTrue(all(c["model"] == "gpt-4o-mini" for c in created))
        self.assertTrue(summary_row["_summary"])
        self.assertEqual(summary_row["llm_judge_model"], "gpt-4o-mini")
        self.assertEqual(summary_row["llm_judge_calls"], {
            "attempted": 3, "request_succeeded": 2, "request_failed": 1,
            "scored": 1, "invalid_response": 1,
        })
        self.assertEqual(summary_row["llm_judge_usage"],
                         {"input_tokens": 1400, "output_tokens": 60})
        self.assertEqual(summary_row["llm_judge_estimated_prompt_tokens"], 333)
        self.assertIsNone(summary_row["llm_judge_estimate_error"])
        # 1400/1000*0.00015 + 60/1000*0.0006
        self.assertAlmostEqual(summary_row["llm_judge_estimated_cost_usd"], 0.000246)
        self.assertNotIn("llm_judge_total_cost_usd", summary_row)

        self.assertEqual([r["llm_judge_status"] for r in rows],
                         ["scored", "request_failed", "invalid_response"])
        self.assertEqual(rows[0]["llm_judge"]["faithfulness"]["score"], 5)
        self.assertIn("ConnectionError: boom", rows[1]["llm_judge"]["error"])
        self.assertIn("Invalid judge response", rows[2]["llm_judge"]["error"])
        self.assertAlmostEqual(rows[0]["_judge_estimated_cost_usd"], 0.00018)
        self.assertEqual(rows[1]["_judge_estimated_cost_usd"], 0.0)
        self.assertAlmostEqual(rows[2]["_judge_estimated_cost_usd"], 0.000066)
        self.assertFalse(any("_judge_cost_usd" in r for r in rows))

        printed = _rich_output(results, summarise(results))
        self.assertIn("scored=1/3", printed)
        self.assertIn("requests_succeeded=2/3", printed)
        self.assertIn("estimated_cost=~$0.0002", printed)
        self.assertIn("not a bill", printed)
        self.assertIn("llm_judge: PARTIAL — scored 1/3 rows", printed)
        self.assertNotIn("total_cost", printed)

    def test_list_prices_per_million_tokens(self):
        # Standard API list prices, USD per 1M tokens: (input, output).
        expected = {
            "gpt-4o-mini": (0.15, 0.60),
            HAIKU: (1.00, 5.00),
        }
        self.assertEqual(set(rag_eval.LLM_COST_PER_1K), set(expected))
        self.assertEqual(set(rag_eval.JUDGE_MODELS), set(expected))
        for model, (per_m_in, per_m_out) in expected.items():
            with self.subTest(model=model):
                rates = rag_eval.LLM_COST_PER_1K[model]
                self.assertAlmostEqual(rates["input"] * 1000, per_m_in)
                self.assertAlmostEqual(rates["output"] * 1000, per_m_out)
                self.assertAlmostEqual(
                    estimate_cost({"input_tokens": 1_000_000, "output_tokens": 0}, model), per_m_in)
                self.assertAlmostEqual(
                    estimate_cost({"input_tokens": 0, "output_tokens": 1_000_000}, model), per_m_out)

    def test_unknown_model_price_is_none_not_zero(self):
        self.assertIsNone(estimate_cost({"input_tokens": 10, "output_tokens": 5}, "unpriced"))
        self.assertEqual(estimate_cost({"input_tokens": 0, "output_tokens": 0}, "gpt-4o-mini"), 0.0)

    def test_cli_all_judge_calls_failed_exits_1(self):
        with _tempdir() as temp:
            temp = Path(temp)
            stubs = _stub_dir(temp, openai=STUB_OPENAI, tiktoken=STUB_TIKTOKEN)
            out = temp / "judge-fail"
            proc, _ = _run_cli_offline(
                temp, "--golden", str(GOLDEN), "--no-embeddings", "--llm-judge",
                "--output", str(out),
                extra_env={"OPENAI_API_KEY": "test-not-real", "PYTHONPATH": str(stubs),
                           "FAIL_FIRST": "99"},
            )
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            self.assertIn("llm_judge: ERROR — scored 0/8 rows (request_failed=8)", proc.stderr)
            self.assertIn("scored=0/8", proc.stdout)
            self.assertIn("estimated_cost=~$0.0000", proc.stdout)
            data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(data["summary"]["incomplete"]["llm_judge"],
                         {"status": "error", "scored": 0, "request_failed": 8})
        for r in data["results"]:
            self.assertIn("fake API unreachable", r["llm_judge"]["error"])
            self.assertEqual(r["_judge_estimated_cost_usd"], 0.0)


class TestJudgeNoSubstitution(unittest.TestCase):
    def test_default_model_with_only_anthropic_key_is_refused(self):
        with patch.dict(os.environ, _env_without_keys(ANTHROPIC_API_KEY="test-not-real"),
                        clear=True):
            with self.assertRaisesRegex(JudgeConfigError, "requires OPENAI_API_KEY"):
                evaluate(_entries(1), _retriever, _generator,
                         use_llm_judge=True, no_embeddings=True)

    def test_anthropic_model_with_only_openai_key_is_refused(self):
        with patch.dict(os.environ, _env_without_keys(OPENAI_API_KEY="test-not-real"),
                        clear=True):
            with self.assertRaisesRegex(JudgeConfigError, "requires ANTHROPIC_API_KEY"):
                evaluate(_entries(1), _retriever, _generator, use_llm_judge=True,
                         llm_judge_model=HAIKU, no_embeddings=True)

    def test_unsupported_model_is_refused_even_with_both_keys(self):
        env = _env_without_keys(OPENAI_API_KEY="test-not-real", ANTHROPIC_API_KEY="test-not-real")
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(JudgeConfigError, "Unsupported judge model 'claude-opus-x'"):
                evaluate(_entries(1), _retriever, _generator, use_llm_judge=True,
                         llm_judge_model="claude-opus-x", no_embeddings=True)

    def test_cli_unsupported_model_exits_2(self):
        with _tempdir() as temp:
            proc, _ = _run_cli_offline(
                Path(temp), "--golden", str(GOLDEN), "--no-embeddings", "--llm-judge",
                "--judge-model", "gpt-4o", "--no-json",
                extra_env={"OPENAI_API_KEY": "test-not-real", "ANTHROPIC_API_KEY": "test-not-real"},
            )
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("Unsupported judge model 'gpt-4o'", proc.stderr)

    def test_requested_anthropic_model_is_the_model_called(self):
        fake, created = _fake_anthropic(_judge_json(), 300, 40)
        env = _env_without_keys(OPENAI_API_KEY="test-not-real", ANTHROPIC_API_KEY="test-not-real")
        with patch.dict(sys.modules, {"anthropic": fake}), patch.dict(os.environ, env, clear=True), \
                patch.object(rag_eval, "_tiktoken_count", return_value=1):
            results = evaluate(_entries(2), _retriever, _generator, use_llm_judge=True,
                               llm_judge_model=HAIKU, no_embeddings=True)
        self.assertEqual([c["model"] for c in created], [HAIKU, HAIKU])
        self.assertEqual(results[-1]["llm_judge_model"], HAIKU)
        # 600/1000*0.001 + 80/1000*0.005
        self.assertAlmostEqual(results[-1]["llm_judge_estimated_cost_usd"], 0.001)


class TestJudgeResponseValidation(unittest.TestCase):
    def test_invalid_replies_are_billed_but_not_scored(self):
        invalid = [
            _judge_json(faith=6),                               # out of range
            _judge_json(rel=True),                              # bool is not a score
            _judge_json(faith=4.5),                             # not an integer
            json.dumps({"faithfulness": {"score": 3, "reason": "x"}}),   # missing relevance
            json.dumps({"faithfulness": {"score": 3, "reason": 7},
                        "relevance": {"score": 3, "reason": "x"}}),      # non-string reason
            json.dumps([1, 2]),                                 # not an object
            None,                                               # empty message content
        ]
        rows, summary_row, _, results = _run_judge([(c, 10, 1) for c in invalid])
        self.assertEqual(summary_row["llm_judge_calls"], {
            "attempted": 7, "request_succeeded": 7, "request_failed": 0,
            "scored": 0, "invalid_response": 7,
        })
        self.assertEqual(summary_row["llm_judge_usage"], {"input_tokens": 70, "output_tokens": 7})
        for r in rows:
            self.assertEqual(r["llm_judge_status"], "invalid_response")
            self.assertIn("Invalid judge response", r["llm_judge"]["error"])
            self.assertGreater(r["_judge_estimated_cost_usd"], 0)
        self.assertIn("faithfulness.score must be an integer 1-5, got 6", rows[0]["llm_judge"]["error"])
        self.assertIn("missing or non-object 'relevance'", rows[3]["llm_judge"]["error"])
        self.assertEqual(summarise(results)["incomplete"]["llm_judge"],
                         {"status": "error", "scored": 0, "invalid_response": 7})

    def test_valid_reply_is_normalised(self):
        rows, summary_row, _, results = _run_judge([(_judge_json(model_notes="extra"), 10, 1)])
        self.assertEqual(rows[0]["llm_judge"], {
            "faithfulness": {"score": 5, "reason": "grounded"},
            "relevance": {"score": 4, "reason": "on topic"},
        })
        self.assertEqual(rows[0]["llm_judge_status"], "scored")
        self.assertEqual(summarise(results)["incomplete"], {})


class TestTokenizerEstimateFailure(unittest.TestCase):
    def test_tokenizer_failure_does_not_stop_judging(self):
        tk_mock = Mock(side_effect=OSError("cannot fetch o200k_base\ndetail"))
        with redirect_stderr(StringIO()) as err:
            rows, summary_row, created, results = _run_judge(
                [(_judge_json(), 100, 10)] * 3,
                tiktoken=patch.object(rag_eval, "_tiktoken_count", tk_mock),
            )
        tk_mock.assert_called_once()
        self.assertEqual(len(created), 3)
        self.assertEqual(err.getvalue().count("Warning:"), 1)
        self.assertIn("estimate unavailable (OSError: cannot fetch o200k_base)", err.getvalue())
        self.assertIsNone(summary_row["llm_judge_estimated_prompt_tokens"])
        self.assertIn("OSError: cannot fetch o200k_base", summary_row["llm_judge_estimate_error"])
        self.assertEqual(summary_row["llm_judge_calls"]["scored"], 3)
        self.assertEqual(summary_row["llm_judge_usage"], {"input_tokens": 300, "output_tokens": 30})
        self.assertIsNotNone(summary_row["llm_judge_estimated_cost_usd"])
        self.assertTrue(all(r["llm_judge_status"] == "scored" for r in rows))
        self.assertIn("estimated_prompt_tokens=unavailable", _rich_output(results, summarise(results)))


class TestJudgeMachineReadableOutput(unittest.TestCase):
    def test_json_and_csv_include_judge_usage_cost_and_status(self):
        rows, summary_row, _, results = _run_judge([
            (_judge_json(), 1000, 50),
            ConnectionError("boom"),
        ])
        summary = summarise(results)
        with _tempdir() as temp, redirect_stdout(StringIO()):
            save_json(results, summary, str(Path(temp) / "out.json"))
            save_csv(results, str(Path(temp) / "out.csv"))
            saved = json.loads((Path(temp) / "out.json").read_text(encoding="utf-8"))
            with open(Path(temp) / "out.csv", newline="", encoding="utf-8") as f:
                csv_rows = list(csv.DictReader(f))

        judge = saved["summary"]["llm_judge"]
        self.assertEqual(judge["llm_judge_model"], "gpt-4o-mini")
        self.assertEqual(judge["llm_judge_calls"]["scored"], 1)
        self.assertEqual(judge["llm_judge_calls"]["request_failed"], 1)
        self.assertEqual(judge["llm_judge_usage"], {"input_tokens": 1000, "output_tokens": 50})
        self.assertEqual(judge["llm_judge_estimated_prompt_tokens"], 222)
        self.assertAlmostEqual(judge["llm_judge_estimated_cost_usd"], 0.00018)
        self.assertNotIn("_summary", judge)
        self.assertEqual(saved["summary"]["incomplete"]["llm_judge"],
                         {"status": "partial", "scored": 1, "request_failed": 1})
        self.assertEqual(len(saved["results"]), 2)           # no pseudo-row in results
        self.assertEqual([r["llm_judge_status"] for r in saved["results"]],
                         ["scored", "request_failed"])

        self.assertEqual([r["llm_judge_status"] for r in csv_rows], ["scored", "request_failed"])
        self.assertAlmostEqual(float(csv_rows[0]["judge_estimated_cost_usd"]), 0.00018)
        self.assertEqual(float(csv_rows[1]["judge_estimated_cost_usd"]), 0.0)

    def test_cli_partial_judge_failure_exits_1_with_outputs(self):
        with _tempdir() as temp:
            temp = Path(temp)
            stubs = _stub_dir(temp, openai=STUB_OPENAI, tiktoken=STUB_TIKTOKEN)
            out = temp / "judge-partial"
            proc, _ = _run_cli_offline(
                temp, "--golden", str(GOLDEN), "--no-embeddings", "--llm-judge", "--csv",
                "--output", str(out),
                extra_env={"OPENAI_API_KEY": "test-not-real", "PYTHONPATH": str(stubs),
                           "FAIL_FIRST": "2"},
            )
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            self.assertIn("llm_judge: PARTIAL — scored 6/8 rows (request_failed=2)", proc.stdout)
            self.assertIn("llm_judge: PARTIAL", proc.stderr)
            self.assertIn("results are incomplete", proc.stderr)
            data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
            with open(out.with_suffix(".csv"), newline="", encoding="utf-8") as f:
                csv_rows = list(csv.DictReader(f))
        summary = data["summary"]
        self.assertEqual(summary["llm_judge"]["llm_judge_calls"]["scored"], 6)
        self.assertEqual(summary["llm_judge"]["llm_judge_usage"],
                         {"input_tokens": 600, "output_tokens": 120})
        self.assertEqual(summary["incomplete"],
                         {"llm_judge": {"status": "partial", "scored": 6, "request_failed": 2}})
        self.assertEqual([r["llm_judge_status"] for r in csv_rows][:3],
                         ["request_failed", "request_failed", "scored"])
        # P0 semantics unaffected.
        self.assertEqual(summary["metric_counts"]["retrieval_recall_at_k"], 7)


if __name__ == "__main__":
    unittest.main()
