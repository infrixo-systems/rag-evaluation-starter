# Contributing

Contributions are welcome. The two most useful things you can add:

## New metrics

Add your metric function to `rag_eval.py` following the existing pattern:
- Function signature: `metric_<name>(args) -> float`
- Return a score in [0.0, 1.0]
- Add a threshold entry in `THRESHOLDS` at the top of the file
- Add the score to the `metrics` dict inside `evaluate()`
- Add a column to `print_rich_table()`

## New example golden sets

Add a JSON file to `examples/` following the schema in `examples/golden_set.json`. Use a fictional domain so the example is universally relatable. Open a PR with a short description of the domain and why the example is useful.

## Bug reports

Use the issue template in `.github/ISSUE_TEMPLATE/bug_report.md`.
