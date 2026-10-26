# Contributing

Contributions are welcome. The two most useful things you can add:

## New metrics

Add your metric function to `rag_eval.py` following the existing pattern:
- Function signature: `metric_<name>(args) -> float`
- Return a score in [0.0, 1.0]
- Add a threshold entry in `THRESHOLDS` at the top of the file
- Add the score to the `metrics` dict inside `evaluate()`
- Add a column to `print_rich_table()`
- Add score and verdict columns to `save_csv()`, appended after the existing columns

## New example golden sets

Add a JSON file to `examples/` following the schema in `examples/golden_set.json`. Use a fictional domain so the example is universally relatable. Open a PR with a short description of the domain and why the example is useful.

## Bug reports

[Open an issue](https://github.com/infrixo-systems/rag-evaluation-starter/issues/new). Include the command you ran, your Python version, the full error output, and a minimal golden set entry that reproduces the problem.
