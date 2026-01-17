# Graduate Assist - Getting Started

This project implements a LangGraph-based agentic workflow for graduate-level experimental data analysis. It produces structured artifacts for context, data quality, analysis planning, execution results, and academic reporting. A rebuttal cycle is supported when reviewer/advisor feedback is provided.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Start Command

```bash
PYTHONPATH=src python -m graduate_assist.run \
  --data examples/soil_npk_growth.csv \
  --description examples/experiment_description.md \
  --artifacts artifacts \
  --verbose
```

Optional reviewer comment (triggers rebuttal and conditional reroute):

```bash
PYTHONPATH=src python -m graduate_assist.run \
  --data examples/soil_npk_growth.csv \
  --description examples/experiment_description.md \
  --artifacts artifacts \
  --reviewer-comment "Please reanalyze the data; the current analysis appears incorrect."
```

Auto self-review comment (LLM-generated, triggers rebuttal):

```bash
PYTHONPATH=src python -m graduate_assist.run \
  --data examples/soil_npk_growth.csv \
  --description examples/experiment_description.md \
  --artifacts artifacts \
  --auto-reviewer-comment
```

Two-step workflow (initial run, then reviewer-driven rerun):

```bash
PYTHONPATH=src python -m graduate_assist.run \
  --data examples/soil_npk_growth.csv \
  --description examples/experiment_description.md \
  --artifacts artifacts
```

```bash
PYTHONPATH=src python -m graduate_assist.run \
  --data examples/soil_npk_growth.csv \
  --description examples/experiment_description.md \
  --artifacts artifacts \
  --reviewer-comment "Please reanalyze the data; the current analysis appears incorrect."
```

## Outputs

Each run writes a unique folder under `artifacts/` containing:
- `context.json`
- `schema.yaml`
- `data_quality_report.md`
- `analysis_plan.yaml`
- `results.json`
- `figures/`
- `execution_log.txt`
- `report.md`
- `reviewer_comment.md` (when review comment exists)
- `rebuttal_decision.yaml` (when review comment exists)
- `rebuttal_response.md` (when review comment exists)

## Notes

- The example dataset in `examples/soil_npk_growth.csv` is a realistic synthetic dataset suitable for testing the workflow.
- A larger, mixed-type dataset is available in `examples/complex/complex_analysis_data.csv` (1,200 rows, 19 columns) for stress-testing schema inference and analysis selection.
- A matching description for the complex dataset lives at `examples/complex/experiment_description.md`.
- If you modify the dataset or description, the plan and outputs will adjust accordingly.
- Use `--verbose` to print agent logs (including prompts/responses) to stdout during a run.

## Analysis Selection

The analysis plan is chosen using rule-based heuristics plus optional LLM input (when enabled). Supported analyses include:
- Welch t-test (2 groups, numeric outcome)
- Mann-Whitney U (2 groups, nonparametric)
- One-way ANOVA (3+ groups, numeric outcome)
- Kruskal-Wallis (3+ groups, nonparametric)
- Paired t-test / Wilcoxon signed-rank (paired 2-condition data)
- Friedman test (paired 3+ conditions)
- Chi-square / Fisher exact (categorical outcome vs categorical predictor)
- Pearson/Spearman correlation
- Simple linear regression
- Descriptive statistics fallback

## Optional LLM Configuration

If you want LLM-authored prose or LLM-assisted analysis selection, set `LLM_PROVIDER` and `LLM_MODEL`. The pipeline will fall back to rule-based planning and templated text if the provider is unavailable.

OpenAI (gpt-4.1):
```bash
pip install openai
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4.1
export OPENAI_API_KEY=your_key_here
```

Ollama (free local models):
```bash
export LLM_PROVIDER=ollama
export LLM_MODEL=llama3.1:8b
```
