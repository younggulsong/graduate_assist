from __future__ import annotations

import json
import math
import os
import traceback
import urllib.parse
import urllib.request
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Tuple
import re

import numpy as np
import pandas as pd

from .artifacts import artifacts_root, write_json, write_markdown, write_text, write_yaml
from .llm import generate_text
from .state import GraphState


def _append_log(state: GraphState, message: str) -> None:
    state.setdefault("execution_log", [])
    state["execution_log"].append(message)
    if state.get("verbose"):
        print(message)


def _ensure_matplotlib_ready() -> None:
    if "MPLCONFIGDIR" not in os.environ:
        mpl_dir = Path(tempfile.gettempdir()) / "graduate_assist_mpl"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    if "XDG_CACHE_HOME" not in os.environ:
        cache_dir = Path(tempfile.gettempdir()) / "graduate_assist_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(cache_dir)
    os.environ.setdefault("MPLBACKEND", "Agg")


def _infer_column_kind(series: pd.Series, name: str) -> str:
    name_lower = name.lower()
    id_names = {"id", "subject", "participant", "user_id", "patient_id", "sample_id"}
    time_names = {"date", "time", "timestamp", "day", "week", "month", "year"}

    if name_lower in id_names:
        return "id"
    if any(token in name_lower for token in time_names):
        return "datetime"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    non_null = series.dropna()
    if non_null.empty:
        return "unknown"

    unique_count = int(non_null.nunique())
    unique_ratio = unique_count / float(len(non_null))

    if pd.api.types.is_numeric_dtype(series):
        if unique_count == 2:
            return "binary"
        if pd.api.types.is_integer_dtype(series) and non_null.min() >= 0:
            if unique_count <= 10:
                return "categorical"
            return "count"
        return "numeric"

    if unique_count == 2:
        return "binary"
    if unique_count <= 20 or unique_ratio <= 0.05:
        return "categorical"
    return "text"


def _load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _infer_schema(df: pd.DataFrame) -> Dict[str, Any]:
    schema: Dict[str, Any] = {
        "rows": int(df.shape[0]),
        "columns": {},
    }
    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        sample_values = series.dropna().head(3).tolist()
        non_null = int(series.notna().sum())
        unique_count = int(series.nunique(dropna=True))
        unique_ratio = float(unique_count / non_null) if non_null else 0.0
        kind = _infer_column_kind(series, col)
        schema["columns"][col] = {
            "dtype": dtype,
            "sample_values": sample_values,
            "non_null": non_null,
            "unique_count": unique_count,
            "unique_ratio": unique_ratio,
            "kind": kind,
        }
    return schema


def _detect_outliers(series: pd.Series, z_threshold: float = 3.0) -> List[int]:
    if series.empty:
        return []
    values = series.dropna().astype(float)
    if values.std(ddof=0) == 0:
        return []
    z_scores = (values - values.mean()) / values.std(ddof=0)
    return values.index[z_scores.abs() > z_threshold].tolist()


def _data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "missing_values": {},
        "duplicate_rows": int(df.duplicated().sum()),
        "outliers": {},
    }
    for col in df.columns:
        report["missing_values"][col] = int(df[col].isna().sum())
        if pd.api.types.is_numeric_dtype(df[col]):
            report["outliers"][col] = _detect_outliers(df[col])
    return report


def _format_quality_report_md(report: Dict[str, Any]) -> str:
    lines = ["# Data Quality Report", ""]
    lines.append("## Missing Values")
    for col, count in report["missing_values"].items():
        lines.append(f"- {col}: {count}")
    lines.append("")
    lines.append("## Duplicate Rows")
    lines.append(f"- {report['duplicate_rows']}")
    lines.append("")
    lines.append("## Outliers (z-score > 3)")
    if report["outliers"]:
        for col, idxs in report["outliers"].items():
            lines.append(f"- {col}: {idxs}")
    else:
        lines.append("- None detected")
    lines.append("")
    return "\n".join(lines)


def _choose_group_and_outcome_from_schema(schema: Dict[str, Any]) -> Tuple[str | None, str | None]:
    columns = list(schema.get("columns", {}).keys())
    group_col = None
    for col in columns:
        if col.lower() in {"group", "condition", "treatment"}:
            group_col = col
            break
    numeric_cols = []
    for col, meta in schema.get("columns", {}).items():
        dtype = str(meta.get("dtype", ""))
        if any(token in dtype for token in ["int", "float"]):
            numeric_cols.append(col)
    outcome_col = None
    for col in numeric_cols:
        if any(token in col.lower() for token in ["outcome", "growth", "response", "yield"]):
            outcome_col = col
            break
    if outcome_col is None and numeric_cols:
        outcome_col = numeric_cols[-1]
    return group_col, outcome_col


SUPPORTED_ANALYSES = {
    "welch_ttest",
    "mannwhitney_u",
    "paired_ttest",
    "wilcoxon_signed_rank",
    "anova",
    "kruskal_wallis",
    "friedman",
    "chi_square",
    "fisher_exact",
    "pearson_correlation",
    "spearman_correlation",
    "linear_regression",
    "descriptive_only",
}


def _pick_group_column(schema: Dict[str, Any]) -> str | None:
    columns = schema.get("columns", {})
    preferred = {"group", "condition", "treatment"}
    for col in columns:
        if col.lower() in preferred and columns[col].get("kind") in {"categorical", "binary"}:
            return col

    candidates = []
    for col, meta in columns.items():
        if meta.get("kind") in {"categorical", "binary"}:
            unique_count = int(meta.get("unique_count", 0))
            if 2 <= unique_count <= 20:
                candidates.append((unique_count, col))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def _pick_outcome_column(schema: Dict[str, Any]) -> str | None:
    columns = schema.get("columns", {})
    numeric_cols = [col for col, meta in columns.items() if meta.get("kind") in {"numeric", "count"}]
    preferred_tokens = {"outcome", "growth", "response", "yield", "score", "rate", "amount", "result"}

    for col in numeric_cols:
        if any(token in col.lower() for token in preferred_tokens):
            return col
    if numeric_cols:
        return numeric_cols[-1]

    categorical_cols = [col for col, meta in columns.items() if meta.get("kind") in {"categorical", "binary"}]
    for col in categorical_cols:
        if any(token in col.lower() for token in preferred_tokens):
            return col
    if categorical_cols:
        return categorical_cols[-1]
    return None


def _pick_numeric_predictor(schema: Dict[str, Any], outcome_col: str | None) -> str | None:
    columns = schema.get("columns", {})
    candidates = [
        col
        for col, meta in columns.items()
        if meta.get("kind") in {"numeric", "count"} and col != outcome_col
    ]
    if not candidates:
        return None
    preferred_tokens = {"predictor", "feature", "input", "x"}
    for col in candidates:
        if any(token in col.lower() for token in preferred_tokens):
            return col
    return candidates[0]


def _pick_id_column(schema: Dict[str, Any]) -> str | None:
    for col, meta in schema.get("columns", {}).items():
        if meta.get("kind") == "id":
            return col
    return None


def _skew_abs(series: pd.Series) -> float:
    try:
        return float(series.skew())
    except Exception:
        return 0.0


def _should_use_nonparametric(groups: List[pd.Series]) -> bool:
    for group in groups:
        if len(group) < 20:
            return True
        if abs(_skew_abs(group)) > 1.0:
            return True
    try:
        from scipy import stats

        for group in groups:
            if 3 <= len(group) <= 5000:
                stat, p_value = stats.shapiro(group)
                if p_value < 0.05:
                    return True
    except Exception:
        pass
    return False


def _paired_pivot(
    df: pd.DataFrame, pairing_col: str, group_col: str, outcome_col: str
) -> pd.DataFrame:
    data = df[[pairing_col, group_col, outcome_col]].dropna()
    pivot = data.pivot_table(index=pairing_col, columns=group_col, values=outcome_col, aggfunc="mean")
    return pivot.dropna()


def _select_rule_based_plan(
    df: pd.DataFrame, schema: Dict[str, Any], quality: Dict[str, Any]
) -> Dict[str, Any]:
    group_col = _pick_group_column(schema)
    outcome_col = _pick_outcome_column(schema)
    predictor_col = _pick_numeric_predictor(schema, outcome_col)
    pairing_col = _pick_id_column(schema)

    plan: Dict[str, Any] = {
        "analysis_type": "descriptive_only",
        "group_column": group_col,
        "outcome_column": outcome_col,
        "predictor_columns": [predictor_col] if predictor_col else [],
        "pairing_column": pairing_col,
        "assumptions": [],
        "validation": [],
        "steps": [],
        "selection_method": "rules",
        "selection_rationale": "",
    }

    if not outcome_col:
        plan["selection_rationale"] = "No outcome candidate detected; defaulting to descriptive statistics."
        plan["steps"] = ["Summarize distributions per column"]
        return plan

    outcome_kind = schema.get("columns", {}).get(outcome_col, {}).get("kind")

    if outcome_kind in {"categorical", "binary"}:
        if group_col and group_col != outcome_col:
            contingency = pd.crosstab(df[group_col], df[outcome_col])
            if contingency.shape == (2, 2):
                try:
                    from scipy import stats

                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency, correction=False)
                    if (expected < 5).any():
                        plan["analysis_type"] = "fisher_exact"
                        plan["selection_rationale"] = "2x2 contingency with low expected counts."
                    else:
                        plan["analysis_type"] = "chi_square"
                        plan["selection_rationale"] = "Categorical outcome with categorical predictor."
                except Exception:
                    plan["analysis_type"] = "chi_square"
                    plan["selection_rationale"] = "Categorical outcome with categorical predictor."
            else:
                plan["analysis_type"] = "chi_square"
                plan["selection_rationale"] = "Categorical outcome with categorical predictor."
            plan["assumptions"] = ["Observations are independent", "Expected cell counts are adequate"]
            plan["validation"] = ["Inspect contingency table for sparse cells"]
            plan["steps"] = ["Build contingency table", f"Run {plan['analysis_type']} test"]
            return plan

        plan["selection_rationale"] = "Categorical outcome without clear predictor; using descriptive statistics."
        plan["steps"] = ["Summarize category frequencies"]
        return plan

    if group_col and group_col != outcome_col:
        group_series = df[group_col].dropna()
        group_count = int(group_series.nunique())
        data_groups = []
        for _, subset in df[[group_col, outcome_col]].dropna().groupby(group_col):
            data_groups.append(subset[outcome_col])

        if pairing_col:
            pivot = _paired_pivot(df, pairing_col, group_col, outcome_col)
            if not pivot.empty:
                if pivot.shape[1] == 2:
                    if _should_use_nonparametric([pivot.iloc[:, 0], pivot.iloc[:, 1]]):
                        plan["analysis_type"] = "wilcoxon_signed_rank"
                        plan["selection_rationale"] = "Paired data with nonparametric distribution."
                    else:
                        plan["analysis_type"] = "paired_ttest"
                        plan["selection_rationale"] = "Paired data with two conditions."
                    plan["steps"] = ["Align paired observations", f"Run {plan['analysis_type']}"]
                    plan["assumptions"] = ["Paired observations across conditions"]
                    plan["validation"] = ["Check paired sample sizes per condition"]
                    return plan
                if pivot.shape[1] > 2:
                    plan["analysis_type"] = "friedman"
                    plan["selection_rationale"] = "Repeated measures across multiple conditions."
                    plan["steps"] = ["Align paired observations", "Run Friedman test"]
                    plan["assumptions"] = ["Paired observations across all conditions"]
                    plan["validation"] = ["Check repeated measures coverage per condition"]
                    return plan

        if group_count == 2:
            if _should_use_nonparametric(data_groups):
                plan["analysis_type"] = "mannwhitney_u"
                plan["selection_rationale"] = "Two groups with nonparametric distribution."
            else:
                plan["analysis_type"] = "welch_ttest"
                plan["selection_rationale"] = "Two independent groups with numeric outcome."
            plan["assumptions"] = ["Independent samples across groups"]
            plan["validation"] = ["Report group sizes and variance"]
            plan["steps"] = ["Compute group summary statistics", f"Run {plan['analysis_type']}"]
            return plan

        if group_count > 2:
            if _should_use_nonparametric(data_groups):
                plan["analysis_type"] = "kruskal_wallis"
                plan["selection_rationale"] = "Multiple groups with nonparametric distribution."
            else:
                plan["analysis_type"] = "anova"
                plan["selection_rationale"] = "Multiple groups with numeric outcome."
            plan["assumptions"] = ["Independent samples across groups"]
            plan["validation"] = ["Report group sizes and variance"]
            plan["steps"] = ["Compute group summary statistics", f"Run {plan['analysis_type']}"]
            return plan

    if predictor_col and predictor_col != outcome_col:
        outcome_series = df[outcome_col].dropna()
        predictor_series = df[predictor_col].dropna()
        if _should_use_nonparametric([outcome_series, predictor_series]):
            plan["analysis_type"] = "spearman_correlation"
            plan["selection_rationale"] = "Nonparametric relationship between numeric variables."
            plan["steps"] = ["Compute Spearman correlation"]
            plan["assumptions"] = ["Monotonic relationship between variables"]
        else:
            plan["analysis_type"] = "linear_regression"
            plan["selection_rationale"] = "Numeric outcome with numeric predictor."
            plan["steps"] = ["Fit simple linear regression", "Report slope and fit statistics"]
            plan["assumptions"] = ["Linear relationship between variables", "Residuals are approximately normal"]
        plan["validation"] = ["Inspect scatter plot and residuals"]
        return plan

    plan["selection_rationale"] = "No suitable grouping or predictor detected; using descriptive statistics."
    plan["steps"] = ["Summarize distributions per column"]
    return plan


def _extract_json_block(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _build_llm_plan_prompt(
    schema: Dict[str, Any],
    rule_plan: Dict[str, Any],
    context: Dict[str, Any],
) -> Tuple[str, str]:
    columns = {
        col: {
            "kind": meta.get("kind"),
            "dtype": meta.get("dtype"),
            "unique_count": meta.get("unique_count"),
            "unique_ratio": meta.get("unique_ratio"),
            "non_null": meta.get("non_null"),
        }
        for col, meta in schema.get("columns", {}).items()
    }
    system_prompt = (
        "You are a statistical method selector. Choose the most appropriate analysis from the allowed list. "
        "Return JSON only."
    )
    user_prompt = (
        "Select an analysis plan for the dataset.\n\n"
        f"Rows: {schema.get('rows')}\n"
        f"Columns: {json.dumps(columns, ensure_ascii=True)}\n"
        f"Rule-based suggestion: {json.dumps(rule_plan, ensure_ascii=True)}\n"
        f"Allowed analysis types: {sorted(SUPPORTED_ANALYSES)}\n\n"
        "Return JSON with keys: analysis_type, group_column, outcome_column, predictor_columns, "
        "pairing_column, assumptions, validation, steps, selection_rationale."
    )
    analysis_requests = context.get("analysis_requests") or []
    if analysis_requests:
        user_prompt += (
            "\nAnalysis requests:\n"
            + "\n".join(f"- {item}" for item in analysis_requests)
        )
    return system_prompt, user_prompt


def _parse_llm_plan(llm_text: str | None) -> Dict[str, Any] | None:
    if not llm_text:
        return None
    json_block = _extract_json_block(llm_text)
    if not json_block:
        return None
    try:
        return json.loads(json_block)
    except Exception:
        return None


def _validate_llm_plan(schema: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any] | None:
    if not isinstance(plan, dict):
        return None
    analysis_type = plan.get("analysis_type")
    if analysis_type not in SUPPORTED_ANALYSES:
        return None

    requirements = {
        "welch_ttest": {"group": True, "outcome": True},
        "mannwhitney_u": {"group": True, "outcome": True},
        "anova": {"group": True, "outcome": True},
        "kruskal_wallis": {"group": True, "outcome": True},
        "paired_ttest": {"group": True, "outcome": True, "pairing": True},
        "wilcoxon_signed_rank": {"group": True, "outcome": True, "pairing": True},
        "friedman": {"group": True, "outcome": True, "pairing": True},
        "chi_square": {"group": True, "outcome": True},
        "fisher_exact": {"group": True, "outcome": True},
        "pearson_correlation": {"outcome": True, "predictor": True},
        "spearman_correlation": {"outcome": True, "predictor": True},
        "linear_regression": {"outcome": True, "predictor": True},
        "descriptive_only": {},
    }

    columns = set(schema.get("columns", {}).keys())

    group_col = plan.get("group_column")
    if group_col and group_col not in columns:
        return None
    outcome_col = plan.get("outcome_column")
    if outcome_col and outcome_col not in columns:
        return None
    predictor_cols = plan.get("predictor_columns") or []
    if not isinstance(predictor_cols, list):
        return None
    for col in predictor_cols:
        if col not in columns:
            return None
    pairing_col = plan.get("pairing_column")
    if pairing_col and pairing_col not in columns:
        return None

    required = requirements.get(analysis_type, {})
    if required.get("group") and not group_col:
        return None
    if required.get("outcome") and not outcome_col:
        return None
    if required.get("predictor") and not predictor_cols:
        return None
    if required.get("pairing") and not pairing_col:
        return None

    cleaned = {
        "analysis_type": analysis_type,
        "group_column": group_col,
        "outcome_column": outcome_col,
        "predictor_columns": predictor_cols,
        "pairing_column": pairing_col,
        "assumptions": plan.get("assumptions") or [],
        "validation": plan.get("validation") or [],
        "steps": plan.get("steps") or [],
        "selection_rationale": plan.get("selection_rationale") or "",
    }
    return cleaned


def _grouped_numeric_arrays(
    df: pd.DataFrame, group_col: str, outcome_col: str
) -> List[Tuple[Any, np.ndarray]]:
    grouped = df[[group_col, outcome_col]].dropna().groupby(group_col)
    groups: List[Tuple[Any, np.ndarray]] = []
    for name, group in grouped:
        values = group[outcome_col].to_numpy()
        if len(values) > 0:
            groups.append((name, values))
    return groups


def _group_summary(groups: List[Tuple[Any, np.ndarray]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for name, values in groups:
        summary[str(name)] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
            "n": int(len(values)),
        }
    return summary

def _welch_ttest(group_a: np.ndarray, group_b: np.ndarray) -> Dict[str, Any]:
    mean_a = float(np.mean(group_a))
    mean_b = float(np.mean(group_b))
    var_a = float(np.var(group_a, ddof=1))
    var_b = float(np.var(group_b, ddof=1))
    n_a = len(group_a)
    n_b = len(group_b)
    t_num = mean_a - mean_b
    t_den = math.sqrt(var_a / n_a + var_b / n_b)
    t_stat = t_num / t_den if t_den != 0 else float("nan")
    df_num = (var_a / n_a + var_b / n_b) ** 2
    df_den = ((var_a / n_a) ** 2) / (n_a - 1) + ((var_b / n_b) ** 2) / (n_b - 1)
    df = df_num / df_den if df_den != 0 else float("nan")

    p_value = None
    try:
        from scipy import stats

        p_value = float(stats.t.sf(abs(t_stat), df) * 2)
    except Exception:
        p_value = None

    pooled_sd = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    cohen_d = (mean_a - mean_b) / pooled_sd if pooled_sd != 0 else float("nan")

    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "n_a": n_a,
        "n_b": n_b,
        "t_stat": t_stat,
        "df": df,
        "p_value": p_value,
        "cohen_d": cohen_d,
    }

def _extract_python_code(text: str) -> str:
    if "```" not in text:
        return text.strip()
    start = text.find("```")
    if start == -1:
        return text.strip()
    end = text.find("```", start + 3)
    if end == -1:
        return text[start + 3 :].strip()
    block = text[start + 3 : end]
    if block.startswith("python"):
        block = block[len("python") :].strip()
    return block.strip()


def _build_llm_execution_prompt(
    schema: Dict[str, Any],
    plan: Dict[str, Any],
    context: Dict[str, Any],
) -> Tuple[str, str]:
    system_prompt = (
        "You are a data analysis engineer. Return only executable Python code, no prose or markdown."
    )
    user_prompt = (
        "Write Python code to execute the analysis plan on a CSV dataset.\n"
        "Requirements:\n"
        "- DATA_PATH and FIGURES_DIR are provided as variables; do NOT redefine or overwrite them.\n"
        "- Read the CSV from DATA_PATH.\n"
        "- Use pandas/numpy/scipy/matplotlib (seaborn optional). statsmodels/pingouin/scikit-posthocs are optional; handle ImportError.\n"
        "- Save one or more figures to FIGURES_DIR (PNG files).\n"
        "- Use clear titles, axis labels, and tight layout; call plt.close() after saving.\n"
        "- Produce a RESULTS dict (JSON-serializable) with keys:\n"
        "  analysis_type, summary, test, figures, errors, warnings.\n"
        "- Avoid accessing private attributes (names starting with underscore) on third-party objects.\n"
        "- Do NOT call exit(), quit(), or sys.exit().\n"
        "- Do NOT write results.json or other files outside FIGURES_DIR.\n"
        "- If required columns are missing or analysis fails, add a message to errors.\n"
        "- Keep code deterministic; do not prompt for input.\n\n"
        f"Analysis plan: {json.dumps(plan, ensure_ascii=True)}\n"
        f"Data schema: {json.dumps(schema, ensure_ascii=True)}\n"
    )
    analysis_requests = context.get("analysis_requests") or []
    if analysis_requests:
        user_prompt += (
            "Additional analysis requests to address with figures or summaries:\n"
            + "\n".join(f"- {item}" for item in analysis_requests)
            + "\n"
        )
    return system_prompt, user_prompt


def _validate_llm_execution_code(code: str) -> List[str]:
    violations: List[str] = []
    if re.search(r"^\s*DATA_PATH\s*=", code, flags=re.MULTILINE):
        violations.append("Do not redefine DATA_PATH.")
    if re.search(r"^\s*FIGURES_DIR\s*=", code, flags=re.MULTILINE):
        violations.append("Do not redefine FIGURES_DIR.")
    if re.search(r"\b(exit|quit)\s*\(", code):
        violations.append("Do not call exit() or quit().")
    if re.search(r"\bsys\.exit\s*\(", code):
        violations.append("Do not call sys.exit().")
    if re.search(r"results\.json", code):
        violations.append("Do not write results.json files.")
    return violations


def _try_llm_execution(
    state: GraphState,
    plan: Dict[str, Any],
    schema: Dict[str, Any],
    artifacts_dir: Path,
    figures_dir: Path,
    max_attempts: int = 3,
) -> Tuple[Dict[str, Any] | None, List[str]]:
    context = state.get("experiment_context", {})
    system_prompt, user_prompt = _build_llm_execution_prompt(schema, plan, context)
    errors: List[str] = []
    retry_note = ""

    for attempt in range(1, max_attempts + 1):
        _append_log(state, f"[ExecutionAgent] LLM code generation attempt {attempt}")
        llm_text = generate_text(system_prompt, user_prompt + retry_note)
        if not llm_text:
            errors.append(f"LLM code generation attempt {attempt} returned no output.")
            continue

        code = _extract_python_code(llm_text)
        violations = _validate_llm_execution_code(code)
        if violations:
            errors.append(
                "LLM code violated execution constraints: " + "; ".join(violations)
            )
            retry_note = (
                "\n\nPrevious attempt violations:\n"
                + "\n".join(f"- {v}" for v in violations)
                + "\nReturn corrected Python code only."
            )
            continue
        script_path = artifacts_dir / f"execution_script_attempt_{attempt}.py"
        write_text(script_path, code + "\n")

        pre_existing = {p.name for p in figures_dir.glob("*.png")}
        exec_globals = {
            "DATA_PATH": state["raw_experimental_data"],
            "FIGURES_DIR": str(figures_dir),
        }
        try:
            exec(code, exec_globals)
            results = exec_globals.get("RESULTS")
            if not isinstance(results, dict):
                raise ValueError("RESULTS dict not found after execution.")

            post_existing = {p.name for p in figures_dir.glob("*.png")}
            new_figures = sorted(post_existing - pre_existing)
            if new_figures:
                results.setdefault("figures", [])
                results["figures"].extend([str(figures_dir / name) for name in new_figures])

            results.setdefault("analysis_type", plan.get("analysis_type"))
            results.setdefault("errors", [])
            results.setdefault("warnings", [])
            results["llm_execution"] = {
                "attempts": attempt,
                "code_path": str(script_path),
            }
            return results, errors
        except Exception as exc:
            tb = traceback.format_exc(limit=8)
            errors.append(f"LLM execution attempt {attempt} failed: {exc}")
            retry_note = (
                "\n\nPrevious attempt error:\n"
                f"{tb}\n"
                "Fix the error and return corrected Python code only."
            )

    return None, errors


def context_agent_node(state: GraphState) -> GraphState:
    """
    Parse experiment description into structured context.

    Inputs:
    - raw_experiment_description

    Outputs:
    - experiment_context (persisted to context.json)
    """
    raw = state.get("raw_experiment_description", "").strip()
    _append_log(state, "[ContextAgent] start parsing experiment description")
    context = {
        "objective": "",
        "hypothesis": "",
        "variables": {
            "independent": [],
            "dependent": [],
            "controls": [],
        },
        "constraints": [],
        "analysis_requests": [],
        "raw": raw,
    }

    for line in raw.splitlines():
        lower = line.lower()
        if lower.startswith("objective:"):
            context["objective"] = line.split(":", 1)[1].strip()
        elif lower.startswith("hypothesis:"):
            context["hypothesis"] = line.split(":", 1)[1].strip()
        elif lower.startswith("independent variables:"):
            values = line.split(":", 1)[1].strip()
            context["variables"]["independent"] = [v.strip() for v in values.split(",") if v.strip()]
        elif lower.startswith("dependent variables:"):
            values = line.split(":", 1)[1].strip()
            context["variables"]["dependent"] = [v.strip() for v in values.split(",") if v.strip()]
        elif lower.startswith("controls:"):
            values = line.split(":", 1)[1].strip()
            context["variables"]["controls"] = [v.strip() for v in values.split(",") if v.strip()]
        elif lower.startswith("constraints:"):
            values = line.split(":", 1)[1].strip()
            context["constraints"] = [v.strip() for v in values.split(",") if v.strip()]
        elif lower.startswith("analysis focus:") or lower.startswith("analysis requests:"):
            values = line.split(":", 1)[1].strip()
            context["analysis_requests"] = [v.strip() for v in values.split(";") if v.strip()]

    artifacts_dir = artifacts_root(state["artifacts_path"], state["run_id"])
    write_json(artifacts_dir / "context.json", context)
    state["experiment_context"] = context
    _append_log(
        state,
        f"[ContextAgent] parsed objective={context['objective']!r} hypothesis={context['hypothesis']!r}",
    )
    return state


def data_understanding_agent_node(state: GraphState) -> GraphState:
    """
    Load experimental data, infer schema, and assess data quality.

    Inputs:
    - experiment_context
    - raw_experimental_data

    Outputs:
    - data_schema
    - data_quality_report
    """
    _append_log(state, "[DataUnderstandingAgent] start data load and quality checks")
    df = _load_data(state["raw_experimental_data"])
    schema = _infer_schema(df)
    quality = _data_quality_report(df)

    artifacts_dir = artifacts_root(state["artifacts_path"], state["run_id"])
    write_yaml(artifacts_dir / "schema.yaml", schema)
    write_markdown(artifacts_dir / "data_quality_report.md", _format_quality_report_md(quality))

    state["data_schema"] = schema
    state["data_quality_report"] = quality
    _append_log(
        state,
        f"[DataUnderstandingAgent] rows={schema.get('rows')} cols={len(schema.get('columns', {}))} "
        f"duplicate_rows={quality.get('duplicate_rows')}",
    )
    return state


def analysis_planning_agent_node(state: GraphState) -> GraphState:
    """
    Decide analysis steps, statistical assumptions, and validation strategy.

    Inputs:
    - experiment_context
    - data_schema
    - data_quality_report

    Outputs:
    - analysis_plan (persisted to analysis_plan.yaml)
    """
    _append_log(state, "[AnalysisPlanningAgent] start analysis plan selection")
    schema = state.get("data_schema", {})
    quality = state.get("data_quality_report", {})
    df = _load_data(state["raw_experimental_data"])

    rule_plan = _select_rule_based_plan(df, schema, quality)
    plan = rule_plan

    context = state.get("experiment_context", {})
    system_prompt, user_prompt = _build_llm_plan_prompt(schema, rule_plan, context)
    _append_log(state, f"[AnalysisPlanningAgent] system_prompt={system_prompt!r}")
    _append_log(state, f"[AnalysisPlanningAgent] user_prompt={user_prompt!r}")
    llm_text = generate_text(system_prompt, user_prompt)
    if llm_text:
        _append_log(state, f"[AnalysisPlanningAgent] llm_response={llm_text.strip()!r}")
    else:
        _append_log(state, "[AnalysisPlanningAgent] llm unavailable; using rule-based plan")
    llm_plan_raw = _parse_llm_plan(llm_text)
    llm_plan = _validate_llm_plan(schema, llm_plan_raw) if llm_plan_raw else None
    if llm_plan:
        llm_plan["selection_method"] = "llm"
        llm_plan["rule_based_plan"] = {
            "analysis_type": rule_plan.get("analysis_type"),
            "group_column": rule_plan.get("group_column"),
            "outcome_column": rule_plan.get("outcome_column"),
            "predictor_columns": rule_plan.get("predictor_columns"),
            "pairing_column": rule_plan.get("pairing_column"),
            "selection_rationale": rule_plan.get("selection_rationale"),
        }
        plan = llm_plan

    artifacts_dir = artifacts_root(state["artifacts_path"], state["run_id"])
    write_yaml(artifacts_dir / "analysis_plan.yaml", plan)

    state["analysis_plan"] = plan
    _append_log(
        state,
        f"[AnalysisPlanningAgent] analysis_type={plan['analysis_type']} "
        f"group_column={plan.get('group_column')!r} outcome_column={plan.get('outcome_column')!r} "
        f"selection_method={plan.get('selection_method')!r}",
    )
    return state


def execution_agent_node(state: GraphState) -> GraphState:
    """
    Execute analysis plan with robust error handling.

    Inputs:
    - analysis_plan
    - raw_experimental_data

    Outputs:
    - execution_results (persisted to results.json and figures/)
    """
    _append_log(state, "[ExecutionAgent] start execution")
    plan = state.get("analysis_plan", {})
    schema = state.get("data_schema", {})
    df = _load_data(state["raw_experimental_data"])
    analysis_type = plan.get("analysis_type")
    results: Dict[str, Any] = {
        "analysis_type": analysis_type,
        "errors": [],
        "warnings": [],
    }

    artifacts_dir = artifacts_root(state["artifacts_path"], state["run_id"])
    figures_dir = artifacts_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    llm_results, llm_errors = _try_llm_execution(state, plan, schema, artifacts_dir, figures_dir)
    if llm_results is not None:
        if llm_errors:
            llm_results.setdefault("warnings", [])
            llm_results["warnings"].extend(llm_errors)
        results = llm_results
    else:
        if llm_errors:
            _append_log(
                state,
                f"[ExecutionAgent] LLM execution failed after retries; falling back. errors={llm_errors}",
            )
            results["warnings"].extend(llm_errors)

    group_col = plan.get("group_column")
    outcome_col = plan.get("outcome_column")
    predictor_cols = plan.get("predictor_columns") or []
    pairing_col = plan.get("pairing_column")

    if llm_results is not None:
        log_text = "\n".join(state.get("execution_log", []))
        write_text(artifacts_dir / "execution_log.txt", log_text)
        write_json(artifacts_dir / "results.json", results)
        state["execution_results"] = results
        if results.get("errors"):
            _append_log(state, f"[ExecutionAgent] errors={results['errors']}")
        else:
            _append_log(state, "[ExecutionAgent] completed execution without errors")
        return state

    if analysis_type in {
        "welch_ttest",
        "mannwhitney_u",
        "anova",
        "kruskal_wallis",
    }:
        if not group_col or not outcome_col:
            results["errors"].append("Missing group or outcome column.")
        else:
            groups = _grouped_numeric_arrays(df, group_col, outcome_col)
            results["summary"] = _group_summary(groups)
            if analysis_type in {"welch_ttest", "mannwhitney_u"} and len(groups) != 2:
                results["errors"].append("Two-group analysis requires exactly 2 groups.")
            elif analysis_type in {"anova", "kruskal_wallis"} and len(groups) < 2:
                results["errors"].append("Multi-group analysis requires at least 2 groups.")
            else:
                try:
                    from scipy import stats

                    if analysis_type == "welch_ttest":
                        (name_a, values_a), (name_b, values_b) = groups
                        results["group_a"] = name_a
                        results["group_b"] = name_b
                        results["test"] = _welch_ttest(values_a, values_b)
                    elif analysis_type == "mannwhitney_u":
                        (name_a, values_a), (name_b, values_b) = groups
                        stat, p_value = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
                        results["test"] = {
                            "u_stat": float(stat),
                            "p_value": float(p_value),
                        }
                        results["group_a"] = name_a
                        results["group_b"] = name_b
                    elif analysis_type == "anova":
                        stat, p_value = stats.f_oneway(*[values for _, values in groups])
                        results["test"] = {
                            "f_stat": float(stat),
                            "p_value": float(p_value),
                            "groups": [str(name) for name, _ in groups],
                        }
                    elif analysis_type == "kruskal_wallis":
                        stat, p_value = stats.kruskal(*[values for _, values in groups])
                        results["test"] = {
                            "h_stat": float(stat),
                            "p_value": float(p_value),
                            "groups": [str(name) for name, _ in groups],
                        }
                except Exception as exc:
                    results["errors"].append(f"Analysis failed: {exc}")

            if not results["errors"]:
                try:
                    _ensure_matplotlib_ready()
                    import matplotlib.pyplot as plt

                    fig_path = figures_dir / "group_comparison.png"
                    plt.figure(figsize=(6, 4))
                    df.boxplot(column=outcome_col, by=group_col)
                    plt.title(f"{outcome_col} by {group_col}")
                    plt.suptitle("")
                    plt.xlabel(group_col)
                    plt.ylabel(outcome_col)
                    plt.tight_layout()
                    plt.savefig(fig_path)
                    plt.close()
                    results["figures"] = [str(fig_path)]
                except Exception as exc:
                    results["warnings"].append(f"Figure generation failed: {exc}")

    elif analysis_type in {"paired_ttest", "wilcoxon_signed_rank", "friedman"}:
        if not pairing_col or not group_col or not outcome_col:
            results["errors"].append("Missing pairing, group, or outcome column.")
        else:
            pivot = _paired_pivot(df, pairing_col, group_col, outcome_col)
            if pivot.empty:
                results["errors"].append("No paired observations available for analysis.")
            else:
                try:
                    from scipy import stats

                    if analysis_type in {"paired_ttest", "wilcoxon_signed_rank"} and pivot.shape[1] != 2:
                        results["errors"].append("Paired two-condition analysis requires exactly 2 conditions.")
                    elif analysis_type == "friedman" and pivot.shape[1] < 3:
                        results["errors"].append("Friedman test requires at least 3 conditions.")
                    else:
                        if analysis_type == "paired_ttest":
                            stat, p_value = stats.ttest_rel(pivot.iloc[:, 0], pivot.iloc[:, 1])
                            results["test"] = {
                                "t_stat": float(stat),
                                "p_value": float(p_value),
                                "conditions": list(map(str, pivot.columns)),
                            }
                        elif analysis_type == "wilcoxon_signed_rank":
                            stat, p_value = stats.wilcoxon(pivot.iloc[:, 0], pivot.iloc[:, 1])
                            results["test"] = {
                                "w_stat": float(stat),
                                "p_value": float(p_value),
                                "conditions": list(map(str, pivot.columns)),
                            }
                        else:
                            arrays = [pivot[col].to_numpy() for col in pivot.columns]
                            stat, p_value = stats.friedmanchisquare(*arrays)
                            results["test"] = {
                                "chi2_stat": float(stat),
                                "p_value": float(p_value),
                                "conditions": list(map(str, pivot.columns)),
                            }
                        results["summary"] = {
                            str(col): {
                                "mean": float(pivot[col].mean()),
                                "std": float(pivot[col].std(ddof=1)) if len(pivot[col]) > 1 else float("nan"),
                                "n": int(pivot[col].shape[0]),
                            }
                            for col in pivot.columns
                        }
                except Exception as exc:
                    results["errors"].append(f"Analysis failed: {exc}")

    elif analysis_type in {"chi_square", "fisher_exact"}:
        if not group_col or not outcome_col:
            results["errors"].append("Missing group or outcome column.")
        else:
            contingency = pd.crosstab(df[group_col], df[outcome_col])
            results["contingency_table"] = contingency.to_dict()
            try:
                from scipy import stats

                if analysis_type == "fisher_exact":
                    if contingency.shape != (2, 2):
                        results["errors"].append("Fisher exact test requires a 2x2 table.")
                    else:
                        stat, p_value = stats.fisher_exact(contingency.to_numpy())
                        results["test"] = {
                            "odds_ratio": float(stat),
                            "p_value": float(p_value),
                        }
                else:
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                    results["test"] = {
                        "chi2_stat": float(chi2),
                        "p_value": float(p_value),
                        "dof": int(dof),
                    }
                    results["expected_counts"] = expected.tolist()
            except Exception as exc:
                results["errors"].append(f"Analysis failed: {exc}")

    elif analysis_type in {"pearson_correlation", "spearman_correlation", "linear_regression"}:
        if not outcome_col or not predictor_cols:
            results["errors"].append("Missing outcome or predictor column.")
        else:
            predictor_col = predictor_cols[0]
            paired = df[[predictor_col, outcome_col]].dropna()
            if paired.empty:
                results["errors"].append("No paired observations available for analysis.")
            else:
                x = paired[predictor_col].to_numpy()
                y = paired[outcome_col].to_numpy()
                try:
                    from scipy import stats

                    if analysis_type == "pearson_correlation":
                        stat, p_value = stats.pearsonr(x, y)
                        results["test"] = {
                            "r": float(stat),
                            "p_value": float(p_value),
                        }
                    elif analysis_type == "spearman_correlation":
                        stat, p_value = stats.spearmanr(x, y)
                        results["test"] = {
                            "rho": float(stat),
                            "p_value": float(p_value),
                        }
                    else:
                        regression = stats.linregress(x, y)
                        results["test"] = {
                            "slope": float(regression.slope),
                            "intercept": float(regression.intercept),
                            "r_value": float(regression.rvalue),
                            "p_value": float(regression.pvalue),
                            "stderr": float(regression.stderr),
                        }
                    results["summary"] = {
                        "n": int(len(paired)),
                        "predictor": predictor_col,
                        "outcome": outcome_col,
                    }
                except Exception as exc:
                    results["errors"].append(f"Analysis failed: {exc}")

    else:
        results["summary"] = df.describe(include="all").to_dict()

    log_text = "\n".join(state.get("execution_log", []))
    write_text(artifacts_dir / "execution_log.txt", log_text)
    write_json(artifacts_dir / "results.json", results)

    state["execution_results"] = results
    if results.get("errors"):
        _append_log(state, f"[ExecutionAgent] errors={results['errors']}")
    else:
        _append_log(state, "[ExecutionAgent] completed execution without errors")
    return state


def reporting_agent_node(state: GraphState) -> GraphState:
    """
    Generate a conservative academic report.

    Inputs:
    - experiment_context
    - analysis_plan
    - execution_results

    Outputs:
    - report_text (persisted to report.md)
    """
    _append_log(state, "[ReportingAgent] start report generation")
    context = state.get("experiment_context", {})
    plan = state.get("analysis_plan", {})
    results = state.get("execution_results", {})

    system_prompt = (
        "You are an academic analyst. Write conservative, publication-ready prose. "
        "Avoid overstating significance and clearly state assumptions and limitations."
    )
    user_prompt = (
        "Write a concise report based on the following context.\n\n"
        f"Objective: {context.get('objective')}\n"
        f"Hypothesis: {context.get('hypothesis')}\n"
        f"Analysis plan: {json.dumps(plan, ensure_ascii=True)}\n"
        f"Execution results: {json.dumps(results, ensure_ascii=True)}\n"
    )

    _append_log(state, f"[ReportingAgent] system_prompt={system_prompt!r}")
    _append_log(state, f"[ReportingAgent] user_prompt={user_prompt!r}")
    llm_text = generate_text(system_prompt, user_prompt)
    if llm_text:
        _append_log(state, f"[ReportingAgent] llm_response={llm_text.strip()!r}")
        report_text = llm_text.strip() + "\n"
    else:
        _append_log(state, "[ReportingAgent] llm unavailable; using template report")
        lines = ["# Analysis Report", ""]
        lines.append("## Objective")
        lines.append(context.get("objective") or "Objective not specified.")
        lines.append("")
        lines.append("## Hypothesis")
        lines.append(context.get("hypothesis") or "Hypothesis not specified.")
        lines.append("")
        lines.append("## Methods")
        lines.append(f"Analysis type: {plan.get('analysis_type', 'unknown')}")
        for assumption in plan.get("assumptions", []):
            lines.append(f"- Assumption: {assumption}")
        lines.append("")
        lines.append("## Results")

        if results.get("errors"):
            lines.append("Analysis could not be completed due to the following issues:")
            for err in results.get("errors", []):
                lines.append(f"- {err}")
        elif plan.get("analysis_type") in {
            "welch_ttest",
            "mannwhitney_u",
            "anova",
            "kruskal_wallis",
        } and "test" in results:
            analysis_type = plan.get("analysis_type")
            if analysis_type == "welch_ttest":
                test = results["test"]
                lines.append(
                    f"Welch's t-test between {results.get('group_a')} and {results.get('group_b')} yielded "
                    f"t = {test['t_stat']:.3f}, df = {test['df']:.2f}"
                    + (f", p = {test['p_value']:.4f}." if test.get("p_value") is not None else ".")
                )
                lines.append(f"Effect size (Cohen's d): {test['cohen_d']:.3f}")
            elif analysis_type == "mannwhitney_u":
                test = results["test"]
                lines.append(
                    "Mann-Whitney U test yielded "
                    f"U = {test['u_stat']:.3f}, p = {test['p_value']:.4f}."
                )
            elif analysis_type == "anova":
                test = results["test"]
                lines.append(
                    f"One-way ANOVA yielded F = {test['f_stat']:.3f}, p = {test['p_value']:.4f}."
                )
            else:
                test = results["test"]
                lines.append(
                    f"Kruskal-Wallis test yielded H = {test['h_stat']:.3f}, p = {test['p_value']:.4f}."
                )
            lines.append("Group summaries:")
            for group, summary in results.get("summary", {}).items():
                lines.append(
                    f"- {group}: mean = {summary['mean']:.3f}, SD = {summary['std']:.3f}, n = {summary['n']}"
                )
        elif plan.get("analysis_type") in {"paired_ttest", "wilcoxon_signed_rank", "friedman"} and "test" in results:
            analysis_type = plan.get("analysis_type")
            test = results["test"]
            if analysis_type == "paired_ttest":
                lines.append(
                    f"Paired t-test yielded t = {test['t_stat']:.3f}, p = {test['p_value']:.4f}."
                )
            elif analysis_type == "wilcoxon_signed_rank":
                lines.append(
                    f"Wilcoxon signed-rank test yielded W = {test['w_stat']:.3f}, p = {test['p_value']:.4f}."
                )
            else:
                lines.append(
                    f"Friedman test yielded chi2 = {test['chi2_stat']:.3f}, p = {test['p_value']:.4f}."
                )
            lines.append("Condition summaries:")
            for group, summary in results.get("summary", {}).items():
                lines.append(
                    f"- {group}: mean = {summary['mean']:.3f}, SD = {summary['std']:.3f}, n = {summary['n']}"
                )
        elif plan.get("analysis_type") in {"chi_square", "fisher_exact"} and "test" in results:
            test = results["test"]
            if plan.get("analysis_type") == "chi_square":
                lines.append(
                    f"Chi-square test yielded chi2 = {test['chi2_stat']:.3f}, p = {test['p_value']:.4f}."
                )
            else:
                lines.append(
                    f"Fisher exact test yielded odds ratio = {test['odds_ratio']:.3f}, "
                    f"p = {test['p_value']:.4f}."
                )
        elif plan.get("analysis_type") in {
            "pearson_correlation",
            "spearman_correlation",
            "linear_regression",
        } and "test" in results:
            test = results["test"]
            if plan.get("analysis_type") == "pearson_correlation":
                lines.append(f"Pearson correlation r = {test['r']:.3f}, p = {test['p_value']:.4f}.")
            elif plan.get("analysis_type") == "spearman_correlation":
                lines.append(f"Spearman correlation rho = {test['rho']:.3f}, p = {test['p_value']:.4f}.")
            else:
                lines.append(
                    f"Linear regression slope = {test['slope']:.3f}, intercept = {test['intercept']:.3f}, "
                    f"p = {test['p_value']:.4f}."
                )
        else:
            lines.append("Descriptive statistics available in results artifact.")

        lines.append("")
        lines.append("## Limitations")
        lines.append("- Results are contingent on data quality checks and assumptions listed above.")
        lines.append("- Statistical significance should be interpreted conservatively with domain context.")

        report_text = "\n".join(lines) + "\n"

    prior_comment = (state.get("reviewer_comment") or "").strip()
    followup_requested = bool(prior_comment) and int(state.get("rebuttal_rounds", 0)) > 0
    if state.get("auto_reviewer_comment") and (not prior_comment or followup_requested):
        _append_log(state, "[ReportingAgent] generating self-review comment")
        if results.get("errors"):
            comment_text = (
                "Please reanalyze the data; the execution reported errors that could affect the "
                "validity of the conclusions."
            )
        else:
            review_system_prompt = (
                "You are a critical but fair scientific reviewer. "
                "Write a concise review comment that improves rigor and clarity."
            )
            review_user_prompt = (
                "Provide 2 to 4 sentences of reviewer feedback in plain text. "
                "Focus on methodological clarity, assumptions, limitations, and interpretation. "
                "Request reanalysis only if there are clear errors or inconsistencies.\n\n"
                f"Objective: {context.get('objective')}\n"
                f"Hypothesis: {context.get('hypothesis')}\n"
                f"Analysis plan: {json.dumps(plan, ensure_ascii=True)}\n"
                f"Execution results: {json.dumps(results, ensure_ascii=True)}\n"
                f"Report: {report_text}\n"
            )
            if followup_requested:
                review_user_prompt += (
                    "Previous reviewer comment:\n"
                    f"{prior_comment}\n\n"
                    "Provide a new comment that adds a different perspective or requests different clarifications. "
                    "Do not repeat the same points."
                )
            _append_log(state, f"[ReportingAgent] review_system_prompt={review_system_prompt!r}")
            _append_log(state, f"[ReportingAgent] review_user_prompt={review_user_prompt!r}")
            review_text = generate_text(review_system_prompt, review_user_prompt)
            if review_text:
                _append_log(state, f"[ReportingAgent] review_llm_response={review_text.strip()!r}")
                comment_text = review_text.strip()
            else:
                _append_log(state, "[ReportingAgent] reviewer LLM unavailable; using template comment")
                comment_text = (
                    "Please clarify the analysis assumptions, justify the method choice, and "
                    "temper interpretations with stated limitations."
                )

        state["reviewer_comment"] = comment_text

    artifacts_dir = artifacts_root(state["artifacts_path"], state["run_id"])
    write_markdown(artifacts_dir / "report.md", report_text)
    if state.get("reviewer_comment"):
        write_markdown(artifacts_dir / "reviewer_comment.md", state["reviewer_comment"].strip() + "\n")

    state["report_text"] = report_text
    return state


def _build_search_query(context: Dict[str, Any], plan: Dict[str, Any]) -> str:
    objective = (context.get("objective") or "").strip()
    hypothesis = (context.get("hypothesis") or "").strip()
    analysis_type = (plan.get("analysis_type") or "").replace("_", " ").strip()
    terms = " ".join(token for token in [objective, hypothesis] if token)
    if analysis_type and analysis_type.lower() not in terms.lower():
        terms = f"{terms} {analysis_type}".strip()
    return terms or "experimental data analysis"


def _safe_api_key(env_name: str) -> str:
    value = os.getenv(env_name, "").strip()
    if not value or "your-" in value:
        return ""
    return value


def _fetch_openalex(query: str) -> List[Dict[str, Any]]:
    params = {"search": query, "per-page": "5"}
    api_key = _safe_api_key("OPENALEX_API_KEY")
    if api_key:
        params["api_key"] = api_key
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "graduate-assist/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    results = []
    for item in payload.get("results", []):
        venue = ""
        host = item.get("host_venue") or {}
        venue = host.get("display_name") or ""
        landing = ""
        primary_location = item.get("primary_location") or {}
        if primary_location:
            landing = primary_location.get("landing_page_url") or ""
        results.append(
            {
                "source": "openalex",
                "title": item.get("title"),
                "year": item.get("publication_year"),
                "venue": venue,
                "url": landing or item.get("doi") or item.get("id"),
            }
        )
    return results


def _fetch_semantic_scholar(query: str) -> List[Dict[str, Any]]:
    params = {
        "query": query,
        "limit": "5",
        "fields": "title,year,abstract,url,venue,authors,citationCount",
    }
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(params)
    headers = {"User-Agent": "graduate-assist/1.0"}
    api_key = _safe_api_key("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    results = []
    for item in payload.get("data", []):
        results.append(
            {
                "source": "semantic_scholar",
                "title": item.get("title"),
                "year": item.get("year"),
                "venue": item.get("venue"),
                "url": item.get("url"),
                "abstract": item.get("abstract"),
                "citation_count": item.get("citationCount"),
            }
        )
    return results


def _format_web_context_comment(
    query: str,
    openalex_hits: List[Dict[str, Any]],
    semantic_hits: List[Dict[str, Any]],
    report_text: str,
) -> str:
    findings = {
        "query": query,
        "openalex": openalex_hits,
        "semantic_scholar": semantic_hits,
    }
    system_prompt = (
        "You are a scientific reviewer. Summarize whether the analysis results add new meaning "
        "relative to related literature, using cautious language."
    )
    user_prompt = (
        "Based on the report and search results, write 3-5 sentences assessing novelty and context. "
        "Mention if findings align with existing work or appear incremental. Avoid definitive claims.\n\n"
        f"Report: {report_text}\n"
        f"Search results: {json.dumps(findings, ensure_ascii=True)}\n"
    )
    llm_text = generate_text(system_prompt, user_prompt)
    if llm_text:
        return llm_text.strip() + "\n"

    if openalex_hits or semantic_hits:
        return (
            "Related literature was found in web searches. The results appear broadly consistent with "
            "existing work, so novelty should be framed as incremental unless further domain review "
            "supports a stronger claim.\n"
        )
    return (
        "No closely matching literature was returned by the web search. Novelty remains uncertain and "
        "should be validated with a deeper, domain-specific review.\n"
    )


def web_context_agent_node(state: GraphState) -> GraphState:
    """
    Append a web-context comment using OpenAlex and Semantic Scholar searches.

    Inputs:
    - experiment_context
    - analysis_plan
    - execution_results
    - report_text

    Outputs:
    - web_search_results (persisted to web_search_results.json)
    - web_context_comment (persisted to web_context_comment.md and appended to report.md)
    """
    _append_log(state, "[WebContextAgent] start web-context assessment")
    context = state.get("experiment_context", {})
    plan = state.get("analysis_plan", {})
    report_text = state.get("report_text", "")

    query = _build_search_query(context, plan)
    openalex_hits: List[Dict[str, Any]] = []
    semantic_hits: List[Dict[str, Any]] = []
    errors: List[str] = []

    try:
        openalex_hits = _fetch_openalex(query)
    except Exception as exc:
        errors.append(f"OpenAlex search failed: {exc}")

    try:
        semantic_hits = _fetch_semantic_scholar(query)
    except Exception as exc:
        errors.append(f"Semantic Scholar search failed: {exc}")

    comment_text = _format_web_context_comment(query, openalex_hits, semantic_hits, report_text)
    if errors:
        comment_text += "Search limitations: " + "; ".join(errors) + "\n"

    lines = ["# Web Context Comment", ""]
    lines.append("## Query")
    lines.append(query)
    lines.append("")
    lines.append("## Findings")
    if openalex_hits:
        lines.append("OpenAlex:")
        for item in openalex_hits:
            title = item.get("title") or "Untitled"
            year = item.get("year") or "n.d."
            venue = item.get("venue") or "unknown venue"
            url = item.get("url") or ""
            lines.append(f"- {title} ({year}, {venue}) {url}".strip())
    if semantic_hits:
        lines.append("Semantic Scholar:")
        for item in semantic_hits:
            title = item.get("title") or "Untitled"
            year = item.get("year") or "n.d."
            venue = item.get("venue") or "unknown venue"
            url = item.get("url") or ""
            lines.append(f"- {title} ({year}, {venue}) {url}".strip())
    if not openalex_hits and not semantic_hits:
        lines.append("No results returned.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append(comment_text.strip())
    lines.append("")
    md_comment = "\n".join(lines)

    artifacts_dir = artifacts_root(state["artifacts_path"], state["run_id"])
    write_json(
        artifacts_dir / "web_search_results.json",
        {
            "query": query,
            "openalex": openalex_hits,
            "semantic_scholar": semantic_hits,
            "errors": errors,
        },
    )
    write_markdown(artifacts_dir / "web_context_comment.md", md_comment)

    appended_report = report_text.rstrip() + "\n\n## Web Context Check\n\n" + comment_text.strip() + "\n"
    write_markdown(artifacts_dir / "report.md", appended_report)

    state["web_search_results"] = {
        "query": query,
        "openalex": openalex_hits,
        "semantic_scholar": semantic_hits,
        "errors": errors,
    }
    state["web_context_comment"] = comment_text
    state["report_text"] = appended_report
    _append_log(state, "[WebContextAgent] completed web-context assessment")
    return state


def rebuttal_agent_node(state: GraphState) -> GraphState:
    """
    Process reviewer comments and decide response strategy.

    Inputs:
    - reviewer_comment
    - experiment_context
    - analysis_plan
    - execution_results
    - report_text

    Outputs:
    - rebuttal_decision
    - rebuttal_response_text
    """
    _append_log(state, "[RebuttalAgent] start rebuttal drafting")
    comment = (state.get("reviewer_comment") or "").strip()
    decision = {
        "strategy": "explain_only",
        "severity": "minor",
        "rationale": "",
    }

    override = (state.get("rebuttal_strategy_override") or "").strip().lower()
    if override in {"explain_only", "concede_and_limit", "reanalyze"}:
        decision["strategy"] = override
        decision["severity"] = "major" if override == "reanalyze" else "moderate"
        decision["rationale"] = "Strategy overridden by runtime configuration."
    else:
        lower = comment.lower()
        if any(token in lower for token in ["reanalyze", "redo", "incorrect analysis", "major error", "fatal"]):
            decision["strategy"] = "reanalyze"
            decision["severity"] = "major"
            decision["rationale"] = "Reviewer requests reanalysis or flags major errors."
        elif any(token in lower for token in ["overstated", "exaggerated", "unsupported"]):
            decision["strategy"] = "concede_and_limit"
            decision["severity"] = "moderate"
            decision["rationale"] = "Reviewer flags overstatement; interpretation will be constrained."
        else:
            decision["rationale"] = "Clarify methods and assumptions without reanalysis."

    system_prompt = (
        "You are preparing a rebuttal response for a scientific reviewer. "
        "Be polite, concise, and academically conservative."
    )
    user_prompt = (
        "Draft a rebuttal response consistent with the decision strategy.\n\n"
        f"Decision: {json.dumps(decision, ensure_ascii=True)}\n"
        f"Reviewer comment: {comment}\n"
    )

    _append_log(state, f"[RebuttalAgent] system_prompt={system_prompt!r}")
    _append_log(state, f"[RebuttalAgent] user_prompt={user_prompt!r}")
    llm_text = generate_text(system_prompt, user_prompt)
    if llm_text:
        _append_log(state, f"[RebuttalAgent] llm_response={llm_text.strip()!r}")
        response_text = llm_text.strip() + "\n"
    else:
        _append_log(state, "[RebuttalAgent] llm unavailable; using template response")
        response_lines = ["# Rebuttal Response", ""]
        response_lines.append("## Reviewer Comment")
        response_lines.append(comment or "No reviewer comment provided.")
        response_lines.append("")
        response_lines.append("## Response")
        if decision["strategy"] == "reanalyze":
            response_lines.append("We acknowledge the concern and will rerun the analysis with updated assumptions.")
        elif decision["strategy"] == "concede_and_limit":
            response_lines.append("We agree that the prior interpretation was too strong and will revise it.")
        else:
            response_lines.append("We clarify the analysis steps, assumptions, and limitations as requested.")

        response_lines.append("")
        response_lines.append("## Action")
        response_lines.append(f"Strategy: {decision['strategy']}")

        response_text = "\n".join(response_lines) + "\n"

    artifacts_dir = artifacts_root(state["artifacts_path"], state["run_id"])
    write_yaml(artifacts_dir / "rebuttal_decision.yaml", decision)
    write_markdown(artifacts_dir / "rebuttal_response.md", response_text)

    state["rebuttal_rounds"] = int(state.get("rebuttal_rounds", 0)) + 1
    state["rebuttal_decision"] = decision
    state["rebuttal_response_text"] = response_text
    max_rounds = int(state.get("max_rebuttal_rounds", 2))
    if decision["strategy"] == "reanalyze" and state["rebuttal_rounds"] >= max_rounds:
        # Stop the loop after two rebuttal cycles.
        state["reviewer_comment"] = None
        state["auto_reviewer_comment"] = False
    return state
