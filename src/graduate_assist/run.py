from __future__ import annotations

import argparse
import datetime as dt
import os
import uuid
from pathlib import Path

from .artifacts import artifacts_root, write_text
from .graph import build_graph

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def _load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _build_run_id() -> str:
    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    short_id = uuid.uuid4().hex[:8]
    return f"run_{timestamp}_{short_id}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Graduate Assist LangGraph runner")
    parser.add_argument("--data", required=True, help="Path to experimental CSV data")
    parser.add_argument("--description", required=True, help="Path to experiment description markdown")
    parser.add_argument("--artifacts", default="artifacts", help="Artifacts output directory")
    parser.add_argument("--reviewer-comment", default=None, help="Optional reviewer/advisor comment")
    parser.add_argument(
        "--auto-reviewer-comment",
        action="store_true",
        help="Generate a self-review comment to trigger rebuttal",
    )
    parser.add_argument(
        "--rebuttal-strategy",
        choices=["explain_only", "concede_and_limit", "reanalyze"],
        default=None,
        help="Override rebuttal strategy selection.",
    )
    parser.add_argument(
        "--max-rebuttal-rounds",
        type=int,
        default=2,
        help="Maximum number of rebuttal rounds when reanalysis is requested.",
    )
    parser.add_argument("--llm-model", default="gpt-4.1", help="LLM model name (default: gpt-4.1)")
    parser.add_argument("--verbose", action="store_true", help="Print agent logs to stdout")
    parser.add_argument("--test-mode", action="store_true", help="Enable test-only flow shortcuts")
    args = parser.parse_args()

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["LLM_MODEL"] = args.llm_model

    run_id = _build_run_id()

    state = {
        "raw_experiment_description": _load_text(args.description),
        "raw_experimental_data": args.data,
        "artifacts_path": args.artifacts,
        "run_id": run_id,
        "verbose": args.verbose,
        "test_mode": args.test_mode,
        "auto_reviewer_comment": args.auto_reviewer_comment,
        "max_rebuttal_rounds": args.max_rebuttal_rounds,
    }
    if args.rebuttal_strategy:
        state["rebuttal_strategy_override"] = args.rebuttal_strategy

    if args.reviewer_comment:
        state["reviewer_comment"] = args.reviewer_comment

    graph = build_graph().compile()
    artifacts_dir = artifacts_root(args.artifacts, run_id)
    write_text(artifacts_dir / "langgraph.mmd", graph.get_graph().draw_mermaid())
    graph.invoke(state)

    print(f"Run complete. Artifacts stored in: {Path(args.artifacts) / run_id}")


if __name__ == "__main__":
    main()
