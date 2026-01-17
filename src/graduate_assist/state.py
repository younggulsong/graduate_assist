from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class GraphState(TypedDict, total=False):
    """Shared state passed through the LangGraph StateGraph."""

    raw_experiment_description: str
    raw_experimental_data: str

    experiment_context: Dict[str, Any]
    data_schema: Dict[str, Any]
    data_quality_report: Dict[str, Any]
    analysis_plan: Dict[str, Any]
    execution_results: Dict[str, Any]
    report_text: str
    web_search_results: Dict[str, Any]
    web_context_comment: str

    reviewer_comment: Optional[str]
    auto_reviewer_comment: bool
    rebuttal_decision: Optional[Dict[str, Any]]
    rebuttal_response_text: Optional[str]
    rebuttal_strategy_override: Optional[str]
    rebuttal_rounds: int
    max_rebuttal_rounds: int
    test_mode: bool

    artifacts_path: str
    run_id: str
    verbose: bool

    # For internal routing/diagnostics
    execution_log: List[str]
