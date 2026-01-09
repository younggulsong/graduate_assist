from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph

from .nodes import (
    analysis_planning_agent_node,
    context_agent_node,
    data_understanding_agent_node,
    execution_agent_node,
    rebuttal_agent_node,
    reporting_agent_node,
    web_context_agent_node,
)
from .state import GraphState


def _reviewer_route(state: GraphState) -> Literal["rebuttal", "end"]:
    if state.get("reviewer_comment"):
        return "rebuttal"
    return "end"


def _rebuttal_route(state: GraphState) -> Literal["explain_only", "concede_and_limit", "reanalyze"]:
    decision = state.get("rebuttal_decision", {})
    strategy = decision.get("strategy", "explain_only")
    if strategy not in {"explain_only", "concede_and_limit", "reanalyze"}:
        return "explain_only"
    return strategy


def build_graph() -> StateGraph:
    """Construct the LangGraph StateGraph for the analysis workflow."""
    graph = StateGraph(GraphState)

    graph.add_node("ContextAgentNode", context_agent_node)
    graph.add_node("DataUnderstandingAgentNode", data_understanding_agent_node)
    graph.add_node("AnalysisPlanningAgentNode", analysis_planning_agent_node)
    graph.add_node("ExecutionAgentNode", execution_agent_node)
    graph.add_node("ReportingAgentNode", reporting_agent_node)
    graph.add_node("RebuttalAgentNode", rebuttal_agent_node)
    graph.add_node("WebContextAgentNode", web_context_agent_node)

    graph.set_entry_point("ContextAgentNode")
    graph.add_edge("ContextAgentNode", "DataUnderstandingAgentNode")
    graph.add_edge("DataUnderstandingAgentNode", "AnalysisPlanningAgentNode")
    graph.add_edge("AnalysisPlanningAgentNode", "ExecutionAgentNode")
    graph.add_edge("ExecutionAgentNode", "ReportingAgentNode")

    graph.add_conditional_edges(
        "ReportingAgentNode",
        _reviewer_route,
        {
            "rebuttal": "RebuttalAgentNode",
            "end": "WebContextAgentNode",
        },
    )

    graph.add_conditional_edges(
        "RebuttalAgentNode",
        _rebuttal_route,
        {
            "explain_only": "WebContextAgentNode",
            "concede_and_limit": "ReportingAgentNode",
            "reanalyze": "AnalysisPlanningAgentNode",
        },
    )

    graph.add_edge("WebContextAgentNode", END)

    return graph
