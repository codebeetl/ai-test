"""LangGraph state machine — wires all nodes and edges together."""

from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import (
    classify_intent,
    execute_analysis,
    confirmation_gate,
    execute_destructive,
    mask_and_format,
)


def _route_intent(state: AgentState) -> str:
    """Determine which node to route to after intent classification."""
    if state.get("pending_destructive_op"):
        return "destructive"
    if (state.get("raw_result") or {}).get("out_of_scope"):
        return "out_of_scope"
    return "analysis"


def build_graph() -> StateGraph:
    """Construct and compile the retail data agent LangGraph."""
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("execute_analysis", execute_analysis)
    graph.add_node("confirmation_gate", confirmation_gate)
    graph.add_node("execute_destructive", execute_destructive)
    graph.add_node("mask_and_format", mask_and_format)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        _route_intent,
        {
            "destructive":  "confirmation_gate",
            "analysis":     "execute_analysis",
            "out_of_scope": "mask_and_format",
        },
    )

    graph.add_edge("execute_analysis", "mask_and_format")
    graph.add_edge("confirmation_gate", "execute_destructive")
    graph.add_edge("execute_destructive", "mask_and_format")
    graph.add_edge("mask_and_format", END)

    return graph.compile()
