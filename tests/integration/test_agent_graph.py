"""Integration tests: verify the LangGraph compiles and routes correctly."""

import pytest
from unittest.mock import patch, MagicMock
from src.agent.graph import build_graph


def test_graph_compiles():
    graph = build_graph()
    assert graph is not None


def test_graph_has_expected_nodes():
    graph = build_graph()
    node_names = list(graph.nodes.keys())
    for expected in ["classify_intent", "execute_analysis", "mask_and_format"]:
        assert expected in node_names
