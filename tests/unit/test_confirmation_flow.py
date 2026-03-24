"""Unit tests for the high-stakes confirmation flow."""

import pytest
from src.oversight.confirmation_flow import require_confirmation


def test_abort_on_wrong_phrase(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    assert require_confirmation("Delete 3 reports for Client X") is False


def test_abort_on_empty_input(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert require_confirmation("Delete reports") is False


def test_confirm_on_exact_phrase(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "YES DELETE")
    assert require_confirmation("Delete 3 reports for Client X") is True


def test_abort_on_lowercase_phrase(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "yes delete")
    assert require_confirmation("Delete reports") is False
