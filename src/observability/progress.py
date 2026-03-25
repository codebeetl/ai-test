"""Lightweight single-line progress feedback for the CLI.

Prints a short status line at the start of each agent node so the user
knows work is underway. Uses carriage-return overwriting to avoid clutter —
each stage replaces the previous on the same line.  A final newline is
printed when the agent completes.

Usage:
    from src.observability.progress import show, clear

    show("Classifying question...")   # overwrites previous stage line
    clear()                           # call once after final output is printed
"""

import sys

_STAGES = {
    "classify":    ("🔍", "Analysing your question...  "),
    "analysis":    ("⚙️ ", "Generating SQL query...      "),
    "executing":   ("🔄", "Running query...             "),
    "reporting":   ("✍️ ", "Preparing report...          "),
    "destructive": ("⚠️ ", "Processing delete request... "),
    "formatting":  ("📋", "Formatting results...        "),
    "done":        ("✅", "Done.                        "),
    "error":       ("❌", "Something went wrong.        "),
}

_CURRENT: list[str] = []   # mutable so we can track whether a line is active


def show(stage: str) -> None:
    """Print a single-line progress indicator, overwriting the previous one.

    Args:
        stage: One of the keys in _STAGES, or a raw string to display directly.
    """
    if stage in _STAGES:
        icon, label = _STAGES[stage]
        text = f"  {icon}  {label}"
    else:
        text = f"  ⏳  {stage:<40}"

    # \r moves cursor to start of line; padding ensures previous text is erased
    sys.stdout.write(f"\r{text}")
    sys.stdout.flush()
    _CURRENT.clear()
    _CURRENT.append(text)


def clear() -> None:
    """Print a newline to cleanly end the progress line before final output."""
    if _CURRENT:
        sys.stdout.write("\r" + " " * len(_CURRENT[0]) + "\r")
        sys.stdout.flush()
        _CURRENT.clear()
