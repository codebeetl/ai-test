"""High-stakes oversight: CLI confirmation gate for all destructive operations.

Any operation that mutates or deletes data in the Saved Reports library MUST
pass through require_confirmation() before execution. This satisfies:

  - The assignment's High-Stakes Oversight requirement (GDPR delete flows).
  - General "human-in-the-loop" best practice for agentic systems that manage
    persistent state.

Design intent:
  The function is intentionally synchronous and blocking. The LangGraph
  confirmation_gate node calls this function and only proceeds to the
  execute_destructive node when it returns True. A False return causes the
  graph to route to a safe abort terminal node instead.

Confirmation phrase:
  The operator must type the exact phrase ``YES DELETE`` (case-sensitive).
  Partial matches, 'yes', 'y', or 'confirm' are all rejected. This reduces
  the risk of accidental confirmation from casual keyboard input.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

CONFIRM_PHRASE = "YES DELETE"


def require_confirmation(action_description: str) -> bool:
    """Block execution until the operator explicitly confirms a destructive action.

    Renders a high-visibility warning block in the terminal, describes the
    pending action in plain language, and waits for exact phrase confirmation.
    All outcomes (confirmed / aborted) are logged at WARNING level so the
    observability layer can alert on unusual delete patterns.

    Args:
        action_description: Human-readable plain-English summary of the
            operation that will be performed if confirmed. Should include
            the scope of deletion (e.g. 'Delete 3 reports mentioning Client X').

    Returns:
        True if the operator typed the exact confirmation phrase.
        False if the operator typed anything else or pressed Ctrl+C.
    """
    print("\n" + "!" * 60)
    print("  \u26a0  HIGH-STAKES OPERATION \u2014 CONFIRMATION REQUIRED")
    print("!" * 60)
    print(f"\nPending action: {action_description}")
    print(f'\nType exactly "{CONFIRM_PHRASE}" to proceed, or anything else to abort:\n')

    try:
        user_input = input("> ").strip()
    except (KeyboardInterrupt, EOFError):
        logger.warning("Destructive operation aborted via interrupt")
        print("\nOperation aborted. No changes made.")
        return False

    if user_input == CONFIRM_PHRASE:
        logger.warning(
            "Destructive operation CONFIRMED by operator",
            extra={"action": action_description},
        )
        return True

    logger.info(
        "Destructive operation aborted — incorrect confirmation phrase",
        extra={"action": action_description},
    )
    print("\nOperation aborted. No changes made.")
    return False
