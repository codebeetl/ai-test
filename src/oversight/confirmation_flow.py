"""High-stakes oversight: CLI confirmation gate for all destructive operations.

Any tool marked as destructive MUST call require_confirmation() before
mutating state. The operator must type the exact phrase 'YES DELETE'
to proceed. Any other input aborts cleanly.
"""

import logging

logger = logging.getLogger(__name__)

CONFIRM_PHRASE = "YES DELETE"


def require_confirmation(action_description: str) -> bool:
    """Block execution until the operator explicitly confirms a destructive action.

    Args:
        action_description: Human-readable summary of what will be changed/deleted.

    Returns:
        True if the user confirmed, False if aborted.
    """
    print("\n" + "!" * 60)
    print("  ⚠  HIGH-STAKES OPERATION — CONFIRMATION REQUIRED")
    print("!" * 60)
    print(f"\nAction: {action_description}")
    print(f'\nType exactly "{CONFIRM_PHRASE}" to proceed, or anything else to abort:\n')

    user_input = input("> ").strip()

    if user_input == CONFIRM_PHRASE:
        logger.warning("Destructive operation confirmed", extra={"action": action_description})
        return True

    logger.info("Destructive operation aborted by user")
    print("\nOperation aborted. No changes made.")
    return False
