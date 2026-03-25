"""CLI entry point for the Retail Data Agent."""

from dotenv import load_dotenv
load_dotenv()

from src.observability.logger import setup_logging
setup_logging()

from langchain_core.messages import HumanMessage
from src.agent.graph import build_graph
from src.agent.state import AgentState
from src.memory.user_prefs import UserPrefsStore
from src.config.settings import load_settings
from src.observability.metrics import write_snapshot
from src.observability.progress import clear as progress_clear
from src.resilience.quota_check import check_quota_or_exit


def _get_startup_hints() -> list[str]:
    """Retrieve top 3 trios from the Golden Bucket for startup hints."""
    import sqlite3
    import logging

    log = logging.getLogger(__name__)

    try:
        settings = load_settings()
        db_path = settings.memory.resolve_path(settings.memory.golden_bucket_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"  [debug] Reading Golden Bucket from: {db_path}")

        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trios (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              question TEXT NOT NULL,
              sql TEXT NOT NULL,
              report TEXT NOT NULL
            )
        """)
        rows = conn.execute("SELECT question FROM trios LIMIT 3").fetchall()
        count = conn.execute("SELECT COUNT(*) FROM trios").fetchone()[0]
        conn.close()

        print(f"  [debug] Found {count} trios in Golden Bucket.")

        if rows:
            return [row[0] for row in rows]

    except Exception as e:
        log.warning(f"Could not load startup hints: {e}", exc_info=True)

    return [
        "What are the top 10 products by total revenue?",
        "Show me monthly orders and revenue this year",
        "Who are the top 10 customers by total spend?",
    ]


def print_banner(hints: list[str]) -> None:
    """Print welcome banner with example questions and persona tone."""
    settings = load_settings()
    print("=" * 60)
    print("  Retail Data Agent - BigQuery Analytics Chat")
    print(f"  Model: {settings.llm.model}")
    print(f"  Tone:  {settings.persona.tone}")
    print("=" * 60)
    print("\nExample questions you can ask:")
    for i, hint in enumerate(hints, 1):
        print(f"  {i}. {hint}")
    print("\nCommands:")
    print("  /quit or /exit           - Exit")
    print("  /format table            - Switch to table output")
    print("  /format bullets          - Switch to bullet output")
    print("  /whoami <user_id>        - Switch user (e.g. manager_a, manager_b)")
    print("=" * 60)
    print()


def run_cli() -> None:
    """Run an interactive terminal session with the agent."""
    check_quota_or_exit()
    hints = _get_startup_hints()
    settings = load_settings()
    graph = build_graph()
    prefs_store = UserPrefsStore(
        str(settings.memory.resolve_path(settings.memory.user_prefs_path))
    )

    state: AgentState = {
        "messages": [],
        "user_id": load_settings().safety.default_user_id,
        "pending_destructive_op": None,
        "last_sql": None,
        "retry_count": 0,
        "raw_result": None,
        "final_output": None,
    }

    print_banner(hints)

    while True:
        try:
            prompt_label = f"[{state['user_id']}]"
            user_input = input(f"\nYou {prompt_label}: ").strip()
            if not user_input:
                continue

            if user_input.lower() in {"/quit", "/exit"}:
                write_snapshot("data/metrics_snapshot.json")
                print("\nGoodbye!")
                break

            if user_input.lower().startswith("/format "):
                fmt = user_input.split(maxsplit=1)[1].strip().lower()
                if fmt in {"table", "bullets"}:
                    prefs_store.set_output_format(state["user_id"], fmt)
                    print(f"Output format set to '{fmt}' for {state['user_id']}.")
                else:
                    print("Unknown format. Use: /format table  or  /format bullets")
                continue

            if user_input.lower().startswith("/whoami "):
                new_user = user_input.split(maxsplit=1)[1].strip()
                state["user_id"] = new_user
                state["messages"] = []
                fmt = prefs_store.get(new_user)["output_format"]
                print(f"Switched to user '{new_user}' (format: {fmt}).")
                continue

            state["messages"].append(HumanMessage(content=user_input))
            state = graph.invoke(state)

            progress_clear()
            print("\nAssistant:")
            print(state.get("final_output") or "[No output]")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /quit to exit.")
        except EOFError:
            write_snapshot("data/metrics_snapshot.json")
            print("\n\nGoodbye!")
            break


def main() -> None:
    """Main entry point."""
    run_cli()


if __name__ == "__main__":
    main()
