"""CLI entry point for the Retail Data Agent."""

from dotenv import load_dotenv
load_dotenv()

from src.agent.graph import build_graph
from src.agent.state import AgentState


def print_banner() -> None:
    """Print a simple welcome banner with usage hints."""
    print("=" * 60)
    print("  Retail Data Agent - BigQuery Analytics Chat")
    print("=" * 60)
    print("Commands:")
    print("  /quit or /exit - Exit")
    print("  /format table  - Switch to table output")
    print("  /format bullets - Switch to bullet output")
    print("=" * 60)
    print()


def run_cli() -> None:
    """Run an interactive terminal session with the agent."""
    graph = build_graph()
    state: AgentState = {
        "messages": [],
        "user_id": "manager_a",
        "pending_destructive_op": None,
        "last_sql": None,
        "retry_count": 0,
        "raw_result": None,
        "final_output": None,
    }

    print_banner()

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"/quit", "/exit"}:
                print("\nGoodbye!")
                break

            state["messages"].append({"role": "user", "content": user_input})
            state = graph.invoke(state)  # type: ignore[arg-type]

            print("\nAssistant:")
            print(state.get("final_output") or "[No output]")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /quit to exit.")
        except EOFError:
            print("\n\nGoodbye!")
            break


def main() -> None:
    """Main entry point."""
    run_cli()


if __name__ == "__main__":
    main()
