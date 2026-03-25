"""Pre-seeded Golden Bucket trios and admin CLI for expert-authored trio management."""

from dataclasses import dataclass
from typing import List


@dataclass
class SeedTrio:
    question: str
    sql: str
    report: str


def seed_golden_bucket_if_empty(golden_bucket) -> bool:
    """Populate the Golden Bucket with seed trios if it is empty.

    Returns True if seeding was performed, False if bucket already had data.
    """
    from src.memory.golden_bucket import Trio

    existing = golden_bucket._conn.execute("SELECT COUNT(*) FROM trios").fetchone()[0]
    if existing > 0:
        return False

    # Import seed trios from init_data to avoid duplication
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    try:
        from scripts.init_data import EXPERT_TRIOS
        trios = [Trio(question=t.question, sql=t.sql, report=t.report) for t in EXPERT_TRIOS]
    except ImportError:
        return False

    golden_bucket.add_trios(trios)
    return True


def add_expert_trio_interactive(golden_bucket) -> None:
    """Interactive CLI for a human expert to author and save a new trio.

    This is the only sanctioned route for adding new trios to the Golden
    Bucket, ensuring all entries are human expert-authored as per spec.

    Usage:
        python -m src.memory.seed_trios
    """
    from src.memory.golden_bucket import Trio

    print("\n--- Add Expert Trio to Golden Bucket ---")
    question = input("Question: ").strip()
    if not question:
        print("Aborted — question cannot be empty.")
        return

    print("SQL (paste, then enter a line with just '.' to finish):")
    sql_lines = []
    while True:
        line = input()
        if line.strip() == ".":
            break
        sql_lines.append(line)
    sql = "\n".join(sql_lines).strip()

    report = input("Report summary: ").strip()

    print(f"\nSaving trio:\n  Q: {question}\n  SQL: {sql[:80]}...\n  R: {report}")
    confirm = input("Confirm? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Aborted.")
        return

    golden_bucket.add_trios([Trio(question=question, sql=sql, report=report)])
    print("✅ Expert trio saved to Golden Bucket.")


if __name__ == "__main__":
    from src.config.settings import load_settings
    from src.memory.golden_bucket import GoldenBucket
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from dotenv import load_dotenv
    load_dotenv()

    settings = load_settings()
    embedder = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    db_path = settings.memory.resolve_path(settings.memory.golden_bucket_path)
    gb = GoldenBucket(str(db_path), embedder)
    seed_golden_bucket_if_empty(gb)
    add_expert_trio_interactive(gb)
