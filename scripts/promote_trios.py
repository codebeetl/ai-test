"""Promote expert-approved candidate trios into the Golden Bucket (Req 1).

Golden Bucket governance flow:
  1. The agent logs every successful query to data/candidate_trios.jsonl
     with "promoted": false and "ingested": false.
  2. A human expert opens that file, reviews each entry, and sets
     "promoted": true on entries worth keeping.
  3. The expert runs this script:
         python scripts/promote_trios.py
  4. The script ingests all promoted-but-not-yet-ingested trios into the
     Golden Bucket vector store, then marks them "ingested": true so they
     are not double-counted on the next run.

This keeps the Golden Bucket aligned with the assignment requirement that
golden knowledge comes from expert human judgment, not raw user queries.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from src.config.settings import load_settings
from src.memory.golden_bucket import GoldenBucket
from langchain_google_genai import GoogleGenerativeAIEmbeddings


@dataclass
class Trio:
    question: str
    sql: str
    report: str


def main() -> None:
    settings = load_settings()
    path = settings.memory.resolve_path(settings.memory.candidate_trios_path)

    if not path.exists():
        print("No candidate trios file found at:", path)
        return

    lines = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    to_promote = [
        entry for entry in lines
        if entry.get("promoted") and not entry.get("ingested")
    ]

    if not to_promote:
        print('No approved trios to promote (set "promoted": true to approve).')
        return

    print(f"Promoting {len(to_promote)} trio(s) into the Golden Bucket...")

    embedder = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    gb = GoldenBucket(settings.memory.golden_bucket_path, embedder)

    trios = [
        Trio(
            question=e["question"],
            sql=e["sql"],
            report=e.get("report", f"Returned {e.get('row_count', 0)} rows."),
        )
        for e in to_promote
    ]
    gb.add_trios(trios)

    ingested_questions = {e["question"] for e in to_promote}
    for entry in lines:
        if entry.get("promoted") and entry["question"] in ingested_questions:
            entry["ingested"] = True

    newline = "\n"
    path.write_text(
        newline.join(json.dumps(e) for e in lines) + newline,
        encoding="utf-8",
    )

    print(f"Done. {len(to_promote)} trio(s) added to Golden Bucket.")


if __name__ == "__main__":
    main()
