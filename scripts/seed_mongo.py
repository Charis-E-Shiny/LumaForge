"""Load results/summary.json into MongoDB and create the indexes the API relies on.

Requires MONGODB_URI. Safe to re-run: results are versioned by generatedAt.

    MONGODB_URI=... python scripts/seed_mongo.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from pymongo import ASCENDING, DESCENDING, MongoClient

RUN_RETENTION_DAYS = 180


def main() -> None:
    uri = os.environ["MONGODB_URI"]
    database = MongoClient(uri, serverSelectionTimeoutMS=8000)[os.environ.get("MONGODB_DB", "lumaforge")]
    summary = json.loads(Path("results/summary.json").read_text())
    database.results.update_one({"generatedAt": summary["generatedAt"]}, {"$set": summary}, upsert=True)
    database.results.create_index([("generatedAt", DESCENDING)])
    # Doubles as the sort index for recent runs, and keeps a free cluster from filling up.
    database.runs.create_index([("createdAt", ASCENDING)], expireAfterSeconds=RUN_RETENTION_DAYS * 86400)
    print("results:", database.results.count_documents({}), "runs:", database.runs.count_documents({}))


if __name__ == "__main__":
    main()
