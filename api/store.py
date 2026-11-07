"""MongoDB persistence for run telemetry and evaluation results.

The database is optional by design. Image processing must never fail or slow down because
the database is unreachable, so every call is short-timeout and swallows connection errors.
Only run metadata is stored: images, including uploads, are never written to the database.
"""
from __future__ import annotations

import os
from typing import Any

from pymongo import DESCENDING, MongoClient
from pymongo.errors import PyMongoError

_client: MongoClient | None = None
DATABASE = os.environ.get("MONGODB_DB", "lumaforge")


def _db():
    global _client
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        return None
    if _client is None:
        # A module-level client is reused across warm invocations of the function.
        _client = MongoClient(uri, serverSelectionTimeoutMS=1500, connectTimeoutMS=1500,
                              socketTimeoutMS=2500, maxPoolSize=2, appname="lumaforge")
    return _client[DATABASE]


def status() -> str:
    db = _db()
    if db is None:
        return "not_configured"
    try:
        db.command("ping")
        return "connected"
    except PyMongoError:
        return "unreachable"


def log_run(document: dict[str, Any]) -> bool:
    db = _db()
    if db is None:
        return False
    try:
        db.runs.insert_one(dict(document))
        return True
    except PyMongoError:
        return False


def latest_results() -> dict | None:
    db = _db()
    if db is None:
        return None
    try:
        return db.results.find_one({}, {"_id": 0}, sort=[("generatedAt", DESCENDING)])
    except PyMongoError:
        return None


def telemetry() -> dict[str, Any]:
    """Aggregate live usage; the response always has the same shape."""
    empty = {"database": status(), "total": 0, "by_route": [], "recent": []}
    db = _db()
    if db is None:
        return empty
    try:
        by_route = list(
            db.runs.aggregate(
                [
                    {
                        "$group": {
                            "_id": "$route",
                            "runs": {"$sum": 1},
                            "avg_total_ms": {"$avg": "$total_ms"},
                            "avg_hybrid_ms": {"$avg": "$timings_ms.hybrid_net"},
                            "avg_psnr": {"$avg": "$metrics.hybrid.psnr"},
                        }
                    },
                    {"$sort": {"runs": -1}},
                ]
            )
        )
        recent = list(
            db.runs.find(
                {},
                {"_id": 0, "createdAt": 1, "route": 1, "camera": 1, "source": 1, "total_ms": 1,
                 "metrics.hybrid": 1, "stats.sigma": 1},
                sort=[("createdAt", DESCENDING)],
                limit=8,
            )
        )
        for row in recent:
            row["createdAt"] = row["createdAt"].isoformat()
        return {
            "database": "connected",
            "total": sum(r["runs"] for r in by_route),
            "by_route": [{"route": r.pop("_id"), **r} for r in by_route],
            "recent": recent,
        }
    except PyMongoError:
        return {**empty, "database": "unreachable"}
