"""
Real data source for the "Figure 6.4: Adaptive Memory Coverage Growth"
dashboard. Uses:
  - The live ChromaDB vector count (rag/vector_db.py AdaptiveMemory) for
    "Total Vectors" (current state).
  - models/correction_log.json for the day-by-day growth curve of
    corrections learned (each correction = one embedded vector).
  - models/training_history.json for real accuracy-over-time from
    actual retrain runs.

Honesty note: the reference figure shows three stacked layers (Base
Coverage / Learned Patterns / Adaptive Improvements). This project's
memory store only tracks one real signal — corrections added over
time — so we report ONE real cumulative-growth curve plus real
accuracy-over-time, instead of fabricating three synthetic layers.
"""
import json
import os
from datetime import datetime
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORRECTION_LOG = os.path.join(BASE_DIR, "models", "correction_log.json")
TRAINING_HISTORY = os.path.join(BASE_DIR, "models", "training_history.json")


def _load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def get_memory_growth(memory_instance):
    corrections = _load_json(CORRECTION_LOG, [])
    history = _load_json(TRAINING_HISTORY, [])

    total_vectors = memory_instance.get_stats().get("total_memories", 0)

    if not corrections:
        return {
            "total_vectors": total_vectors,
            "unique_patterns_learned": 0,
            "total_corrections_logged": 0,
            "avg_accuracy_gain_pct": 0,
            "daily_growth_rate": 0,
            "days": [],
            "cumulative_vectors": [],
            "accuracy_over_time": [],
        }

    parsed = []
    for c in corrections:
        ts = c.get("corrected_at")
        if not ts:
            continue
        try:
            parsed.append(datetime.fromisoformat(ts))
        except Exception:
            continue
    parsed.sort()

    if not parsed:
        return {
            "total_vectors": total_vectors,
            "unique_patterns_learned": 0,
            "total_corrections_logged": len(corrections),
            "avg_accuracy_gain_pct": 0,
            "daily_growth_rate": 0,
            "days": [],
            "cumulative_vectors": [],
            "accuracy_over_time": [],
        }

    first_day = parsed[0].date()
    day_counts = defaultdict(int)
    for ts in parsed:
        day_counts[(ts.date() - first_day).days] += 1

    max_day = max(day_counts.keys())
    days = list(range(0, max_day + 1))
    cumulative = []
    running = 0
    for d in days:
        running += day_counts.get(d, 0)
        cumulative.append(running)

    elapsed_days = max(max_day, 1)
    daily_growth_rate = round(len(parsed) / elapsed_days, 2)

    unique_patterns = len({c.get("correct_category") for c in corrections if c.get("correct_category")})

    accuracy_over_time = []
    avg_accuracy_gain_pct = 0
    if history:
        sorted_hist = sorted(history, key=lambda h: h.get("timestamp", ""))
        accuracy_over_time = [
            {"timestamp": h.get("timestamp"), "accuracy": round(h.get("accuracy", 0) * 100, 2)}
            for h in sorted_hist
        ]
        if len(sorted_hist) >= 2:
            avg_accuracy_gain_pct = round(
                (sorted_hist[-1].get("accuracy", 0) - sorted_hist[0].get("accuracy", 0)) * 100, 2
            )

    return {
        "total_vectors": total_vectors,
        "unique_patterns_learned": unique_patterns,
        "total_corrections_logged": len(corrections),
        "avg_accuracy_gain_pct": avg_accuracy_gain_pct,
        "daily_growth_rate": daily_growth_rate,
        "days": days,
        "cumulative_vectors": cumulative,
        "accuracy_over_time": accuracy_over_time,
    }