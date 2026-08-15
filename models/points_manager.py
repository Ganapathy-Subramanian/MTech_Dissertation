import json
import os
from threading import Lock


class PointsManager:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.file = os.path.join(self.base_dir, "penalty_rewards.json")
        self._lock = Lock()
        self._data = {}
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.file):
                with open(self.file, 'r') as f:
                    self._data = json.load(f) or {}
            else:
                self._data = {}
                self._save()
        except Exception:
            self._data = {}

    def _save(self):
        try:
            with open(self.file, 'w') as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            print(f"Failed to save points data: {e}")

    def adjust_points(self, category: str, delta: int):
        if not category:
            return None
        cat = category.strip()
        with self._lock:
            entry = self._data.get(cat, {"points": 0, "history": []})
            entry["points"] = entry.get("points", 0) + int(delta)
            entry["history"].append({"delta": int(delta)})
            self._data[cat] = entry
            self._save()
            return entry

    def get_summary(self):
        with self._lock:
            # return a copy
            return {k: {"points": v.get("points", 0), "history_len": len(v.get("history", []))} for k, v in self._data.items()}


# Singleton instance
points_manager = PointsManager()
