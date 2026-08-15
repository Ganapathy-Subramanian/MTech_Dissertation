"""
Real test-coverage measurement for the "Figure 7.1: Test Coverage Summary"
dashboard. Runs the actual pytest suite under coverage.py and buckets
per-file line coverage into the 8 module groups shown on the dashboard.

Honesty note: this project's current tests/ suite is not split into
unit / integration / e2e tiers (no pytest markers for that), so we
report ONE real "Test Coverage %" per module rather than fabricating
a 3-way Unit/Integration/E2E split. If unit/integration/e2e markers
are added later (e.g. @pytest.mark.unit), this module can be extended
to run coverage per marker and populate all three bars honestly.

Requires: pip install pytest coverage
"""
import subprocess
import json
import os
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_coverage_data.json")

# Maps the dashboard's module labels to the source files/folders that
# implement them, so raw per-file coverage can be rolled up correctly.
MODULE_MAP = {
    "Core Triage Engine": ["models/triage.py", "models/enhanced_triage.py", "models/bert_triage.py", "models/auto_retrain.py"],
    "RAG/Memory System": ["rag/"],
    "LLM Integration": ["llm/"],
    "Salesforce Integration": ["integration/"],
    "Workflow Automation": ["workflow/"],
    "Analytics Module": ["analytics/"],
    "API Endpoints": ["main_enhanced.py", "main.py"],
    "Security/Auth": ["security/"],
}

TARGET_COVERAGE = 85


def _bucket_for_file(rel_path: str):
    norm = rel_path.replace("\\", "/")
    for module, patterns in MODULE_MAP.items():
        for p in patterns:
            if norm == p or norm.startswith(p):
                return module
    return None


def run_coverage():
    """Executes `coverage run -m pytest tests/` + `coverage json`, then
    aggregates real line coverage into the 8 dashboard modules. Writes
    the result to test_coverage_data.json so /analytics/test-coverage
    can serve it without re-running the suite on every page load."""
    json_out = os.path.join(BASE_DIR, "coverage.json")

    try:
        run_proc = subprocess.run(
            [sys.executable, "-m", "coverage", "run", "--source=.", "-m", "pytest", "tests/", "-q"],
            cwd=BASE_DIR, capture_output=True, text=True, timeout=300
        )
        subprocess.run(
            [sys.executable, "-m", "coverage", "json", "-o", json_out],
            cwd=BASE_DIR, capture_output=True, text=True, timeout=60
        )
        with open(json_out) as f:
            raw = json.load(f)
    except FileNotFoundError:
        return {"error": "pytest/coverage not installed. Run: pip install pytest coverage"}
    except Exception as e:
        return {"error": f"Coverage run failed: {e}"}

    buckets = {m: {"covered": 0, "total": 0} for m in MODULE_MAP}
    for filepath, filedata in raw.get("files", {}).items():
        rel = os.path.relpath(filepath, BASE_DIR).replace("\\", "/")
        bucket = _bucket_for_file(rel)
        if not bucket:
            continue
        summary = filedata.get("summary", {})
        buckets[bucket]["covered"] += summary.get("covered_lines", 0)
        buckets[bucket]["total"] += summary.get("num_statements", 0)

    modules = []
    total_covered = total_lines = 0
    for name, agg in buckets.items():
        pct = round(agg["covered"] / agg["total"] * 100, 1) if agg["total"] else 0.0
        modules.append({
            "name": name,
            "coverage": pct,
            "lines_covered": agg["covered"],
            "lines_total": agg["total"],
            "status": "PASS" if pct >= TARGET_COVERAGE else ("N/A" if agg["total"] == 0 else "FAIL"),
        })
        total_covered += agg["covered"]
        total_lines += agg["total"]

    overall = round(total_covered / total_lines * 100, 1) if total_lines else 0.0
    result = {
        "generated_at": datetime.now().isoformat(),
        "target": TARGET_COVERAGE,
        "overall_coverage": overall,
        "overall_status": "PASS" if overall >= TARGET_COVERAGE else "FAIL",
        "modules": modules,
        "tests_stdout_tail": (run_proc.stdout or "")[-800:],
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2)
    return result


def get_last_result():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return None