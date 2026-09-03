"""Continuously aggregate the main and extra worker pools into one status file."""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "multisource_cross_session_20260903"
COMBINED_PATH = OUTPUT_ROOT / "suite_status_combined.json"
PID_PATH = OUTPUT_ROOT / "suite_monitor.pid"
STOP_PATH = OUTPUT_ROOT / "STOP_SUITE_MONITOR"
POOL_STATUS_PATHS = (
    OUTPUT_ROOT / "suite_status.json",
    OUTPUT_ROOT / "suite_status_extra.json",
)
SOURCE_BANKS = ("single_recent", "recent3", "diverse3", "all_past")
CALIBRATION_TASKS = (20, 40, 80)
SEEDS = (42, 43, 44)


def now_iso() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def build_tests() -> list[dict[str, Any]]:
    tests = []
    for source_bank in SOURCE_BANKS:
        for calibration_tasks in CALIBRATION_TASKS:
            for seed in SEEDS:
                tests.append(
                    {
                        "number": len(tests) + 1,
                        "source_bank": source_bank,
                        "calibration_tasks": calibration_tasks,
                        "seed": seed,
                    }
                )
    return tests


def test_dir(test: dict[str, Any]) -> Path:
    return (
        OUTPUT_ROOT
        / str(test["source_bank"])
        / f"cal{test['calibration_tasks']}"
        / f"seed{test['seed']}"
    )


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def worker_assignments() -> dict[int, dict[str, Any]]:
    assignments = {}
    for status_path in POOL_STATUS_PATHS:
        status = read_json(status_path)
        if status is None:
            continue
        for worker_name, current in status.get("workers", {}).items():
            if current is not None:
                assignments[int(current["test_number"])] = {
                    "worker": worker_name,
                    **current,
                }
    return assignments


def build_status() -> dict[str, Any]:
    assignments = worker_assignments()
    tests = []
    for definition in build_tests():
        directory = test_dir(definition)
        summary = read_json(directory / "run_summary.json")
        if summary is not None and summary.get("status") == "complete":
            state = "complete"
        elif (directory / ".run.lock").exists():
            state = "running"
        else:
            state = "pending"
        record = {**definition, "status": state}
        if state == "running":
            record.update(assignments.get(int(definition["number"]), {}))
        if state == "complete" and summary is not None:
            record["selected_candidate"] = summary.get("selected_candidate")
            record["selected_test"] = summary.get("selected_test")
            record["scratch_test"] = summary.get("scratch_test")
        tests.append(record)
    counts = {
        state: sum(test["status"] == state for test in tests)
        for state in ("pending", "running", "complete")
    }
    return {
        "suite": "multisource_cross_session_20260903",
        "mode": "four_worker_two_pool_aggregate",
        "status": "complete" if counts["complete"] == len(tests) else "running",
        "total_tests": len(tests),
        "counts": counts,
        "running_tests": [test for test in tests if test["status"] == "running"],
        "updated_at": now_iso(),
        "tests": tests,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    STOP_PATH.unlink(missing_ok=True)
    PID_PATH.write_text(str(os.getpid()), encoding="ascii")
    try:
        while not STOP_PATH.exists():
            status = build_status()
            atomic_json(COMBINED_PATH, status)
            if status["status"] == "complete":
                break
            time.sleep(10)
    finally:
        PID_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
