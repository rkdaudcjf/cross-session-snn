"""Run and monitor the complete 36-condition multi-source pilot matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
RUNNER = EXPERIMENT_DIR / "run_multisource_pilot.py"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "multisource_cross_session_20260903"
STATUS_PATH = OUTPUT_ROOT / "suite_status.json"
LOG_PATH = OUTPUT_ROOT / "suite.log"

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
                        "status": "pending",
                    }
                )
    return tests


def result_path(test: dict[str, Any]) -> Path:
    return (
        OUTPUT_ROOT
        / str(test["source_bank"])
        / f"cal{test['calibration_tasks']}"
        / f"seed{test['seed']}"
        / "run_summary.json"
    )


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def counts(tests: list[dict[str, Any]]) -> dict[str, int]:
    return {
        status: sum(test["status"] == status for test in tests)
        for status in ("pending", "running", "complete", "failed")
    }


def status_payload(tests: list[dict[str, Any]], started_at: str) -> dict[str, Any]:
    running = next((test for test in tests if test["status"] == "running"), None)
    return {
        "suite": "multisource_cross_session_20260903",
        "status": (
            "failed"
            if any(test["status"] == "failed" for test in tests)
            and not any(test["status"] in {"pending", "running"} for test in tests)
            else "complete"
            if all(test["status"] == "complete" for test in tests)
            else "running"
        ),
        "total_tests": len(tests),
        "counts": counts(tests),
        "current_test": running,
        "started_at": started_at,
        "updated_at": now_iso(),
        "tests": tests,
        "status_path": str(STATUS_PATH),
        "log_path": str(LOG_PATH),
    }


def emit(message: str, log) -> None:
    print(message, flush=True)
    log.write(message + "\n")
    log.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tests = build_tests()
    if args.list_only:
        print(json.dumps({"total_tests": len(tests), "tests": tests}, indent=2))
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    started_at = now_iso()
    with LOG_PATH.open("a", encoding="utf-8", buffering=1) as log:
        emit(f"[{started_at}] START suite total={len(tests)}", log)
        for test in tests:
            summary_path = result_path(test)
            if summary_path.exists():
                saved = json.loads(summary_path.read_text(encoding="utf-8"))
                if saved.get("status") == "complete":
                    test["status"] = "complete"
                    test["finished_at"] = saved.get("finished_at", "completed_before_suite")
                    atomic_json(STATUS_PATH, status_payload(tests, started_at))
                    emit(f"SKIP complete {test['number']:02d}/36 {summary_path.parent}", log)
                    continue

            test["status"] = "running"
            test["started_at"] = now_iso()
            atomic_json(STATUS_PATH, status_payload(tests, started_at))
            label = (
                f"{test['number']:02d}/36 bank={test['source_bank']} "
                f"cal={test['calibration_tasks']} seed={test['seed']}"
            )
            emit(f"\n[{test['started_at']}] START {label}", log)
            command = [
                sys.executable,
                str(RUNNER),
                "--source-bank",
                str(test["source_bank"]),
                "--calibration-tasks",
                str(test["calibration_tasks"]),
                "--seed",
                str(test["seed"]),
                "--epochs",
                str(args.epochs),
                "--cpu-threads",
                str(args.cpu_threads),
                "--resume",
            ]
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                emit(line.rstrip(), log)
            return_code = process.wait()
            test["finished_at"] = now_iso()
            test["return_code"] = return_code
            if return_code == 0 and summary_path.exists():
                saved = json.loads(summary_path.read_text(encoding="utf-8"))
                test["status"] = "complete" if saved.get("status") == "complete" else "failed"
                test["selected_candidate"] = saved.get("selected_candidate")
                test["selected_test"] = saved.get("selected_test")
                test["scratch_test"] = saved.get("scratch_test")
            else:
                test["status"] = "failed"
            atomic_json(STATUS_PATH, status_payload(tests, started_at))
            emit(f"[{test['finished_at']}] {test['status'].upper()} {label}", log)
            if test["status"] == "failed" and args.stop_on_error:
                break

        final = status_payload(tests, started_at)
        atomic_json(STATUS_PATH, final)
        emit(f"[{final['updated_at']}] SUITE {final['status'].upper()} counts={final['counts']}", log)


if __name__ == "__main__":
    main()
