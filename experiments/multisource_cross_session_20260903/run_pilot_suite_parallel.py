"""Resume the pilot matrix with a lock-safe dynamic worker queue."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
RUNNER = EXPERIMENT_DIR / "run_multisource_pilot.py"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "multisource_cross_session_20260903"
STATUS_PATH = OUTPUT_ROOT / "suite_status.json"
PARALLEL_LOG_PATH = OUTPUT_ROOT / "suite_parallel.log"
PID_PATH = OUTPUT_ROOT / "suite_parallel.pid"
STOP_PATH = OUTPUT_ROOT / "STOP_PARALLEL_SUITE"

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


def result_dir(test: dict[str, Any]) -> Path:
    return (
        OUTPUT_ROOT
        / str(test["source_bank"])
        / f"cal{test['calibration_tasks']}"
        / f"seed{test['seed']}"
    )


def result_path(test: dict[str, Any]) -> Path:
    return result_dir(test) / "run_summary.json"


def completed_summary(test: dict[str, Any]) -> dict[str, Any] | None:
    path = result_path(test)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if payload.get("status") == "complete" else None


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def acquire_test_lock(test: dict[str, Any]) -> tuple[int, Path] | None:
    directory = result_dir(test)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".run.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    os.write(descriptor, f"pid={os.getpid()} started={now_iso()}\n".encode())
    return descriptor, lock_path


def release_test_lock(lock: tuple[int, Path]) -> None:
    descriptor, path = lock
    os.close(descriptor)
    path.unlink(missing_ok=True)


class ParallelSuite:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.tests = build_tests()
        self.started_at = now_iso()
        self.state_lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.workers: dict[str, dict[str, Any] | None] = {
            f"worker_{index}": None for index in range(1, args.workers + 1)
        }
        self.tasks: queue.Queue[dict[str, Any]] = queue.Queue()
        for test in self.tests:
            saved = completed_summary(test)
            if saved is None:
                self.tasks.put(test)
            else:
                test["status"] = "complete"
                test["selected_candidate"] = saved.get("selected_candidate")
                test["selected_test"] = saved.get("selected_test")
                test["scratch_test"] = saved.get("scratch_test")

    def counts(self) -> dict[str, int]:
        return {
            status: sum(test["status"] == status for test in self.tests)
            for status in ("pending", "running", "complete", "failed")
        }

    def payload(self) -> dict[str, Any]:
        counts = self.counts()
        if counts["running"] or counts["pending"]:
            suite_status = "running"
        elif counts["failed"]:
            suite_status = "failed"
        else:
            suite_status = "complete"
        return {
            "suite": "multisource_cross_session_20260903",
            "mode": "parallel_dynamic_queue",
            "status": suite_status,
            "workers_requested": self.args.workers,
            "cpu_threads_per_worker": self.args.cpu_threads,
            "total_tests": len(self.tests),
            "counts": counts,
            "workers": self.workers,
            "started_at": self.started_at,
            "updated_at": now_iso(),
            "tests": self.tests,
            "status_path": str(STATUS_PATH),
            "parallel_log_path": str(PARALLEL_LOG_PATH),
            "stop_file": str(STOP_PATH),
        }

    def write_status(self) -> None:
        with self.state_lock:
            atomic_json(STATUS_PATH, self.payload())

    def emit(self, message: str, *logs: TextIO) -> None:
        line = f"[{now_iso()}] {message}"
        with self.log_lock:
            print(line, flush=True)
            for log in logs:
                log.write(line + "\n")
                log.flush()

    def run_test(self, worker_name: str, test: dict[str, Any], worker_log: TextIO) -> None:
        lock = acquire_test_lock(test)
        if lock is None:
            self.emit(
                f"{worker_name} lock busy; requeue test={test['number']:02d}", worker_log
            )
            self.tasks.put(test)
            return
        try:
            saved = completed_summary(test)
            if saved is not None:
                test["status"] = "complete"
                return
            test["status"] = "running"
            test["started_at"] = now_iso()
            test["worker"] = worker_name
            self.workers[worker_name] = {
                "test_number": test["number"],
                "source_bank": test["source_bank"],
                "calibration_tasks": test["calibration_tasks"],
                "seed": test["seed"],
                "started_at": test["started_at"],
            }
            self.write_status()
            label = (
                f"test={test['number']:02d}/36 bank={test['source_bank']} "
                f"cal={test['calibration_tasks']} seed={test['seed']}"
            )
            self.emit(f"{worker_name} START {label}", worker_log)
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
                str(self.args.epochs),
                "--cpu-threads",
                str(self.args.cpu_threads),
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
            test["process_id"] = process.pid
            self.write_status()
            assert process.stdout is not None
            for output_line in process.stdout:
                self.emit(f"{worker_name} {label} | {output_line.rstrip()}", worker_log)
            return_code = process.wait()
            test["return_code"] = return_code
            test["finished_at"] = now_iso()
            saved = completed_summary(test)
            if return_code == 0 and saved is not None:
                test["status"] = "complete"
                test["selected_candidate"] = saved.get("selected_candidate")
                test["selected_test"] = saved.get("selected_test")
                test["scratch_test"] = saved.get("scratch_test")
            else:
                test["status"] = "failed"
            self.emit(f"{worker_name} {test['status'].upper()} {label}", worker_log)
        finally:
            self.workers[worker_name] = None
            release_test_lock(lock)
            self.write_status()

    def worker(self, worker_number: int) -> None:
        worker_name = f"worker_{worker_number}"
        worker_log_path = OUTPUT_ROOT / f"suite_{worker_name}.log"
        with worker_log_path.open("a", encoding="utf-8", buffering=1) as worker_log:
            while not STOP_PATH.exists():
                try:
                    test = self.tasks.get_nowait()
                except queue.Empty:
                    return
                try:
                    self.run_test(worker_name, test, worker_log)
                finally:
                    self.tasks.task_done()
            self.emit(f"{worker_name} STOP requested", worker_log)

    def run(self) -> None:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        STOP_PATH.unlink(missing_ok=True)
        PID_PATH.write_text(str(os.getpid()), encoding="ascii")
        with PARALLEL_LOG_PATH.open("a", encoding="utf-8", buffering=1) as main_log:
            self.emit(
                f"PARALLEL START workers={self.args.workers} "
                f"threads_per_worker={self.args.cpu_threads} counts={self.counts()}",
                main_log,
            )
            self.write_status()
            threads = [
                threading.Thread(target=self.worker, args=(index,), name=f"worker_{index}")
                for index in range(1, self.args.workers + 1)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.write_status()
            self.emit(f"PARALLEL END counts={self.counts()}", main_log)
        PID_PATH.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--cpu-threads", type=int, default=3)
    parser.add_argument("--list-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_only:
        tests = build_tests()
        pending = [test for test in tests if completed_summary(test) is None]
        print(json.dumps({"complete": len(tests) - len(pending), "pending": pending}, indent=2))
        return
    ParallelSuite(args).run()


if __name__ == "__main__":
    main()
