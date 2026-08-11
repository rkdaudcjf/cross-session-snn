"""Train one reconstructed-task Indy session with correlation-selected channels.

This extends the completed 96-channel baseline experiments.
Channel ranking follows Leone et al. (DATE 2025): Pearson correlation between
training firing rate and behavior, using only the chronological first half of
training entries.  The paper does not define how vx/vy correlations are reduced
to one rank, so the documented RMS magnitude is used here.

The primary result is a freshly trained structural K-input SNN.  It is not a
post-hoc zero mask applied to a 96-channel checkpoint.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import threading
import time
import traceback
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data" / "sabes_zenodo" / "master_mat"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
BASELINE_EXPERIMENT_NAME = "baseline_96ch"
SESSIONS = (
    "indy_20170124_01",
    "indy_20170127_03",
    "indy_20170131_02",
    "indy_20160630_01",
    "indy_20160622_01",
)


def load_masking_module() -> Any:
    path = SCRIPT_DIR / "internal" / "channel_selection_core.py"
    name = "channel_selection_core"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


masking = load_masking_module()
base = masking.base


def reconstructed_task_windows(
    session: Any,
    window_steps: int = 3_876,
    offset_ms: int = -32,
) -> list[Any]:
    """Apply the fixed three-reach task rule used by the baseline experiments."""
    starts = [0]
    starts.extend(window.start + offset_ms for window in session.task_windows[2::3])
    starts = [max(0, int(start)) for start in starts]
    windows = []
    for index, start in enumerate(starts):
        end = min(start + window_steps, len(session.velocity))
        if end - start == window_steps:
            windows.append(base.TaskWindow(task_index=index, start=start, end=end))
    return windows


def floor_chronological_split(windows: Sequence[Any]) -> tuple[list[Any], list[Any], list[Any]]:
    n_total = len(windows)
    n_train = int(0.8 * n_total)
    n_validation = int(0.1 * n_total)
    return (
        list(windows[:n_train]),
        list(windows[n_train : n_train + n_validation]),
        list(windows[n_train + n_validation :]),
    )


def baseline_summary_path(session: str) -> Path:
    return (
        PROJECT_ROOT
        / "outputs"
        / BASELINE_EXPERIMENT_NAME
        / session
        / "run_summary.json"
    )


def output_dir(args: argparse.Namespace) -> Path:
    root = OUTPUT_ROOT / args.experiment_name
    if args.smoke:
        root = root / "_smoke"
    return root / args.session / f"top{args.keep_channels}"


def checkpoint_resume_state(path: Path, resume: bool) -> tuple[int, float]:
    if not resume or not path.exists():
        return 1, 0.0
    saved = torch.load(path, map_location="cpu", weights_only=False)
    return int(saved["epoch"]) + 1, float(saved.get("elapsed_seconds", 0.0))


class BatchHeartbeat:
    """Emit progress on a wall-clock timer, including during a long batch."""

    def __init__(
        self,
        *,
        path: Path,
        session: str,
        keep_channels: int,
        interval_seconds: float,
        start_epoch: int,
        display_max_epochs: int,
        train_batches: int,
        validation_batches: int,
        run_index: int,
        total_runs: int,
        elapsed_before: float,
    ) -> None:
        self.path = path
        self.session = session
        self.keep_channels = keep_channels
        self.interval_seconds = interval_seconds
        self.display_max_epochs = display_max_epochs
        self.train_batches = train_batches
        self.validation_batches = validation_batches
        self.run_index = run_index
        self.total_runs = total_runs
        self.elapsed_before = elapsed_before
        self.started = time.perf_counter()
        self.next_epoch = start_epoch
        self.epoch = min(start_epoch, display_max_epochs)
        self.phase = "training_start"
        self.batch_current = 0
        self.batch_total = 0
        self.batch_completed = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self.emit()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_seconds + 1.0))

    def begin_train_epoch(self) -> int:
        with self._lock:
            self.epoch = self.next_epoch
            self.next_epoch += 1
            self.phase = "train"
            self.batch_current = 0
            self.batch_total = self.train_batches
            self.batch_completed = 0
            return self.epoch

    def current_epoch(self) -> int:
        with self._lock:
            return self.epoch

    def set_batch(self, phase: str, batch_current: int, batch_total: int) -> None:
        with self._lock:
            self.phase = phase
            self.batch_current = batch_current
            self.batch_total = batch_total
            self.batch_completed = max(0, batch_current - 1)

    def complete_batch(self, phase: str, batch_completed: int, batch_total: int) -> None:
        with self._lock:
            self.phase = phase
            self.batch_current = batch_completed
            self.batch_total = batch_total
            self.batch_completed = batch_completed

    def set_external_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase
            self.batch_current = 0
            self.batch_total = 0
            self.batch_completed = 0

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.emit()

    def _snapshot(self) -> dict[str, Any]:
        with self._lock:
            epoch = self.epoch
            phase = self.phase
            batch_current = self.batch_current
            batch_total = self.batch_total
            batch_completed = self.batch_completed

        batches_per_epoch = self.train_batches + self.validation_batches
        completed_before_epoch = max(0, epoch - 1) * batches_per_epoch
        if phase == "train":
            completed_in_epoch = batch_completed
        elif phase == "validation":
            completed_in_epoch = self.train_batches + batch_completed
        elif phase in {"validation_continuous", "test_continuous", "test_per_window", "saving"}:
            completed_in_epoch = batches_per_epoch
        else:
            completed_in_epoch = 0
        completed_slots = completed_before_epoch + completed_in_epoch
        total_slots = max(1, self.display_max_epochs * batches_per_epoch)
        completed_slots = min(completed_slots, total_slots)
        run_fraction = completed_slots / total_slots
        sweep_fraction = min(
            1.0,
            ((self.run_index - 1) + run_fraction) / max(1, self.total_runs),
        )
        elapsed = self.elapsed_before + (time.perf_counter() - self.started)
        eta_seconds = None
        if completed_slots > 0 and completed_slots < total_slots:
            eta_seconds = elapsed * (total_slots - completed_slots) / completed_slots
        return {
            "status": "running",
            "session": self.session,
            "keep_channels": self.keep_channels,
            "experiment_index": self.run_index,
            "experiment_total": self.total_runs,
            "phase": phase,
            "epoch": epoch,
            "display_max_epochs": self.display_max_epochs,
            "batch_current": batch_current,
            "batch_total": batch_total,
            "epoch_batch_slots_completed": completed_in_epoch,
            "epoch_batch_slots_total": batches_per_epoch,
            "run_batch_slots_completed": completed_slots,
            "run_batch_slots_total": total_slots,
            "run_max_progress_percent": 100.0 * run_fraction,
            "sweep_max_progress_percent": 100.0 * sweep_fraction,
            "elapsed_seconds": elapsed,
            "elapsed_hhmmss": base.format_elapsed(elapsed),
            "eta_max_seconds": eta_seconds,
            "eta_max_hhmmss": base.format_elapsed(eta_seconds) if eta_seconds is not None else None,
            "early_stopping_note": "Percent and ETA use the 100-epoch maximum; early stopping may finish sooner.",
            "updated_at": base.now_iso(),
        }

    def emit(self) -> None:
        snapshot = self._snapshot()
        base.atomic_json(self.path, snapshot)
        batch_text = (
            f"{snapshot['batch_current']:03d}/{snapshot['batch_total']:03d}"
            if snapshot["batch_total"]
            else "---/---"
        )
        eta_text = snapshot["eta_max_hhmmss"] or "--:--:--"
        print(
            "heartbeat "
            f"run={self.run_index:02d}/{self.total_runs:02d} "
            f"session={self.session} channels={self.keep_channels:02d} "
            f"phase={snapshot['phase']} "
            f"epoch={snapshot['epoch']:03d}/{self.display_max_epochs:03d} "
            f"batch={batch_text} "
            f"epoch_batches={snapshot['epoch_batch_slots_completed']:03d}/"
            f"{snapshot['epoch_batch_slots_total']:03d} "
            f"run_max={snapshot['run_max_progress_percent']:.2f}% "
            f"sweep_max={snapshot['sweep_max_progress_percent']:.2f}% "
            f"elapsed={snapshot['elapsed_hhmmss']} eta_max={eta_text}",
            flush=True,
        )


class ReportingLoader:
    def __init__(self, loader: Any, heartbeat: BatchHeartbeat, phase: str) -> None:
        self.loader = loader
        self.heartbeat = heartbeat
        self.phase = phase

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        if self.phase == "train":
            self.heartbeat.begin_train_epoch()
        else:
            self.heartbeat.current_epoch()
        total = len(self.loader)
        for index, batch in enumerate(self.loader, start=1):
            self.heartbeat.set_batch(self.phase, index, total)
            yield batch
            self.heartbeat.complete_batch(self.phase, index, total)


def metric_mean(frame: pd.DataFrame) -> dict[str, float]:
    row = frame.loc[frame["axis"] == "mean"].iloc[0]
    return {key: float(row[key]) for key in ("R2", "CC", "RMSE")}


def evaluate_window_average(
    model: Any,
    session: Any,
    windows: Sequence[Any],
    neural_lead_ms: int,
    device: torch.device,
    selected_channels: np.ndarray,
    *,
    max_steps: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float], np.ndarray, np.ndarray]:
    rows = []
    targets = []
    predictions = []
    for window in windows:
        _, target, prediction = masking.predict_continuous(
            model,
            session,
            [window],
            neural_lead_ms,
            device,
            selected_channels=selected_channels,
            max_steps=max_steps,
        )
        metrics = base.regression_metrics(target, prediction)
        for axis in ("vx", "vy"):
            row = metrics.loc[metrics["axis"] == axis].iloc[0]
            rows.append(
                {
                    "window": window.task_index,
                    "axis": axis,
                    "R2": float(row["R2"]),
                    "CC": float(row["CC"]),
                    "RMSE": float(row["RMSE"]),
                }
            )
        targets.append(target)
        predictions.append(prediction)
    frame = pd.DataFrame(rows)
    summary = {
        "R2": float(frame["R2"].mean()),
        "CC": float(frame["CC"].mean()),
        "RMSE": float(frame["RMSE"].mean()),
    }
    return frame, summary, np.stack(targets), np.stack(predictions)


def paper_channel_reference(session: str, keep_channels: int) -> dict[str, float] | None:
    if session != "indy_20170127_03":
        return None
    validation_r2 = {96: 0.65, 64: 0.65, 32: 0.56, 16: 0.51, 8: 0.44, 4: 0.28}
    reference: dict[str, float] = {"validation_R2": validation_r2[keep_channels]}
    if keep_channels == 64:
        reference.update({"test_R2": 0.73, "test_CC": 0.86})
    return reference


def make_protocol(
    args: argparse.Namespace,
    config: Any,
    split_counts: dict[str, int],
    baseline_path: Path,
) -> dict[str, Any]:
    baseline_summary = baseline_path.relative_to(PROJECT_ROOT).as_posix()
    return {
        "experiment_name": args.experiment_name,
        "session": args.session,
        "analysis_type": "fresh_structural_retraining_with_selected_channels",
        "baseline_96_summary": baseline_summary,
        "channel_selection_source": {
            "paper": "Leone et al., Enabling SNN-Based Near-MEA Neural Decoding with Channel Selection, DATE 2025",
            "section": "III-D Channel Selection Mechanism",
            "doi": "https://doi.org/10.23919/DATE64628.2025.10993220",
            "repository_reference": "docs/REFERENCES.md",
        },
        "ranking": {
            "source": "chronological first 50% of training entries only",
            "feature": "1 ms binary GT MUA; mean is proportional to firing rate",
            "targets": ["vx", "vy"],
            "statistic": "Pearson correlation",
            "axis_reduction": "sqrt(mean([corr_vx^2, corr_vy^2]))",
            "axis_reduction_warning": "The paper does not specify how multiple behavioral-variable correlations are collapsed; RMS magnitude is a documented reproduction assumption.",
            "nested_subsets": "top32 is a subset of top64 for the same session ranking",
            "leakage_control": "validation and test entries are excluded from ranking",
        },
        "shared_baseline_preprocessing": {
            "task": "three consecutive reaches",
            "task_start": "recording start, then every third reach boundary -32 ms",
            "training_window_steps": 3_876,
            "split": "chronological floor 80/10/remainder",
            "neural_lead_ms": 0,
            "test": "continuous from first held-out task through recording end",
        },
        "split_counts": split_counts,
        "config": asdict(config),
        "progress_logging": {
            "interval_seconds": args.batch_log_interval_seconds,
            "file": "batch_progress.json",
            "percent_basis": f"maximum {args.display_max_epochs} epochs; early stopping may finish sooner",
        },
    }


def run_experiment(args: argparse.Namespace) -> None:
    out = output_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "last_checkpoint.pt"
    start_epoch, elapsed_before = checkpoint_resume_state(checkpoint_path, args.resume)
    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device("cpu")
    config = replace(
        masking.MaskingConfig(),
        keep_channels=args.keep_channels,
        ranking_fraction=0.5,
        epochs=args.epochs,
        neural_lead_ms=0,
        learning_rate=1e-3,
        optimizer_name="adamw",
        weight_decay=1e-4,
        lr_plateau_patience=3,
        lr_plateau_factor=0.5,
        early_stopping_patience=10,
        beta_max=0.999,
        shuffle_training_windows=True,
        validation_drop_last=False,
        test_evaluation="continuous_single_state_reset",
        posthoc_random_repeats=0,
    )

    base.atomic_json(
        out / "progress.json",
        {
            "status": "preprocessing",
            "epoch": max(0, start_epoch - 1),
            "epochs": args.epochs,
            "session": args.session,
            "keep_channels": args.keep_channels,
            "updated_at": base.now_iso(),
        },
    )
    pipeline_started = time.perf_counter()
    preprocessing_started = time.perf_counter()
    session = base.prepare_sabes_session(
        DATA_ROOT / f"{args.session}.mat",
        mua_mode=config.mua_mode,
        moving_average_mode=config.moving_average_mode,
    )
    all_windows = reconstructed_task_windows(session)
    train_windows, validation_windows, test_windows = floor_chronological_split(all_windows)
    if args.session == "indy_20170131_02" and len(train_windows) != 169:
        raise RuntimeError(f"Expected 169 reconstructed training windows, found {len(train_windows)}")
    ranking, kept, masked = masking.rank_channels(session, train_windows, config)
    preprocessing_seconds = time.perf_counter() - preprocessing_started

    split_counts = {
        "all": len(all_windows),
        "train": len(train_windows),
        "validation": len(validation_windows),
        "test": len(test_windows),
    }
    baseline_path = baseline_summary_path(args.session)
    if not baseline_path.exists() and not args.smoke:
        raise FileNotFoundError(f"Completed 96-channel baseline is missing: {baseline_path}")
    protocol = make_protocol(args, config, split_counts, baseline_path)
    base.atomic_json(out / "protocol.json", protocol)
    ranking.to_csv(out / "channel_ranking.csv", index=False)
    base.atomic_json(
        out / "channel_mask.json",
        {
            "session": args.session,
            "keep_channels": args.keep_channels,
            "kept_indices_0based": kept.tolist(),
            "kept_channel_names": [session.channel_names[index] for index in kept],
            "masked_indices_0based": masked.tolist(),
            "masked_channel_names": [session.channel_names[index] for index in masked],
            "ranking_fraction": config.ranking_fraction,
            "ranking_axis_reduction": config.ranking_axis_reduction,
            "ranking_steps": int(ranking["ranking_steps"].iloc[0]),
        },
    )
    masking.plot_ranking(out, ranking)
    pd.DataFrame(
        [
            {
                "task": window.task_index,
                "start_step": window.start,
                "end_step": window.end,
                "split": (
                    "train"
                    if window in train_windows
                    else "validation"
                    if window in validation_windows
                    else "test"
                ),
            }
            for window in all_windows
        ]
    ).to_csv(out / "task_windows.csv", index=False)

    loader_train_windows = train_windows[:12] if args.smoke else train_windows
    loader_validation_windows = validation_windows[:4] if args.smoke else validation_windows
    loader_test_windows = test_windows[:2] if args.smoke else test_windows
    truncate_steps = 256 if args.smoke else None
    evaluation_max_steps = 512 if args.smoke else None
    train_loader, validation_loader = masking.make_selected_loaders(
        session,
        loader_train_windows,
        loader_validation_windows,
        kept,
        config,
        truncate_steps=truncate_steps,
    )
    heartbeat = BatchHeartbeat(
        path=out / "batch_progress.json",
        session=args.session,
        keep_channels=args.keep_channels,
        interval_seconds=args.batch_log_interval_seconds,
        start_epoch=start_epoch,
        display_max_epochs=args.display_max_epochs,
        train_batches=len(train_loader),
        validation_batches=len(validation_loader),
        run_index=args.run_index,
        total_runs=args.total_runs,
        elapsed_before=elapsed_before,
    )
    reporting_train_loader = ReportingLoader(train_loader, heartbeat, "train")
    reporting_validation_loader = ReportingLoader(validation_loader, heartbeat, "validation")
    base.set_reproducible_seed(config.seed)
    model = base.MartisSNN(
        input_size=args.keep_channels,
        output_size=2,
        hidden_sizes=config.hidden_sizes,
        threshold=config.threshold,
        beta_init=config.beta_init,
        optimized_forward=args.optimized_forward,
    )
    print(
        f"session={args.session} channels={args.keep_channels} device={device} "
        f"tasks={len(train_windows)}/{len(validation_windows)}/{len(test_windows)} "
        f"batches={len(train_loader)}/{len(validation_loader)}",
        flush=True,
    )
    heartbeat.start()
    try:
        (
            history,
            best_epoch,
            best_validation_loss,
            best_state,
            training_elapsed_seconds,
            training_started_at,
            training_finished_at,
        ) = base.train_resumable(
            model,
            reporting_train_loader,
            reporting_validation_loader,
            config,
            device,
            out,
            resume=args.resume,
        )
        model.load_state_dict(best_state)
        completed_epoch = int(history["epoch"].max()) if not history.empty else start_epoch - 1

        if args.training_only:
            history.to_csv(out / "training_history.csv", index=False)
            base.atomic_torch_save(
                out / "best_model.pt",
                {
                    "model_state_dict": best_state,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation_loss,
                    "kept_channels_0based": kept.tolist(),
                    "config": asdict(config),
                    "protocol": protocol,
                },
            )
            summary = {
                "status": "training_complete",
                "session": args.session,
                "keep_channels": args.keep_channels,
                "completed_epoch": completed_epoch,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "training_elapsed_seconds": training_elapsed_seconds,
                "protocol": protocol,
            }
            base.atomic_json(out / "run_summary.json", summary)
            base.atomic_json(out / "progress.json", summary)
            print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
            return

        validation_continuous = [
            base.TaskWindow(
                task_index=-1,
                start=validation_windows[0].start,
                end=test_windows[0].start,
            )
        ]
        test_continuous = [
            base.TaskWindow(
                task_index=-1,
                start=test_windows[0].start,
                end=len(session.velocity),
            )
        ]
        heartbeat.set_external_phase("validation_continuous")
        validation_started = time.perf_counter()
        validation_time, validation_target, validation_prediction = masking.predict_continuous(
            model,
            session,
            validation_continuous,
            config.neural_lead_ms,
            device,
            selected_channels=kept,
            max_steps=evaluation_max_steps,
        )
        validation_seconds = time.perf_counter() - validation_started
        heartbeat.set_external_phase("test_continuous")
        test_started = time.perf_counter()
        test_time, test_target, test_prediction = masking.predict_continuous(
            model,
            session,
            test_continuous,
            config.neural_lead_ms,
            device,
            selected_channels=kept,
            max_steps=evaluation_max_steps,
        )
        test_seconds = time.perf_counter() - test_started
        validation_metrics = base.regression_metrics(validation_target, validation_prediction)
        test_metrics = base.regression_metrics(test_target, test_prediction)
        heartbeat.set_external_phase("test_per_window")
        window_started = time.perf_counter()
        window_metrics, window_mean, window_targets, window_predictions = evaluate_window_average(
            model,
            session,
            loader_test_windows,
            config.neural_lead_ms,
            device,
            kept,
            max_steps=evaluation_max_steps,
        )
        window_seconds = time.perf_counter() - window_started
        heartbeat.set_external_phase("saving")

        history.to_csv(out / "training_history.csv", index=False)
        validation_metrics.to_csv(out / "validation_metrics_continuous.csv", index=False)
        test_metrics.to_csv(out / "test_metrics_continuous.csv", index=False)
        window_metrics.to_csv(out / "test_metrics_per_window.csv", index=False)
        np.savez_compressed(
            out / "test_predictions.npz",
            validation_time_sec=validation_time,
            validation_target=validation_target,
            validation_prediction=validation_prediction,
            test_time_sec=test_time,
            test_target=test_target,
            test_prediction=test_prediction,
            window_target=window_targets,
            window_prediction=window_predictions,
        )
        base.atomic_torch_save(
            out / "best_model.pt",
            {
                "model_state_dict": best_state,
                "session": args.session,
                "keep_channels": args.keep_channels,
                "kept_channels_0based": kept.tolist(),
                "masked_channels_0based": masked.tolist(),
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "config": asdict(config),
                "protocol": protocol,
            },
        )

        baseline_summary = None
        baseline_test = None
        delta = None
        if baseline_path.exists() and not args.smoke:
            baseline_summary = json.loads(baseline_path.read_text(encoding="utf-8"))
            baseline_test = baseline_summary["test_continuous"]
            current_test = metric_mean(test_metrics)
            delta = {
                key: current_test[key] - float(baseline_test[key])
                for key in ("R2", "CC", "RMSE")
            }
        baseline_model = base.MartisSNN(
            input_size=96,
            output_size=2,
            hidden_sizes=config.hidden_sizes,
            threshold=config.threshold,
            beta_init=config.beta_init,
            optimized_forward=args.optimized_forward,
        )
        selected_parameters = base.count_trainable_parameters(model)
        baseline_parameters = base.count_trainable_parameters(baseline_model)
        summary = {
            "status": "complete",
            "analysis_type": "fresh_structural_retraining_with_selected_channels",
            "session": args.session,
            "experiment_name": args.experiment_name,
            "smoke": args.smoke,
            "device": str(device),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "keep_channels": args.keep_channels,
            "mask_channels": 96 - args.keep_channels,
            "kept_indices_0based": kept.tolist(),
            "masked_indices_0based": masked.tolist(),
            "completed_epoch": completed_epoch,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "training_started_at": training_started_at,
            "training_finished_at": training_finished_at,
            "training_elapsed_seconds": training_elapsed_seconds,
            "timing_seconds": {
                "preprocessing": preprocessing_seconds,
                "training": training_elapsed_seconds,
                "validation_continuous": validation_seconds,
                "test_continuous": test_seconds,
                "test_per_window": window_seconds,
                "pipeline_process": time.perf_counter() - pipeline_started,
            },
            "split_counts": split_counts,
            "trainable_parameters": selected_parameters,
            "baseline_96_trainable_parameters": baseline_parameters,
            "parameter_reduction_fraction": 1.0 - selected_parameters / baseline_parameters,
            "validation_continuous": metric_mean(validation_metrics),
            "test_continuous": metric_mean(test_metrics),
            "test_per_window_mean": window_mean,
            "baseline_96": baseline_test,
            "delta_vs_baseline_96": delta,
            "paper_channel_selection_reference": paper_channel_reference(
                args.session, args.keep_channels
            ),
            "protocol": protocol,
        }
        base.atomic_json(out / "run_summary.json", summary)
        base.atomic_json(out / "progress.json", summary)
        (out / "failure.json").unlink(missing_ok=True)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    finally:
        heartbeat.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", choices=SESSIONS, required=True)
    parser.add_argument("--keep-channels", type=int, choices=(64, 32), required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--experiment-name", default="channel_selection_64_32ch")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--optimized-forward", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--training-only", action="store_true")
    parser.add_argument("--batch-log-interval-seconds", type=float, default=10.0)
    parser.add_argument("--display-max-epochs", type=int, default=100)
    parser.add_argument("--run-index", type=int, default=1)
    parser.add_argument("--total-runs", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        masking.self_test()
        return
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be at least 1")
    if args.epochs < 1 or args.display_max_epochs < args.epochs:
        raise ValueError("Epoch values are invalid")
    if args.batch_log_interval_seconds <= 0:
        raise ValueError("--batch-log-interval-seconds must be positive")
    if not 1 <= args.run_index <= args.total_runs:
        raise ValueError("--run-index must be within --total-runs")
    out = output_dir(args)
    try:
        run_experiment(args)
    except Exception as error:
        out.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "failed_at": base.now_iso(),
        }
        base.atomic_json(out / "failure.json", failure)
        base.atomic_json(out / "progress.json", failure)
        raise


if __name__ == "__main__":
    main()
