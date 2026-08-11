"""Train one indy session using reconstructed three-reach author task windows."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data" / "sabes_zenodo" / "master_mat"
PAPER_RESULTS = {
    "indy_20170124_01": (0.74, 0.87),
    "indy_20170127_03": (0.72, 0.86),
    "indy_20170131_02": (0.72, 0.85),
    "indy_20160630_01": (0.57, 0.76),
    "indy_20160622_01": (0.72, 0.86),
}


def load_base_module():
    path = SCRIPT_DIR / "internal" / "reproduction_core.py"
    spec = importlib.util.spec_from_file_location("baseline_reproduction_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def reconstructed_task_windows(base, session, window_steps: int = 3_876, offset_ms: int = -32):
    """Group three reaches per task, matching all 169 public training windows.

    Window zero begins at the recording start. Subsequent starts use every
    third target-change reach and a global -32 ms alignment inferred only from
    the public training targets. The rule is then fixed for validation/test.
    """

    starts = [0]
    starts.extend(window.start + offset_ms for window in session.task_windows[2::3])
    starts = [max(0, int(start)) for start in starts]
    windows = []
    for index, start in enumerate(starts):
        end = min(start + window_steps, len(session.velocity))
        if end - start == window_steps:
            windows.append(base.TaskWindow(task_index=index, start=start, end=end))
    return windows


def floor_chronological_split(windows):
    n_total = len(windows)
    n_train = int(0.8 * n_total)
    n_validation = int(0.1 * n_total)
    return (
        windows[:n_train],
        windows[n_train : n_train + n_validation],
        windows[n_train + n_validation :],
    )


def metric_mean(frame: pd.DataFrame) -> dict[str, float]:
    row = frame.loc[frame["axis"] == "mean"].iloc[0]
    return {key: float(row[key]) for key in ("R2", "CC", "RMSE")}


def evaluate_window_average(base, model, session, windows, neural_lead_ms, device):
    rows = []
    predictions = []
    targets = []
    for window in windows:
        _, target, prediction = base.predict_continuous_lagged(
            model, session, [window], neural_lead_ms, device
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="indy_20170131_02", choices=list(PAPER_RESULTS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--experiment-name", default="baseline_96ch")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--neural-lead-ms", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-plateau-patience", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--beta-max", type=float, default=1.0)
    parser.add_argument("--shuffle-training-windows", action="store_true")
    parser.add_argument("--keep-validation-remainder", action="store_true")
    parser.add_argument("--training-only", action="store_true")
    parser.add_argument(
        "--optimized-forward",
        action="store_true",
        help="Use the equation-equivalent buffer-free LIF forward path.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    base = load_base_module()
    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / args.experiment_name
        / args.session
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        pipeline_started = time.perf_counter()
        torch.set_num_threads(args.cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        device = torch.device("cpu")
        config = replace(
            base.ReconstructionConfig(),
            epochs=args.epochs,
            neural_lead_ms=args.neural_lead_ms,
            learning_rate=args.learning_rate,
            optimizer_name="adamw" if args.weight_decay > 0 else "adam",
            weight_decay=args.weight_decay,
            lr_plateau_patience=args.lr_plateau_patience,
            early_stopping_patience=args.early_stopping_patience,
            beta_max=args.beta_max,
            shuffle_training_windows=args.shuffle_training_windows,
            validation_drop_last=not args.keep_validation_remainder,
            test_evaluation="continuous_single_state_reset",
        )

        base.atomic_json(
            output_dir / "progress.json",
            {"status": "preprocessing", "epoch": 0, "epochs": args.epochs},
        )
        preprocessing_started = time.perf_counter()
        session = base.prepare_sabes_session(
            DATA_ROOT / f"{args.session}.mat",
            mua_mode=config.mua_mode,
            moving_average_mode=config.moving_average_mode,
        )
        all_tasks = reconstructed_task_windows(base, session)
        train_windows, validation_windows, test_windows = floor_chronological_split(all_tasks)
        if args.session == "indy_20170131_02" and len(train_windows) != 169:
            raise RuntimeError(
                f"Expected 169 reconstructed training tasks, found {len(train_windows)}"
            )
        preprocessing_seconds = time.perf_counter() - preprocessing_started

        protocol = {
            "experiment_name": args.experiment_name,
            "session": args.session,
            "evidence": {
                "public_training_windows": 169,
                "public_window_steps": 3876,
                "public_output_median_correlation": 0.9866623080535781,
                "reach_increment": 3,
                "global_alignment_offset_ms": -32,
                "public_input_note": (
                    "Event pattern does not match MAT GT MUA; this run intentionally uses "
                    "provided GT MUA for the Table IV GT comparison."
                ),
            },
            "inferred_choices": {
                "task": "three consecutive reaches",
                "task_start": "recording start, then every third reach boundary -32 ms",
                "training_window": "fixed 3876 ms",
                "split": "chronological floor 80/10/remainder",
                "training_state": "reset per task window",
                "validation": "task windows; drop_last=True following public notebook",
                "test": "continuous from first held-out task through recording end",
                "neural_behavior_lag_ms": 0,
                "execution": (
                    "buffer-free equation-equivalent LIF forward"
                    if args.optimized_forward
                    else "snnTorch module-buffer forward"
                ),
            },
            "split_counts": {
                "all": len(all_tasks),
                "train": len(train_windows),
                "validation": len(validation_windows),
                "test": len(test_windows),
            },
            "config": asdict(config),
        }
        base.atomic_json(output_dir / "protocol_choices.json", protocol)
        pd.DataFrame(
            [
                {
                    "task": window.task_index,
                    "start_step": window.start,
                    "end_step": window.end,
                    "split": (
                        "train" if window in train_windows else
                        "validation" if window in validation_windows else "test"
                    ),
                }
                for window in all_tasks
            ]
        ).to_csv(output_dir / "task_windows.csv", index=False)

        train_loader, validation_loader = base.make_loaders(
            session, train_windows, validation_windows, config
        )
        base.set_reproducible_seed(config.seed)
        model = base.MartisSNN(
            input_size=96,
            output_size=2,
            hidden_sizes=config.hidden_sizes,
            threshold=config.threshold,
            beta_init=config.beta_init,
            optimized_forward=args.optimized_forward,
        )
        print(
            f"session={args.session} device={device} tasks="
            f"{len(train_windows)}/{len(validation_windows)}/{len(test_windows)}",
            flush=True,
        )

        (
            history,
            best_epoch,
            best_validation_loss,
            best_state,
            elapsed_seconds,
            training_started_at,
            training_finished_at,
        ) = base.train_resumable(
            model,
            train_loader,
            validation_loader,
            config,
            device,
            output_dir,
            resume=args.resume,
        )
        model.load_state_dict(best_state)

        if args.training_only:
            history.to_csv(output_dir / "training_history.csv", index=False)
            base.atomic_torch_save(
                output_dir / "best_model.pt",
                {
                    "model_state_dict": best_state,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation_loss,
                    "config": asdict(config),
                    "protocol": protocol,
                },
            )
            summary = {
                "status": "training_complete",
                "session": args.session,
                "experiment_name": args.experiment_name,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "training_elapsed_seconds": elapsed_seconds,
                "protocol": protocol,
            }
            base.atomic_json(output_dir / "run_summary.json", summary)
            base.atomic_json(output_dir / "progress.json", summary)
            print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
            return

        base.atomic_json(
            output_dir / "progress.json",
            {
                "status": "validation_continuous",
                "epoch": args.epochs,
                "epochs": args.epochs,
                "training_elapsed_seconds": elapsed_seconds,
            },
        )

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
        validation_started = time.perf_counter()
        _, validation_target, validation_prediction = base.predict_continuous_lagged(
            model, session, validation_continuous, config.neural_lead_ms, device
        )
        validation_seconds = time.perf_counter() - validation_started
        base.atomic_json(
            output_dir / "progress.json",
            {
                "status": "test_continuous",
                "epoch": args.epochs,
                "epochs": args.epochs,
                "training_elapsed_seconds": elapsed_seconds,
                "validation_elapsed_seconds": validation_seconds,
            },
        )
        test_started = time.perf_counter()
        test_time, test_target, test_prediction = base.predict_continuous_lagged(
            model, session, test_continuous, config.neural_lead_ms, device
        )
        test_seconds = time.perf_counter() - test_started
        validation_metrics = base.regression_metrics(validation_target, validation_prediction)
        test_metrics = base.regression_metrics(test_target, test_prediction)
        base.atomic_json(
            output_dir / "progress.json",
            {
                "status": "test_per_window",
                "epoch": args.epochs,
                "epochs": args.epochs,
                "training_elapsed_seconds": elapsed_seconds,
                "validation_elapsed_seconds": validation_seconds,
                "test_continuous_elapsed_seconds": test_seconds,
            },
        )
        window_test_started = time.perf_counter()
        window_metrics, window_mean, window_targets, window_predictions = evaluate_window_average(
            base, model, session, test_windows, config.neural_lead_ms, device
        )
        window_test_seconds = time.perf_counter() - window_test_started

        history.to_csv(output_dir / "training_history.csv", index=False)
        validation_metrics.to_csv(output_dir / "validation_metrics_continuous.csv", index=False)
        test_metrics.to_csv(output_dir / "test_metrics_continuous.csv", index=False)
        window_metrics.to_csv(output_dir / "test_metrics_per_window.csv", index=False)
        np.savez_compressed(
            output_dir / "test_predictions.npz",
            time_sec=test_time,
            target=test_target,
            prediction=test_prediction,
            window_target=window_targets,
            window_prediction=window_predictions,
        )
        base.atomic_torch_save(
            output_dir / "best_model.pt",
            {
                "model_state_dict": best_state,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "config": asdict(config),
                "protocol": protocol,
            },
        )

        paper_r2, paper_cc = PAPER_RESULTS[args.session]
        comparison = pd.DataFrame(
            [
                {"result": "paper Table IV GT", "R2": paper_r2, "CC": paper_cc},
                {
                    "result": "current 96-channel continuous test",
                    **{k: v for k, v in metric_mean(test_metrics).items() if k in ("R2", "CC")},
                },
                {
                    "result": "current 96-channel per-window mean",
                    "R2": window_mean["R2"],
                    "CC": window_mean["CC"],
                },
            ]
        )
        comparison.to_csv(output_dir / "paper_baseline_comparison.csv", index=False)

        summary = {
            "status": "complete",
            "session": args.session,
            "experiment_name": args.experiment_name,
            "device": str(device),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "optimized_forward": args.optimized_forward,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "training_elapsed_seconds": elapsed_seconds,
            "training_started_at": training_started_at,
            "training_finished_at": training_finished_at,
            "timing_seconds": {
                "preprocessing": preprocessing_seconds,
                "training": elapsed_seconds,
                "validation_continuous": validation_seconds,
                "test_continuous": test_seconds,
                "test_per_window": window_test_seconds,
                "pipeline_total": time.perf_counter() - pipeline_started,
            },
            "split_counts": protocol["split_counts"],
            "validation_continuous": metric_mean(validation_metrics),
            "test_continuous": metric_mean(test_metrics),
            "test_per_window_mean": window_mean,
            "paper": {"R2": paper_r2, "CC": paper_cc},
            "protocol": protocol,
        }
        base.atomic_json(output_dir / "run_summary.json", summary)
        base.atomic_json(output_dir / "progress.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    except Exception as error:
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "failure.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (output_dir / "progress.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()
