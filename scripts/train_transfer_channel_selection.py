"""Run transfer-aware channel selection and source-to-target SNN fine-tuning."""

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
from typing import Any

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data" / "sabes_zenodo" / "master_mat"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
SESSIONS = (
    "indy_20160622_01",
    "indy_20160630_01",
    "indy_20170124_01",
    "indy_20170127_03",
    "indy_20170131_02",
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


within = load_module(SCRIPT_DIR / "train_channel_selection.py", "within_session_trainer")
transfer = load_module(
    SCRIPT_DIR / "internal" / "transfer_selection_core.py",
    "transfer_selection_core",
)
masking = within.masking
base = within.base


def output_dir(args: argparse.Namespace) -> Path:
    root = OUTPUT_ROOT / args.experiment_name
    if args.smoke:
        root = root / "_smoke"
    pair = f"{args.source_session}_to_{args.target_session}"
    return root / pair / f"top{args.keep_channels}"


def metric_mean(frame: pd.DataFrame) -> dict[str, float]:
    row = frame.loc[frame["axis"] == "mean"].iloc[0]
    return {key: float(row[key]) for key in ("R2", "CC", "RMSE")}


def velocity_normalization_stats(
    session: Any,
    windows: list[Any],
    *,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit per-axis velocity z-score statistics on explicitly allowed windows only."""
    if not windows:
        raise ValueError("At least one window is required for velocity normalization")
    values = np.concatenate(
        [session.velocity[window.start : window.end] for window in windows],
        axis=0,
    ).astype(np.float64, copy=False)
    mean = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0), epsilon)
    return mean.astype(np.float32), scale.astype(np.float32), len(values)


def normalized_velocity_session(
    session: Any,
    mean: np.ndarray,
    scale: np.ndarray,
) -> Any:
    velocity = ((session.velocity - mean) / scale).astype(np.float32)
    return replace(session, velocity=velocity)


def restore_velocity(
    values: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return values * scale + mean


def evaluate_window_average_raw_units(
    model: Any,
    session: Any,
    windows: list[Any],
    neural_lead_ms: int,
    device: torch.device,
    selected_channels: np.ndarray,
    velocity_mean: np.ndarray,
    velocity_scale: np.ndarray,
    *,
    max_steps: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate normalized decoding while reporting all metrics in original units."""
    rows: list[dict[str, Any]] = []
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
        target = restore_velocity(target, velocity_mean, velocity_scale)
        prediction = restore_velocity(prediction, velocity_mean, velocity_scale)
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
    summary = {metric: float(frame[metric].mean()) for metric in ("R2", "CC", "RMSE")}
    return frame, summary, np.stack(targets), np.stack(predictions)


def target_baseline_path(target_session: str) -> Path:
    return OUTPUT_ROOT / "baseline_96ch" / target_session / "run_summary.json"


def target_within_selection_path(target_session: str, keep_channels: int) -> Path:
    return (
        OUTPUT_ROOT
        / "channel_selection_64_32ch"
        / target_session
        / f"top{keep_channels}"
        / "run_summary.json"
    )


def training_config(
    *,
    keep_channels: int,
    seed: int,
    epochs: int,
    learning_rate: float,
    early_stopping_patience: int,
) -> Any:
    return replace(
        masking.MaskingConfig(),
        keep_channels=keep_channels,
        seed=seed,
        ranking_fraction=0.5,
        epochs=epochs,
        neural_lead_ms=0,
        learning_rate=learning_rate,
        optimizer_name="adamw",
        weight_decay=1e-4,
        lr_plateau_patience=3,
        lr_plateau_factor=0.5,
        early_stopping_patience=early_stopping_patience,
        beta_max=0.999,
        shuffle_training_windows=True,
        validation_drop_last=False,
        test_evaluation="continuous_single_state_reset",
        posthoc_random_repeats=0,
    )


def run_or_load_stage(
    *,
    model: Any,
    train_loader: Any,
    validation_loader: Any,
    config: Any,
    device: torch.device,
    stage_dir: Path,
    resume: bool,
    stage_name: str,
) -> tuple[pd.DataFrame, int, float, dict[str, torch.Tensor], float, str, str]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    summary_path = stage_dir / "stage_summary.json"
    model_path = stage_dir / "best_model.pt"
    history_path = stage_dir / "training_history.csv"
    if resume and summary_path.exists() and model_path.exists() and history_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") == "complete":
            saved = torch.load(model_path, map_location="cpu", weights_only=False)
            saved_config = saved.get("config")
            current_config = asdict(config)
            if saved_config != current_config:
                raise RuntimeError(
                    f"Completed stage configuration changed for {stage_name}; "
                    "use a new experiment name so results from different protocols "
                    "cannot be mixed"
                )
            print(f"SKIP complete stage: {stage_name}", flush=True)
            return (
                pd.read_csv(history_path),
                int(saved["best_epoch"]),
                float(saved["best_validation_loss"]),
                saved["model_state_dict"],
                float(summary["training_elapsed_seconds"]),
                str(summary["training_started_at"]),
                str(summary["training_finished_at"]),
            )

    result = base.train_resumable(
        model,
        train_loader,
        validation_loader,
        config,
        device,
        stage_dir,
        resume=resume,
    )
    history, best_epoch, best_loss, best_state, elapsed, started, finished = result
    history.to_csv(history_path, index=False)
    base.atomic_torch_save(
        model_path,
        {
            "stage": stage_name,
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "config": asdict(config),
        },
    )
    base.atomic_json(
        summary_path,
        {
            "status": "complete",
            "stage": stage_name,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "training_elapsed_seconds": elapsed,
            "training_started_at": started,
            "training_finished_at": finished,
        },
    )
    return result


def make_protocol(
    args: argparse.Namespace,
    selection_config: Any,
    source_split: dict[str, int],
    target_split: dict[str, int],
    calibration_split: dict[str, int],
    normalization: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment_name": args.experiment_name,
        "analysis_type": "transfer_aware_channel_selection_pretrain_finetune",
        "source_session": args.source_session,
        "target_session": args.target_session,
        "selection": {
            "concept": "stationarity + continuous-velocity importance - redundancy",
            "stationarity": "1 - Jensen-Shannon distance of unpaired 50 ms count histograms",
            "importance": (
                "RMS symmetrical uncertainty for source-defined quantile bins of vx/vy; "
                "source and target calibration receive equal session-level weight by default"
            ),
            "redundancy": (
                "mean of source and target-calibration pairwise channel SU matrices; "
                "greedy penalty against channels already selected"
            ),
            "fixed_k": args.keep_channels,
            "leakage_control": (
                "source train and target calibration only; target validation/test excluded"
            ),
            "config": asdict(selection_config),
        },
        "target_calibration": {
            "reconstructed_tasks": args.calibration_tasks,
            "approximate_reaches": 3 * args.calibration_tasks,
            "fine_tune_train_tasks": args.calibration_train_tasks,
            "fine_tune_validation_tasks": (args.calibration_tasks - args.calibration_train_tasks),
        },
        "shared_preprocessing": {
            "input": "96-channel binary GT MUA at 1 kHz",
            "task": "three consecutive reaches",
            "task_window_steps": 3_876,
            "task_offset_ms": -32,
            "split": "chronological floor 80/10/remainder independently per session",
            "neural_lead_ms": 0,
            "target": "continuous vx/vy",
        },
        "training": {
            "seed": args.seed,
            "source_pretrain_epochs": args.pretrain_epochs,
            "target_finetune_epochs": args.finetune_epochs,
            "target_scratch_epochs": args.scratch_epochs,
            "source_learning_rate": args.pretrain_learning_rate,
            "target_learning_rate": args.finetune_learning_rate,
            "target_scratch_learning_rate": args.scratch_learning_rate,
            "source_early_stopping_patience": args.pretrain_early_stopping_patience,
            "target_early_stopping_patience": args.finetune_early_stopping_patience,
            "target_scratch_early_stopping_patience": args.scratch_early_stopping_patience,
            "initialization": (
                "fresh K-input source model; target stage starts from the best source state"
            ),
            "target_test_policy": "continuous held-out span, state reset once",
            "target_scratch_control": args.run_target_scratch_control,
            "velocity_normalization": normalization,
        },
        "source_split_counts": source_split,
        "target_split_counts": target_split,
        "calibration_split_counts": calibration_split,
        "smoke": args.smoke,
    }


def run_experiment(args: argparse.Namespace) -> None:
    if args.source_session == args.target_session:
        raise ValueError("source_session and target_session must differ")
    if not 1 <= args.calibration_train_tasks < args.calibration_tasks:
        raise ValueError(
            "calibration_train_tasks must be positive and smaller than calibration_tasks"
        )
    out = output_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device("cpu")
    pretrain_epochs = min(args.pretrain_epochs, 2) if args.smoke else args.pretrain_epochs
    finetune_epochs = min(args.finetune_epochs, 2) if args.smoke else args.finetune_epochs
    scratch_epochs = min(args.scratch_epochs, 2) if args.smoke else args.scratch_epochs
    calibration_tasks = min(args.calibration_tasks, 4) if args.smoke else args.calibration_tasks
    calibration_train_tasks = (
        min(args.calibration_train_tasks, calibration_tasks - 1)
        if args.smoke
        else args.calibration_train_tasks
    )
    selection_config = transfer.TransferSelectionConfig(
        aggregation_bin_ms=args.aggregation_bin_ms,
        importance_quantile_bins=args.importance_quantile_bins,
        source_importance_weight=args.source_importance_weight,
        stationarity_weight=args.stationarity_weight,
        redundancy_weight=args.redundancy_weight,
        histogram_pseudocount=args.histogram_pseudocount,
    )
    base.atomic_json(
        out / "progress.json",
        {
            "status": "preprocessing",
            "source_session": args.source_session,
            "target_session": args.target_session,
            "keep_channels": args.keep_channels,
            "updated_at": base.now_iso(),
        },
    )
    pipeline_started = time.perf_counter()
    try:
        source_session = base.prepare_sabes_session(DATA_ROOT / f"{args.source_session}.mat")
        target_session = base.prepare_sabes_session(DATA_ROOT / f"{args.target_session}.mat")
        if source_session.channel_names != target_session.channel_names:
            raise RuntimeError("Physical channel names do not match across sessions")

        source_all = within.reconstructed_task_windows(source_session)
        target_all = within.reconstructed_task_windows(target_session)
        source_train, source_validation, source_test = within.floor_chronological_split(source_all)
        target_train, target_validation, target_test = within.floor_chronological_split(target_all)
        if calibration_tasks > len(target_train):
            raise ValueError("Target training split is smaller than the calibration request")
        target_calibration = target_train[:calibration_tasks]
        calibration_train = target_calibration[:calibration_train_tasks]
        calibration_validation = target_calibration[calibration_train_tasks:]

        if args.velocity_normalization == "zscore":
            source_velocity_mean, source_velocity_scale, source_fit_samples = (
                velocity_normalization_stats(source_session, source_train)
            )
            target_velocity_mean, target_velocity_scale, target_fit_samples = (
                velocity_normalization_stats(target_session, target_calibration)
            )
            source_decode_session = normalized_velocity_session(
                source_session, source_velocity_mean, source_velocity_scale
            )
            target_decode_session = normalized_velocity_session(
                target_session, target_velocity_mean, target_velocity_scale
            )
        else:
            source_velocity_mean = np.zeros(2, dtype=np.float32)
            source_velocity_scale = np.ones(2, dtype=np.float32)
            target_velocity_mean = np.zeros(2, dtype=np.float32)
            target_velocity_scale = np.ones(2, dtype=np.float32)
            source_fit_samples = 0
            target_fit_samples = 0
            source_decode_session = source_session
            target_decode_session = target_session

        normalization = {
            "method": args.velocity_normalization,
            "source_fit_scope": "source_train_only",
            "target_fit_scope": "target_calibration_only",
            "validation_and_test_excluded": True,
            "source_mean": source_velocity_mean.tolist(),
            "source_scale": source_velocity_scale.tolist(),
            "source_fit_samples": source_fit_samples,
            "target_mean": target_velocity_mean.tolist(),
            "target_scale": target_velocity_scale.tolist(),
            "target_fit_samples": target_fit_samples,
            "reported_metrics": "original_velocity_units",
        }

        source_aggregated = transfer.aggregate_windows(
            source_session,
            source_train,
            bin_ms=selection_config.aggregation_bin_ms,
            neural_lead_ms=0,
        )
        target_aggregated = transfer.aggregate_windows(
            target_session,
            target_calibration,
            bin_ms=selection_config.aggregation_bin_ms,
            neural_lead_ms=0,
        )
        ranking_result = transfer.rank_transfer_channels(
            source_aggregated,
            target_aggregated,
            selection_config,
            channel_names=source_session.channel_names,
        )
        kept = np.sort(ranking_result.order[: args.keep_channels])
        masked = np.sort(ranking_result.order[args.keep_channels :])
        ranking = ranking_result.ranking.copy()
        ranking["kept"] = ranking["rank"] <= args.keep_channels

        source_split = {
            "all": len(source_all),
            "train": len(source_train),
            "validation": len(source_validation),
            "test": len(source_test),
        }
        target_split = {
            "all": len(target_all),
            "train": len(target_train),
            "validation": len(target_validation),
            "test": len(target_test),
        }
        calibration_split = {
            "all": len(target_calibration),
            "fine_tune_train": len(calibration_train),
            "fine_tune_validation": len(calibration_validation),
        }
        protocol_args = argparse.Namespace(**vars(args))
        protocol_args.pretrain_epochs = pretrain_epochs
        protocol_args.finetune_epochs = finetune_epochs
        protocol_args.scratch_epochs = scratch_epochs
        protocol_args.calibration_tasks = calibration_tasks
        protocol_args.calibration_train_tasks = calibration_train_tasks
        protocol = make_protocol(
            protocol_args,
            selection_config,
            source_split,
            target_split,
            calibration_split,
            normalization,
        )
        base.atomic_json(out / "protocol.json", protocol)
        ranking.to_csv(out / "channel_ranking.csv", index=False)
        np.save(out / "channel_redundancy.npy", ranking_result.redundancy_matrix)
        base.atomic_json(
            out / "channel_mask.json",
            {
                "source_session": args.source_session,
                "target_session": args.target_session,
                "keep_channels": args.keep_channels,
                "kept_indices_0based": kept.tolist(),
                "kept_channel_names": [source_session.channel_names[index] for index in kept],
                "masked_indices_0based": masked.tolist(),
                "masked_channel_names": [source_session.channel_names[index] for index in masked],
                "greedy_order_0based": ranking_result.order.tolist(),
                "source_velocity_bin_edges": [
                    edge.tolist() for edge in ranking_result.velocity_bin_edges
                ],
                "source_selection_bins": len(source_aggregated.counts),
                "target_calibration_bins": len(target_aggregated.counts),
            },
        )
        transfer.plot_transfer_ranking(
            out / "channel_ranking.png",
            ranking,
            keep_channels=args.keep_channels,
            title=f"{args.source_session} -> {args.target_session}",
        )
        print(
            f"selected source={args.source_session} target={args.target_session} "
            f"top{args.keep_channels} calibration_tasks={len(target_calibration)}",
            flush=True,
        )

        if args.selection_only:
            summary = {
                "status": "selection_complete",
                "analysis_type": "transfer_aware_channel_selection_only",
                "source_session": args.source_session,
                "target_session": args.target_session,
                "keep_channels": args.keep_channels,
                "kept_indices_0based": kept.tolist(),
                "timing_seconds": {"pipeline": time.perf_counter() - pipeline_started},
                "protocol": protocol,
            }
            base.atomic_json(out / "run_summary.json", summary)
            base.atomic_json(out / "progress.json", summary)
            print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
            return

        truncate_steps = 256 if args.smoke else None
        evaluation_max_steps = 512 if args.smoke else None
        source_loader_train = source_train[:12] if args.smoke else source_train
        source_loader_validation = source_validation[:4] if args.smoke else source_validation
        pretrain_config = training_config(
            keep_channels=args.keep_channels,
            seed=args.seed,
            epochs=pretrain_epochs,
            learning_rate=args.pretrain_learning_rate,
            early_stopping_patience=args.pretrain_early_stopping_patience,
        )
        source_train_loader, source_validation_loader = masking.make_selected_loaders(
            source_decode_session,
            source_loader_train,
            source_loader_validation,
            kept,
            pretrain_config,
            truncate_steps=truncate_steps,
        )
        base.set_reproducible_seed(pretrain_config.seed)
        source_model = base.MartisSNN(
            input_size=args.keep_channels,
            output_size=2,
            hidden_sizes=pretrain_config.hidden_sizes,
            threshold=pretrain_config.threshold,
            beta_init=pretrain_config.beta_init,
            optimized_forward=args.optimized_forward,
        )
        source_stage = run_or_load_stage(
            model=source_model,
            train_loader=source_train_loader,
            validation_loader=source_validation_loader,
            config=pretrain_config,
            device=device,
            stage_dir=out / "source_pretrain",
            resume=args.resume,
            stage_name="source_pretrain",
        )
        (
            _,
            source_best_epoch,
            source_best_loss,
            source_best_state,
            source_elapsed,
            _,
            _,
        ) = source_stage
        source_model.load_state_dict(source_best_state)

        finetune_config = training_config(
            keep_channels=args.keep_channels,
            seed=args.seed,
            epochs=finetune_epochs,
            learning_rate=args.finetune_learning_rate,
            early_stopping_patience=args.finetune_early_stopping_patience,
        )
        calibration_train_loader, calibration_validation_loader = masking.make_selected_loaders(
            target_decode_session,
            calibration_train,
            calibration_validation,
            kept,
            finetune_config,
            truncate_steps=truncate_steps,
        )
        target_model = base.MartisSNN(
            input_size=args.keep_channels,
            output_size=2,
            hidden_sizes=finetune_config.hidden_sizes,
            threshold=finetune_config.threshold,
            beta_init=finetune_config.beta_init,
            optimized_forward=args.optimized_forward,
        )
        target_model.load_state_dict(source_best_state)
        base.set_reproducible_seed(finetune_config.seed)
        target_stage = run_or_load_stage(
            model=target_model,
            train_loader=calibration_train_loader,
            validation_loader=calibration_validation_loader,
            config=finetune_config,
            device=device,
            stage_dir=out / "target_finetune",
            resume=args.resume,
            stage_name="target_finetune",
        )
        (
            _,
            target_best_epoch,
            target_best_loss,
            target_best_state,
            target_elapsed,
            _,
            _,
        ) = target_stage
        target_model.load_state_dict(target_best_state)

        scratch_model = None
        scratch_best_epoch = None
        scratch_best_loss = None
        scratch_elapsed = None
        if args.run_target_scratch_control:
            scratch_config = training_config(
                keep_channels=args.keep_channels,
                seed=args.seed,
                epochs=scratch_epochs,
                learning_rate=args.scratch_learning_rate,
                early_stopping_patience=args.scratch_early_stopping_patience,
            )
            base.set_reproducible_seed(scratch_config.seed)
            scratch_model = base.MartisSNN(
                input_size=args.keep_channels,
                output_size=2,
                hidden_sizes=scratch_config.hidden_sizes,
                threshold=scratch_config.threshold,
                beta_init=scratch_config.beta_init,
                optimized_forward=args.optimized_forward,
            )
            scratch_stage = run_or_load_stage(
                model=scratch_model,
                train_loader=calibration_train_loader,
                validation_loader=calibration_validation_loader,
                config=scratch_config,
                device=device,
                stage_dir=out / "target_scratch",
                resume=args.resume,
                stage_name="target_scratch_calibration_only",
            )
            (
                _,
                scratch_best_epoch,
                scratch_best_loss,
                scratch_best_state,
                scratch_elapsed,
                _,
                _,
            ) = scratch_stage
            scratch_model.load_state_dict(scratch_best_state)

        validation_continuous = [
            base.TaskWindow(
                task_index=-1,
                start=target_validation[0].start,
                end=target_test[0].start,
            )
        ]
        test_continuous = [
            base.TaskWindow(
                task_index=-1,
                start=target_test[0].start,
                end=len(target_decode_session.velocity),
            )
        ]
        validation_time, validation_target, validation_prediction = masking.predict_continuous(
            target_model,
            target_decode_session,
            validation_continuous,
            finetune_config.neural_lead_ms,
            device,
            selected_channels=kept,
            max_steps=evaluation_max_steps,
        )
        test_time, test_target_values, test_prediction = masking.predict_continuous(
            target_model,
            target_decode_session,
            test_continuous,
            finetune_config.neural_lead_ms,
            device,
            selected_channels=kept,
            max_steps=evaluation_max_steps,
        )
        validation_target = restore_velocity(
            validation_target, target_velocity_mean, target_velocity_scale
        )
        validation_prediction = restore_velocity(
            validation_prediction, target_velocity_mean, target_velocity_scale
        )
        test_target_values = restore_velocity(
            test_target_values, target_velocity_mean, target_velocity_scale
        )
        test_prediction = restore_velocity(
            test_prediction, target_velocity_mean, target_velocity_scale
        )
        validation_metrics = base.regression_metrics(validation_target, validation_prediction)
        test_metrics = base.regression_metrics(test_target_values, test_prediction)
        _, source_only_target, source_only_prediction = masking.predict_continuous(
            source_model,
            target_decode_session,
            test_continuous,
            pretrain_config.neural_lead_ms,
            device,
            selected_channels=kept,
            max_steps=evaluation_max_steps,
        )
        source_only_target = restore_velocity(
            source_only_target, target_velocity_mean, target_velocity_scale
        )
        source_only_prediction = restore_velocity(
            source_only_prediction, target_velocity_mean, target_velocity_scale
        )
        source_only_metrics = base.regression_metrics(source_only_target, source_only_prediction)
        scratch_metrics = None
        if scratch_model is not None:
            _, scratch_target, scratch_prediction = masking.predict_continuous(
                scratch_model,
                target_decode_session,
                test_continuous,
                finetune_config.neural_lead_ms,
                device,
                selected_channels=kept,
                max_steps=evaluation_max_steps,
            )
            scratch_target = restore_velocity(
                scratch_target, target_velocity_mean, target_velocity_scale
            )
            scratch_prediction = restore_velocity(
                scratch_prediction, target_velocity_mean, target_velocity_scale
            )
            scratch_metrics = base.regression_metrics(scratch_target, scratch_prediction)
        test_windows_for_report = target_test[:2] if args.smoke else target_test
        window_metrics, window_mean, window_targets, window_predictions = (
            evaluate_window_average_raw_units(
                target_model,
                target_decode_session,
                test_windows_for_report,
                finetune_config.neural_lead_ms,
                device,
                kept,
                target_velocity_mean,
                target_velocity_scale,
                max_steps=evaluation_max_steps,
            )
        )
        validation_metrics.to_csv(out / "validation_metrics_continuous.csv", index=False)
        test_metrics.to_csv(out / "test_metrics_continuous.csv", index=False)
        source_only_metrics.to_csv(
            out / "source_only_target_test_metrics_continuous.csv", index=False
        )
        if scratch_metrics is not None:
            scratch_metrics.to_csv(out / "target_scratch_test_metrics_continuous.csv", index=False)
        window_metrics.to_csv(out / "test_metrics_per_window.csv", index=False)
        np.savez_compressed(
            out / "test_predictions.npz",
            validation_time_sec=validation_time,
            validation_target=validation_target,
            validation_prediction=validation_prediction,
            test_time_sec=test_time,
            test_target=test_target_values,
            test_prediction=test_prediction,
            window_target=window_targets,
            window_prediction=window_predictions,
        )
        base.atomic_torch_save(
            out / "best_model.pt",
            {
                "model_state_dict": target_best_state,
                "source_session": args.source_session,
                "target_session": args.target_session,
                "keep_channels": args.keep_channels,
                "kept_channels_0based": kept.tolist(),
                "source_best_epoch": source_best_epoch,
                "target_best_epoch": target_best_epoch,
                "target_scratch_best_epoch": scratch_best_epoch,
                "source_best_validation_loss": source_best_loss,
                "target_best_validation_loss": target_best_loss,
                "target_scratch_best_validation_loss": scratch_best_loss,
                "pretrain_config": asdict(pretrain_config),
                "finetune_config": asdict(finetune_config),
                "scratch_config": (
                    asdict(scratch_config) if args.run_target_scratch_control else None
                ),
                "protocol": protocol,
                "velocity_normalization": normalization,
            },
        )

        baseline_path = target_baseline_path(args.target_session)
        baseline_test = None
        delta = None
        within_selection_test = None
        delta_vs_within_selection = None
        if baseline_path.exists() and not args.smoke:
            baseline_summary = json.loads(baseline_path.read_text(encoding="utf-8"))
            baseline_test = baseline_summary["test_continuous"]
            current = metric_mean(test_metrics)
            delta = {key: current[key] - float(baseline_test[key]) for key in ("R2", "CC", "RMSE")}
        within_path = target_within_selection_path(args.target_session, args.keep_channels)
        if within_path.exists() and not args.smoke:
            within_summary = json.loads(within_path.read_text(encoding="utf-8"))
            within_selection_test = within_summary["test_continuous"]
            current = metric_mean(test_metrics)
            delta_vs_within_selection = {
                key: current[key] - float(within_selection_test[key])
                for key in ("R2", "CC", "RMSE")
            }
        summary = {
            "status": "complete",
            "analysis_type": "transfer_aware_channel_selection_pretrain_finetune",
            "source_session": args.source_session,
            "target_session": args.target_session,
            "experiment_name": args.experiment_name,
            "smoke": args.smoke,
            "device": str(device),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "keep_channels": args.keep_channels,
            "kept_indices_0based": kept.tolist(),
            "source_best_epoch": source_best_epoch,
            "target_best_epoch": target_best_epoch,
            "target_scratch_best_epoch": scratch_best_epoch,
            "source_best_validation_loss": source_best_loss,
            "target_best_validation_loss": target_best_loss,
            "target_scratch_best_validation_loss": scratch_best_loss,
            "training_elapsed_seconds": {
                "source_pretrain": source_elapsed,
                "target_finetune": target_elapsed,
                "target_scratch": scratch_elapsed,
            },
            "source_only_target_test": metric_mean(source_only_metrics),
            "target_scratch_test": (
                metric_mean(scratch_metrics) if scratch_metrics is not None else None
            ),
            "validation_continuous": metric_mean(validation_metrics),
            "test_continuous": metric_mean(test_metrics),
            "test_per_window_mean": window_mean,
            "baseline_96": baseline_test,
            "delta_vs_target_baseline_96": delta,
            "within_session_selection": within_selection_test,
            "delta_vs_within_session_selection": delta_vs_within_selection,
            "timing_seconds": {"pipeline_process": time.perf_counter() - pipeline_started},
            "protocol": protocol,
            "velocity_normalization": normalization,
        }
        base.atomic_json(out / "run_summary.json", summary)
        base.atomic_json(out / "progress.json", summary)
        (out / "failure.json").unlink(missing_ok=True)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    except Exception as error:
        failure = {
            "status": "failed",
            "source_session": args.source_session,
            "target_session": args.target_session,
            "keep_channels": args.keep_channels,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "updated_at": base.now_iso(),
        }
        base.atomic_json(out / "failure.json", failure)
        base.atomic_json(out / "progress.json", failure)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-session", choices=SESSIONS)
    parser.add_argument("--target-session", choices=SESSIONS)
    parser.add_argument("--keep-channels", type=int, choices=(32, 48, 64, 80), default=64)
    parser.add_argument("--experiment-name", default="transfer_sutl_64ch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--velocity-normalization",
        choices=("none", "zscore"),
        default="none",
    )
    parser.add_argument("--calibration-tasks", type=int, default=10)
    parser.add_argument("--calibration-train-tasks", type=int, default=8)
    parser.add_argument("--aggregation-bin-ms", type=int, default=50)
    parser.add_argument("--importance-quantile-bins", type=int, default=8)
    parser.add_argument("--source-importance-weight", type=float, default=0.5)
    parser.add_argument("--stationarity-weight", type=float, default=0.5)
    parser.add_argument("--redundancy-weight", type=float, default=0.25)
    parser.add_argument("--histogram-pseudocount", type=float, default=1e-9)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--scratch-epochs", type=int, default=100)
    parser.add_argument("--pretrain-learning-rate", type=float, default=1e-3)
    parser.add_argument("--finetune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--scratch-learning-rate", type=float, default=1e-3)
    parser.add_argument("--pretrain-early-stopping-patience", type=int, default=10)
    parser.add_argument("--finetune-early-stopping-patience", type=int, default=8)
    parser.add_argument("--scratch-early-stopping-patience", type=int, default=10)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--optimized-forward", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-target-scratch-control", action="store_true")
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        transfer.self_test()
        print("transfer channel-selection self-test passed")
        return
    if args.source_session is None or args.target_session is None:
        raise ValueError("--source-session and --target-session are required")
    run_experiment(args)


if __name__ == "__main__":
    main()
