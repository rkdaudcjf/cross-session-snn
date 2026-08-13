"""Run training or quantized evaluation from a validated YAML preset."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BASELINE_TRAINER = SCRIPT_DIR / "train_baseline.py"
SELECTION_TRAINER = SCRIPT_DIR / "train_channel_selection.py"
TRANSFER_SELECTION_TRAINER = SCRIPT_DIR / "train_transfer_channel_selection.py"
FIXED_POINT_EVALUATOR = SCRIPT_DIR / "evaluate_mixed_fixed_point_baseline.py"
FIXED_POINT_SUMMARIZER = SCRIPT_DIR / "summarize_mixed_fixed_point.py"
BASELINE_EXPERIMENT_NAME = "baseline_96ch"
KNOWN_SESSIONS = {
    "indy_20160622_01",
    "indy_20160630_01",
    "indy_20170124_01",
    "indy_20170127_03",
    "indy_20170131_02",
}


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("The config root must be a YAML mapping")
    return payload


def reject_unknown(mapping: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"Unknown {location} keys: {sorted(unknown)}")


def require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"'{key}' must be a mapping")
    return value


def validate_name(value: Any, location: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError(f"{location} must contain only letters, numbers, underscores, and hyphens")
    return value


def validate_sessions(experiment: dict[str, Any]) -> None:
    sessions = experiment.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("experiment.sessions must be a non-empty list")
    unknown_sessions = set(sessions) - KNOWN_SESSIONS
    if unknown_sessions:
        raise ValueError(f"Unknown sessions: {sorted(unknown_sessions)}")
    if len(sessions) != len(set(sessions)):
        raise ValueError("experiment.sessions contains duplicates")


def validate_common(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    reject_unknown(payload, {"schema_version", "experiment", "training"}, "top-level")
    if payload.get("schema_version") != 1:
        raise ValueError("Only schema_version: 1 is supported")
    experiment = require_mapping(payload, "experiment")
    training = require_mapping(payload, "training")
    validate_name(experiment.get("name"), "experiment.name")
    validate_sessions(experiment)
    for key in ("warmup_epochs", "epochs", "cpu_threads"):
        if not isinstance(training.get(key), int) or training[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if training["warmup_epochs"] >= training["epochs"]:
        raise ValueError("training.warmup_epochs must be smaller than training.epochs")
    for key in ("resume", "optimized_forward"):
        if not isinstance(training.get(key), bool):
            raise TypeError(f"training.{key} must be true or false")
    return experiment, training


def validate_mixed_fixed_point_config(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    reject_unknown(
        payload,
        {"schema_version", "experiment", "quantization", "evaluation"},
        "top-level",
    )
    if payload.get("schema_version") != 1:
        raise ValueError("Only schema_version: 1 is supported")
    experiment = require_mapping(payload, "experiment")
    quantization = require_mapping(payload, "quantization")
    evaluation = require_mapping(payload, "evaluation")
    reject_unknown(
        experiment,
        {"type", "name", "source_name", "sessions"},
        "experiment",
    )
    reject_unknown(
        quantization,
        {
            "input_bits",
            "weight_bits",
            "potential_bits",
            "decay_bits",
            "decay_fractional_bits",
            "weight_scale_mode",
            "rounding",
            "saturation",
        },
        "quantization",
    )
    reject_unknown(evaluation, {"cpu_threads", "evaluate_fp32"}, "evaluation")
    validate_name(experiment.get("name"), "experiment.name")
    validate_name(experiment.get("source_name"), "experiment.source_name")
    validate_sessions(experiment)

    reported_widths = {
        "input_bits": 1,
        "weight_bits": 8,
        "potential_bits": 32,
        "decay_bits": 13,
        "decay_fractional_bits": 13,
    }
    for key, expected in reported_widths.items():
        if quantization.get(key) != expected:
            raise ValueError(
                f"quantization.{key} must be {expected} for this paper-precision preset"
            )
    if quantization.get("weight_scale_mode") not in {"pow2", "maxabs"}:
        raise ValueError("quantization.weight_scale_mode must be 'pow2' or 'maxabs'")
    if quantization.get("rounding") != "nearest_half_away_from_zero":
        raise ValueError("quantization.rounding must be 'nearest_half_away_from_zero'")
    if quantization.get("saturation") is not True:
        raise ValueError("quantization.saturation must be true")
    if not isinstance(evaluation.get("cpu_threads"), int) or evaluation["cpu_threads"] < 1:
        raise ValueError("evaluation.cpu_threads must be a positive integer")
    if not isinstance(evaluation.get("evaluate_fp32"), bool):
        raise TypeError("evaluation.evaluate_fp32 must be true or false")
    return experiment, quantization, evaluation


def validate_transfer_config(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    reject_unknown(
        payload,
        {"schema_version", "experiment", "selection", "training"},
        "top-level",
    )
    if payload.get("schema_version") != 1:
        raise ValueError("Only schema_version: 1 is supported")
    experiment = require_mapping(payload, "experiment")
    selection = require_mapping(payload, "selection")
    training = require_mapping(payload, "training")
    training.setdefault("velocity_normalization", "none")
    reject_unknown(
        experiment,
        {"type", "name", "pairs", "channel_counts"},
        "experiment",
    )
    reject_unknown(
        selection,
        {
            "calibration_tasks",
            "calibration_train_tasks",
            "aggregation_bin_ms",
            "importance_quantile_bins",
            "source_importance_weight",
            "stationarity_weight",
            "redundancy_weight",
            "histogram_pseudocount",
        },
        "selection",
    )
    reject_unknown(
        training,
        {
            "seed",
            "pretrain_epochs",
            "finetune_epochs",
            "scratch_epochs",
            "pretrain_learning_rate",
            "finetune_learning_rate",
            "scratch_learning_rate",
            "pretrain_early_stopping_patience",
            "finetune_early_stopping_patience",
            "scratch_early_stopping_patience",
            "cpu_threads",
            "resume",
            "optimized_forward",
            "run_target_scratch_control",
            "velocity_normalization",
        },
        "training",
    )
    validate_name(experiment.get("name"), "experiment.name")
    pairs = experiment.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("experiment.pairs must be a non-empty list")
    normalized_pairs = []
    seen_pairs: set[tuple[str, str]] = set()
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise TypeError(f"experiment.pairs[{index}] must be a mapping")
        reject_unknown(pair, {"source", "target", "role"}, f"experiment.pairs[{index}]")
        source = pair.get("source")
        target = pair.get("target")
        role = pair.get("role")
        if source not in KNOWN_SESSIONS or target not in KNOWN_SESSIONS:
            raise ValueError(f"Unknown transfer pair session: {source} -> {target}")
        if source == target:
            raise ValueError("Transfer source and target must differ")
        validate_name(role, f"experiment.pairs[{index}].role")
        key = (str(source), str(target))
        if key in seen_pairs:
            raise ValueError(f"Duplicate transfer pair: {source} -> {target}")
        seen_pairs.add(key)
        normalized_pairs.append(pair)
    experiment["pairs"] = normalized_pairs

    channel_counts = experiment.get("channel_counts")
    if not isinstance(channel_counts, list) or not channel_counts:
        raise ValueError("experiment.channel_counts must be a non-empty list")
    if not all(isinstance(value, int) for value in channel_counts):
        raise TypeError("experiment.channel_counts must contain integers")
    if not set(channel_counts) <= {32, 48, 64, 80}:
        raise ValueError("Transfer channel counts must be selected from 32, 48, 64, 80")

    positive_integer_keys = {
        "calibration_tasks",
        "calibration_train_tasks",
        "aggregation_bin_ms",
        "importance_quantile_bins",
    }
    for key in positive_integer_keys:
        if not isinstance(selection.get(key), int) or selection[key] < 1:
            raise ValueError(f"selection.{key} must be a positive integer")
    if selection["calibration_train_tasks"] >= selection["calibration_tasks"]:
        raise ValueError("selection.calibration_train_tasks must be smaller than calibration_tasks")
    for key in ("source_importance_weight", "stationarity_weight"):
        value = selection.get(key)
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"selection.{key} must be in [0, 1]")
    for key in ("redundancy_weight", "histogram_pseudocount"):
        value = selection.get(key)
        if not isinstance(value, (int, float)) or float(value) <= 0.0:
            raise ValueError(f"selection.{key} must be positive")

    for key in (
        "seed",
        "pretrain_epochs",
        "finetune_epochs",
        "scratch_epochs",
        "pretrain_early_stopping_patience",
        "finetune_early_stopping_patience",
        "scratch_early_stopping_patience",
        "cpu_threads",
    ):
        if not isinstance(training.get(key), int) or training[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    for key in (
        "pretrain_learning_rate",
        "finetune_learning_rate",
        "scratch_learning_rate",
    ):
        value = training.get(key)
        if not isinstance(value, (int, float)) or float(value) <= 0.0:
            raise ValueError(f"training.{key} must be positive")
    for key in ("resume", "optimized_forward", "run_target_scratch_control"):
        if not isinstance(training.get(key), bool):
            raise TypeError(f"training.{key} must be true or false")
    if training.get("velocity_normalization") not in {"none", "zscore"}:
        raise ValueError("training.velocity_normalization must be 'none' or 'zscore'")
    return experiment, selection, training


def is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    summary = json.loads(path.read_text(encoding="utf-8"))
    return summary.get("status") == "complete"


def checkpoint_epoch(path: Path) -> int:
    if not path.exists():
        return 0
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint["epoch"])


def execute(command: list[str], dry_run: bool) -> None:
    print(f"$ {shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def append_boolean_flag(command: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        command.append(flag)


def baseline_command(
    *,
    session: str,
    experiment_name: str,
    training: dict[str, Any],
    epochs: int,
    resume: bool,
    training_only: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(BASELINE_TRAINER),
        "--session",
        session,
        "--experiment-name",
        experiment_name,
        "--seed",
        str(training["seed"]),
        "--epochs",
        str(epochs),
        "--cpu-threads",
        str(training["cpu_threads"]),
        "--neural-lead-ms",
        str(training["neural_lead_ms"]),
        "--learning-rate",
        str(training["learning_rate"]),
        "--weight-decay",
        str(training["weight_decay"]),
        "--lr-plateau-patience",
        str(training["lr_plateau_patience"]),
        "--early-stopping-patience",
        str(training["early_stopping_patience"]),
        "--beta-max",
        str(training["beta_max"]),
    ]
    append_boolean_flag(command, training["optimized_forward"], "--optimized-forward")
    append_boolean_flag(command, training["shuffle_training_windows"], "--shuffle-training-windows")
    append_boolean_flag(
        command, training["keep_validation_remainder"], "--keep-validation-remainder"
    )
    append_boolean_flag(command, resume, "--resume")
    append_boolean_flag(command, training_only, "--training-only")
    return command


def run_baselines(
    experiment: dict[str, Any],
    training: dict[str, Any],
    selected_sessions: set[str] | None,
    dry_run: bool,
) -> None:
    reject_unknown(
        experiment,
        {"type", "name", "sessions"},
        "experiment",
    )
    reject_unknown(
        training,
        {
            "warmup_epochs",
            "epochs",
            "cpu_threads",
            "resume",
            "optimized_forward",
            "neural_lead_ms",
            "learning_rate",
            "weight_decay",
            "lr_plateau_patience",
            "early_stopping_patience",
            "beta_max",
            "shuffle_training_windows",
            "keep_validation_remainder",
        },
        "training",
    )
    required_training = {
        "neural_lead_ms",
        "learning_rate",
        "weight_decay",
        "lr_plateau_patience",
        "early_stopping_patience",
        "beta_max",
        "shuffle_training_windows",
        "keep_validation_remainder",
    }
    missing = required_training - set(training)
    if missing:
        raise ValueError(f"Missing baseline training keys: {sorted(missing)}")
    experiment_name = str(experiment["name"])

    sessions = [
        session
        for session in experiment["sessions"]
        if selected_sessions is None or session in selected_sessions
    ]
    if not sessions:
        raise ValueError("The session filter selected no baseline experiments")

    for session in sessions:
        output = PROJECT_ROOT / "outputs" / experiment_name / session
        summary = output / "run_summary.json"
        checkpoint = output / "last_checkpoint.pt"
        if is_complete(summary):
            print(f"SKIP complete baseline: {session} ({experiment_name})", flush=True)
            continue
        completed_epoch = checkpoint_epoch(checkpoint)
        if completed_epoch and not training["resume"]:
            raise RuntimeError(
                f"Checkpoint exists for {session}; set training.resume: true or choose a new experiment.name"
            )
        if completed_epoch < training["warmup_epochs"]:
            execute(
                baseline_command(
                    session=session,
                    experiment_name=experiment_name,
                    training=training,
                    epochs=training["warmup_epochs"],
                    resume=completed_epoch > 0,
                    training_only=True,
                ),
                dry_run,
            )
        execute(
            baseline_command(
                session=session,
                experiment_name=experiment_name,
                training=training,
                epochs=training["epochs"],
                resume=True,
                training_only=False,
            ),
            dry_run,
        )


def selection_command(
    *,
    session: str,
    channels: int,
    experiment_name: str,
    training: dict[str, Any],
    epochs: int,
    resume: bool,
    training_only: bool,
    run_index: int,
    total_runs: int,
) -> list[str]:
    command = [
        sys.executable,
        str(SELECTION_TRAINER),
        "--session",
        session,
        "--keep-channels",
        str(channels),
        "--experiment-name",
        experiment_name,
        "--epochs",
        str(epochs),
        "--display-max-epochs",
        str(training["epochs"]),
        "--cpu-threads",
        str(training["cpu_threads"]),
        "--batch-log-interval-seconds",
        str(training["log_interval_seconds"]),
        "--run-index",
        str(run_index),
        "--total-runs",
        str(total_runs),
    ]
    append_boolean_flag(command, training["optimized_forward"], "--optimized-forward")
    append_boolean_flag(command, resume, "--resume")
    append_boolean_flag(command, training_only, "--training-only")
    return command


def baseline_summary_path(session: str) -> Path:
    return PROJECT_ROOT / "outputs" / BASELINE_EXPERIMENT_NAME / session / "run_summary.json"


def run_channel_selection(
    experiment: dict[str, Any],
    training: dict[str, Any],
    selected_sessions: set[str] | None,
    selected_channels: set[int] | None,
    dry_run: bool,
) -> None:
    reject_unknown(
        experiment,
        {"type", "name", "sessions", "channel_counts"},
        "experiment",
    )
    reject_unknown(
        training,
        {
            "warmup_epochs",
            "epochs",
            "cpu_threads",
            "resume",
            "optimized_forward",
            "log_interval_seconds",
        },
        "training",
    )
    experiment_name = str(experiment["name"])
    channel_counts = experiment.get("channel_counts")
    if not isinstance(channel_counts, list) or not channel_counts:
        raise ValueError("experiment.channel_counts must be a non-empty list")
    if not set(channel_counts) <= {32, 64}:
        raise ValueError("Only 32 and 64 selected channels are supported")
    if not isinstance(training.get("log_interval_seconds"), (int, float)):
        raise TypeError("training.log_interval_seconds must be numeric")
    if training["log_interval_seconds"] <= 0:
        raise ValueError("training.log_interval_seconds must be positive")

    sessions = [
        session
        for session in experiment["sessions"]
        if selected_sessions is None or session in selected_sessions
    ]
    channels = [
        count for count in channel_counts if selected_channels is None or count in selected_channels
    ]
    runs = [(session, count) for session in sessions for count in channels]
    if not runs:
        raise ValueError("The filters selected no channel-selection experiments")

    for run_index, (session, count) in enumerate(runs, start=1):
        baseline = baseline_summary_path(session)
        if not dry_run and not is_complete(baseline):
            raise RuntimeError(f"A completed 96-channel baseline is required: {baseline}")
        output = PROJECT_ROOT / "outputs" / experiment_name / session / f"top{count}"
        summary = output / "run_summary.json"
        checkpoint = output / "last_checkpoint.pt"
        if is_complete(summary):
            print(f"SKIP complete selection: {session} top{count}", flush=True)
            continue
        completed_epoch = checkpoint_epoch(checkpoint)
        if completed_epoch and not training["resume"]:
            raise RuntimeError(
                f"Checkpoint exists for {session} top{count}; enable resume or choose a new experiment.name"
            )
        if completed_epoch < training["warmup_epochs"]:
            execute(
                selection_command(
                    session=session,
                    channels=count,
                    experiment_name=experiment_name,
                    training=training,
                    epochs=training["warmup_epochs"],
                    resume=completed_epoch > 0,
                    training_only=True,
                    run_index=run_index,
                    total_runs=len(runs),
                ),
                dry_run,
            )
        execute(
            selection_command(
                session=session,
                channels=count,
                experiment_name=experiment_name,
                training=training,
                epochs=training["epochs"],
                resume=True,
                training_only=False,
                run_index=run_index,
                total_runs=len(runs),
            ),
            dry_run,
        )


def transfer_selection_command(
    *,
    source: str,
    target: str,
    channels: int,
    experiment_name: str,
    selection: dict[str, Any],
    training: dict[str, Any],
    selection_only: bool,
    smoke: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(TRANSFER_SELECTION_TRAINER),
        "--source-session",
        source,
        "--target-session",
        target,
        "--keep-channels",
        str(channels),
        "--experiment-name",
        experiment_name,
        "--seed",
        str(training["seed"]),
        "--velocity-normalization",
        str(training["velocity_normalization"]),
        "--calibration-tasks",
        str(selection["calibration_tasks"]),
        "--calibration-train-tasks",
        str(selection["calibration_train_tasks"]),
        "--aggregation-bin-ms",
        str(selection["aggregation_bin_ms"]),
        "--importance-quantile-bins",
        str(selection["importance_quantile_bins"]),
        "--source-importance-weight",
        str(selection["source_importance_weight"]),
        "--stationarity-weight",
        str(selection["stationarity_weight"]),
        "--redundancy-weight",
        str(selection["redundancy_weight"]),
        "--histogram-pseudocount",
        str(selection["histogram_pseudocount"]),
        "--pretrain-epochs",
        str(training["pretrain_epochs"]),
        "--finetune-epochs",
        str(training["finetune_epochs"]),
        "--scratch-epochs",
        str(training["scratch_epochs"]),
        "--pretrain-learning-rate",
        str(training["pretrain_learning_rate"]),
        "--finetune-learning-rate",
        str(training["finetune_learning_rate"]),
        "--scratch-learning-rate",
        str(training["scratch_learning_rate"]),
        "--pretrain-early-stopping-patience",
        str(training["pretrain_early_stopping_patience"]),
        "--finetune-early-stopping-patience",
        str(training["finetune_early_stopping_patience"]),
        "--scratch-early-stopping-patience",
        str(training["scratch_early_stopping_patience"]),
        "--cpu-threads",
        str(training["cpu_threads"]),
    ]
    append_boolean_flag(command, training["optimized_forward"], "--optimized-forward")
    append_boolean_flag(command, training["resume"], "--resume")
    append_boolean_flag(
        command,
        training["run_target_scratch_control"],
        "--run-target-scratch-control",
    )
    append_boolean_flag(command, selection_only, "--selection-only")
    append_boolean_flag(command, smoke, "--smoke")
    return command


def run_transfer_channel_selection(
    experiment: dict[str, Any],
    selection: dict[str, Any],
    training: dict[str, Any],
    selected_sessions: set[str] | None,
    selected_channels: set[int] | None,
    selection_only: bool,
    smoke: bool,
    dry_run: bool,
) -> None:
    pairs = [
        pair
        for pair in experiment["pairs"]
        if selected_sessions is None or pair["target"] in selected_sessions
    ]
    channels = [
        count
        for count in experiment["channel_counts"]
        if selected_channels is None or count in selected_channels
    ]
    runs = [(pair, count) for pair in pairs for count in channels]
    if not runs:
        raise ValueError("The filters selected no transfer channel-selection experiments")
    experiment_name = str(experiment["name"])
    for pair, count in runs:
        root = PROJECT_ROOT / "outputs" / experiment_name
        if smoke:
            root = root / "_smoke"
        output = root / f"{pair['source']}_to_{pair['target']}" / f"top{count}"
        summary_path = output / "run_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            expected_status = "selection_complete" if selection_only else "complete"
            if summary.get("status") == expected_status:
                print(
                    f"SKIP {expected_status}: {pair['source']} -> {pair['target']} top{count}",
                    flush=True,
                )
                continue
        execute(
            transfer_selection_command(
                source=str(pair["source"]),
                target=str(pair["target"]),
                channels=int(count),
                experiment_name=experiment_name,
                selection=selection,
                training=training,
                selection_only=selection_only,
                smoke=smoke,
            ),
            dry_run,
        )


def mixed_fixed_point_command(
    *,
    session: str,
    experiment_name: str,
    source_experiment_name: str,
    quantization: dict[str, Any],
    evaluation: dict[str, Any],
) -> list[str]:
    command = [
        sys.executable,
        str(FIXED_POINT_EVALUATOR),
        "--session",
        session,
        "--experiment-name",
        experiment_name,
        "--source-experiment-name",
        source_experiment_name,
        "--cpu-threads",
        str(evaluation["cpu_threads"]),
        "--weight-scale-mode",
        str(quantization["weight_scale_mode"]),
    ]
    append_boolean_flag(command, evaluation["evaluate_fp32"], "--evaluate-fp32")
    return command


def run_mixed_fixed_point_evaluation(
    experiment: dict[str, Any],
    quantization: dict[str, Any],
    evaluation: dict[str, Any],
    selected_sessions: set[str] | None,
    dry_run: bool,
) -> None:
    experiment_name = str(experiment["name"])
    source_experiment_name = str(experiment["source_name"])
    sessions = [
        session
        for session in experiment["sessions"]
        if selected_sessions is None or session in selected_sessions
    ]
    if not sessions:
        raise ValueError("The session filter selected no mixed fixed-point evaluations")

    for session in sessions:
        source_summary = (
            PROJECT_ROOT / "outputs" / source_experiment_name / session / "run_summary.json"
        )
        if not dry_run and not is_complete(source_summary):
            raise RuntimeError(f"A completed 96-channel baseline is required: {source_summary}")
        result_summary = (
            PROJECT_ROOT / "outputs" / experiment_name / session / "fixed_point_run_summary.json"
        )
        if is_complete(result_summary):
            result = json.loads(result_summary.read_text(encoding="utf-8"))
            if not result.get("smoke", False):
                print(
                    f"SKIP complete mixed fixed-point evaluation: {session}",
                    flush=True,
                )
                continue
        execute(
            mixed_fixed_point_command(
                session=session,
                experiment_name=experiment_name,
                source_experiment_name=source_experiment_name,
                quantization=quantization,
                evaluation=evaluation,
            ),
            dry_run,
        )
    execute(
        [
            sys.executable,
            str(FIXED_POINT_SUMMARIZER),
            "--experiment-name",
            experiment_name,
        ],
        dry_run,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--session", action="append", choices=sorted(KNOWN_SESSIONS))
    parser.add_argument("--channels", action="append", type=int, choices=(32, 48, 64, 80))
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    payload = load_yaml(config_path)
    selected_sessions = set(args.session) if args.session else None
    selected_channels = set(args.channels) if args.channels else None
    raw_experiment = require_mapping(payload, "experiment")
    experiment_type = raw_experiment.get("type")

    if experiment_type == "transfer_channel_selection":
        experiment, selection, training = validate_transfer_config(payload)
        print(f"config={config_path}", flush=True)
        print(
            f"experiment={experiment_type} name={experiment['name']} "
            f"selection_only={args.selection_only} smoke={args.smoke} "
            f"dry_run={args.dry_run}",
            flush=True,
        )
        run_transfer_channel_selection(
            experiment,
            selection,
            training,
            selected_sessions,
            selected_channels,
            args.selection_only,
            args.smoke,
            args.dry_run,
        )
        return

    if experiment_type == "mixed_fixed_point_baseline_evaluation":
        experiment, quantization, evaluation = validate_mixed_fixed_point_config(payload)
        print(f"config={config_path}", flush=True)
        print(
            f"experiment={experiment_type} name={experiment['name']} dry_run={args.dry_run}",
            flush=True,
        )
        if selected_channels:
            raise ValueError("--channels cannot be used with a mixed fixed-point baseline config")
        if args.selection_only or args.smoke:
            raise ValueError(
                "--selection-only and --smoke can only be used with transfer experiments"
            )
        run_mixed_fixed_point_evaluation(
            experiment,
            quantization,
            evaluation,
            selected_sessions,
            args.dry_run,
        )
        return

    experiment, training = validate_common(payload)
    print(f"config={config_path}", flush=True)
    print(
        f"experiment={experiment_type} name={experiment['name']} dry_run={args.dry_run}",
        flush=True,
    )
    if experiment_type == "baseline":
        if args.selection_only or args.smoke:
            raise ValueError(
                "--selection-only and --smoke can only be used with transfer experiments"
            )
        if selected_channels:
            raise ValueError("--channels can only be used with a channel_selection config")
        run_baselines(experiment, training, selected_sessions, args.dry_run)
    elif experiment_type == "channel_selection":
        if args.selection_only or args.smoke:
            raise ValueError(
                "--selection-only and --smoke can only be used with transfer experiments"
            )
        run_channel_selection(
            experiment,
            training,
            selected_sessions,
            selected_channels,
            args.dry_run,
        )
    else:
        raise ValueError(
            "experiment.type must be 'baseline', 'channel_selection', "
            "'transfer_channel_selection', or 'mixed_fixed_point_baseline_evaluation'"
        )


if __name__ == "__main__":
    main()
