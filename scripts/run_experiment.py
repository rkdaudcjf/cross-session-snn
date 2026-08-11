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
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", value
    ):
        raise ValueError(
            f"{location} must contain only letters, numbers, underscores, and hyphens"
        )
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
        raise ValueError(
            "quantization.rounding must be 'nearest_half_away_from_zero'"
        )
    if quantization.get("saturation") is not True:
        raise ValueError("quantization.saturation must be true")
    if not isinstance(evaluation.get("cpu_threads"), int) or evaluation["cpu_threads"] < 1:
        raise ValueError("evaluation.cpu_threads must be a positive integer")
    if not isinstance(evaluation.get("evaluate_fp32"), bool):
        raise TypeError("evaluation.evaluate_fp32 must be true or false")
    return experiment, quantization, evaluation


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
    append_boolean_flag(command, training["keep_validation_remainder"], "--keep-validation-remainder")
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
        count
        for count in channel_counts
        if selected_channels is None or count in selected_channels
    ]
    runs = [(session, count) for session in sessions for count in channels]
    if not runs:
        raise ValueError("The filters selected no channel-selection experiments")

    for run_index, (session, count) in enumerate(runs, start=1):
        baseline = baseline_summary_path(session)
        if not dry_run and not is_complete(baseline):
            raise RuntimeError(f"A completed 96-channel baseline is required: {baseline}")
        output = (
            PROJECT_ROOT
            / "outputs"
            / experiment_name
            / session
            / f"top{count}"
        )
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
            PROJECT_ROOT
            / "outputs"
            / source_experiment_name
            / session
            / "run_summary.json"
        )
        if not dry_run and not is_complete(source_summary):
            raise RuntimeError(f"A completed 96-channel baseline is required: {source_summary}")
        result_summary = (
            PROJECT_ROOT
            / "outputs"
            / experiment_name
            / session
            / "fixed_point_run_summary.json"
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
    parser.add_argument("--channels", action="append", type=int, choices=(32, 64))
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

    if experiment_type == "mixed_fixed_point_baseline_evaluation":
        experiment, quantization, evaluation = validate_mixed_fixed_point_config(payload)
        print(f"config={config_path}", flush=True)
        print(
            f"experiment={experiment_type} name={experiment['name']} dry_run={args.dry_run}",
            flush=True,
        )
        if selected_channels:
            raise ValueError(
                "--channels cannot be used with a mixed fixed-point baseline config"
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
        if selected_channels:
            raise ValueError("--channels can only be used with a channel_selection config")
        run_baselines(experiment, training, selected_sessions, args.dry_run)
    elif experiment_type == "channel_selection":
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
            "or 'mixed_fixed_point_baseline_evaluation'"
        )


if __name__ == "__main__":
    main()
