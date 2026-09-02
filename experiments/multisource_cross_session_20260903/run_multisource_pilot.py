"""Run the time-ordered, source-budget-matched multi-source Indy pilot.

The script intentionally lives in a dated experiment directory.  It reuses the
published project pipeline, but adds the controls requested in the 2026-08-24
research plan: frozen source banks, equal total source-task budgets, target
calibration learning curves, and validation-gated source-preserving adaptation.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "sabes_zenodo" / "master_mat"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "multisource_cross_session_20260903"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


within = load_module(SCRIPT_DIR / "train_channel_selection.py", "multisource_within")
single = load_module(SCRIPT_DIR / "train_transfer_channel_selection.py", "multisource_single")
transfer = load_module(
    SCRIPT_DIR / "internal" / "transfer_selection_core.py", "multisource_transfer_core"
)
adaptation = load_module(
    SCRIPT_DIR / "train_transfer_adaptation_search.py", "multisource_adaptation"
)
base = within.base
masking = within.masking


class ArrayWindowDataset(Dataset):
    def __init__(self, windows: list[tuple[np.ndarray, np.ndarray]], channels: np.ndarray):
        self.windows = windows
        self.channels = np.asarray(channels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features, targets = self.windows[index]
        return (
            torch.from_numpy(features[:, self.channels].astype(np.float32, copy=False)),
            torch.from_numpy(targets.astype(np.float32, copy=False)),
        )


def loader(
    windows: list[tuple[np.ndarray, np.ndarray]],
    channels: np.ndarray,
    *,
    seed: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        ArrayWindowDataset(windows, channels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=masking.pad_selected_batch,
        generator=torch.Generator().manual_seed(seed),
        drop_last=False,
    )


def session_date(name: str) -> int:
    match = re.fullmatch(r"indy_(\d{8})_\d+", name)
    if match is None:
        raise ValueError(f"Session name does not contain a date: {name}")
    return int(match.group(1))


def read_plan() -> dict[str, Any]:
    payload = yaml.safe_load((EXPERIMENT_DIR / "plan.yaml").read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Only plan schema_version 1 is supported")
    return payload


def allocate_equal(total: int, sessions: list[str]) -> dict[str, int]:
    if total < len(sessions):
        raise ValueError("Source task budget must be at least the source-session count")
    quotient, remainder = divmod(total, len(sessions))
    return {
        session: quotient + (1 if index < remainder else 0)
        for index, session in enumerate(sessions)
    }


def validate_plan(plan: dict[str, Any], data_root: Path, bank_name: str) -> dict[str, Any]:
    experiment = plan["experiment"]
    target = str(experiment["target"])
    banks = experiment["source_banks"]
    if bank_name not in banks:
        raise ValueError(f"Unknown source bank {bank_name}; choose from {sorted(banks)}")
    sources = [str(value) for value in banks[bank_name]]
    if len(set(sources)) != len(sources):
        raise ValueError(f"Duplicate source in bank {bank_name}")
    if target in sources:
        raise ValueError("Target cannot also be a source")
    future = [source for source in sources if session_date(source) >= session_date(target)]
    if future:
        raise ValueError(f"Non-past source sessions violate the frozen-bank rule: {future}")
    required = [*sources, target]
    files = {}
    for session in required:
        path = data_root / f"{session}.mat"
        if not path.is_file():
            raise FileNotFoundError(path)
        files[session] = {"path": str(path.resolve()), "bytes": path.stat().st_size}
    budgets = allocate_equal(int(experiment["source_task_budget_total"]), sources)
    return {
        "status": "valid",
        "source_bank": bank_name,
        "sources": sources,
        "target": target,
        "source_task_allocation": budgets,
        "total_source_tasks": sum(budgets.values()),
        "files": files,
        "controls": {
            "all_sources_precede_target": True,
            "future_target_excluded": True,
            "total_source_tasks_fixed_across_banks": True,
            "target_validation_and_test_excluded_from_channel_ranking": True,
        },
    }


def fit_velocity(session: Any, windows: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    mean, scale, _ = single.velocity_normalization_stats(session, windows)
    return mean, scale


def cache_windows(
    session: Any,
    windows: list[Any],
    mean: np.ndarray,
    scale: np.ndarray,
    *,
    truncate_steps: int | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    cached = []
    for window in windows:
        end = window.end if truncate_steps is None else min(window.end, window.start + truncate_steps)
        features = session.mua_binary[window.start:end].astype(np.uint8, copy=True)
        targets = ((session.velocity[window.start:end] - mean) / scale).astype(np.float32)
        cached.append((features, targets))
    return cached


def merge_aggregated(parts: list[Any]) -> Any:
    return transfer.AggregatedWindows(
        counts=np.concatenate([part.counts for part in parts], axis=0),
        velocity=np.concatenate([part.velocity for part in parts], axis=0),
        window_ids=np.concatenate([part.window_ids for part in parts], axis=0),
    )


def make_model(config: Any, channels: int) -> Any:
    return base.MartisSNN(
        input_size=channels,
        output_size=2,
        hidden_sizes=config.hidden_sizes,
        threshold=config.threshold,
        beta_init=config.beta_init,
        optimized_forward=True,
    )


def evaluate_state(
    state: dict[str, torch.Tensor],
    config: Any,
    target_session: Any,
    windows: list[Any],
    kept: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    max_steps: int | None,
) -> dict[str, float]:
    model = make_model(config, len(kept))
    model.load_state_dict(state)
    _, target, prediction = masking.predict_continuous(
        model,
        target_session,
        windows,
        0,
        torch.device("cpu"),
        selected_channels=kept,
        max_steps=max_steps,
    )
    target = single.restore_velocity(target, mean, scale)
    prediction = single.restore_velocity(prediction, mean, scale)
    return single.metric_mean(base.regression_metrics(target, prediction))


def configure_trainable(model: Any, strategy: str) -> list[str]:
    mapping = {
        "head_only": ("fc_out.", "leaky_out.beta"),
        "last_block": ("fc3.", "lif3.beta", "fc_out.", "leaky_out.beta"),
        "full": (),
    }
    if strategy not in mapping:
        raise ValueError(strategy)
    for parameter in model.parameters():
        parameter.requires_grad = strategy == "full"
    for name, parameter in model.named_parameters():
        if mapping[strategy] and name.startswith(mapping[strategy]):
            parameter.requires_grad = True
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan = read_plan()
    audit = validate_plan(plan, args.data_root, args.source_bank)
    experiment = plan["experiment"]
    if args.validate_only:
        return audit

    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    base.set_reproducible_seed(args.seed)
    started = time.perf_counter()
    sources = audit["sources"]
    target_name = audit["target"]
    allocation = dict(audit["source_task_allocation"])
    if args.smoke:
        allocation = allocate_equal(max(3 * len(sources), len(sources)), sources)
    source_train_arrays: list[tuple[np.ndarray, np.ndarray]] = []
    source_validation_arrays: list[tuple[np.ndarray, np.ndarray]] = []
    source_aggregates = []
    source_metadata = []
    channel_names = None
    truncate_steps = 256 if args.smoke else None

    for source_name in sources:
        session = base.prepare_sabes_session(args.data_root / f"{source_name}.mat")
        if channel_names is None:
            channel_names = list(session.channel_names)
        elif channel_names != list(session.channel_names):
            raise RuntimeError(f"Physical channel mismatch in {source_name}")
        all_windows = within.reconstructed_task_windows(session)
        train_windows, validation_windows, _ = within.floor_chronological_split(all_windows)
        requested = allocation[source_name]
        if requested > len(train_windows):
            raise ValueError(f"{source_name} has only {len(train_windows)} train tasks")
        chosen_train = train_windows[:requested]
        chosen_validation = validation_windows[: int(experiment["source_validation_tasks_per_session"])]
        mean, scale = fit_velocity(session, chosen_train)
        normalized = single.normalized_velocity_session(session, mean, scale)
        source_aggregates.append(
            transfer.aggregate_windows(normalized, chosen_train, bin_ms=50, neural_lead_ms=0)
        )
        source_train_arrays.extend(
            cache_windows(session, chosen_train, mean, scale, truncate_steps=truncate_steps)
        )
        source_validation_arrays.extend(
            cache_windows(session, chosen_validation, mean, scale, truncate_steps=truncate_steps)
        )
        source_metadata.append(
            {
                "session": source_name,
                "train_tasks": len(chosen_train),
                "validation_tasks": len(chosen_validation),
                "velocity_mean": mean.tolist(),
                "velocity_scale": scale.tolist(),
            }
        )
        del normalized, session
        gc.collect()

    target_session = base.prepare_sabes_session(args.data_root / f"{target_name}.mat")
    if channel_names != list(target_session.channel_names):
        raise RuntimeError("Target physical channels do not match the source bank")
    target_all = within.reconstructed_task_windows(target_session)
    target_train, target_validation, target_test = within.floor_chronological_split(target_all)
    calibration_tasks = 4 if args.smoke else args.calibration_tasks
    if calibration_tasks > len(target_train):
        raise ValueError(f"Target has only {len(target_train)} train tasks")
    target_calibration = target_train[:calibration_tasks]
    validation_count = max(1, round(calibration_tasks * 0.2))
    calibration_train = target_calibration[:-validation_count]
    calibration_early_stop = target_calibration[-validation_count:]
    target_mean, target_scale = fit_velocity(target_session, target_calibration)
    normalized_target = single.normalized_velocity_session(target_session, target_mean, target_scale)
    target_aggregated = transfer.aggregate_windows(
        normalized_target, target_calibration, bin_ms=50, neural_lead_ms=0
    )
    ranking = transfer.rank_transfer_channels(
        merge_aggregated(source_aggregates),
        target_aggregated,
        transfer.TransferSelectionConfig(),
        channel_names=channel_names,
    )
    kept = np.sort(ranking.order[: int(experiment["channel_count"])])

    out = args.output_root / args.source_bank / f"cal{args.calibration_tasks}" / f"seed{args.seed}"
    if args.smoke:
        out = out / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    ranking_frame = ranking.ranking.copy()
    ranking_frame["kept"] = ranking_frame["rank"] <= len(kept)
    ranking_frame.to_csv(out / "channel_ranking.csv", index=False)

    epochs = 2 if args.smoke else args.epochs
    config = single.training_config(
        keep_channels=len(kept),
        seed=args.seed,
        epochs=epochs,
        learning_rate=1e-3,
        early_stopping_patience=10,
    )
    source_train_loader = loader(
        source_train_arrays,
        kept,
        seed=args.seed,
        batch_size=config.batch_size,
        shuffle=True,
    )
    source_validation_loader = loader(
        source_validation_arrays,
        kept,
        seed=args.seed,
        batch_size=config.batch_size,
        shuffle=False,
    )
    source_model = make_model(config, len(kept))
    source_stage = single.run_or_load_stage(
        model=source_model,
        train_loader=source_train_loader,
        validation_loader=source_validation_loader,
        config=config,
        device=torch.device("cpu"),
        stage_dir=out / "source_pretrain",
        resume=args.resume,
        stage_name="balanced_multisource_pretrain",
    )
    source_state = source_stage[3]

    calibration_train_loader, calibration_validation_loader = masking.make_selected_loaders(
        normalized_target,
        calibration_train,
        calibration_early_stop,
        kept,
        config,
        truncate_steps=truncate_steps,
    )
    validation_span = [
        base.TaskWindow(task_index=-1, start=target_validation[0].start, end=target_test[0].start)
    ]
    test_span = [
        base.TaskWindow(task_index=-1, start=target_test[0].start, end=len(target_session.velocity))
    ]
    max_steps = 512 if args.smoke else None
    candidate_states = {"source_only": source_state}
    trainable = {"source_only": []}
    for strategy in ("head_only", "last_block", "full"):
        model = make_model(config, len(kept))
        model.load_state_dict(source_state)
        trainable[strategy] = configure_trainable(model, strategy)
        stage = single.run_or_load_stage(
            model=model,
            train_loader=calibration_train_loader,
            validation_loader=calibration_validation_loader,
            config=config,
            device=torch.device("cpu"),
            stage_dir=out / "adaptation" / strategy,
            resume=args.resume,
            stage_name=strategy,
        )
        candidate_states[strategy] = stage[3]

    scratch_model = make_model(config, len(kept))
    scratch_stage = single.run_or_load_stage(
        model=scratch_model,
        train_loader=calibration_train_loader,
        validation_loader=calibration_validation_loader,
        config=config,
        device=torch.device("cpu"),
        stage_dir=out / "scratch",
        resume=args.resume,
        stage_name="scratch",
    )
    scratch_state = scratch_stage[3]

    candidate_validation = {
        name: evaluate_state(
            state,
            config,
            normalized_target,
            validation_span,
            kept,
            target_mean,
            target_scale,
            max_steps,
        )
        for name, state in candidate_states.items()
    }
    selected = max(candidate_validation, key=lambda name: candidate_validation[name]["R2"])
    selected_test = evaluate_state(
        candidate_states[selected],
        config,
        normalized_target,
        test_span,
        kept,
        target_mean,
        target_scale,
        max_steps,
    )
    scratch_test = evaluate_state(
        scratch_state,
        config,
        normalized_target,
        test_span,
        kept,
        target_mean,
        target_scale,
        max_steps,
    )
    summary = {
        "status": "smoke_complete" if args.smoke else "complete",
        "analysis_type": "time_ordered_budget_matched_multisource_transfer",
        "source_bank": args.source_bank,
        "sources": sources,
        "target": target_name,
        "seed": args.seed,
        "requested_calibration_tasks": args.calibration_tasks,
        "executed_calibration_tasks": calibration_tasks,
        "source_task_allocation": allocation,
        "source_metadata": source_metadata,
        "selected_candidate": selected,
        "candidate_validation": candidate_validation,
        "selected_test": selected_test,
        "scratch_test": scratch_test,
        "delta_selected_vs_scratch": {
            metric: selected_test[metric] - scratch_test[metric]
            for metric in ("R2", "CC", "RMSE")
        },
        "kept_indices_0based": kept.tolist(),
        "target_velocity_normalization": {
            "fit_scope": "target_calibration_only",
            "mean": target_mean.tolist(),
            "scale": target_scale.tolist(),
        },
        "trainable_parameters": trainable,
        "controls": audit["controls"],
        "test_policy": "validation-selected adaptation and scratch evaluated once each",
        "smoke_warning": (
            "Smoke metrics use four calibration tasks, two epochs and at most 512 evaluation steps; "
            "they validate execution only and are not scientific estimates."
            if args.smoke
            else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    base.atomic_json(out / "run_summary.json", summary)
    base.atomic_json(out / "protocol_audit.json", audit)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bank", default="recent3")
    parser.add_argument("--calibration-tasks", type=int, choices=(20, 40, 80), default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
