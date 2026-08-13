"""Validation-gated adaptation search using completed normalized transfer runs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
OUTPUT_ROOT = ROOT / "outputs"
DATA_ROOT = ROOT / "data" / "sabes_zenodo" / "master_mat"
PAIRS = (
    ("indy_20160622_01", "indy_20160630_01", "8-day"),
    ("indy_20170124_01", "indy_20170127_03", "3-day"),
    ("indy_20170127_03", "indy_20170131_02", "4-day"),
)
NEW_CANDIDATES = (
    ("full_lr3e4", "full", 3e-4),
    ("full_lr1e4", "full", 1e-4),
    ("head_lr1e3", "head", 1e-3),
    ("last_block_lr3e4", "last_block", 3e-4),
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


trainer = load_module(
    SCRIPT_DIR / "train_transfer_channel_selection.py",
    "transfer_channel_trainer_for_adaptation",
)
base = trainer.base
within = trainer.within
masking = trainer.masking


def pair_key(source: str, target: str) -> str:
    return f"{source}_to_{target}"


def source_run_dir(seed: int, source: str, target: str) -> Path:
    return OUTPUT_ROOT / f"transfer_sutl_norm20_seed{seed}" / pair_key(source, target) / "top64"


def output_dir(seed: int, source: str, target: str, smoke: bool) -> Path:
    root = OUTPUT_ROOT / f"transfer_adaptation_search_seed{seed}"
    if smoke:
        root = root / "_smoke"
    return root / pair_key(source, target) / "top64"


def make_model(config: Any) -> Any:
    return base.MartisSNN(
        input_size=64,
        output_size=2,
        hidden_sizes=config.hidden_sizes,
        threshold=config.threshold,
        beta_init=config.beta_init,
        optimized_forward=True,
    )


def configure_trainable(model: Any, strategy: str) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad = strategy == "full"
    if strategy == "head":
        prefixes = ("fc_out.", "leaky_out.beta")
    elif strategy == "last_block":
        prefixes = ("fc3.", "lif3.beta", "fc_out.", "leaky_out.beta")
    elif strategy == "full":
        prefixes = ()
    else:
        raise ValueError(f"Unknown trainable strategy: {strategy}")
    if prefixes:
        for name, parameter in model.named_parameters():
            if name.startswith(prefixes):
                parameter.requires_grad = True
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError(f"Strategy {strategy} left no trainable parameters")
    return trainable


def load_state(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["model_state_dict"]


def validate_strategy_file(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != payload:
            raise RuntimeError(
                f"Adaptation strategy changed in {path.parent}; use a new experiment name"
            )
    else:
        base.atomic_json(path, payload)


def evaluate(
    model: Any,
    state: dict[str, torch.Tensor],
    session: Any,
    windows: list[Any],
    kept: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    *,
    max_steps: int | None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    model.load_state_dict(state)
    time_sec, target, prediction = masking.predict_continuous(
        model,
        session,
        windows,
        0,
        torch.device("cpu"),
        selected_channels=kept,
        max_steps=max_steps,
    )
    target = trainer.restore_velocity(target, mean, scale)
    prediction = trainer.restore_velocity(prediction, mean, scale)
    return base.regression_metrics(target, prediction), time_sec, target, prediction


def metric_mean(frame: pd.DataFrame) -> dict[str, float]:
    return trainer.metric_mean(frame)


def run_pair(args: argparse.Namespace, source: str, target: str, label: str) -> None:
    original = source_run_dir(args.seed, source, target)
    original_summary_path = original / "run_summary.json"
    if not original_summary_path.exists():
        raise FileNotFoundError(original_summary_path)
    original_summary = json.loads(original_summary_path.read_text(encoding="utf-8"))
    if original_summary.get("status") != "complete":
        raise RuntimeError(f"Source transfer run is incomplete: {original_summary_path}")

    out = output_dir(args.seed, source, target, args.smoke)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "run_summary.json"
    if args.resume and summary_path.exists():
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        if saved.get("status") == "complete":
            print(f"SKIP complete: seed={args.seed} {source}->{target}", flush=True)
            return

    base.atomic_json(
        out / "progress.json",
        {
            "status": "preprocessing",
            "seed": args.seed,
            "source_session": source,
            "target_session": target,
            "updated_at": base.now_iso(),
        },
    )
    started = time.perf_counter()
    try:
        target_session = base.prepare_sabes_session(DATA_ROOT / f"{target}.mat")
        target_all = within.reconstructed_task_windows(target_session)
        target_train, target_validation, target_test = within.floor_chronological_split(target_all)
        calibration = target_train[:20]
        calibration_train = calibration[:16]
        calibration_early_stop = calibration[16:]
        normalization = original_summary["velocity_normalization"]
        if normalization["method"] != "zscore":
            raise RuntimeError("Adaptation search requires the normalized 20-task source run")
        target_mean = np.asarray(normalization["target_mean"], dtype=np.float32)
        target_scale = np.asarray(normalization["target_scale"], dtype=np.float32)
        normalized_session = trainer.normalized_velocity_session(
            target_session, target_mean, target_scale
        )
        kept = np.asarray(original_summary["kept_indices_0based"], dtype=np.int64)

        epochs = 2 if args.smoke else 100
        max_steps = 512 if args.smoke else None
        train_windows = calibration_train[:3] if args.smoke else calibration_train
        early_stop_windows = calibration_early_stop[:1] if args.smoke else calibration_early_stop
        selection_windows = [
            base.TaskWindow(
                task_index=-1,
                start=target_validation[0].start,
                end=target_test[0].start,
            )
        ]
        test_windows = [
            base.TaskWindow(
                task_index=-1,
                start=target_test[0].start,
                end=len(normalized_session.velocity),
            )
        ]
        common_config = trainer.training_config(
            keep_channels=64,
            seed=args.seed,
            epochs=epochs,
            learning_rate=1e-3,
            early_stopping_patience=10,
        )
        train_loader, early_stop_loader = masking.make_selected_loaders(
            normalized_session,
            train_windows,
            early_stop_windows,
            kept,
            common_config,
            truncate_steps=256 if args.smoke else None,
        )

        source_state = load_state(original / "source_pretrain" / "best_model.pt")
        states: dict[str, dict[str, torch.Tensor]] = {
            "source_only": source_state,
            "existing_full_lr1e3": load_state(original / "target_finetune" / "best_model.pt"),
            "existing_scratch": load_state(original / "target_scratch" / "best_model.pt"),
        }
        trainable_records: dict[str, list[str]] = {
            "source_only": [],
            "existing_full_lr1e3": [],
            "existing_scratch": [],
        }

        for candidate, strategy, learning_rate in NEW_CANDIDATES:
            base.atomic_json(
                out / "progress.json",
                {
                    "status": "training_candidate",
                    "seed": args.seed,
                    "source_session": source,
                    "target_session": target,
                    "candidate": candidate,
                    "strategy": strategy,
                    "learning_rate": learning_rate,
                    "updated_at": base.now_iso(),
                },
            )
            config = trainer.training_config(
                keep_channels=64,
                seed=args.seed,
                epochs=epochs,
                learning_rate=learning_rate,
                early_stopping_patience=10,
            )
            model = make_model(config)
            model.load_state_dict(source_state)
            trainable = configure_trainable(model, strategy)
            candidate_dir = out / "candidates" / candidate
            candidate_dir.mkdir(parents=True, exist_ok=True)
            strategy_payload = {
                "candidate": candidate,
                "strategy": strategy,
                "learning_rate": learning_rate,
                "seed": args.seed,
                "initialization": str(original / "source_pretrain" / "best_model.pt"),
                "trainable_parameters": trainable,
                "training_config": asdict(config),
            }
            validate_strategy_file(candidate_dir / "strategy.json", strategy_payload)
            stage = trainer.run_or_load_stage(
                model=model,
                train_loader=train_loader,
                validation_loader=early_stop_loader,
                config=config,
                device=torch.device("cpu"),
                stage_dir=candidate_dir,
                resume=args.resume,
                stage_name=candidate,
            )
            states[candidate] = stage[3]
            trainable_records[candidate] = trainable

        validation_rows = []
        validation_frames = []
        validation_predictions: dict[str, np.ndarray] = {}
        for candidate, state in states.items():
            model = make_model(common_config)
            frame, time_sec, target_values, prediction = evaluate(
                model,
                state,
                normalized_session,
                selection_windows,
                kept,
                target_mean,
                target_scale,
                max_steps=max_steps,
            )
            means = metric_mean(frame)
            validation_rows.append({"candidate": candidate, **means})
            tagged = frame.copy()
            tagged.insert(0, "candidate", candidate)
            validation_frames.append(tagged)
            validation_predictions[f"{candidate}_prediction"] = prediction
            validation_predictions[f"{candidate}_time_sec"] = time_sec
            validation_predictions[f"{candidate}_target"] = target_values

        validation_table = pd.DataFrame(validation_rows).sort_values(
            ["R2", "CC", "RMSE"], ascending=[False, False, True], kind="stable"
        )
        selected = str(validation_table.iloc[0]["candidate"])
        selected_state = states[selected]
        selected_model = make_model(common_config)
        test_frame, test_time, test_target, test_prediction = evaluate(
            selected_model,
            selected_state,
            normalized_session,
            test_windows,
            kept,
            target_mean,
            target_scale,
            max_steps=max_steps,
        )
        test_mean = metric_mean(test_frame)

        validation_table.to_csv(out / "candidate_validation_metrics.csv", index=False)
        pd.concat(validation_frames, ignore_index=True).to_csv(
            out / "candidate_validation_metrics_axes.csv", index=False
        )
        test_frame.to_csv(out / "selected_test_metrics.csv", index=False)
        np.savez_compressed(
            out / "selected_test_predictions.npz",
            test_time_sec=test_time,
            test_target=test_target,
            test_prediction=test_prediction,
        )
        base.atomic_torch_save(
            out / "selected_model.pt",
            {
                "model_state_dict": selected_state,
                "selected_candidate": selected,
                "seed": args.seed,
                "source_session": source,
                "target_session": target,
                "kept_indices_0based": kept.tolist(),
                "normalization": normalization,
            },
        )
        old_transfer = original_summary["test_continuous"]
        old_scratch = original_summary["target_scratch_test"]
        old_source = original_summary["source_only_target_test"]
        summary = {
            "status": "complete",
            "analysis_type": "validation_gated_transfer_adaptation_search",
            "seed": args.seed,
            "pair": label,
            "source_session": source,
            "target_session": target,
            "selected_candidate": selected,
            "candidate_selection_metric": "target_validation_continuous_mean_R2",
            "candidate_validation": validation_table.to_dict(orient="records"),
            "selected_test": test_mean,
            "old_source_only_test": old_source,
            "old_scratch_test": old_scratch,
            "old_transfer_test": old_transfer,
            "delta_selected_vs_old_transfer": {
                key: test_mean[key] - float(old_transfer[key]) for key in ("R2", "CC", "RMSE")
            },
            "delta_selected_vs_scratch": {
                key: test_mean[key] - float(old_scratch[key]) for key in ("R2", "CC", "RMSE")
            },
            "protocol": {
                "calibration_train_tasks": 16,
                "early_stopping_tasks": 4,
                "candidate_selection_data": "original target validation split only",
                "test_policy": "evaluated once for the validation-selected candidate",
                "test_excluded_from_training_early_stopping_and_selection": True,
                "velocity_normalization": normalization,
                "candidates": [
                    "source_only",
                    "existing_full_lr1e3",
                    "existing_scratch",
                    *[candidate for candidate, _, _ in NEW_CANDIDATES],
                ],
                "trainable_parameters": trainable_records,
                "exploratory_warning": (
                    "Candidate families were motivated after observing the prior experiment; "
                    "this is an exploratory optimization, not a pristine confirmatory test."
                ),
            },
            "elapsed_seconds": time.perf_counter() - started,
            "smoke": args.smoke,
        }
        base.atomic_json(summary_path, summary)
        base.atomic_json(out / "progress.json", summary)
        (out / "failure.json").unlink(missing_ok=True)
        print(
            f"COMPLETE seed={args.seed} {label} selected={selected} test_R2={test_mean['R2']:.4f}",
            flush=True,
        )
    except Exception as error:
        failure = {
            "status": "failed",
            "seed": args.seed,
            "source_session": source,
            "target_session": target,
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
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--pair", choices=("8-day", "3-day", "4-day"))
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    pairs = [pair for pair in PAIRS if args.pair is None or pair[2] == args.pair]
    for source, target, label in pairs:
        run_pair(args, source, target, label)


if __name__ == "__main__":
    main()
