"""Five-session GT-MUA SNN reproduction for Martis et al. (2024).

This is a controlled reconstruction of the five Dataset-II rows in Table IV.
Every recording is trained independently.  Paper-reported settings are kept
separate from choices that the paper does not specify, and every choice is
written to the output metadata.

Primary results are the raw SNN outputs.  A validation-only affine calibration
is saved as a diagnostic because Test 1 showed amplitude shrinkage, but the
calibrated metrics are never presented as the paper-faithful SNN result.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cross_session_snn.martis_mua_pipeline import (
    MartisSNN,
    PreparedSession,
    TaskWindow,
    count_trainable_parameters,
    masked_mse,
    prepare_sabes_session,
    regression_metrics,
    set_reproducible_seed,
    split_task_windows,
)

SESSION_REFERENCES: dict[str, dict[str, float | int]] = {
    "indy_20170124_01": {"paper_reaches": 485, "paper_R2": 0.74, "paper_CC": 0.87},
    "indy_20170127_03": {"paper_reaches": 583, "paper_R2": 0.72, "paper_CC": 0.86},
    "indy_20170131_02": {"paper_reaches": 635, "paper_R2": 0.72, "paper_CC": 0.85},
    "indy_20160630_01": {"paper_reaches": 1023, "paper_R2": 0.57, "paper_CC": 0.76},
    "indy_20160622_01": {"paper_reaches": 970, "paper_R2": 0.72, "paper_CC": 0.86},
}

EXPERIMENT_NAME = "legacy_baseline_reproduction"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / EXPERIMENT_NAME
DATA_ROOT = PROJECT_ROOT / "data" / "sabes_zenodo" / "master_mat"


@dataclass(frozen=True)
class ReconstructionConfig:
    hidden_sizes: tuple[int, int, int] = (64, 128, 64)
    threshold: float = 0.1
    beta_init: float = 0.9
    learning_rate: float = 1e-3
    optimizer_name: str = "adam"
    weight_decay: float = 0.0
    lr_plateau_patience: int = 0
    lr_plateau_factor: float = 0.5
    early_stopping_patience: int = 0
    beta_max: float = 1.0
    batch_size: int = 10
    epochs: int = 100
    seed: int = 42
    mua_mode: str = "all_threshold_crossings"
    moving_average_mode: str = "centered"
    neural_lead_ms: int = 80
    zero_search_before_ms: int = 100
    zero_search_after_ms: int = 250
    min_training_window_ms: int = 200
    max_training_window_ms: int = 4000
    target_standardization: bool = False
    shuffle_training_windows: bool = False
    validation_drop_last: bool = False
    test_evaluation: str = "continuous_single_state_reset"


CONFIG = ReconstructionConfig()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def format_elapsed(seconds: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(max(0.0, seconds)))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    temporary.replace(path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def protocol_payload(config: ReconstructionConfig) -> dict[str, Any]:
    return {
        "experiment_name": EXPERIMENT_NAME,
        "paper_reported": {
            "input": "96-channel binary MUA at 1 kHz",
            "target": "position resampled to 1 kHz -> MAF32 -> first difference -> MAF8",
            "topology": "96-64-128-64-2",
            "neuron": "LIF; train weights and beta; threshold=0.1 fixed",
            "reset": "subtract in hidden layers; disabled in output layer",
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "loss": "MSE",
            "batch_size": 10,
            "epochs": 100,
            "split": "80/10/10 by number of tasks within each recording",
            "checkpoint": "lowest validation loss",
            "test": "continuous recording without segmentation",
        },
        "not_reported_choices": {
            "session_training": "five independent models; no cross-session mixing",
            "split_order": "chronological, following the author public notebook",
            "split_rounding": "Python round for train and validation; remainder is test",
            "provided_gt_mua": (
                "union all valid spike slots on each physical M1 electrode, then binary 1 ms bins"
            ),
            "position_resampling": "linear interpolation from 250 Hz to a 1 kHz grid",
            "moving_average_phase": "centered; nearest-value boundary extension",
            "task_boundary_proxy": (
                "target_pos transitions, refined to the minimum 2-D speed within "
                "[-100,+250] ms to approximate the paper's zero-crossing start"
            ),
            "acquisition_pause_rule": (
                "exclude <200 ms or >4000 ms windows from train/validation only; "
                "keep the nominal continuous test span"
            ),
            "neural_behavior_alignment": (
                "fixed 80 ms neural lead for every session, selected previously on validation-only "
                "Ridge analysis; no final-test labels used"
            ),
            "target_scale": (
                "raw velocity units; train-only standardization rejected because Test 2 reduced CC"
            ),
            "initialization": "PyTorch default Linear initialization; beta initialized to 0.9",
            "bias": False,
            "surrogate": "snnTorch fast_sigmoid default slope",
            "batch_order": "chronological, no shuffle, matching the author public notebook",
            "padding": "right-pad variable windows and exclude padding from MSE with a mask",
            "state_during_training": "reset once at the beginning of each task window",
            "state_during_test": "reset once at the start of the continuous test span",
            "seed": config.seed,
            "software": "local locked project environment; versions recorded per session",
            "quantization": "none during training/evaluation; software floating-point comparison",
        },
        "primary_metric_policy": (
            "raw SNN R2/CC only; validation-only affine calibration is a separately labeled diagnostic"
        ),
        "test1_lessons": [
            "Do not claim exact reproduction when task boundaries and lag are inferred.",
            (
                "Do not use train-target standardization as the primary setting: it raised R2 only "
                "slightly and reduced CC in Test 2."
            ),
            (
                "Record prediction amplitude and a validation-only calibration diagnostic because "
                "Test 1 showed strong amplitude shrinkage."
            ),
        ],
        "config": asdict(config),
    }


def write_protocol_files(config: ReconstructionConfig) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUTPUT_ROOT / "protocol_choices.json", protocol_payload(config))


def session_output_dir(session_name: str, *, smoke: bool) -> Path:
    if smoke:
        return OUTPUT_ROOT / "_smoke" / session_name
    return OUTPUT_ROOT / session_name


def align_boundaries_to_speed_minima(
    session: PreparedSession,
    *,
    before_ms: int,
    after_ms: int,
) -> tuple[list[TaskWindow], pd.DataFrame]:
    """Refine target-change boundaries to nearby speed minima.

    The MAT export has no task-id vector.  Target changes reproduce the paper's
    task counts exactly, while the paper additionally says that a training
    window starts at a zero crossing of the target waveform.  The nearest
    low-speed point in a small, pre-declared neighborhood is used as a proxy.
    """

    if not session.task_windows:
        raise ValueError("Session contains no task windows")
    original_boundaries = [session.task_windows[0].start]
    original_boundaries.extend(window.end for window in session.task_windows)
    speed = np.linalg.norm(session.velocity.astype(np.float64), axis=1)
    aligned: list[int] = []
    diagnostics: list[dict[str, float | int]] = []

    for boundary_index, original in enumerate(original_boundaries):
        lower = max(0, original - before_ms)
        upper = min(len(speed), original + after_ms + 1)
        local = int(np.argmin(speed[lower:upper])) + lower
        if aligned and local <= aligned[-1]:
            local = max(aligned[-1] + 1, original)
        local = min(local, len(speed) - 1)
        aligned.append(local)
        diagnostics.append(
            {
                "boundary_index": boundary_index,
                "original_step": original,
                "aligned_step": local,
                "shift_ms": local - original,
                "speed_at_original": float(speed[original]),
                "speed_at_aligned": float(speed[local]),
            }
        )

    windows = [
        TaskWindow(task_index=index, start=aligned[index], end=aligned[index + 1])
        for index in range(len(aligned) - 1)
        if aligned[index + 1] > aligned[index]
    ]
    if len(windows) != len(session.task_windows):
        raise RuntimeError(
            f"Boundary refinement changed task count: {len(session.task_windows)} -> {len(windows)}"
        )
    return windows, pd.DataFrame(diagnostics)


def valid_training_window(window: TaskWindow, config: ReconstructionConfig) -> bool:
    effective_steps = window.steps - config.neural_lead_ms
    return (
        effective_steps >= config.min_training_window_ms
        and window.steps <= config.max_training_window_ms
    )


class LaggedReachDataset(Dataset):
    def __init__(
        self,
        session: PreparedSession,
        windows: Sequence[TaskWindow],
        neural_lead_ms: int,
        truncate_steps: int | None = None,
    ) -> None:
        self.session = session
        self.windows = list(windows)
        self.neural_lead_ms = neural_lead_ms
        self.truncate_steps = truncate_steps

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.windows[index]
        feature_start = window.start
        feature_end = window.end - self.neural_lead_ms
        target_start = window.start + self.neural_lead_ms
        target_end = window.end
        if self.truncate_steps is not None:
            feature_end = min(feature_end, feature_start + self.truncate_steps)
            target_end = target_start + (feature_end - feature_start)
        if feature_end <= feature_start:
            raise ValueError(f"Window {window.task_index} is too short after lag alignment")
        features = torch.from_numpy(
            self.session.mua_binary[feature_start:feature_end].astype(np.float32)
        )
        targets = torch.from_numpy(self.session.velocity[target_start:target_end])
        return features, targets


def pad_lagged_batch(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_steps = max(features.shape[0] for features, _ in batch)
    batch_size = len(batch)
    features = torch.zeros(max_steps, batch_size, 96, dtype=torch.float32)
    targets = torch.zeros(max_steps, batch_size, 2, dtype=torch.float32)
    mask = torch.zeros(max_steps, batch_size, 1, dtype=torch.float32)
    for index, (item_features, item_targets) in enumerate(batch):
        steps = item_features.shape[0]
        features[:steps, index] = item_features
        targets[:steps, index] = item_targets
        mask[:steps, index, 0] = 1.0
    return features, targets, mask


def make_loaders(
    session: PreparedSession,
    train_windows: Sequence[TaskWindow],
    validation_windows: Sequence[TaskWindow],
    config: ReconstructionConfig,
    *,
    truncate_steps: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = LaggedReachDataset(
        session, train_windows, config.neural_lead_ms, truncate_steps
    )
    validation_dataset = LaggedReachDataset(
        session, validation_windows, config.neural_lead_ms, truncate_steps
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_training_windows,
        num_workers=0,
        collate_fn=pad_lagged_batch,
        generator=generator,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pad_lagged_batch,
        drop_last=config.validation_drop_last,
    )
    return train_loader, validation_loader


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def train_resumable(
    model: MartisSNN,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: ReconstructionConfig,
    device: torch.device,
    output_dir: Path,
    *,
    resume: bool,
) -> tuple[pd.DataFrame, int, float, dict[str, torch.Tensor], float, str, str]:
    model.to(device)
    if config.optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    elif config.optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer_name}")
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.lr_plateau_factor,
            patience=config.lr_plateau_patience,
        )
        if config.lr_plateau_patience > 0
        else None
    )
    checkpoint_path = output_dir / "last_checkpoint.pt"
    history_rows: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    start_epoch = 1
    elapsed_before = 0.0
    training_started_at = now_iso()

    if resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        saved_config = saved.get("config")
        current_config = asdict(config)
        saved_comparable = (
            {key: value for key, value in saved_config.items() if key != "epochs"}
            if isinstance(saved_config, dict)
            else saved_config
        )
        current_comparable = {
            key: value for key, value in current_config.items() if key != "epochs"
        }
        if saved_comparable != current_comparable:
            raise RuntimeError(
                "Resume checkpoint configuration does not match current configuration"
            )
        if config.epochs < int(saved["epoch"]):
            raise RuntimeError(
                "Requested epoch count is lower than the checkpoint epoch"
            )
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        optimizer_to_device(optimizer, device)
        if scheduler is not None and saved.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(saved["scheduler_state_dict"])
        history_rows = list(saved["history"])
        best_loss = float(saved["best_validation_loss"])
        best_epoch = int(saved["best_epoch"])
        best_state = saved["best_state_dict"]
        start_epoch = int(saved["epoch"]) + 1
        elapsed_before = float(saved.get("elapsed_seconds", 0.0))
        training_started_at = str(saved.get("training_started_at", training_started_at))
        print(f"Resuming from epoch {start_epoch - 1}", flush=True)

    timer_start = time.perf_counter()
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for features, targets, mask in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            loss = masked_mse(prediction, targets, mask)
            loss.backward()
            optimizer.step()
            if config.beta_max < 1.0:
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        if name.endswith("beta"):
                            parameter.clamp_(0.0, config.beta_max)
            train_loss_sum += float(loss.detach().cpu())
            train_batches += 1

        model.eval()
        validation_loss_sum = 0.0
        validation_batches = 0
        with torch.no_grad():
            for features, targets, mask in validation_loader:
                features = features.to(device)
                targets = targets.to(device)
                mask = mask.to(device)
                prediction = model(features)
                loss = masked_mse(prediction, targets, mask)
                validation_loss_sum += float(loss.detach().cpu())
                validation_batches += 1

        train_loss = train_loss_sum / max(1, train_batches)
        validation_loss = validation_loss_sum / max(1, validation_batches)
        if scheduler is not None:
            scheduler.step(validation_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        beta_values = torch.cat(
            [
                parameter.detach().reshape(-1)
                for name, parameter in model.named_parameters()
                if name.endswith("beta")
            ]
        )
        beta_at_max = int((beta_values >= config.beta_max - 1e-7).sum().item())
        history_rows.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "validation_mse": validation_loss,
                "learning_rate": learning_rate,
                "beta_min": float(beta_values.min().item()),
                "beta_mean": float(beta_values.mean().item()),
                "beta_max": float(beta_values.max().item()),
                "beta_at_max": beta_at_max,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            atomic_torch_save(
                output_dir / "best_model_in_progress.pt",
                {
                    "model_state_dict": best_state,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_loss,
                    "config": asdict(config),
                },
            )

        elapsed = elapsed_before + (time.perf_counter() - timer_start)
        history = pd.DataFrame(history_rows)
        history.to_csv(output_dir / "training_history_in_progress.csv", index=False)
        atomic_torch_save(
            checkpoint_path,
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "best_state_dict": best_state,
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "history": history_rows,
                "elapsed_seconds": elapsed,
                "training_started_at": training_started_at,
                "config": asdict(config),
            },
        )
        atomic_json(
            output_dir / "progress.json",
            {
                "status": "training",
                "epoch": epoch,
                "epochs": config.epochs,
                "train_mse": train_loss,
                "validation_mse": validation_loss,
                "best_epoch": best_epoch,
                "best_validation_mse": best_loss,
                "learning_rate": learning_rate,
                "beta_min": float(beta_values.min().item()),
                "beta_mean": float(beta_values.mean().item()),
                "beta_max": float(beta_values.max().item()),
                "beta_at_max": beta_at_max,
                "elapsed_seconds": elapsed,
                "elapsed_hhmmss": format_elapsed(elapsed),
                "updated_at": now_iso(),
            },
        )
        print(
            f"epoch={epoch:03d}/{config.epochs} train_mse={train_loss:.6f} "
            f"validation_mse={validation_loss:.6f} best={best_epoch:03d} "
            f"elapsed={format_elapsed(elapsed)}",
            flush=True,
        )
        if (
            config.early_stopping_patience > 0
            and epoch - best_epoch >= config.early_stopping_patience
        ):
            print(
                f"early_stopping epoch={epoch:03d} best_epoch={best_epoch:03d}",
                flush=True,
            )
            break

    total_elapsed = elapsed_before + (time.perf_counter() - timer_start)
    return (
        pd.DataFrame(history_rows),
        best_epoch,
        best_loss,
        best_state,
        total_elapsed,
        training_started_at,
        now_iso(),
    )


def predict_continuous_lagged(
    model: MartisSNN,
    session: PreparedSession,
    windows: Sequence[TaskWindow],
    neural_lead_ms: int,
    device: torch.device,
    *,
    max_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not windows:
        raise ValueError("No windows to evaluate")
    feature_start = windows[0].start
    feature_end = windows[-1].end - neural_lead_ms
    target_start = feature_start + neural_lead_ms
    target_end = windows[-1].end
    if max_steps is not None:
        feature_end = min(feature_end, feature_start + max_steps)
        target_end = target_start + (feature_end - feature_start)
    features = torch.from_numpy(
        session.mua_binary[feature_start:feature_end].astype(np.float32)
    ).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        prediction = model(features.to(device)).squeeze(1).cpu().numpy()
    target = session.velocity[target_start:target_end]
    time_sec = session.time_sec[target_start:target_end] - session.time_sec[target_start]
    return time_sec, target, prediction


def fit_validation_calibration(
    target: np.ndarray, prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    gains = np.zeros(2, dtype=np.float64)
    offsets = np.zeros(2, dtype=np.float64)
    for axis in range(2):
        design = np.column_stack([prediction[:, axis], np.ones(len(prediction))])
        gains[axis], offsets[axis] = np.linalg.lstsq(design, target[:, axis], rcond=None)[0]
    return gains, offsets


def amplitude_rows(target: np.ndarray, prediction: np.ndarray) -> pd.DataFrame:
    rows = []
    for axis, name in enumerate(("vx", "vy")):
        target_std = float(np.std(target[:, axis]))
        prediction_std = float(np.std(prediction[:, axis]))
        rows.append(
            {
                "axis": name,
                "target_std": target_std,
                "prediction_std": prediction_std,
                "prediction_to_target_std_ratio": prediction_std / target_std,
            }
        )
    return pd.DataFrame(rows)


def plot_session_outputs(
    output_dir: Path,
    history: pd.DataFrame,
    best_epoch: int,
    time_test: np.ndarray,
    target_test: np.ndarray,
    prediction_test: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.semilogy(history["epoch"], history["train_mse"], label="train")
    ax.semilogy(history["epoch"], history["validation_mse"], label="validation")
    ax.axvline(best_epoch, color="crimson", linestyle="--", label=f"best={best_epoch}")
    ax.set(title="Five-GT reconstruction training", xlabel="epoch", ylabel="MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curve.png", dpi=160)
    plt.close(fig)

    steps = min(10_000, len(time_test))
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    for axis, label in enumerate(("vx", "vy")):
        axes[axis].plot(time_test[:steps], target_test[:steps, axis], label="target", lw=1)
        axes[axis].plot(
            time_test[:steps], prediction_test[:steps, axis], label="raw SNN", lw=1, alpha=0.85
        )
        axes[axis].set_ylabel(label)
    axes[0].set_title("Continuous held-out test trace")
    axes[0].legend(ncol=2)
    axes[-1].set_xlabel("time from aligned test target start (s)")
    fig.tight_layout()
    fig.savefig(output_dir / "test_trace.png", dpi=160)
    plt.close(fig)


def run_session(
    session_name: str,
    config: ReconstructionConfig,
    *,
    resume: bool,
    smoke: bool,
    cpu_threads: int,
) -> None:
    if session_name not in SESSION_REFERENCES:
        raise ValueError(f"Unknown session: {session_name}")
    write_protocol_files(config)
    output_dir = session_output_dir(session_name, smoke=smoke)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output_dir / "progress.json",
        {
            "status": "preprocessing",
            "epoch": 0,
            "epochs": config.epochs,
            "updated_at": now_iso(),
        },
    )

    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device("cpu")
    set_reproducible_seed(config.seed)

    mat_path = DATA_ROOT / f"{session_name}.mat"
    print(f"session={session_name}", flush=True)
    print(f"mat={mat_path}", flush=True)
    print(f"device={device} cpu_threads={torch.get_num_threads()}", flush=True)
    session = prepare_sabes_session(
        mat_path,
        mua_mode=config.mua_mode,
        moving_average_mode=config.moving_average_mode,
    )
    aligned_windows, boundary_diagnostics = align_boundaries_to_speed_minima(
        session,
        before_ms=config.zero_search_before_ms,
        after_ms=config.zero_search_after_ms,
    )
    reference = SESSION_REFERENCES[session_name]
    if len(aligned_windows) != int(reference["paper_reaches"]):
        raise RuntimeError(
            f"Paper reach-count mismatch for {session_name}: "
            f"expected {reference['paper_reaches']}, got {len(aligned_windows)}"
        )

    train_nominal, validation_nominal, test_nominal = split_task_windows(aligned_windows)
    train_windows = [w for w in train_nominal if valid_training_window(w, config)]
    validation_windows = [w for w in validation_nominal if valid_training_window(w, config)]
    rejected_train = [w for w in train_nominal if w not in train_windows]
    rejected_validation = [w for w in validation_nominal if w not in validation_windows]
    boundary_diagnostics.to_csv(output_dir / "boundary_alignment.csv", index=False)
    pd.DataFrame(
        [
            {
                "split": "train",
                "nominal_tasks": len(train_nominal),
                "used_training_windows": len(train_windows),
                "rejected_windows": len(rejected_train),
            },
            {
                "split": "validation",
                "nominal_tasks": len(validation_nominal),
                "used_training_windows": len(validation_windows),
                "rejected_windows": len(rejected_validation),
            },
            {
                "split": "test",
                "nominal_tasks": len(test_nominal),
                "used_training_windows": "continuous nominal span",
                "rejected_windows": 0,
            },
        ]
    ).to_csv(output_dir / "split_summary.csv", index=False)

    truncate_steps = 256 if smoke else None
    if smoke:
        train_windows = train_windows[:12]
        validation_windows = validation_windows[:4]
        test_nominal = test_nominal[:2]
    train_loader, validation_loader = make_loaders(
        session,
        train_windows,
        validation_windows,
        config,
        truncate_steps=truncate_steps,
    )

    set_reproducible_seed(config.seed)
    model = MartisSNN(
        input_size=96,
        output_size=2,
        hidden_sizes=config.hidden_sizes,
        threshold=config.threshold,
        beta_init=config.beta_init,
    )
    if count_trainable_parameters(model) != 22_914:
        raise RuntimeError("Unexpected model parameter count")

    (
        history,
        best_epoch,
        best_validation_loss,
        best_state,
        elapsed_seconds,
        training_started_at,
        training_finished_at,
    ) = train_resumable(
        model,
        train_loader,
        validation_loader,
        config,
        device,
        output_dir,
        resume=resume,
    )
    model.load_state_dict(best_state)

    max_steps = 1_000 if smoke else None
    _validation_time, validation_target, validation_prediction = predict_continuous_lagged(
        model,
        session,
        validation_nominal,
        config.neural_lead_ms,
        device,
        max_steps=max_steps,
    )
    time_test, target_test, prediction_test = predict_continuous_lagged(
        model,
        session,
        test_nominal,
        config.neural_lead_ms,
        device,
        max_steps=max_steps,
    )
    raw_metrics = regression_metrics(target_test, prediction_test)
    gains, offsets = fit_validation_calibration(validation_target, validation_prediction)
    calibrated_prediction = prediction_test * gains[None, :] + offsets[None, :]
    calibrated_metrics = regression_metrics(target_test, calibrated_prediction)
    amplitude = amplitude_rows(target_test, prediction_test)

    history.to_csv(output_dir / "training_history.csv", index=False)
    raw_metrics.to_csv(output_dir / "test_metrics_raw_snn.csv", index=False)
    calibrated_metrics.to_csv(output_dir / "test_metrics_validation_calibrated.csv", index=False)
    amplitude.to_csv(output_dir / "amplitude_diagnostics.csv", index=False)
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        time_sec=time_test,
        target=target_test,
        raw_prediction=prediction_test,
        validation_calibrated_prediction=calibrated_prediction,
        validation_gain=gains,
        validation_offset=offsets,
    )
    checkpoint = {
        "model_state_dict": best_state,
        "session": session_name,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "config": asdict(config),
        "protocol": protocol_payload(config),
    }
    atomic_torch_save(output_dir / "best_model.pt", checkpoint)
    plot_session_outputs(output_dir, history, best_epoch, time_test, target_test, prediction_test)

    raw_mean = raw_metrics.loc[raw_metrics["axis"] == "mean"].iloc[0]
    calibrated_mean = calibrated_metrics.loc[calibrated_metrics["axis"] == "mean"].iloc[0]
    summary = {
        "status": "complete",
        "session": session_name,
        "experiment_name": EXPERIMENT_NAME,
        "smoke": smoke,
        "paper_reference": reference,
        "device": str(device),
        "cpu_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "snntorch": __import__("snntorch").__version__,
        "mat_path": str(mat_path),
        "complete_tasks": len(aligned_windows),
        "nominal_split": {
            "train": len(train_nominal),
            "validation": len(validation_nominal),
            "test": len(test_nominal),
        },
        "used_for_gradient_or_checkpoint": {
            "train": len(train_windows),
            "validation": len(validation_windows),
            "rejected_train": len(rejected_train),
            "rejected_validation": len(rejected_validation),
        },
        "input_density": float(session.mua_binary.mean()),
        "boundary_shift_ms": {
            "median": float(boundary_diagnostics["shift_ms"].median()),
            "mean": float(boundary_diagnostics["shift_ms"].mean()),
            "min": int(boundary_diagnostics["shift_ms"].min()),
            "max": int(boundary_diagnostics["shift_ms"].max()),
        },
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "training_started_at": training_started_at,
        "training_finished_at": training_finished_at,
        "training_elapsed_seconds": elapsed_seconds,
        "training_elapsed_hhmmss": format_elapsed(elapsed_seconds),
        "raw_snn_metrics": raw_metrics.to_dict(orient="records"),
        "validation_only_calibration": {
            "gain": gains.tolist(),
            "offset": offsets.tolist(),
            "metrics": calibrated_metrics.to_dict(orient="records"),
            "warning": "diagnostic only; not the primary paper comparison",
        },
        "primary_result": {
            "mean_R2": float(raw_mean["R2"]),
            "mean_CC": float(raw_mean["CC"]),
            "mean_RMSE": float(raw_mean["RMSE"]),
        },
        "diagnostic_calibrated_result": {
            "mean_R2": float(calibrated_mean["R2"]),
            "mean_CC": float(calibrated_mean["CC"]),
            "mean_RMSE": float(calibrated_mean["RMSE"]),
        },
        "config": asdict(config),
    }
    atomic_json(output_dir / "run_summary.json", summary)
    (output_dir / "failure.json").unlink(missing_ok=True)
    atomic_json(
        output_dir / "progress.json",
        {
            "status": "complete",
            "epoch": config.epochs,
            "epochs": config.epochs,
            "best_epoch": best_epoch,
            "raw_mean_R2": float(raw_mean["R2"]),
            "raw_mean_CC": float(raw_mean["CC"]),
            "elapsed_seconds": elapsed_seconds,
            "elapsed_hhmmss": format_elapsed(elapsed_seconds),
            "updated_at": now_iso(),
        },
    )
    print(
        f"COMPLETE {session_name} raw_R2={raw_mean['R2']:.6f} "
        f"raw_CC={raw_mean['CC']:.6f} elapsed={format_elapsed(elapsed_seconds)}",
        flush=True,
    )


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [headers] + [[str(value) for value in row] for row in frame.to_numpy()]
    widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]
    lines = [
        "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(rows[0])) + " |",
        "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |",
    ]
    for row in rows[1:]:
        lines.append(
            "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
        )
    return "\n".join(lines)


def finalize_report(config: ReconstructionConfig) -> None:
    summaries = []
    for session_name in SESSION_REFERENCES:
        path = OUTPUT_ROOT / session_name / "run_summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing completed summary: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("status") != "complete" or summary.get("smoke"):
            raise RuntimeError(f"Invalid final summary: {path}")
        history_path = OUTPUT_ROOT / session_name / "training_history.csv"
        history = pd.read_csv(history_path)
        if len(history) != config.epochs or int(history.iloc[-1]["epoch"]) != config.epochs:
            raise RuntimeError(f"Invalid training history: {history_path}")
        summaries.append(summary)

    rows = []
    for summary in summaries:
        ref = summary["paper_reference"]
        raw = summary["primary_result"]
        diagnostic = summary["diagnostic_calibrated_result"]
        rows.append(
            {
                "session": summary["session"],
                "paper_R2": float(ref["paper_R2"]),
                "raw_R2": float(raw["mean_R2"]),
                "R2_gap": float(raw["mean_R2"]) - float(ref["paper_R2"]),
                "paper_CC": float(ref["paper_CC"]),
                "raw_CC": float(raw["mean_CC"]),
                "CC_gap": float(raw["mean_CC"]) - float(ref["paper_CC"]),
                "calibrated_R2_diagnostic": float(diagnostic["mean_R2"]),
                "calibrated_CC_diagnostic": float(diagnostic["mean_CC"]),
                "best_epoch": int(summary["best_epoch"]),
                "elapsed_hhmmss": summary["training_elapsed_hhmmss"],
            }
        )
    aggregate = pd.DataFrame(rows)
    aggregate.to_csv(OUTPUT_ROOT / "aggregate_results.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(aggregate))
    labels = [name.replace("indy_", "") for name in aggregate["session"]]
    width = 0.36
    axes[0].bar(x - width / 2, aggregate["paper_R2"], width, label="paper GT")
    axes[0].bar(x + width / 2, aggregate["raw_R2"], width, label="raw SNN reconstruction")
    axes[0].set(title="Mean R2 by recording", xticks=x, xticklabels=labels, ylim=(0, 1))
    axes[1].bar(x - width / 2, aggregate["paper_CC"], width, label="paper GT")
    axes[1].bar(x + width / 2, aggregate["raw_CC"], width, label="raw SNN reconstruction")
    axes[1].set(title="Mean CC by recording", xticks=x, xticklabels=labels, ylim=(0, 1))
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "paper_vs_reconstruction.png", dpi=170)
    plt.close(fig)

    display = aggregate.copy()
    for column in [
        "paper_R2",
        "raw_R2",
        "R2_gap",
        "paper_CC",
        "raw_CC",
        "CC_gap",
        "calibrated_R2_diagnostic",
        "calibrated_CC_diagnostic",
    ]:
        display[column] = display[column].map(lambda value: f"{value:.4f}")

    protocol = protocol_payload(config)
    choices = protocol["not_reported_choices"]
    report = f"""# Martis 2024 다섯 GT 세션 SNN 재현 - 임시 결과

생성 시각: {now_iso()}

## 판정 원칙

이 파일의 논문 비교값은 **보정하지 않은 raw SNN 출력**이다. validation으로 적합한 gain/offset
보정 결과도 진폭 문제 진단용으로 저장했지만, 논문에 없는 후처리이므로 논문 재현 성능으로 사용하지 않는다.

## 세션별 결과

{markdown_table(display)}

평균 raw R²: **{aggregate["raw_R2"].mean():.4f}**  
평균 raw CC: **{aggregate["raw_CC"].mean():.4f}**  
논문 5개 행 평균 R²: **{aggregate["paper_R2"].mean():.4f}**  
논문 5개 행 평균 CC: **{aggregate["paper_CC"].mean():.4f}**

## 논문에 없어서 이번에 고정한 선택

- 세션 처리: {choices["session_training"]}
- 분할 순서: {choices["split_order"]}
- GT MUA 구성: {choices["provided_gt_mua"]}
- 위치 보간: {choices["position_resampling"]}
- 이동평균 위상: {choices["moving_average_phase"]}
- task 경계: {choices["task_boundary_proxy"]}
- pause 처리: {choices["acquisition_pause_rule"]}
- 신경-행동 정렬: {choices["neural_behavior_alignment"]}
- target scale: {choices["target_scale"]}
- 초기화: {choices["initialization"]}
- batch 순서: {choices["batch_order"]}
- 가변 길이 padding: {choices["padding"]}
- 학습 state: {choices["state_during_training"]}
- 시험 state: {choices["state_during_test"]}
- seed: `{choices["seed"]}`
- quantization: {choices["quantization"]}

## TEST1에서 반영한 교훈

1. TEST1의 R² 0.557/CC 0.831은 구현이 동작한다는 증거였지만 논문 수치를 재현하지 못했다.
2. 이번에는 target 변화 시점을 그대로 window 시작으로 쓰지 않고 논문의 zero-crossing 설명에 맞춰
   주변 최소 속도점으로 경계를 재정렬했다.
3. 이전 validation-only Ridge에서 선택된 80 ms neural lead를 전 세션에 고정했다.
4. TEST2의 target 표준화는 R²를 소폭 높였지만 CC를 낮췄으므로 채택하지 않았다.
5. validation calibration은 원인 진단용일 뿐 raw SNN 결과를 대체하지 않는다.

## 해석 제한

- 원 논문은 MAT 파일에서 GT MUA와 학습 window를 만드는 코드를 공개하지 않았다.
- 현재 `spikes` slot 합집합, zero-crossing 근사, 80 ms lead는 명시적 재현 가정이다.
- 따라서 결과가 논문과 다르면 모델 구조의 실패뿐 아니라 MUA 구성·task indicator·window·시간 정렬
  차이가 원인일 수 있다.
- 한 seed만 실행했으므로 초기화 분산은 측정하지 않았다.
- 논문의 FPGA quantization 성능이 아니라 floating-point software SNN 성능을 비교한다.

## 산출물

- `aggregate_results.csv`: 다섯 세션 raw/diagnostic 지표
- `paper_vs_reconstruction.png`: 논문과 raw SNN 비교
- `protocol_choices.json`: 공개 조건과 추정 조건 전체
- 각 세션 폴더: checkpoint, 100-epoch history, raw metrics, calibration 진단, 연속 예측, trace
"""
    (OUTPUT_ROOT / "TEMP_RESULTS.md").write_text(report, encoding="utf-8")
    atomic_json(
        OUTPUT_ROOT / "finalization.json",
        {
            "status": "complete",
            "created_at": now_iso(),
            "sessions": list(SESSION_REFERENCES),
            "mean_raw_R2": float(aggregate["raw_R2"].mean()),
            "mean_raw_CC": float(aggregate["raw_CC"].mean()),
            "report": str(OUTPUT_ROOT / "TEMP_RESULTS.md"),
        },
    )
    print(OUTPUT_ROOT / "TEMP_RESULTS.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", choices=list(SESSION_REFERENCES))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    return parser.parse_args()


def show_status() -> None:
    rows = []
    for session_name in SESSION_REFERENCES:
        path = OUTPUT_ROOT / session_name / "progress.json"
        if path.exists():
            progress = json.loads(path.read_text(encoding="utf-8"))
        else:
            progress = {"status": "not_started", "epoch": 0, "epochs": CONFIG.epochs}
        rows.append({"session": session_name, **progress})
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    args = parse_args()
    config = ReconstructionConfig(epochs=args.epochs)
    if args.status:
        show_status()
        return
    if args.finalize:
        if args.epochs != 100:
            raise ValueError("Final report is defined for the 100-epoch runs")
        finalize_report(config)
        return
    if not args.session:
        raise ValueError("--session is required unless --status or --finalize is used")
    output_dir = session_output_dir(args.session, smoke=args.smoke)
    try:
        run_session(
            args.session,
            config,
            resume=args.resume,
            smoke=args.smoke,
            cpu_threads=args.cpu_threads,
        )
    except Exception as error:
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(
            output_dir / "failure.json",
            {
                "status": "failed",
                "session": args.session,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "failed_at": now_iso(),
            },
        )
        atomic_json(
            output_dir / "progress.json",
            {
                "status": "failed",
                "epoch": 0,
                "epochs": config.epochs,
                "error": str(error),
                "updated_at": now_iso(),
            },
        )
        raise


if __name__ == "__main__":
    main()
