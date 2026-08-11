"""Train and evaluate a correlation-selected channel mask on one Indy session.

The experiment extends the Martis et al. (2024) GT-MUA reconstruction without
modifying its baseline artifacts.  Channel ranking follows Leone et al. (DATE
2025): input channels are ranked by the Pearson correlation between training
firing rate and behavioral variables.  The paper does not state how its two
velocity-axis correlations are collapsed to one rank, so this implementation
uses their root-mean-square magnitude.  Only the first half of chronological
training entries is used for ranking, as stated in that paper.

Primary comparison:
  - existing, completed 96-channel baseline (no retraining here)
  - a fresh SNN trained and inferred with the selected top 64 channels

The script also performs diagnostic frozen-model ablations.  They show the
immediate effect of zero-masking the already-trained 96-channel model and are
explicitly kept separate from the primary retrained result.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import traceback
from collections.abc import Sequence
from dataclasses import asdict, dataclass
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
SESSION_NAME = "indy_20170127_03"
DATA_PATH = PROJECT_ROOT / "data" / "sabes_zenodo" / "master_mat" / f"{SESSION_NAME}.mat"
BASELINE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "legacy_baseline_reproduction"
    / SESSION_NAME
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "legacy_channel_selection_64ch" / SESSION_NAME


def load_reproduction_module() -> Any:
    """Load the numbered baseline script as an importable module."""
    path = SCRIPT_DIR / "reproduction_core.py"
    module_name = "baseline_reproduction_core"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import baseline module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


base = load_reproduction_module()


@dataclass(frozen=True)
class MaskingConfig(base.ReconstructionConfig):
    keep_channels: int = 64
    ranking_fraction: float = 0.5
    ranking_axis_reduction: str = "rms_absolute_pearson"
    ranking_source: str = "chronological_first_half_of_training_entries"
    posthoc_random_repeats: int = 5


REFERENCES = [
    {
        "id": "martis2024",
        "title": (
            "Low-Power FPGA-Based Spiking Neural Networks for Real-Time "
            "Decoding of Intracortical Neural Activity"
        ),
        "authors": "L. Martis, G. Leone, L. Raffo, P. Meloni",
        "venue": "IEEE Sensors Journal 24(24), 2024",
        "doi": "https://doi.org/10.1109/JSEN.2024.3487021",
        "used_for": "SNN topology, target preprocessing, split, training, and metrics",
    },
    {
        "id": "leone2025",
        "title": (
            "Enabling SNN-Based Near-MEA Neural Decoding with Channel Selection: "
            "An Open-HW Approach"
        ),
        "authors": "G. Leone, L. Martis, L. Raffo, P. Meloni",
        "venue": "DATE 2025",
        "doi": "https://doi.org/10.23919/DATE64628.2025.10993220",
        "used_for": (
            "Pearson firing-rate/behavior ranking, first-half training calibration, "
            "and 64-channel operating point"
        ),
    },
    {
        "id": "odoherty2020",
        "title": "Nonhuman Primate Reaching with Multichannel Sensorimotor Cortex Electrophysiology",
        "authors": "J. E. O'Doherty et al.",
        "venue": "Zenodo dataset, 2020",
        "doi": "https://doi.org/10.5281/zenodo.3854034",
        "used_for": "Indy session data and physical M1 electrode labels",
    },
]


def output_dir(config: MaskingConfig, smoke: bool) -> Path:
    name = f"corr_top{config.keep_channels}_seed{config.seed}"
    if smoke:
        return OUTPUT_ROOT / "_smoke" / name
    return OUTPUT_ROOT / name


def prepare_protocol(config: MaskingConfig) -> dict[str, Any]:
    return {
        "session": SESSION_NAME,
        "primary_question": (
            "Performance after masking the lowest-ranked 32 of 96 M1 channels, "
            "then training and inferring a structural 64-input SNN"
        ),
        "baseline": {
            "artifact": str(BASELINE_DIR / "best_model.pt"),
            "role": "completed 96-channel reference; not retrained by this script",
        },
        "ranking": {
            "feature": "1 ms binary MUA; mean is firing rate up to a constant 1000 Hz scale",
            "target": "vx and vy at the established 80 ms neural lead",
            "statistic": "Pearson correlation computed from sufficient statistics",
            "axis_reduction": (
                "sqrt(mean([corr(MUA,vx)^2, corr(MUA,vy)^2])); "
                "a documented extension because Leone et al. do not state the two-axis reduction"
            ),
            "calibration_data": (
                "first chronological half of valid training entries only; validation and test excluded"
            ),
            "mask_rule": (
                f"keep top {config.keep_channels} scores and mask the remaining "
                f"{96 - config.keep_channels} channels"
            ),
        },
        "training": {
            "initialization": "fresh seed-controlled initialization; no baseline weight transfer",
            "input_layer": f"{config.keep_channels}-64",
            "hidden_and_output": "64-128-64-2",
            "epochs": config.epochs,
            "selection": "lowest validation MSE over all epochs",
            "inference": "one continuous held-out test span, state reset only once",
        },
        "diagnostics": {
            "frozen_model_ablation": (
                "zero-mask the completed 96-input model without retraining; never mixed with primary result"
            ),
            "random_repeats": config.posthoc_random_repeats,
        },
        "config": asdict(config),
        "references": REFERENCES,
    }


def split_windows(
    session: Any, config: MaskingConfig
) -> tuple[list[Any], list[Any], list[Any], list[Any], pd.DataFrame]:
    aligned, diagnostics = base.align_boundaries_to_speed_minima(
        session,
        before_ms=config.zero_search_before_ms,
        after_ms=config.zero_search_after_ms,
    )
    reference = base.SESSION_REFERENCES[SESSION_NAME]
    if len(aligned) != int(reference["paper_reaches"]):
        raise RuntimeError(
            f"Paper reach-count mismatch: {len(aligned)} != {reference['paper_reaches']}"
        )
    train_nominal, validation_nominal, test_nominal = base.split_task_windows(aligned)
    train = [window for window in train_nominal if base.valid_training_window(window, config)]
    validation = [
        window for window in validation_nominal if base.valid_training_window(window, config)
    ]
    return train, validation, validation_nominal, test_nominal, diagnostics


def rank_channels(
    session: Any,
    train_windows: Sequence[Any],
    config: MaskingConfig,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Rank channels from train-only Pearson sufficient statistics."""
    if not 0 < config.ranking_fraction <= 1:
        raise ValueError("ranking_fraction must be in (0, 1]")
    total_steps = sum(window.steps - config.neural_lead_ms for window in train_windows)
    requested_steps = max(2, int(total_steps * config.ranking_fraction))

    n = 0
    sum_x = np.zeros(96, dtype=np.float64)
    sum_x2 = np.zeros(96, dtype=np.float64)
    sum_y = np.zeros(2, dtype=np.float64)
    sum_y2 = np.zeros(2, dtype=np.float64)
    sum_xy = np.zeros((96, 2), dtype=np.float64)

    for window in train_windows:
        remaining = requested_steps - n
        if remaining <= 0:
            break
        steps = min(window.steps - config.neural_lead_ms, remaining)
        if steps <= 0:
            continue
        x = session.mua_binary[window.start : window.start + steps].astype(
            np.float64, copy=False
        )
        target_start = window.start + config.neural_lead_ms
        y = session.velocity[target_start : target_start + steps].astype(
            np.float64, copy=False
        )
        n += steps
        sum_x += x.sum(axis=0)
        sum_x2 += np.square(x).sum(axis=0)
        sum_y += y.sum(axis=0)
        sum_y2 += np.square(y).sum(axis=0)
        sum_xy += x.T @ y

    if n != requested_steps:
        raise RuntimeError(f"Ranking used {n} steps, expected {requested_steps}")

    numerator = n * sum_xy - sum_x[:, None] * sum_y[None, :]
    x_term = n * sum_x2 - np.square(sum_x)
    y_term = n * sum_y2 - np.square(sum_y)
    denominator = np.sqrt(np.maximum(x_term[:, None] * y_term[None, :], 0.0))
    correlations = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    scores = np.sqrt(np.mean(np.square(correlations), axis=1))
    order = np.argsort(-scores, kind="stable")
    kept = np.sort(order[: config.keep_channels])
    masked = np.sort(order[config.keep_channels :])
    selected = np.zeros(96, dtype=bool)
    selected[kept] = True

    ranking = pd.DataFrame(
        {
            "rank": np.arange(1, 97),
            "channel_index_0based": order,
            "channel_label": [f"M1 {index + 1:03d}" for index in order],
            "corr_vx": correlations[order, 0],
            "corr_vy": correlations[order, 1],
            "score_rms_abs_corr": scores[order],
            "firing_rate_hz_calibration": 1000.0 * sum_x[order] / n,
            "kept": selected[order],
            "masked": ~selected[order],
            "ranking_steps": n,
        }
    )
    return ranking, kept, masked


class SelectedChannelDataset(Dataset):
    def __init__(
        self,
        session: Any,
        windows: Sequence[Any],
        neural_lead_ms: int,
        channels: np.ndarray,
        truncate_steps: int | None = None,
    ) -> None:
        self.session = session
        self.windows = list(windows)
        self.neural_lead_ms = neural_lead_ms
        self.channels = np.asarray(channels, dtype=np.int64)
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
            self.session.mua_binary[feature_start:feature_end, self.channels].astype(
                np.float32
            )
        )
        targets = torch.from_numpy(self.session.velocity[target_start:target_end])
        return features, targets


def pad_selected_batch(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_steps = max(item[0].shape[0] for item in batch)
    batch_size = len(batch)
    input_size = batch[0][0].shape[1]
    features = torch.zeros(max_steps, batch_size, input_size, dtype=torch.float32)
    targets = torch.zeros(max_steps, batch_size, 2, dtype=torch.float32)
    mask = torch.zeros(max_steps, batch_size, 1, dtype=torch.float32)
    for index, (item_features, item_targets) in enumerate(batch):
        steps = item_features.shape[0]
        features[:steps, index] = item_features
        targets[:steps, index] = item_targets
        mask[:steps, index, 0] = 1.0
    return features, targets, mask


def make_selected_loaders(
    session: Any,
    train_windows: Sequence[Any],
    validation_windows: Sequence[Any],
    channels: np.ndarray,
    config: MaskingConfig,
    truncate_steps: int | None,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = SelectedChannelDataset(
        session, train_windows, config.neural_lead_ms, channels, truncate_steps
    )
    validation_dataset = SelectedChannelDataset(
        session, validation_windows, config.neural_lead_ms, channels, truncate_steps
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_training_windows,
        num_workers=0,
        collate_fn=pad_selected_batch,
        generator=generator,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pad_selected_batch,
        drop_last=config.validation_drop_last,
    )
    return train_loader, validation_loader


def predict_continuous(
    model: Any,
    session: Any,
    windows: Sequence[Any],
    neural_lead_ms: int,
    device: torch.device,
    *,
    selected_channels: np.ndarray | None = None,
    zero_mask: np.ndarray | None = None,
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
    features = session.mua_binary[feature_start:feature_end].astype(np.float32)
    if selected_channels is not None:
        features = features[:, selected_channels]
    elif zero_mask is not None:
        features = features.copy()
        features[:, zero_mask] = 0.0
    feature_tensor = torch.from_numpy(features).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        prediction = model(feature_tensor.to(device)).squeeze(1).cpu().numpy()
    target = session.velocity[target_start:target_end]
    time_sec = session.time_sec[target_start:target_end] - session.time_sec[target_start]
    return time_sec, target, prediction


def mean_metric(frame: pd.DataFrame, name: str) -> float:
    return float(frame.loc[frame["axis"] == "mean", name].iloc[0])


def evaluate_frozen_masks(
    session: Any,
    test_windows: Sequence[Any],
    ranking: pd.DataFrame,
    kept: np.ndarray,
    masked: np.ndarray,
    config: MaskingConfig,
    device: torch.device,
    out: Path,
    max_steps: int | None,
) -> pd.DataFrame:
    """Diagnostic only: apply masks to the trained 96-channel baseline."""
    checkpoint = torch.load(BASELINE_DIR / "best_model.pt", map_location=device, weights_only=False)
    model = base.MartisSNN(
        input_size=96,
        output_size=2,
        hidden_sizes=config.hidden_sizes,
        threshold=config.threshold,
        beta_init=config.beta_init,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    score_order = ranking["channel_index_0based"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(config.seed)
    conditions: list[tuple[str, int, np.ndarray]] = [
        ("correlation_keep_top", -1, masked),
        ("adversarial_mask_top", -1, np.sort(score_order[: 96 - config.keep_channels])),
    ]
    for repeat in range(config.posthoc_random_repeats):
        random_mask = np.sort(
            rng.choice(96, size=96 - config.keep_channels, replace=False)
        )
        conditions.append(("random_mask", repeat, random_mask))

    rows: list[dict[str, Any]] = []
    for condition, repeat, zero_mask in conditions:
        _time, target, prediction = predict_continuous(
            model,
            session,
            test_windows,
            config.neural_lead_ms,
            device,
            zero_mask=zero_mask,
            max_steps=max_steps,
        )
        metrics = base.regression_metrics(target, prediction)
        rows.append(
            {
                "analysis_type": "frozen_96_input_model_zero_mask_without_retraining",
                "condition": condition,
                "repeat": repeat,
                "kept_channels": config.keep_channels,
                "masked_channels_0based": json.dumps(zero_mask.tolist()),
                "mean_R2": mean_metric(metrics, "R2"),
                "mean_CC": mean_metric(metrics, "CC"),
                "mean_RMSE": mean_metric(metrics, "RMSE"),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(out / "posthoc_frozen_ablation.csv", index=False)
    return result


def plot_ranking(out: Path, ranking: pd.DataFrame) -> None:
    by_channel = ranking.sort_values("channel_index_0based")
    colors = np.where(by_channel["kept"], "#2563eb", "#ef4444")
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [2, 1]})
    axes[0].bar(
        by_channel["channel_index_0based"] + 1,
        by_channel["score_rms_abs_corr"],
        color=colors,
        width=0.85,
    )
    axes[0].set(
        title="Train-only correlation channel score (blue=kept, red=masked)",
        xlabel="M1 physical channel (1-based)",
        ylabel="RMS |Pearson r| across vx, vy",
        xlim=(0, 97),
    )
    axes[1].plot(
        ranking["rank"], ranking["score_rms_abs_corr"], color="#334155", linewidth=1.5
    )
    boundary = int(ranking["kept"].sum())
    axes[1].axvline(boundary + 0.5, color="#ef4444", linestyle="--", label="mask boundary")
    axes[1].set(xlabel="rank", ylabel="score", xlim=(1, 96))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "channel_ranking.png", dpi=180)
    plt.close(fig)


def plot_primary_outputs(
    out: Path,
    history: pd.DataFrame,
    best_epoch: int,
    time_test: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.semilogy(history["epoch"], history["train_mse"], label="train")
    ax.semilogy(history["epoch"], history["validation_mse"], label="validation")
    ax.axvline(best_epoch, color="#ef4444", linestyle="--", label=f"best={best_epoch}")
    ax.set(title="Correlation-selected SNN training", xlabel="epoch", ylabel="MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "training_curve.png", dpi=180)
    plt.close(fig)

    steps = min(10_000, len(time_test))
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    for axis, label in enumerate(("vx", "vy")):
        axes[axis].plot(time_test[:steps], target[:steps, axis], label="target", lw=1)
        axes[axis].plot(
            time_test[:steps], prediction[:steps, axis], label="top-64 retrained SNN", lw=1
        )
        axes[axis].set_ylabel(label)
    axes[0].legend(ncol=2)
    axes[0].set_title("Continuous held-out test trace")
    axes[-1].set_xlabel("time from test target start (s)")
    fig.tight_layout()
    fig.savefig(out / "test_trace_top64.png", dpi=180)
    plt.close(fig)


def run_experiment(
    config: MaskingConfig,
    *,
    resume: bool,
    smoke: bool,
    cpu_threads: int,
    score_only: bool,
) -> None:
    out = output_dir(config, smoke)
    out.mkdir(parents=True, exist_ok=True)
    base.atomic_json(out / "protocol.json", prepare_protocol(config))
    base.atomic_json(out / "references.json", {"references": REFERENCES})
    base.atomic_json(
        out / "progress.json",
        {
            "status": "preprocessing",
            "epoch": 0,
            "epochs": config.epochs,
            "updated_at": base.now_iso(),
        },
    )

    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device("cpu")
    base.set_reproducible_seed(config.seed)

    session = base.prepare_sabes_session(
        DATA_PATH,
        mua_mode=config.mua_mode,
        moving_average_mode=config.moving_average_mode,
    )
    train_windows, validation_windows, validation_nominal, test_nominal, diagnostics = (
        split_windows(session, config)
    )
    diagnostics.to_csv(out / "boundary_alignment.csv", index=False)
    ranking, kept, masked = rank_channels(session, train_windows, config)
    ranking.to_csv(out / "channel_ranking.csv", index=False)
    base.atomic_json(
        out / "channel_mask.json",
        {
            "keep_channels": int(config.keep_channels),
            "kept_indices_0based": kept.tolist(),
            "kept_labels": [f"M1 {index + 1:03d}" for index in kept],
            "masked_indices_0based": masked.tolist(),
            "masked_labels": [f"M1 {index + 1:03d}" for index in masked],
            "ranking_rule": config.ranking_axis_reduction,
            "ranking_fraction": config.ranking_fraction,
            "ranking_steps": int(ranking["ranking_steps"].iloc[0]),
        },
    )
    plot_ranking(out, ranking)

    if smoke:
        train_windows = train_windows[:12]
        validation_windows = validation_windows[:4]
        validation_nominal = validation_nominal[:2]
        test_nominal = test_nominal[:2]
    max_steps = 1_000 if smoke else None
    base.atomic_json(
        out / "progress.json",
        {
            "status": "frozen_model_diagnostics",
            "epoch": 0,
            "epochs": config.epochs,
            "updated_at": base.now_iso(),
        },
    )
    frozen = evaluate_frozen_masks(
        session,
        test_nominal,
        ranking,
        kept,
        masked,
        config,
        device,
        out,
        max_steps,
    )

    if score_only:
        base.atomic_json(
            out / "progress.json",
            {
                "status": "score_only_complete",
                "epoch": 0,
                "epochs": config.epochs,
                "updated_at": base.now_iso(),
            },
        )
        print(f"SCORE_ONLY_COMPLETE {out}", flush=True)
        return

    train_loader, validation_loader = make_selected_loaders(
        session,
        train_windows,
        validation_windows,
        kept,
        config,
        truncate_steps=256 if smoke else None,
    )
    base.set_reproducible_seed(config.seed)
    model = base.MartisSNN(
        input_size=config.keep_channels,
        output_size=2,
        hidden_sizes=config.hidden_sizes,
        threshold=config.threshold,
        beta_init=config.beta_init,
    )
    parameter_count = base.count_trainable_parameters(model)

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
        out,
        resume=resume,
    )
    model.load_state_dict(best_state)

    _validation_time, validation_target, validation_prediction = predict_continuous(
        model,
        session,
        validation_nominal,
        config.neural_lead_ms,
        device,
        selected_channels=kept,
        max_steps=max_steps,
    )
    time_test, target_test, prediction_test = predict_continuous(
        model,
        session,
        test_nominal,
        config.neural_lead_ms,
        device,
        selected_channels=kept,
        max_steps=max_steps,
    )
    raw_metrics = base.regression_metrics(target_test, prediction_test)
    gains, offsets = base.fit_validation_calibration(validation_target, validation_prediction)
    calibrated_prediction = prediction_test * gains[None, :] + offsets[None, :]
    calibrated_metrics = base.regression_metrics(target_test, calibrated_prediction)

    history.to_csv(out / "training_history.csv", index=False)
    raw_metrics.to_csv(out / "test_metrics_raw_snn.csv", index=False)
    calibrated_metrics.to_csv(out / "test_metrics_validation_calibrated.csv", index=False)
    base.amplitude_rows(target_test, prediction_test).to_csv(
        out / "amplitude_diagnostics.csv", index=False
    )
    np.savez_compressed(
        out / "test_predictions.npz",
        time_sec=time_test,
        target=target_test,
        raw_prediction=prediction_test,
        validation_calibrated_prediction=calibrated_prediction,
        validation_gain=gains,
        validation_offset=offsets,
    )
    base.atomic_torch_save(
        out / "best_model.pt",
        {
            "model_state_dict": best_state,
            "session": SESSION_NAME,
            "kept_channels_0based": kept.tolist(),
            "masked_channels_0based": masked.tolist(),
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "config": asdict(config),
            "protocol": prepare_protocol(config),
        },
    )
    plot_primary_outputs(
        out, history, best_epoch, time_test, target_test, prediction_test
    )

    baseline_summary = json.loads((BASELINE_DIR / "run_summary.json").read_text(encoding="utf-8"))
    raw_mean = raw_metrics.loc[raw_metrics["axis"] == "mean"].iloc[0]
    summary = {
        "status": "complete",
        "analysis_type": "fresh_structural_retraining_with_selected_channels",
        "session": SESSION_NAME,
        "smoke": smoke,
        "device": str(device),
        "cpu_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "snntorch": __import__("snntorch").__version__,
        "keep_channels": config.keep_channels,
        "mask_channels": 96 - config.keep_channels,
        "kept_indices_0based": kept.tolist(),
        "masked_indices_0based": masked.tolist(),
        "trainable_parameters": parameter_count,
        "baseline_trainable_parameters": 22_914,
        "parameter_reduction_fraction": 1.0 - parameter_count / 22_914,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "training_started_at": training_started_at,
        "training_finished_at": training_finished_at,
        "training_elapsed_seconds": elapsed_seconds,
        "training_elapsed_hhmmss": base.format_elapsed(elapsed_seconds),
        "primary_result": {
            "mean_R2": float(raw_mean["R2"]),
            "mean_CC": float(raw_mean["CC"]),
            "mean_RMSE": float(raw_mean["RMSE"]),
        },
        "raw_snn_metrics": raw_metrics.to_dict(orient="records"),
        "validation_only_calibration": {
            "gain": gains.tolist(),
            "offset": offsets.tolist(),
            "metrics": calibrated_metrics.to_dict(orient="records"),
            "warning": "diagnostic only; not the primary comparison",
        },
        "baseline_reference": baseline_summary["primary_result"],
        "delta_vs_baseline": {
            "mean_R2": float(raw_mean["R2"]) - baseline_summary["primary_result"]["mean_R2"],
            "mean_CC": float(raw_mean["CC"]) - baseline_summary["primary_result"]["mean_CC"],
            "mean_RMSE": float(raw_mean["RMSE"])
            - baseline_summary["primary_result"]["mean_RMSE"],
        },
        "diagnostic_frozen_ablation": frozen.to_dict(orient="records"),
        "config": asdict(config),
        "references": REFERENCES,
    }
    base.atomic_json(out / "run_summary.json", summary)
    (out / "failure.json").unlink(missing_ok=True)
    base.atomic_json(
        out / "progress.json",
        {
            "status": "complete",
            "epoch": config.epochs,
            "epochs": config.epochs,
            "best_epoch": best_epoch,
            "raw_mean_R2": float(raw_mean["R2"]),
            "raw_mean_CC": float(raw_mean["CC"]),
            "elapsed_seconds": elapsed_seconds,
            "elapsed_hhmmss": base.format_elapsed(elapsed_seconds),
            "updated_at": base.now_iso(),
        },
    )
    print(
        f"COMPLETE {SESSION_NAME} top{config.keep_channels} "
        f"raw_R2={raw_mean['R2']:.6f} raw_CC={raw_mean['CC']:.6f} "
        f"elapsed={base.format_elapsed(elapsed_seconds)}",
        flush=True,
    )


def show_status(config: MaskingConfig, smoke: bool) -> None:
    path = output_dir(config, smoke) / "progress.json"
    if not path.exists():
        print(json.dumps({"status": "not_started", "path": str(path)}, indent=2))
        return
    print(path.read_text(encoding="utf-8"))


def self_test() -> None:
    rng = np.random.default_rng(7)
    n = 5_000
    x = rng.binomial(1, 0.05, size=(n, 96)).astype(np.uint8)
    y = rng.normal(size=(n, 2))
    y[:, 0] += 3.0 * x[:, 3]
    y[:, 1] -= 2.0 * x[:, 71]

    class SyntheticSession:
        mua_binary = x
        velocity = y

    class Window:
        start = 0
        end = n
        steps = n

    config = MaskingConfig(neural_lead_ms=0, keep_channels=2, ranking_fraction=1.0)
    ranking, kept, masked = rank_channels(SyntheticSession(), [Window()], config)
    if set(kept.tolist()) != {3, 71}:
        raise AssertionError(f"Synthetic informative channels not selected: {kept.tolist()}")
    if len(masked) != 94 or len(ranking) != 96:
        raise AssertionError("Unexpected mask/ranking shape")
    batch = [
        (torch.ones(3, 2), torch.ones(3, 2)),
        (torch.ones(5, 2), torch.ones(5, 2)),
    ]
    padded_x, padded_y, padded_mask = pad_selected_batch(batch)
    if padded_x.shape != (5, 2, 2) or padded_y.shape != (5, 2, 2):
        raise AssertionError("Batch padding shape mismatch")
    if float(padded_mask.sum()) != 8.0:
        raise AssertionError("Batch padding mask mismatch")
    print("SELF_TEST_OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-channels", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--posthoc-random-repeats", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.keep_channels <= 96:
        raise ValueError("--keep-channels must be between 1 and 96")
    config = MaskingConfig(
        keep_channels=args.keep_channels,
        epochs=args.epochs,
        posthoc_random_repeats=args.posthoc_random_repeats,
    )
    if args.self_test:
        self_test()
        return
    if args.status:
        show_status(config, args.smoke)
        return
    out = output_dir(config, args.smoke)
    try:
        run_experiment(
            config,
            resume=args.resume,
            smoke=args.smoke,
            cpu_threads=args.cpu_threads,
            score_only=args.score_only,
        )
    except Exception as error:
        out.mkdir(parents=True, exist_ok=True)
        base.atomic_json(
            out / "failure.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "failed_at": base.now_iso(),
            },
        )
        base.atomic_json(
            out / "progress.json",
            {
                "status": "failed",
                "epoch": 0,
                "epochs": config.epochs,
                "error": str(error),
                "updated_at": base.now_iso(),
            },
        )
        raise


if __name__ == "__main__":
    main()
