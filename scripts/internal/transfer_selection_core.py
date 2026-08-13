"""Transfer-aware channel ranking for continuous-velocity SNN decoding.

The implementation keeps the stationarity / importance / redundancy structure
of Zhang et al. (TNSRE 2021), but adapts it to unpaired recording sessions and
continuous vx/vy targets:

* stationarity: 1 - Jensen-Shannon distance between 50 ms count histograms;
* importance: symmetrical uncertainty after source-defined velocity quantiles;
* redundancy: equal-session-weighted channel-to-channel symmetrical uncertainty;
* selection: greedy fixed-K ranking with an explicit redundancy penalty.

Only source-train and target-calibration windows may be passed to this module.
Validation and test windows are intentionally outside its API.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TransferSelectionConfig:
    aggregation_bin_ms: int = 50
    importance_quantile_bins: int = 8
    source_importance_weight: float = 0.5
    stationarity_weight: float = 0.5
    redundancy_weight: float = 0.25
    histogram_pseudocount: float = 1e-9


@dataclass(frozen=True)
class AggregatedWindows:
    counts: np.ndarray
    velocity: np.ndarray
    window_ids: np.ndarray


@dataclass(frozen=True)
class TransferRankingResult:
    ranking: pd.DataFrame
    order: np.ndarray
    redundancy_matrix: np.ndarray
    velocity_bin_edges: tuple[np.ndarray, np.ndarray]


def validate_config(config: TransferSelectionConfig) -> None:
    if config.aggregation_bin_ms < 1:
        raise ValueError("aggregation_bin_ms must be positive")
    if config.importance_quantile_bins < 2:
        raise ValueError("importance_quantile_bins must be at least 2")
    for name, value in (
        ("source_importance_weight", config.source_importance_weight),
        ("stationarity_weight", config.stationarity_weight),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if config.redundancy_weight < 0.0:
        raise ValueError("redundancy_weight must be non-negative")
    if config.histogram_pseudocount <= 0.0:
        raise ValueError("histogram_pseudocount must be positive")


def aggregate_windows(
    session: Any,
    windows: Sequence[Any],
    *,
    bin_ms: int,
    neural_lead_ms: int = 0,
) -> AggregatedWindows:
    """Aggregate 1 ms binary MUA and aligned velocity inside each task window."""
    if bin_ms < 1:
        raise ValueError("bin_ms must be positive")
    if neural_lead_ms < 0:
        raise ValueError("neural_lead_ms must be non-negative")
    count_parts: list[np.ndarray] = []
    velocity_parts: list[np.ndarray] = []
    window_id_parts: list[np.ndarray] = []
    n_channels = int(session.mua_binary.shape[1])

    for local_index, window in enumerate(windows):
        available_steps = int(window.end - window.start - neural_lead_ms)
        usable_steps = available_steps // bin_ms * bin_ms
        if usable_steps < bin_ms:
            continue
        feature_start = int(window.start)
        feature_end = feature_start + usable_steps
        target_start = feature_start + neural_lead_ms
        target_end = target_start + usable_steps
        counts = session.mua_binary[feature_start:feature_end]
        velocity = session.velocity[target_start:target_end]
        count_parts.append(counts.reshape(-1, bin_ms, n_channels).sum(axis=1).astype(np.int16))
        velocity_parts.append(velocity.reshape(-1, bin_ms, 2).mean(axis=1).astype(np.float32))
        task_index = int(getattr(window, "task_index", local_index))
        window_id_parts.append(np.full(usable_steps // bin_ms, task_index, dtype=np.int32))

    if not count_parts:
        raise ValueError("No complete aggregation bins were available")
    return AggregatedWindows(
        counts=np.concatenate(count_parts, axis=0),
        velocity=np.concatenate(velocity_parts, axis=0),
        window_ids=np.concatenate(window_id_parts, axis=0),
    )


def _entropy(values: np.ndarray) -> float:
    encoded = np.asarray(values, dtype=np.int64).reshape(-1)
    if encoded.size == 0:
        return 0.0
    if encoded.min() < 0:
        _, encoded = np.unique(encoded, return_inverse=True)
    counts = np.bincount(encoded)
    probabilities = counts[counts > 0].astype(np.float64) / encoded.size
    return float(-(probabilities * np.log(probabilities)).sum())


def symmetrical_uncertainty(left: np.ndarray, right: np.ndarray) -> float:
    """Compute normalized discrete mutual information in [0, 1]."""
    x = np.asarray(left, dtype=np.int64).reshape(-1)
    y = np.asarray(right, dtype=np.int64).reshape(-1)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("SU inputs must be non-empty and have the same shape")
    if x.min() < 0:
        _, x = np.unique(x, return_inverse=True)
    if y.min() < 0:
        _, y = np.unique(y, return_inverse=True)
    x_states = int(x.max()) + 1
    y_states = int(y.max()) + 1
    joint = np.bincount(x * y_states + y, minlength=x_states * y_states)
    joint = joint.reshape(x_states, y_states).astype(np.float64)
    joint /= x.size
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    expected = px[:, None] * py[None, :]
    nonzero = joint > 0
    mutual_information = float(np.sum(joint[nonzero] * np.log(joint[nonzero] / expected[nonzero])))
    denominator = _entropy(x) + _entropy(y)
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(2.0 * mutual_information / denominator, 0.0, 1.0))


def _jensen_shannon_distance(
    source: np.ndarray,
    target: np.ndarray,
    *,
    support_size: int,
    pseudocount: float,
) -> float:
    source_hist = np.bincount(source, minlength=support_size).astype(np.float64)
    target_hist = np.bincount(target, minlength=support_size).astype(np.float64)
    source_hist += pseudocount
    target_hist += pseudocount
    source_hist /= source_hist.sum()
    target_hist /= target_hist.sum()
    midpoint = 0.5 * (source_hist + target_hist)
    divergence = 0.5 * np.sum(source_hist * np.log2(source_hist / midpoint))
    divergence += 0.5 * np.sum(target_hist * np.log2(target_hist / midpoint))
    return float(np.sqrt(max(divergence, 0.0)))


def stationarity_scores(
    source_counts: np.ndarray,
    target_counts: np.ndarray,
    *,
    pseudocount: float,
) -> tuple[np.ndarray, np.ndarray]:
    if source_counts.shape[1] != target_counts.shape[1]:
        raise ValueError("Source and target must have the same channels")
    support_size = int(max(source_counts.max(), target_counts.max())) + 1
    distances = np.array(
        [
            _jensen_shannon_distance(
                source_counts[:, channel],
                target_counts[:, channel],
                support_size=support_size,
                pseudocount=pseudocount,
            )
            for channel in range(source_counts.shape[1])
        ],
        dtype=np.float64,
    )
    return 1.0 - distances, distances


def source_velocity_bin_edges(
    source_velocity: np.ndarray,
    quantile_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    quantiles = np.linspace(0.0, 1.0, quantile_bins + 1)[1:-1]
    edges = []
    for axis in range(2):
        axis_edges = np.unique(np.quantile(source_velocity[:, axis], quantiles))
        edges.append(np.asarray(axis_edges, dtype=np.float64))
    return edges[0], edges[1]


def discretize_velocity(
    velocity: np.ndarray,
    edges: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    return np.column_stack(
        [np.digitize(velocity[:, axis], edges[axis]).astype(np.int16) for axis in range(2)]
    )


def importance_scores(
    counts: np.ndarray,
    velocity_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_channels = counts.shape[1]
    axis_scores = np.zeros((n_channels, 2), dtype=np.float64)
    for channel in range(n_channels):
        for axis in range(2):
            axis_scores[channel, axis] = symmetrical_uncertainty(
                counts[:, channel], velocity_bins[:, axis]
            )
    combined = np.sqrt(np.mean(np.square(axis_scores), axis=1))
    return axis_scores[:, 0], axis_scores[:, 1], combined


def redundancy_matrix(counts: np.ndarray) -> np.ndarray:
    n_channels = counts.shape[1]
    matrix = np.eye(n_channels, dtype=np.float64)
    for left in range(n_channels):
        for right in range(left + 1, n_channels):
            value = symmetrical_uncertainty(counts[:, left], counts[:, right])
            matrix[left, right] = value
            matrix[right, left] = value
    return matrix


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    if len(values) <= 1:
        return np.ones_like(ranks)
    return ranks / (len(values) - 1)


def rank_transfer_channels(
    source: AggregatedWindows,
    target_calibration: AggregatedWindows,
    config: TransferSelectionConfig,
    *,
    channel_names: Sequence[str] | None = None,
) -> TransferRankingResult:
    """Return a full greedy rank so top-K masks remain nested."""
    validate_config(config)
    source_counts = np.asarray(source.counts, dtype=np.int64)
    target_counts = np.asarray(target_calibration.counts, dtype=np.int64)
    if source_counts.ndim != 2 or target_counts.ndim != 2:
        raise ValueError("Count arrays must be two-dimensional")
    if source_counts.shape[1] != target_counts.shape[1]:
        raise ValueError("Source and target channel counts differ")
    n_channels = source_counts.shape[1]
    names = (
        list(channel_names)
        if channel_names is not None
        else [f"channel_{i + 1}" for i in range(n_channels)]
    )
    if len(names) != n_channels:
        raise ValueError("channel_names length does not match the count arrays")

    stationarity, js_distance = stationarity_scores(
        source_counts,
        target_counts,
        pseudocount=config.histogram_pseudocount,
    )
    edges = source_velocity_bin_edges(
        source.velocity,
        config.importance_quantile_bins,
    )
    source_velocity_bins = discretize_velocity(source.velocity, edges)
    target_velocity_bins = discretize_velocity(target_calibration.velocity, edges)
    source_ix, source_iy, source_importance = importance_scores(source_counts, source_velocity_bins)
    target_ix, target_iy, target_importance = importance_scores(target_counts, target_velocity_bins)
    source_weight = config.source_importance_weight
    combined_importance = (
        source_weight * source_importance + (1.0 - source_weight) * target_importance
    )

    source_redundancy = redundancy_matrix(source_counts)
    target_redundancy = redundancy_matrix(target_counts)
    combined_redundancy = 0.5 * (source_redundancy + target_redundancy)

    stationarity_rank = _percentile_rank(stationarity)
    importance_rank = _percentile_rank(combined_importance)
    base_score = (
        config.stationarity_weight * stationarity_rank
        + (1.0 - config.stationarity_weight) * importance_rank
    )

    selected: list[int] = []
    selection_penalty = np.zeros(n_channels, dtype=np.float64)
    selection_score = np.full(n_channels, np.nan, dtype=np.float64)
    for _ in range(n_channels):
        if selected:
            penalties = combined_redundancy[:, selected].mean(axis=1)
        else:
            penalties = np.zeros(n_channels, dtype=np.float64)
        candidate_score = base_score - config.redundancy_weight * penalties
        if selected:
            candidate_score[np.asarray(selected, dtype=np.int64)] = -np.inf
        chosen = int(np.argmax(candidate_score))
        selection_penalty[chosen] = penalties[chosen]
        selection_score[chosen] = candidate_score[chosen]
        selected.append(chosen)

    order = np.asarray(selected, dtype=np.int64)
    source_rate_hz = 1000.0 * source_counts.mean(axis=0) / config.aggregation_bin_ms
    target_rate_hz = 1000.0 * target_counts.mean(axis=0) / config.aggregation_bin_ms
    frame = pd.DataFrame(
        {
            "rank": np.arange(1, n_channels + 1),
            "channel_index_0based": order,
            "channel_name": [names[index] for index in order],
            "stationarity_similarity": stationarity[order],
            "jensen_shannon_distance": js_distance[order],
            "stationarity_percentile": stationarity_rank[order],
            "importance_source_vx_su": source_ix[order],
            "importance_source_vy_su": source_iy[order],
            "importance_source_rms_su": source_importance[order],
            "importance_target_vx_su": target_ix[order],
            "importance_target_vy_su": target_iy[order],
            "importance_target_rms_su": target_importance[order],
            "importance_combined": combined_importance[order],
            "importance_percentile": importance_rank[order],
            "base_score": base_score[order],
            "redundancy_penalty_at_selection": selection_penalty[order],
            "greedy_score_at_selection": selection_score[order],
            "source_firing_rate_hz": source_rate_hz[order],
            "target_calibration_firing_rate_hz": target_rate_hz[order],
        }
    )
    return TransferRankingResult(
        ranking=frame,
        order=order,
        redundancy_matrix=combined_redundancy,
        velocity_bin_edges=edges,
    )


def plot_transfer_ranking(
    output_path: Path,
    ranking: pd.DataFrame,
    *,
    keep_channels: int,
    title: str,
) -> None:
    by_channel = ranking.sort_values("channel_index_0based")
    selected = set(
        ranking.loc[ranking["rank"] <= keep_channels, "channel_index_0based"].astype(int)
    )
    colors = [
        "#2563eb" if int(index) in selected else "#ef4444"
        for index in by_channel["channel_index_0based"]
    ]
    figure, axes = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)
    x = by_channel["channel_index_0based"].to_numpy() + 1
    axes[0].bar(x, by_channel["stationarity_similarity"], color=colors)
    axes[0].set_ylabel("1 - JSD")
    axes[0].set_title(title)
    axes[1].bar(x, by_channel["importance_combined"], color=colors)
    axes[1].set_ylabel("combined importance")
    axes[2].bar(
        np.arange(1, len(ranking) + 1),
        ranking["greedy_score_at_selection"],
        color=["#2563eb"] * keep_channels + ["#ef4444"] * (len(ranking) - keep_channels),
    )
    axes[2].axvline(keep_channels + 0.5, color="black", linestyle="--", linewidth=1)
    axes[2].set_xlabel("greedy rank")
    axes[2].set_ylabel("selection score")
    for axis in axes[:2]:
        axis.set_xlim(0, len(ranking) + 1)
        axis.grid(alpha=0.2, axis="y")
    axes[2].grid(alpha=0.2, axis="y")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def self_test() -> None:
    rng = np.random.default_rng(7)
    rows = 2_000
    latent_source = rng.integers(0, 5, size=rows)
    latent_target = rng.integers(0, 5, size=rows)
    source_counts = np.column_stack(
        [
            latent_source,
            latent_source,
            rng.integers(0, 5, size=rows),
            rng.integers(0, 5, size=rows),
        ]
    )
    target_counts = np.column_stack(
        [
            latent_target,
            latent_target,
            rng.integers(6, 11, size=rows),
            rng.integers(0, 5, size=rows),
        ]
    )
    source_velocity = np.column_stack(
        [latent_source + rng.normal(0, 0.1, rows), rng.normal(0, 1, rows)]
    ).astype(np.float32)
    target_velocity = np.column_stack(
        [latent_target + rng.normal(0, 0.1, rows), rng.normal(0, 1, rows)]
    ).astype(np.float32)
    result = rank_transfer_channels(
        AggregatedWindows(source_counts, source_velocity, np.zeros(rows, dtype=np.int32)),
        AggregatedWindows(target_counts, target_velocity, np.zeros(rows, dtype=np.int32)),
        TransferSelectionConfig(),
        channel_names=["stable", "redundant", "shifted", "noise"],
    )
    if result.order[0] not in {0, 1}:
        raise AssertionError("A stable informative channel should rank first")
    if len(set(result.order.tolist())) != 4:
        raise AssertionError("Ranking must contain each channel exactly once")
    if not np.isfinite(result.ranking.select_dtypes(include=[np.number])).all().all():
        raise AssertionError("Ranking contains a non-finite value")


if __name__ == "__main__":
    self_test()
    print("transfer_selection_core self-test passed")
