"""Export compact fixed-point result tables and figures for the public repository."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FIXED_POINT_ROOT = PROJECT_ROOT / "outputs" / "baseline_96ch_mixed_fixed_point"
RESULTS_ROOT = PROJECT_ROOT / "results"
FIGURES_ROOT = RESULTS_ROOT / "figures"
SESSIONS = (
    "indy_20170124_01",
    "indy_20170127_03",
    "indy_20170131_02",
    "indy_20160630_01",
    "indy_20160622_01",
)
SESSION_LABELS = {
    "indy_20170124_01": "Jan 24\n2017",
    "indy_20170127_03": "Jan 27\n2017",
    "indy_20170131_02": "Jan 31\n2017",
    "indy_20160630_01": "Jun 30\n2016",
    "indy_20160622_01": "Jun 22\n2016",
}
TRACE_SESSION = "indy_20170127_03"
TRACE_SECONDS = 5.0

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILION = "#D55E00"
GRAY = "#4D4D4D"
LIGHT_GRAY = "#D0D0D0"


def require_complete_results() -> pd.DataFrame:
    summary_path = FIXED_POINT_ROOT / "all_results.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing aggregated result: {summary_path}")
    frame = pd.read_csv(summary_path)
    if set(frame["session"]) != set(SESSIONS):
        raise RuntimeError("Fixed-point summary does not contain exactly the five expected sessions")
    frame = frame.set_index("session").loc[list(SESSIONS)].reset_index()
    for session in SESSIONS:
        session_root = FIXED_POINT_ROOT / session
        summary = json.loads(
            (session_root / "fixed_point_run_summary.json").read_text(encoding="utf-8")
        )
        if summary.get("status") != "complete" or summary.get("smoke"):
            raise RuntimeError(f"Full fixed-point result is not complete: {session}")
        reevaluated = summary.get("reevaluated_fp32_test_continuous")
        source = summary["source_fp32_test_continuous"]
        if reevaluated is None or any(
            not np.isclose(reevaluated[key], source[key], rtol=0.0, atol=1e-12)
            for key in ("R2", "CC", "RMSE")
        ):
            raise RuntimeError(f"Re-evaluated FP32 metrics differ from the source: {session}")
    return frame


def prediction_agreement() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for session in SESSIONS:
        path = FIXED_POINT_ROOT / session / "fixed_point_test_predictions.npz"
        with np.load(path) as arrays:
            fp32 = arrays["fp32_prediction"]
            fixed = arrays["mixed_fixed_point_prediction"]
        difference = fixed - fp32
        rows.append(
            {
                "session": session,
                "evaluated_steps": len(fp32),
                "prediction_MAE": float(np.mean(np.abs(difference))),
                "prediction_RMSE": float(np.sqrt(np.mean(difference**2))),
                "prediction_max_abs_error": float(np.max(np.abs(difference))),
                "prediction_corr_vx": float(np.corrcoef(fp32[:, 0], fixed[:, 0])[0, 1]),
                "prediction_corr_vy": float(np.corrcoef(fp32[:, 1], fixed[:, 1])[0, 1]),
            }
        )
    return pd.DataFrame(rows)


def layer_quantization() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for session in SESSIONS:
        frame = pd.read_csv(FIXED_POINT_ROOT / session / "layer_quantization.csv")
        frame.insert(0, "session", session)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def save_performance_figure(frame: pd.DataFrame) -> None:
    metrics = (
        ("delta_R2", 1_000.0, "Delta R-squared (x 1e-3)", True),
        ("delta_CC", 1_000.0, "Delta CC (x 1e-3)", True),
        ("delta_RMSE", 1.0, "Delta RMSE", False),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    x = np.arange(len(frame))
    for axis, (column, multiplier, ylabel, higher_is_better) in zip(axes, metrics, strict=True):
        values = frame[column].to_numpy(dtype=float) * multiplier
        beneficial = values >= 0 if higher_is_better else values <= 0
        colors = [GREEN if is_beneficial else VERMILION for is_beneficial in beneficial]
        axis.bar(x, values, color=colors, width=0.66)
        axis.axhline(0.0, color=GRAY, linewidth=0.9)
        mean = float(values.mean())
        axis.axhline(mean, color=BLUE, linewidth=1.4, linestyle="--")
        limit = max(float(np.max(np.abs(values))), abs(mean), 1e-9) * 1.35
        axis.set_ylim(-limit, limit)
        axis.set_xticks(x, [SESSION_LABELS[name] for name in frame["session"]])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.7, alpha=0.65)
        axis.set_axisbelow(True)
        direction = "higher is better" if higher_is_better else "lower is better"
        axis.set_title(direction)
        axis.text(
            0.98,
            0.96,
            f"mean {mean:+.3f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            color=BLUE,
            fontsize=10,
        )
        for index, value in enumerate(values):
            offset = limit * 0.035
            axis.text(
                index,
                value + (offset if value >= 0 else -offset),
                f"{value:+.3f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8.5,
            )
    figure.suptitle("Mixed fixed-point minus FP32 performance by session", fontsize=15)
    figure.legend(
        handles=(
            Patch(facecolor=GREEN, label="beneficial change"),
            Patch(facecolor=VERMILION, label="adverse change"),
            plt.Line2D([0], [0], color=BLUE, linestyle="--", label="five-session mean"),
        ),
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    figure.savefig(
        FIGURES_ROOT / "mixed_fixed_point_performance.png",
        dpi=180,
        facecolor="white",
    )
    plt.close(figure)


def save_prediction_figure(frame: pd.DataFrame, agreement: pd.DataFrame) -> None:
    path = FIXED_POINT_ROOT / TRACE_SESSION / "fixed_point_test_predictions.npz"
    with np.load(path) as arrays:
        time = arrays["test_time_sec"]
        target = arrays["test_target"]
        fp32 = arrays["fp32_prediction"]
        fixed = arrays["mixed_fixed_point_prediction"]
    time = time - time[0]
    keep = time <= TRACE_SECONDS
    time = time[keep]
    target = target[keep]
    fp32 = fp32[keep]
    fixed = fixed[keep]
    residual = fixed - fp32

    figure, axes = plt.subplots(3, 1, figsize=(13, 8.2), sharex=True, constrained_layout=True)
    for axis_index, (axis, label) in enumerate(zip(axes[:2], ("vx", "vy"), strict=True)):
        axis.plot(time, target[:, axis_index], color=GRAY, linewidth=1.0, label="target")
        axis.plot(time, fp32[:, axis_index], color=BLUE, linewidth=1.15, label="FP32")
        axis.plot(
            time,
            fixed[:, axis_index],
            color=ORANGE,
            linewidth=1.0,
            linestyle="--",
            label="mixed fixed-point",
        )
        axis.set_ylabel(f"{label} velocity")
        axis.grid(color=LIGHT_GRAY, linewidth=0.7, alpha=0.65)
        axis.set_axisbelow(True)
    axes[0].legend(loc="upper right", ncol=3, frameon=False)
    axes[2].plot(time, residual[:, 0], color=BLUE, linewidth=0.9, label="vx residual")
    axes[2].plot(time, residual[:, 1], color=ORANGE, linewidth=0.9, label="vy residual")
    axes[2].axhline(0.0, color=GRAY, linewidth=0.8)
    axes[2].set_ylabel("fixed - FP32")
    axes[2].set_xlabel("Time from held-out test start (s)")
    axes[2].grid(color=LIGHT_GRAY, linewidth=0.7, alpha=0.65)
    axes[2].set_axisbelow(True)
    axes[2].legend(loc="upper right", ncol=2, frameon=False)

    result = frame.loc[frame["session"] == TRACE_SESSION].iloc[0]
    match = agreement.loc[agreement["session"] == TRACE_SESSION].iloc[0]
    figure.suptitle(
        "Prediction trace: indy_20170127_03 (first 5 s of held-out test)\n"
        f"Full test R-squared: FP32 {result.fp32_R2:.4f}, fixed {result.fixed_R2:.4f}; "
        f"prediction correlation: vx {match.prediction_corr_vx:.6f}, "
        f"vy {match.prediction_corr_vy:.6f}",
        fontsize=14,
    )
    figure.savefig(
        FIGURES_ROOT / "mixed_fixed_point_prediction_trace.png",
        dpi=180,
        facecolor="white",
    )
    plt.close(figure)


def main() -> None:
    frame = require_complete_results()
    agreement = prediction_agreement()
    quantization = layer_quantization()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS_ROOT / "fixed_point_summary.csv", index=False)
    agreement.to_csv(RESULTS_ROOT / "fixed_point_prediction_agreement.csv", index=False)
    quantization.to_csv(RESULTS_ROOT / "fixed_point_layer_quantization.csv", index=False)
    save_performance_figure(frame)
    save_prediction_figure(frame, agreement)
    print(f"Exported fixed-point public results to {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
