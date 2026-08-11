"""Evaluate a baseline with the paper-reported mixed fixed-point widths.

Implemented path: binary spike input, signed INT8 weights and synaptic sums,
unsigned 13-bit decay, signed INT32 membrane/threshold/reset arithmetic, then a
conversion of the final two regression outputs to real units.  Undisclosed Q-format
choices are recorded in every result rather than presented as paper facts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data" / "sabes_zenodo" / "master_mat"
SESSIONS = (
    "indy_20170124_01",
    "indy_20170127_03",
    "indy_20170131_02",
    "indy_20160630_01",
    "indy_20160622_01",
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_base_module() -> Any:
    return load_module(
        SCRIPT_DIR / "internal" / "reproduction_core.py",
        "mixed_fixed_point_reproduction_core",
    )


def load_fixed_point_module() -> Any:
    return load_module(
        SCRIPT_DIR / "internal" / "fixed_point_core.py",
        "mixed_fixed_point_integer_core",
    )


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
    temporary.replace(path)


def reconstructed_task_windows(
    base: Any,
    session: Any,
    window_steps: int = 3_876,
    offset_ms: int = -32,
) -> list[Any]:
    starts = [0]
    starts.extend(window.start + offset_ms for window in session.task_windows[2::3])
    windows = []
    for index, raw_start in enumerate(starts):
        start = max(0, int(raw_start))
        end = min(start + window_steps, len(session.velocity))
        if end - start == window_steps:
            windows.append(base.TaskWindow(task_index=index, start=start, end=end))
    return windows


def floor_chronological_split(
    windows: Sequence[Any],
) -> tuple[list[Any], list[Any], list[Any]]:
    n_total = len(windows)
    n_train = int(0.8 * n_total)
    n_validation = int(0.1 * n_total)
    return (
        list(windows[:n_train]),
        list(windows[n_train : n_train + n_validation]),
        list(windows[n_train + n_validation :]),
    )


def metric_mean(frame: pd.DataFrame) -> dict[str, float]:
    row = frame.loc[frame["axis"] == "mean"].iloc[0]
    return {key: float(row[key]) for key in ("R2", "CC", "RMSE")}


def predict_fixed_point(
    model: Any,
    session: Any,
    window: Any,
    neural_lead_ms: int,
    *,
    max_steps: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_start = int(window.start)
    feature_end = int(window.end) - neural_lead_ms
    target_start = feature_start + neural_lead_ms
    if max_steps is not None:
        feature_end = min(feature_end, feature_start + max_steps)
    target_end = target_start + (feature_end - feature_start)
    if feature_end <= feature_start:
        raise ValueError("The selected test span is empty after lead alignment")
    features = torch.from_numpy(
        session.mua_binary[feature_start:feature_end].astype(np.int32)
    ).unsqueeze(1)
    with torch.no_grad():
        prediction = model.forward(features).squeeze(1).numpy()
    target = session.velocity[target_start:target_end]
    time_sec = session.time_sec[target_start:target_end] - session.time_sec[target_start]
    return time_sec, target, prediction


def evaluate(args: argparse.Namespace) -> None:
    base = load_base_module()
    fixed = load_fixed_point_module()
    source_dir = PROJECT_ROOT / "outputs" / args.source_experiment_name / args.session
    output_dir = PROJECT_ROOT / "outputs" / args.experiment_name / args.session
    checkpoint_path = source_dir / "best_model.pt"
    summary_path = source_dir / "run_summary.json"
    if not checkpoint_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Completed 96-channel baseline is missing: {source_dir}")
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if source_summary.get("status") != "complete":
        raise RuntimeError(f"Baseline summary is not complete: {summary_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    started_at = base.now_iso()
    base.atomic_json(
        output_dir / "progress.json",
        {"status": "loading", "session": args.session, "started_at": started_at},
    )
    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    float_model = base.MartisSNN(
        input_size=96,
        output_size=2,
        hidden_sizes=tuple(config["hidden_sizes"]),
        threshold=float(config["threshold"]),
        beta_init=float(config["beta_init"]),
        optimized_forward=True,
    )
    float_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    float_model.eval()
    fixed_model = fixed.FixedPointSNN.from_float_model(
        float_model,
        weight_scale_mode=args.weight_scale_mode,
    )

    preprocessing_started = time.perf_counter()
    session = base.prepare_sabes_session(
        DATA_ROOT / f"{args.session}.mat",
        mua_mode=config["mua_mode"],
        moving_average_mode=config["moving_average_mode"],
    )
    all_windows = reconstructed_task_windows(base, session)
    train_windows, validation_windows, test_windows = floor_chronological_split(all_windows)
    if not test_windows:
        raise RuntimeError("Chronological split produced no test windows")
    continuous_test = base.TaskWindow(
        task_index=-1,
        start=test_windows[0].start,
        end=len(session.velocity),
    )
    preprocessing_seconds = time.perf_counter() - preprocessing_started

    fp32_metrics: dict[str, float] | None = None
    fp32_prediction: np.ndarray | None = None
    fp32_seconds: float | None = None
    if args.evaluate_fp32:
        base.atomic_json(
            output_dir / "progress.json",
            {"status": "fp32_test", "session": args.session},
        )
        fp32_started = time.perf_counter()
        _, fp32_target, fp32_prediction = base.predict_continuous_lagged(
            float_model,
            session,
            [continuous_test],
            int(config["neural_lead_ms"]),
            torch.device("cpu"),
            max_steps=args.max_test_steps,
        )
        fp32_seconds = time.perf_counter() - fp32_started
        fp32_metrics = metric_mean(base.regression_metrics(fp32_target, fp32_prediction))

    base.atomic_json(
        output_dir / "progress.json",
        {"status": "mixed_fixed_point_test", "session": args.session},
    )
    fixed_started = time.perf_counter()
    test_time, test_target, fixed_prediction = predict_fixed_point(
        fixed_model,
        session,
        continuous_test,
        int(config["neural_lead_ms"]),
        max_steps=args.max_test_steps,
    )
    fixed_seconds = time.perf_counter() - fixed_started
    fixed_frame = base.regression_metrics(test_target, fixed_prediction)
    fixed_metrics = metric_mean(fixed_frame)

    source_fp32 = {
        key: float(source_summary["test_continuous"][key]) for key in ("R2", "CC", "RMSE")
    }
    comparison_fp32 = fp32_metrics if fp32_metrics is not None else source_fp32
    delta = {key: fixed_metrics[key] - comparison_fp32[key] for key in fixed_metrics}
    layer_frame = pd.DataFrame(fixed_model.quantization_rows())
    diagnostics = fixed_model.last_diagnostics
    for layer_name, values in diagnostics.items():
        mask = layer_frame["layer"] == layer_name
        for key, value in values.items():
            layer_frame.loc[mask, key] = value

    atomic_csv(output_dir / "fixed_point_test_metrics_continuous.csv", fixed_frame)
    atomic_csv(output_dir / "layer_quantization.csv", layer_frame)
    prediction_arrays: dict[str, Any] = {
        "test_time_sec": test_time,
        "test_target": test_target,
        "mixed_fixed_point_prediction": fixed_prediction,
    }
    if fp32_prediction is not None:
        prediction_arrays["fp32_prediction"] = fp32_prediction
    atomic_npz(output_dir / "fixed_point_test_predictions.npz", **prediction_arrays)
    atomic_npz(
        output_dir / "quantized_parameters.npz",
        **fixed_model.artifact_arrays(),
    )

    is_smoke = args.max_test_steps is not None
    summary = {
        "status": "complete",
        "analysis_type": "paper_precision_fixed_point_integer_emulation",
        "experiment_name": args.experiment_name,
        "source_experiment_name": args.source_experiment_name,
        "session": args.session,
        "channels": 96,
        "smoke": is_smoke,
        "started_at": started_at,
        "finished_at": base.now_iso(),
        "device": "cpu",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "source_model_dir": source_dir.relative_to(PROJECT_ROOT).as_posix(),
        "source_checkpoint": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
        "test_span": {
            "start_sample": int(continuous_test.start),
            "recording_end_sample": int(continuous_test.end),
            "evaluated_steps": len(test_target),
            "max_test_steps": args.max_test_steps,
            "protocol": "continuous held-out test; single state reset",
        },
        "split_counts": {
            "all": len(all_windows),
            "train": len(train_windows),
            "validation": len(validation_windows),
            "test": len(test_windows),
        },
        "paper_reported_precision": {
            "input": "binary MUA spike events",
            "weights": "8-bit fixed-point",
            "membrane_potential": "32-bit fixed-point",
            "decay": "13-bit fixed-point",
        },
        "implementation": fixed_model.metadata(),
        "implementation_assumptions": {
            "weight_scale": (
                "one power-of-two scale per layer inferred from checkpoint max-abs"
                if args.weight_scale_mode == "pow2"
                else "one max-abs symmetric scale per layer"
            ),
            "decay_q_format": "UQ0.13",
            "potential_scale": "the corresponding layer weight scale",
            "threshold_scale": "the corresponding layer potential scale",
            "rounding": "nearest, half away from zero",
            "reset_timing": "snnTorch reset_delay=True checkpoint semantics",
            "limit": (
                "The paper and author notebook do not publish trained observer values, "
                "all fractional-bit positions, or RTL timing. This is fixed-point "
                "integer arithmetic emulation, not bit-exact FPGA reproduction."
            ),
        },
        "source_fp32_test_continuous": source_fp32,
        "reevaluated_fp32_test_continuous": fp32_metrics,
        "mixed_fixed_point_test_continuous": fixed_metrics,
        "delta_fixed_point_minus_fp32": delta,
        "saturation_diagnostics": diagnostics,
        "timing_seconds": {
            "preprocessing": preprocessing_seconds,
            "fp32_test": fp32_seconds,
            "mixed_fixed_point_test": fixed_seconds,
            "pipeline_total": time.perf_counter() - started,
        },
        "files": {
            "metrics": "fixed_point_test_metrics_continuous.csv",
            "predictions": "fixed_point_test_predictions.npz",
            "integer_parameters_and_scales": "quantized_parameters.npz",
            "layer_quantization_report": "layer_quantization.csv",
        },
    }
    base.atomic_json(output_dir / "fixed_point_run_summary.json", summary)
    base.atomic_json(output_dir / "progress.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def self_test() -> None:
    base = load_base_module()
    fixed = load_fixed_point_module()
    torch.manual_seed(17)
    model = base.MartisSNN(
        input_size=5,
        output_size=2,
        hidden_sizes=(4, 3, 2),
        threshold=0.1,
        beta_init=0.9,
        optimized_forward=True,
    )
    integer_model = fixed.FixedPointSNN.from_float_model(model, weight_scale_mode="pow2")
    features = torch.randint(0, 2, (11, 2, 5), dtype=torch.int32)
    prediction = integer_model.forward(features)
    if prediction.shape != (11, 2, 2) or prediction.dtype != torch.float32:
        raise AssertionError("Unexpected fixed-point output")
    arrays = integer_model.artifact_arrays()
    weight_arrays = [value for key, value in arrays.items() if key.endswith("weight_int8")]
    decay_arrays = [value for key, value in arrays.items() if key.endswith("decay_u13")]
    if len(weight_arrays) != 4 or any(value.dtype != np.int8 for value in weight_arrays):
        raise AssertionError("Weights are not stored as four INT8 tensors")
    if len(decay_arrays) != 4 or any(value.dtype != np.uint16 for value in decay_arrays):
        raise AssertionError("Decay values are not stored as four integer tensors")
    if any(int(value.min()) < 0 or int(value.max()) > 8191 for value in decay_arrays):
        raise AssertionError("A decay value is outside unsigned 13-bit range")
    zero_prediction = integer_model.forward(torch.zeros((3, 1, 5), dtype=torch.int32))
    if not bool(torch.all(zero_prediction == 0)):
        raise AssertionError("Zero input from reset state must produce zero output")
    try:
        integer_model.forward(torch.full((2, 1, 5), 0.5))
    except ValueError as error:
        if "binary" not in str(error):
            raise
    else:
        raise AssertionError("Non-binary input was not rejected")
    if not torch.equal(
        fixed._round_shift_signed(torch.tensor([-12, -4, 4, 12]), 3),
        torch.tensor([-2, -1, 1, 2]),
    ):
        raise AssertionError("Signed fixed-point rounding is inconsistent")
    print("SELF_TEST_OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", choices=SESSIONS)
    parser.add_argument("--source-experiment-name", default="baseline_96ch")
    parser.add_argument("--experiment-name", default="baseline_96ch_mixed_fixed_point")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--weight-scale-mode", choices=("pow2", "maxabs"), default="pow2")
    parser.add_argument("--evaluate-fp32", action="store_true")
    parser.add_argument("--max-test-steps", type=int)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.session is None:
        raise ValueError("--session is required unless --self-test is used")
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be at least 1")
    if args.max_test_steps is not None and args.max_test_steps < 10:
        raise ValueError("--max-test-steps must be at least 10")
    if args.max_test_steps is not None and not args.evaluate_fp32:
        raise ValueError("Smoke comparisons require --evaluate-fp32")

    output_dir = PROJECT_ROOT / "outputs" / args.experiment_name / args.session
    try:
        evaluate(args)
    except Exception as error:
        base = load_base_module()
        failure = {
            "status": "failed",
            "session": args.session,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "failed_at": base.now_iso(),
        }
        base.atomic_json(output_dir / "failure.json", failure)
        base.atomic_json(output_dir / "progress.json", failure)
        raise


if __name__ == "__main__":
    main()
