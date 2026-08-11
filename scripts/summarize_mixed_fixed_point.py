"""Aggregate completed mixed fixed-point baseline evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SESSIONS = (
    "indy_20170124_01",
    "indy_20170127_03",
    "indy_20170131_02",
    "indy_20160630_01",
    "indy_20160622_01",
)


def markdown_table(frame: pd.DataFrame) -> str:
    def display(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value)

    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(display(value) for value in row) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default="baseline_96ch_mixed_fixed_point")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = PROJECT_ROOT / "outputs" / args.experiment_name
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for session in SESSIONS:
        path = root / session / "fixed_point_run_summary.json"
        if not path.is_file():
            missing.append(session)
            continue
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("status") != "complete" or item.get("smoke"):
            missing.append(session)
            continue
        fp32 = item["source_fp32_test_continuous"]
        quantized = item["mixed_fixed_point_test_continuous"]
        delta = item["delta_fixed_point_minus_fp32"]
        total_saturations = sum(
            int(layer["potential_saturation_count"])
            for layer in item["saturation_diagnostics"].values()
        )
        rows.append(
            {
                "session": session,
                "channels": 96,
                "weight_scale_mode": item["implementation"]["weight"]["scale_mode"],
                "evaluated_steps": item["test_span"]["evaluated_steps"],
                "potential_saturations": total_saturations,
                "fp32_R2": fp32["R2"],
                "fixed_R2": quantized["R2"],
                "delta_R2": delta["R2"],
                "fp32_CC": fp32["CC"],
                "fixed_CC": quantized["CC"],
                "delta_CC": delta["CC"],
                "fp32_RMSE": fp32["RMSE"],
                "fixed_RMSE": quantized["RMSE"],
                "delta_RMSE": delta["RMSE"],
            }
        )
    if not rows:
        raise RuntimeError(f"No completed full fixed-point results found under {root}")

    order = {session: index for index, session in enumerate(SESSIONS)}
    frame = pd.DataFrame(rows)
    frame["session_order"] = frame["session"].map(order)
    frame = frame.sort_values("session_order").drop(columns="session_order")
    metrics = [
        "potential_saturations",
        "fp32_R2",
        "fixed_R2",
        "delta_R2",
        "fp32_CC",
        "fixed_CC",
        "delta_CC",
        "fp32_RMSE",
        "fixed_RMSE",
        "delta_RMSE",
    ]
    mean = pd.DataFrame(
        [{"completed_sessions": len(frame), **{key: frame[key].mean() for key in metrics}}]
    )
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / "all_results.csv", index=False)
    mean.to_csv(root / "mean_results.csv", index=False)
    aggregate = {
        "experiment_name": args.experiment_name,
        "completed": len(frame),
        "expected": len(SESSIONS),
        "missing": missing,
        "results": frame.to_dict(orient="records"),
        "mean": mean.iloc[0].to_dict(),
    }
    (root / "all_results.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        "# Fixed-point SNN 평가 결과",
        "",
        f"- Experiment: `{args.experiment_name}`",
        f"- Completed: {len(frame)}/{len(SESSIONS)}",
        f"- Missing: {', '.join(missing) if missing else 'none'}",
        "- 적용 정밀도: input 1-bit event, weight 8-bit, membrane 32-bit, decay 13-bit",
        "- 해석 범위: 정수 연산 평가이며 cycle-accurate FPGA 재현은 아님",
        "",
        "## Mean",
        "",
        markdown_table(mean),
        "",
        "## Sessions",
        "",
        markdown_table(frame),
        "",
    ]
    (root / "MIXED_FIXED_POINT_RESULTS.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Aggregated {len(frame)}/{len(SESSIONS)} mixed fixed-point results under {root}")


if __name__ == "__main__":
    main()
