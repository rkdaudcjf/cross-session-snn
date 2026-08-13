"""Summarize transfer-SUTL masks and completed decoding runs."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def dice(left: set[int], right: set[int]) -> float:
    return 2.0 * len(left & right) / (len(left) + len(right))


def metric(summary: dict[str, Any], group: str, name: str) -> float | None:
    payload = summary.get(group)
    if not isinstance(payload, dict) or payload.get(name) is None:
        return None
    return float(payload[name])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default="transfer_sutl_64ch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = OUTPUT_ROOT / args.experiment_name
    summaries = sorted(root.glob("*_to_*/top*/run_summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No transfer summaries found below {root}")

    rows = []
    masks: dict[tuple[str, str, int], set[int]] = {}
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_dir = summary_path.parent
        ranking = pd.read_csv(run_dir / "channel_ranking.csv")
        mask = json.loads((run_dir / "channel_mask.json").read_text(encoding="utf-8"))
        source = str(summary["source_session"])
        target = str(summary["target_session"])
        channels = int(summary["keep_channels"])
        kept = {int(value) for value in mask["kept_indices_0based"]}
        masks[(source, target, channels)] = kept
        selected = ranking.loc[ranking["rank"] <= channels]
        excluded = ranking.loc[ranking["rank"] > channels]

        within_path = (
            OUTPUT_ROOT
            / "channel_selection_64_32ch"
            / target
            / f"top{channels}"
            / "channel_mask.json"
        )
        within_dice = None
        if within_path.exists():
            within_mask = json.loads(within_path.read_text(encoding="utf-8"))
            within_kept = {
                int(value) for value in within_mask["kept_indices_0based"]
            }
            within_dice = dice(kept, within_kept)

        rows.append(
            {
                "source_session": source,
                "target_session": target,
                "channels": channels,
                "status": summary["status"],
                "selected_mean_stationarity": float(selected["stationarity_similarity"].mean()),
                "excluded_mean_stationarity": float(excluded["stationarity_similarity"].mean()),
                "selected_mean_importance": float(selected["importance_combined"].mean()),
                "excluded_mean_importance": float(excluded["importance_combined"].mean()),
                "dice_vs_target_within_session_mask": within_dice,
                "source_only_target_R2": metric(summary, "source_only_target_test", "R2"),
                "target_scratch_R2": metric(summary, "target_scratch_test", "R2"),
                "transfer_finetune_R2": metric(summary, "test_continuous", "R2"),
                "transfer_finetune_CC": metric(summary, "test_continuous", "CC"),
                "transfer_finetune_RMSE": metric(summary, "test_continuous", "RMSE"),
                "delta_R2_vs_96": metric(summary, "delta_vs_target_baseline_96", "R2"),
                "delta_R2_vs_within_selection": metric(
                    summary, "delta_vs_within_session_selection", "R2"
                ),
            }
        )

    summary_frame = pd.DataFrame(rows).sort_values(["channels", "target_session", "source_session"])
    summary_frame.to_csv(root / "selection_summary.csv", index=False)

    overlap_rows = []
    for (left_key, left), (right_key, right) in combinations(masks.items(), 2):
        if left_key[2] != right_key[2]:
            continue
        overlap_rows.append(
            {
                "channels": left_key[2],
                "left_pair": f"{left_key[0]}->{left_key[1]}",
                "right_pair": f"{right_key[0]}->{right_key[1]}",
                "intersection": len(left & right),
                "dice": dice(left, right),
            }
        )
    overlap_frame = pd.DataFrame(overlap_rows)
    overlap_frame.to_csv(root / "pairwise_mask_overlap.csv", index=False)
    print(summary_frame.to_string(index=False))
    if not overlap_frame.empty:
        print("\nPairwise mask overlap")
        print(overlap_frame.to_string(index=False))


if __name__ == "__main__":
    main()
