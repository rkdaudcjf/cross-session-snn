"""Summarize validation-gated transfer adaptation search across three seeds."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs"
REPORT = OUTPUT_ROOT / "transfer_adaptation_search_multiseed"
SEEDS = (42, 43, 44)
PAIRS = (
    ("indy_20160622_01", "indy_20160630_01", "8-day"),
    ("indy_20170124_01", "indy_20170127_03", "3-day"),
    ("indy_20170127_03", "indy_20170131_02", "4-day"),
)
METRICS = ("R2", "CC", "RMSE")


def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    candidates = []
    missing = []
    for seed in SEEDS:
        for source, target, label in PAIRS:
            path = (
                OUTPUT_ROOT
                / f"transfer_adaptation_search_seed{seed}"
                / f"{source}_to_{target}"
                / "top64"
                / "run_summary.json"
            )
            if not path.exists():
                missing.append(str(path))
                continue
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("status") != "complete":
                missing.append(f"{path}: status={summary.get('status')}")
                continue
            rows.append(
                {
                    "seed": seed,
                    "pair": label,
                    "selected_candidate": summary["selected_candidate"],
                    **{
                        f"selected_{metric}": float(summary["selected_test"][metric])
                        for metric in METRICS
                    },
                    **{
                        f"old_transfer_{metric}": float(summary["old_transfer_test"][metric])
                        for metric in METRICS
                    },
                    **{
                        f"scratch_{metric}": float(summary["old_scratch_test"][metric])
                        for metric in METRICS
                    },
                    **{
                        f"source_only_{metric}": float(summary["old_source_only_test"][metric])
                        for metric in METRICS
                    },
                }
            )
            for candidate in summary["candidate_validation"]:
                candidates.append(
                    {
                        "seed": seed,
                        "pair": label,
                        "candidate": candidate["candidate"],
                        **{metric: float(candidate[metric]) for metric in METRICS},
                        "selected": candidate["candidate"] == summary["selected_candidate"],
                    }
                )
    if missing:
        raise RuntimeError("Incomplete adaptation search:\n" + "\n".join(missing))
    return pd.DataFrame(rows), pd.DataFrame(candidates)


def make_summary(raw: pd.DataFrame) -> pd.DataFrame:
    records = []
    for pair in [label for _, _, label in PAIRS]:
        part = raw[raw["pair"] == pair]
        for method, prefix in (
            ("Source-only", "source_only"),
            ("Scratch", "scratch"),
            ("Previous transfer", "old_transfer"),
            ("Validation-gated", "selected"),
        ):
            record = {"pair": pair, "method": method}
            for metric in METRICS:
                values = part[f"{prefix}_{metric}"]
                record[f"{metric}_mean"] = float(values.mean())
                record[f"{metric}_std"] = float(values.std())
            records.append(record)
    return pd.DataFrame(records)


def make_figures(raw: pd.DataFrame, summary: pd.DataFrame) -> None:
    pair_order = [label for _, _, label in PAIRS]
    methods = ("Scratch", "Previous transfer", "Validation-gated")
    colors = ("#e6a032", "#3d78bf", "#2e9a69")
    x = np.arange(len(pair_order))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    for index, (method, color) in enumerate(zip(methods, colors, strict=True)):
        part = summary[summary["method"] == method].set_index("pair").loc[pair_order]
        axes[0].bar(
            x + (index - 1) * width,
            part["R2_mean"],
            width,
            yerr=part["R2_std"],
            capsize=3,
            label=method,
            color=color,
        )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(x, pair_order)
    axes[0].set_ylabel("Test R2 (mean +/- SD, 3 seeds)")
    axes[0].set_title("Accuracy comparison")
    axes[0].legend(frameon=False)

    deltas = []
    errors = []
    for pair in pair_order:
        part = raw[raw["pair"] == pair]
        values = part["selected_R2"] - part["old_transfer_R2"]
        deltas.append(float(values.mean()))
        errors.append(float(values.std()))
    axes[1].bar(
        x,
        deltas,
        yerr=errors,
        capsize=4,
        color=["#2e9a69" if value >= 0 else "#c84c4c" for value in deltas],
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, pair_order)
    axes[1].set_ylabel("Validation-gated R2 - previous transfer R2")
    axes[1].set_title("Accuracy gain")
    fig.suptitle("Validation-gated transfer adaptation search")
    fig.savefig(REPORT / "adaptation_accuracy_summary.png", dpi=180)
    plt.close(fig)

    counts = raw.groupby(["pair", "selected_candidate"]).size().unstack(fill_value=0)
    counts = counts.reindex(pair_order)
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    bottom = np.zeros(len(counts))
    for candidate in counts.columns:
        values = counts[candidate].to_numpy()
        ax.bar(x, values, bottom=bottom, label=candidate)
        bottom += values
    ax.set_xticks(x, pair_order)
    ax.set_yticks((0, 1, 2, 3))
    ax.set_ylabel("Selected seeds (out of 3)")
    ax.set_title("Strategy selected using target validation only")
    ax.legend(frameon=False, fontsize=8, ncols=2)
    fig.savefig(REPORT / "selected_strategy_counts.png", dpi=180)
    plt.close(fig)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def make_readme(raw: pd.DataFrame, summary: pd.DataFrame) -> None:
    rows = []
    for pair in [label for _, _, label in PAIRS]:
        part = summary.set_index(["pair", "method"])
        old = part.loc[(pair, "Previous transfer")]
        selected = part.loc[(pair, "Validation-gated")]
        scratch = part.loc[(pair, "Scratch")]
        gains = (
            raw[raw["pair"] == pair]["selected_R2"] - raw[raw["pair"] == pair]["old_transfer_R2"]
        )
        rows.append(
            f"| {pair} | {fmt(scratch.R2_mean)} ± {fmt(scratch.R2_std)} | "
            f"{fmt(old.R2_mean)} ± {fmt(old.R2_std)} | "
            f"{fmt(selected.R2_mean)} ± {fmt(selected.R2_std)} | "
            f"{fmt(gains.mean())} ± {fmt(gains.std())} |"
        )
    counts = raw["selected_candidate"].value_counts().to_dict()
    global_gain = (
        raw.groupby("seed")["selected_R2"].mean() - raw.groupby("seed")["old_transfer_R2"].mean()
    )
    wins = int((raw["selected_R2"] > raw["old_transfer_R2"]).sum())
    candidate_text = ", ".join(f"{key}={value}" for key, value in counts.items())
    content = f"""# 검증 기반 전이학습 정확도 개선 실험

## 한 줄 결론

목표 validation에서 적응 전략을 선택한 결과, 9개 seed·pair 비교에서 이전 전이학습보다 높은 test R2를 얻은 횟수는 **{wins}/9**입니다. 세 쌍 평균 R2 개선량은 seed 기준 **{fmt(global_gain.mean())} ± {fmt(global_gain.std())}**입니다.

![정확도 비교](adaptation_accuracy_summary.png)

## 세션별 결과

| 세션 차이 | Scratch R2 | 이전 Transfer R2 | 검증 선택 R2 | 개선량 |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## 선택된 전략

{candidate_text}

![선택 전략](selected_strategy_counts.png)

후보는 source-only, 기존 full fine-tuning(LR 1e-3), scratch, full LR 3e-4/1e-4, output head-only LR 1e-3, last block LR 3e-4입니다.

## 데이터 사용 규칙

- calibration 16 tasks: 목표 모델 학습
- calibration 4 tasks: 각 후보의 early stopping
- 별도 target validation: 후보 선택
- target test: validation으로 선택된 후보 하나에만 최종 적용
- test는 학습, early stopping, 후보 선택에 사용하지 않았습니다.
- 모든 후보는 같은 64채널, 속도 정규화, seed, calibration 및 test span을 사용합니다.

## 중요한 한계

이 실험의 후보군은 이전 test 결과를 관찰한 뒤 설계했습니다. 실행 내부에서는 test 누수가 없지만, 연구 전체 관점에서는 **탐색적 정확도 개선 실험**입니다. 완전히 독립적인 새 세션에서 확인하기 전에는 최종 일반화 성능으로 단정하면 안 됩니다. seed와 세션 쌍도 각각 3개뿐입니다.

## 파일

- `raw_selected_results.csv`: seed·pair별 최종 결과와 선택 전략
- `candidate_validation_results.csv`: 후보별 validation 결과
- `metric_summary.csv`: 방법별 평균·표준편차
"""
    (REPORT / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    raw, candidates = load_results()
    summary = make_summary(raw)
    raw.to_csv(REPORT / "raw_selected_results.csv", index=False)
    candidates.to_csv(REPORT / "candidate_validation_results.csv", index=False)
    summary.to_csv(REPORT / "metric_summary.csv", index=False)
    make_figures(raw, summary)
    make_readme(raw, summary)
    print(f"Wrote report to {REPORT}")


if __name__ == "__main__":
    main()
