"""Summarize the leakage-controlled 20-task, three-seed transfer experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs"
REPORT_DIR = OUTPUT_ROOT / "transfer_sutl_norm20_multiseed"
SEEDS = (42, 43, 44)
PAIRS = (
    ("indy_20160622_01", "indy_20160630_01", "8-day"),
    ("indy_20170124_01", "indy_20170127_03", "3-day"),
    ("indy_20170127_03", "indy_20170131_02", "4-day"),
)
METHODS = {
    "source_only_target_test": "Source-only",
    "target_scratch_test": "Scratch",
    "test_continuous": "Transfer",
}
METRICS = ("R2", "CC", "RMSE")


def pair_key(source: str, target: str) -> str:
    return f"{source}_to_{target}"


def load_complete_results() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    metric_rows = []
    convergence_rows = []
    details: dict = {}
    missing = []
    for seed in SEEDS:
        experiment = f"transfer_sutl_norm20_seed{seed}"
        for source, target, label in PAIRS:
            key = pair_key(source, target)
            run_dir = OUTPUT_ROOT / experiment / key / "top64"
            path = run_dir / "run_summary.json"
            if not path.exists():
                missing.append(str(path))
                continue
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("status") != "complete":
                missing.append(f"{path} (status={summary.get('status')})")
                continue
            details[(seed, key)] = summary
            for json_key, method in METHODS.items():
                values = summary.get(json_key)
                if values is None:
                    missing.append(f"{path} ({json_key} missing)")
                    continue
                metric_rows.append(
                    {
                        "seed": seed,
                        "pair": label,
                        "pair_key": key,
                        "source": source,
                        "target": target,
                        "method": method,
                        **{metric: float(values[metric]) for metric in METRICS},
                    }
                )
            for stage, best_key in (
                ("source_pretrain", "source_best_epoch"),
                ("target_finetune", "target_best_epoch"),
                ("target_scratch", "target_scratch_best_epoch"),
            ):
                history_path = run_dir / stage / "training_history.csv"
                if not history_path.exists():
                    missing.append(str(history_path))
                    continue
                history = pd.read_csv(history_path)
                completed_epoch = int(history["epoch"].max())
                convergence_rows.append(
                    {
                        "seed": seed,
                        "pair": label,
                        "stage": stage,
                        "best_epoch": int(summary[best_key]),
                        "completed_epoch": completed_epoch,
                        "early_stopped": completed_epoch < 100,
                    }
                )
    if missing:
        joined = "\n".join(missing)
        raise RuntimeError(f"The full 3-seed experiment is not complete:\n{joined}")
    return pd.DataFrame(metric_rows), pd.DataFrame(convergence_rows), details


def summarize_metrics(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        raw.groupby(["pair", "method"], sort=False)[list(METRICS)]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in summary.columns
    ]
    wide = raw.pivot(index=["seed", "pair"], columns="method", values=list(METRICS))
    effects = []
    for (seed, pair), row in wide.iterrows():
        effects.append(
            {
                "seed": seed,
                "pair": pair,
                **{
                    f"delta_{metric}_transfer_minus_scratch": (
                        float(row[(metric, "Transfer")]) - float(row[(metric, "Scratch")])
                    )
                    for metric in METRICS
                },
                "transfer_beats_scratch_R2": bool(row[("R2", "Transfer")] > row[("R2", "Scratch")]),
            }
        )
    effects_frame = pd.DataFrame(effects)
    effect_summary = (
        effects_frame.groupby("pair", sort=False)
        .agg(
            delta_R2_mean=("delta_R2_transfer_minus_scratch", "mean"),
            delta_R2_std=("delta_R2_transfer_minus_scratch", "std"),
            delta_CC_mean=("delta_CC_transfer_minus_scratch", "mean"),
            delta_CC_std=("delta_CC_transfer_minus_scratch", "std"),
            delta_RMSE_mean=("delta_RMSE_transfer_minus_scratch", "mean"),
            delta_RMSE_std=("delta_RMSE_transfer_minus_scratch", "std"),
            wins=("transfer_beats_scratch_R2", "sum"),
        )
        .reset_index()
    )
    return summary, effects_frame, effect_summary


def channel_stability(details: dict) -> pd.DataFrame:
    rows = []
    for source, target, label in PAIRS:
        key = pair_key(source, target)
        masks = {seed: set(details[(seed, key)]["kept_indices_0based"]) for seed in SEEDS}
        for left, right in ((42, 43), (42, 44), (43, 44)):
            union = masks[left] | masks[right]
            rows.append(
                {
                    "pair": label,
                    "seed_a": left,
                    "seed_b": right,
                    "intersection": len(masks[left] & masks[right]),
                    "union": len(union),
                    "jaccard": len(masks[left] & masks[right]) / len(union),
                }
            )
    return pd.DataFrame(rows)


def make_plots(
    raw: pd.DataFrame,
    effect_summary: pd.DataFrame,
    convergence: pd.DataFrame,
) -> None:
    pair_order = [label for _, _, label in PAIRS]
    colors = {"Source-only": "#8b95a5", "Scratch": "#e59f38", "Transfer": "#3977c3"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    x = np.arange(len(pair_order))
    width = 0.24
    for index, method in enumerate(("Source-only", "Scratch", "Transfer")):
        values = []
        errors = []
        for pair in pair_order:
            selected = raw[(raw["pair"] == pair) & (raw["method"] == method)]["R2"]
            values.append(float(selected.mean()))
            errors.append(float(selected.std()))
        axes[0].bar(
            x + (index - 1) * width,
            values,
            width,
            yerr=errors,
            capsize=3,
            color=colors[method],
            label=method,
        )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(x, pair_order)
    axes[0].set_ylabel("Test R2 (mean +/- SD, 3 seeds)")
    axes[0].set_title("Matched comparison")
    axes[0].legend(frameon=False)

    effect = effect_summary.set_index("pair").loc[pair_order]
    gain_colors = ["#2c9a6b" if value > 0 else "#c84c4c" for value in effect["delta_R2_mean"]]
    axes[1].bar(
        x,
        effect["delta_R2_mean"],
        yerr=effect["delta_R2_std"],
        capsize=4,
        color=gain_colors,
    )
    axes[1].axhline(0, color="black", linewidth=0.9)
    axes[1].set_xticks(x, pair_order)
    axes[1].set_ylabel("Transfer R2 - Scratch R2")
    axes[1].set_title("Transfer effect (positive is better)")
    fig.suptitle("20-task calibration, velocity z-score, 64 selected channels")
    fig.savefig(REPORT_DIR / "transfer_effect_summary.png", dpi=180)
    plt.close(fig)

    stage_order = ["source_pretrain", "target_finetune", "target_scratch"]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    positions = np.arange(len(stage_order))
    for pair_index, pair in enumerate(pair_order):
        subset = convergence[convergence["pair"] == pair]
        means = [
            subset[subset["stage"] == stage]["completed_epoch"].mean() for stage in stage_order
        ]
        ax.plot(positions, means, marker="o", linewidth=2, label=pair)
    ax.set_xticks(positions, ["Source pretrain", "Target fine-tune", "Target scratch"])
    ax.set_ylabel("Completed epochs (mean across seeds)")
    ax.set_ylim(bottom=0)
    ax.set_title("Convergence / early stopping")
    ax.legend(frameon=False)
    fig.savefig(REPORT_DIR / "convergence_summary.png", dpi=180)
    plt.close(fig)


def fmt(value: float) -> str:
    return f"{value:.3f}"


def make_readme(
    raw: pd.DataFrame,
    effects: pd.DataFrame,
    effect_summary: pd.DataFrame,
    convergence: pd.DataFrame,
    stability: pd.DataFrame,
    details: dict,
) -> None:
    pair_rows = []
    for _, _, pair in PAIRS:
        transfer = raw[(raw["pair"] == pair) & (raw["method"] == "Transfer")]
        scratch = raw[(raw["pair"] == pair) & (raw["method"] == "Scratch")]
        source = raw[(raw["pair"] == pair) & (raw["method"] == "Source-only")]
        effect = effect_summary[effect_summary["pair"] == pair].iloc[0]
        pair_rows.append(
            "| {pair} | {so} | {sc} | {tr} | {delta} | {wins}/3 |".format(
                pair=pair,
                so=f"{fmt(source.R2.mean())} ± {fmt(source.R2.std())}",
                sc=f"{fmt(scratch.R2.mean())} ± {fmt(scratch.R2.std())}",
                tr=f"{fmt(transfer.R2.mean())} ± {fmt(transfer.R2.std())}",
                delta=f"{fmt(effect.delta_R2_mean)} ± {fmt(effect.delta_R2_std)}",
                wins=int(effect.wins),
            )
        )

    per_seed_pair_mean = raw.groupby(["seed", "method"])["R2"].mean().unstack("method")
    global_delta = per_seed_pair_mean["Transfer"] - per_seed_pair_mean["Scratch"]
    global_wins = int(effects["transfer_beats_scratch_R2"].sum())
    early_stopped = int(convergence["early_stopped"].sum())
    all_masks_stable = bool(np.allclose(stability["jaccard"], 1.0))

    first = details[(42, pair_key(PAIRS[0][0], PAIRS[0][1]))]
    protocol = first["protocol"]
    baseline_rows = []
    for source, target, label in PAIRS:
        summary = details[(42, pair_key(source, target))]
        baseline = summary.get("baseline_96")
        within = summary.get("within_session_selection")
        baseline_rows.append(
            f"| {label} | {fmt(float(baseline['R2'])) if baseline else 'N/A'} | "
            f"{fmt(float(within['R2'])) if within else 'N/A'} |"
        )

    text = f"""# 전이학습 효과 검증: 20-task 정규화 3-seed 실험

## 한 줄 결론

3개 세션 쌍과 3개 seed에서 `Transfer - Scratch`의 평균 R2 차이는 **{fmt(global_delta.mean())} ± {fmt(global_delta.std())}**였습니다. 총 9번의 직접 비교 중 전이학습이 Scratch보다 높은 R2를 낸 횟수는 **{global_wins}/9**입니다. 양수이면 전이학습 이득, 음수이면 negative transfer입니다.

![전이학습 효과 요약](transfer_effect_summary.png)

## 핵심 결과

아래 값은 test R2의 `평균 ± 표준편차`(seed 42, 43, 44)입니다.

| 세션 간격 | Source-only | Scratch | Transfer | Transfer - Scratch | Transfer 승리 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(pair_rows)}

- **Scratch**: 목표 세션 calibration 데이터만으로 처음부터 학습
- **Transfer**: source 세션에서 미리 학습한 뒤 같은 목표 calibration 데이터로 미세조정
- 두 방법의 목표 데이터, 채널, seed, epoch 상한, learning rate, early stopping 규칙은 동일합니다.
- 그러므로 가장 중요한 비교는 `Transfer - Scratch`입니다. Source-only와 기존 96/64채널 결과는 보조 설명용입니다.

## 실험 조건

- 대상: 8일, 3일, 4일 차이의 primary 세션 쌍 3개
- 반복: seed 42, 43, 44
- 채널: 논문의 안정성 + 중요도 - 중복도 방식으로 96개 중 64개 선택
- 목표 calibration: 20 task 중 16개 학습, 4개 검증(약 60 reaches)
- 속도 정규화: source는 source train만, target은 target calibration 20개만 사용한 축별 z-score
- 평가지표: 원래 속도 단위로 복원한 R2, CC, RMSE
- 최대 epoch: source/transfer/scratch 모두 100, patience 10
- 테스트: 시간상 뒤쪽의 완전 분리된 연속 구간, 상태는 시작 시 한 번만 초기화
- 208일 stress pair는 primary 결론에서 제외했습니다.

## 기존 실험과의 참고 비교

이 값들은 동일한 직접 대조군이 아니므로 효과 판정에는 사용하지 않습니다.

| 세션 간격 | 기존 96채널 R2 | 기존 within-session 64채널 R2 |
|---|---:|---:|
{chr(10).join(baseline_rows)}

## 유효성 점검

- 데이터 누수: **없도록 설계됨**. 채널 선택과 target 정규화에는 calibration 20개까지만 사용했고 target validation/test는 제외했습니다.
- 공정한 대조: Transfer와 Scratch는 동일 seed, 동일 calibration, 동일 test span 및 동일 학습 조건을 사용합니다.
- seed 분리: 출력 폴더를 seed별로 나눠 checkpoint가 섞이지 않습니다.
- 채널 선택 안정성: seed 간 mask Jaccard가 모두 1인지 확인한 결과 **{"통과" if all_masks_stable else "불일치 발견"}**입니다. 선택 알고리즘 자체에는 난수가 없습니다.
- 수렴: 27개 stage 중 {early_stopped}개가 100 epoch 전에 종료되었습니다. 자세한 값은 `convergence.csv`와 아래 그림에 있습니다.

![수렴 요약](convergence_summary.png)

## 해석할 때의 한계

- seed가 3개뿐이므로 표준편차는 볼 수 있지만 강한 통계적 유의성 결론에는 부족합니다.
- 세션 쌍도 3개뿐이므로 다른 동물·날짜에 일반화된다고 단정할 수 없습니다.
- calibration을 10개에서 20개로 늘린 효과와 속도 정규화 효과가 함께 바뀌었습니다. 둘을 따로 분리하려면 이후 ablation이 필요합니다.
- 기존 96채널 및 within-session 64채널은 학습 데이터 양과 프로토콜이 달라 직접적인 전이학습 효과 대조군이 아닙니다.

## 생성 파일

- `raw_metrics.csv`: 모든 seed·pair·방법의 R2/CC/RMSE
- `metric_summary.csv`: pair·방법별 평균과 표준편차
- `transfer_effects_per_seed.csv`: seed별 Transfer - Scratch
- `transfer_effect_summary.csv`: pair별 전이 효과 평균·표준편차와 승리 횟수
- `convergence.csv`: best/completed epoch와 early stopping 여부
- `channel_stability.csv`: seed 간 선택 채널 Jaccard

프로토콜 기준 seed는 {protocol["training"]["seed"]}이며, 각 seed 폴더의 protocol에는 해당 seed가 별도로 기록되어 있습니다.
"""
    (REPORT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    raw, convergence, details = load_complete_results()
    metric_summary, effects, effect_summary = summarize_metrics(raw)
    stability = channel_stability(details)
    raw.to_csv(REPORT_DIR / "raw_metrics.csv", index=False)
    metric_summary.to_csv(REPORT_DIR / "metric_summary.csv", index=False)
    effects.to_csv(REPORT_DIR / "transfer_effects_per_seed.csv", index=False)
    effect_summary.to_csv(REPORT_DIR / "transfer_effect_summary.csv", index=False)
    convergence.to_csv(REPORT_DIR / "convergence.csv", index=False)
    stability.to_csv(REPORT_DIR / "channel_stability.csv", index=False)
    make_plots(raw, effect_summary, convergence)
    make_readme(raw, effects, effect_summary, convergence, stability, details)
    print(f"Wrote multi-seed report to {REPORT_DIR}")


if __name__ == "__main__":
    main()
