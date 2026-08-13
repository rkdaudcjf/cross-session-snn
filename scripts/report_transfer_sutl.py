"""Build the final transfer-SUTL tables, figures, README, and validity audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

PAIR_META = [
    ("indy_20160622_01", "indy_20160630_01", "8-day primary", "P1\n8d", "primary"),
    ("indy_20170124_01", "indy_20170127_03", "3-day primary", "P2\n3d", "primary"),
    ("indy_20170127_03", "indy_20170131_02", "4-day primary", "P3\n4d", "primary"),
    ("indy_20160630_01", "indy_20170124_01", "208-day stress", "S\n208d", "stress"),
]

METHODS = [
    ("source_only_target_test", "Source-only"),
    ("test_continuous", "Transfer"),
    ("target_scratch_test", "Scratch"),
    ("within_session_selection", "Target full top-64"),
    ("baseline_96", "Target full 96-ch"),
]

STAGES = [
    ("source_pretrain", "Source pretrain"),
    ("target_finetune", "Transfer fine-tune"),
    ("target_scratch", "Target scratch"),
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pair_dir(root: Path, source: str, target: str) -> Path:
    return root / f"{source}_to_{target}" / "top64"


def collect_results(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict] = []
    convergence_rows: list[dict] = []
    pair_rows: list[dict] = []
    for source, target, role, short, group in PAIR_META:
        run_dir = pair_dir(root, source, target)
        summary = load_json(run_dir / "run_summary.json")
        protocol = load_json(run_dir / "protocol.json")
        ranking = pd.read_csv(run_dir / "channel_ranking.csv")
        pair_label = f"{source.removeprefix('indy_')} → {target.removeprefix('indy_')}"
        for key, method in METHODS:
            values = summary[key]
            metric_rows.append(
                {
                    "source_session": source,
                    "target_session": target,
                    "role": role,
                    "group": group,
                    "short_label": short.replace("\n", " "),
                    "pair": pair_label,
                    "method": method,
                    "R2": float(values["R2"]),
                    "CC": float(values["CC"]),
                    "RMSE": float(values["RMSE"]),
                }
            )
        for stage, stage_label in STAGES:
            history = pd.read_csv(run_dir / stage / "training_history.csv")
            stage_summary = load_json(run_dir / stage / "stage_summary.json")
            best_epoch = int(stage_summary["best_epoch"])
            for row in history.itertuples(index=False):
                convergence_rows.append(
                    {
                        "source_session": source,
                        "target_session": target,
                        "role": role,
                        "short_label": short.replace("\n", " "),
                        "stage": stage_label,
                        "epoch": int(row.epoch),
                        "train_mse": float(row.train_mse),
                        "validation_mse": float(row.validation_mse),
                        "learning_rate": float(row.learning_rate),
                        "best_epoch": best_epoch,
                    }
                )
        selected = ranking[ranking["kept"]]
        excluded = ranking[~ranking["kept"]]
        transfer = summary["test_continuous"]
        scratch = summary["target_scratch_test"]
        source_only = summary["source_only_target_test"]
        pair_rows.append(
            {
                "source_session": source,
                "target_session": target,
                "role": role,
                "group": group,
                "short_label": short.replace("\n", " "),
                "pair": pair_label,
                "source_only_R2": float(source_only["R2"]),
                "transfer_R2": float(transfer["R2"]),
                "scratch_R2": float(scratch["R2"]),
                "transfer_minus_scratch_R2": float(transfer["R2"] - scratch["R2"]),
                "finetune_minus_source_R2": float(transfer["R2"] - source_only["R2"]),
                "selected_stationarity": float(selected["stationarity_similarity"].mean()),
                "excluded_stationarity": float(excluded["stationarity_similarity"].mean()),
                "selected_importance": float(selected["importance_combined"].mean()),
                "excluded_importance": float(excluded["importance_combined"].mean()),
                "dice_vs_target_top64": float(
                    load_json(run_dir / "channel_mask.json").get(
                        "dice_vs_target_within_session_mask", np.nan
                    )
                ),
                "source_best_epoch": int(summary["source_best_epoch"]),
                "finetune_best_epoch": int(summary["target_best_epoch"]),
                "scratch_best_epoch": int(summary["target_scratch_best_epoch"]),
                "seed": int(protocol["training"]["seed"]),
                "calibration_tasks": int(protocol["target_calibration"]["reconstructed_tasks"]),
                "calibration_train_tasks": int(
                    protocol["target_calibration"]["fine_tune_train_tasks"]
                ),
                "calibration_validation_tasks": int(
                    protocol["target_calibration"]["fine_tune_validation_tasks"]
                ),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    convergence = pd.DataFrame(convergence_rows)
    pairs = pd.DataFrame(pair_rows)
    selection = pd.read_csv(root / "selection_summary.csv")
    pairs = pairs.drop(columns="dice_vs_target_top64").merge(
        selection[
            [
                "source_session",
                "target_session",
                "dice_vs_target_within_session_mask",
            ]
        ],
        on=["source_session", "target_session"],
        how="left",
    )
    return metrics, convergence, pairs


def validity_audit(root: Path, pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(check: str, passed: bool, evidence: str) -> None:
        rows.append(
            {
                "check": check,
                "status": "PASS" if passed else "FAIL",
                "evidence": evidence,
            }
        )

    run_dirs = [pair_dir(root, source, target) for source, target, *_ in PAIR_META]
    summaries = [load_json(path / "run_summary.json") for path in run_dirs]
    protocols = [load_json(path / "protocol.json") for path in run_dirs]
    add(
        "All configured pairs completed",
        all(item["status"] == "complete" for item in summaries),
        f"{sum(item['status'] == 'complete' for item in summaries)}/4 complete",
    )
    failures = [path / "failure.json" for path in run_dirs if (path / "failure.json").exists()]
    add("No run failure artifacts", not failures, f"failure.json count={len(failures)}")
    required = [
        "run_summary.json",
        "protocol.json",
        "channel_mask.json",
        "channel_ranking.csv",
        "best_model.pt",
        "test_metrics_continuous.csv",
        "target_scratch_test_metrics_continuous.csv",
        "source_only_target_test_metrics_continuous.csv",
        "test_predictions.npz",
    ]
    missing = [
        str(path / name) for path in run_dirs for name in required if not (path / name).exists()
    ]
    add("Required output files present", not missing, f"missing count={len(missing)}")
    add(
        "Chronological session split",
        all("chronological" in p["shared_preprocessing"]["split"] for p in protocols),
        "floor 80/10/remainder independently per session",
    )
    add(
        "Selection excludes target validation/test",
        all("excluded" in p["selection"]["leakage_control"] for p in protocols),
        "source train + first 10 target calibration tasks only",
    )
    add(
        "Target test is held out and continuous",
        all("held-out" in p["training"]["target_test_policy"] for p in protocols),
        "continuous held-out span; state reset once",
    )
    add(
        "Matched transfer/scratch data budget",
        all(
            p["target_calibration"]["reconstructed_tasks"] == 10
            and p["target_calibration"]["fine_tune_train_tasks"] == 8
            and p["target_calibration"]["fine_tune_validation_tasks"] == 2
            for p in protocols
        ),
        "10 tasks total = 8 train + 2 validation for every pair",
    )
    add(
        "Matched transfer/scratch optimization budget",
        all(
            p["training"]["target_finetune_epochs"] == p["training"]["target_scratch_epochs"]
            and p["training"]["target_learning_rate"]
            == p["training"]["target_scratch_learning_rate"]
            and p["training"]["target_early_stopping_patience"]
            == p["training"]["target_scratch_early_stopping_patience"]
            for p in protocols
        ),
        "100 epochs max, LR=1e-3, patience=10",
    )
    seeds = sorted({int(p["training"]["seed"]) for p in protocols})
    add("Same recorded seed", seeds == [42], f"seeds={seeds}")
    early_stop_ok = True
    details: list[str] = []
    for run_dir in run_dirs:
        for stage, _ in STAGES:
            history = pd.read_csv(run_dir / stage / "training_history.csv")
            stage_summary = load_json(run_dir / stage / "stage_summary.json")
            best = int(stage_summary["best_epoch"])
            last = int(history["epoch"].iloc[-1])
            early_stop_ok &= last - best == 10 or last == 100
            details.append(f"{run_dir.parent.name}/{stage}:{best}->{last}")
    add("Early-stopping histories consistent", early_stop_ok, "; ".join(details))
    add(
        "Selected channels score above excluded channels",
        bool(
            (
                (pairs["selected_stationarity"] > pairs["excluded_stationarity"])
                & (pairs["selected_importance"] > pairs["excluded_importance"])
            ).all()
        ),
        "stationarity and importance means higher in all 4 masks",
    )
    return pd.DataFrame(rows)


def performance_figure(path: Path, metrics: pd.DataFrame, pairs: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    labels = [item[3] for item in PAIR_META]
    x = np.arange(len(labels))
    colors = ["#7f8c8d", "#2f6b9a", "#e67e22", "#5b8c5a", "#8e5d9f"]
    width = 0.15
    for metric, axis, title in [
        ("R2", axes[0, 0], "Held-out target R² (higher is better)"),
        ("CC", axes[1, 0], "Held-out target correlation (higher is better)"),
        ("RMSE", axes[1, 1], "Held-out target RMSE (lower is better)"),
    ]:
        for index, (_, method) in enumerate(METHODS):
            values = [
                float(
                    metrics[
                        (metrics["source_session"] == source)
                        & (metrics["target_session"] == target)
                        & (metrics["method"] == method)
                    ][metric].iloc[0]
                )
                for source, target, *_ in PAIR_META
            ]
            axis.bar(x + (index - 2) * width, values, width, label=method, color=colors[index])
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.set_xticks(x, labels)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    gain_axis = axes[0, 1]
    gain_axis.bar(
        x - 0.18,
        pairs["transfer_minus_scratch_R2"],
        0.36,
        label="Transfer − scratch",
        color="#2f6b9a",
    )
    gain_axis.bar(
        x + 0.18,
        pairs["finetune_minus_source_R2"],
        0.36,
        label="Fine-tune − source-only",
        color="#58a6a6",
    )
    gain_axis.axhline(0, color="#333333", linewidth=1)
    gain_axis.set_xticks(x, labels)
    gain_axis.set_title("R² gains isolate two different effects")
    gain_axis.grid(axis="y", alpha=0.25)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=5,
        frameon=False,
    )
    gain_axis.legend(loc="lower left", frameon=False)
    fig.suptitle("Transfer-aware top-64 decoding: final test results", fontsize=17, y=0.985)
    fig.subplots_adjust(top=0.875, bottom=0.08, hspace=0.32, wspace=0.18)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def convergence_figure(path: Path, convergence: pd.DataFrame) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(16, 15))
    for row_index, (source, target, role, short, _) in enumerate(PAIR_META):
        for column_index, (_, stage_label) in enumerate(STAGES):
            axis = axes[row_index, column_index]
            data = convergence[
                (convergence["source_session"] == source)
                & (convergence["target_session"] == target)
                & (convergence["stage"] == stage_label)
            ]
            axis.plot(data["epoch"], data["train_mse"], label="train", color="#9aa0a6")
            axis.plot(
                data["epoch"],
                data["validation_mse"],
                label="validation",
                color="#2f6b9a",
            )
            best_epoch = int(data["best_epoch"].iloc[0])
            best_value = float(data.loc[data["epoch"] == best_epoch, "validation_mse"].iloc[0])
            axis.scatter([best_epoch], [best_value], color="#c0392b", zorder=3, s=30)
            axis.axvline(best_epoch, color="#c0392b", linestyle="--", alpha=0.6)
            axis.set_title(
                f"{short.replace(chr(10), ' ')} · {stage_label}\nbest={best_epoch}, val={best_value:.0f}"
            )
            axis.grid(alpha=0.2)
            if row_index == 3:
                axis.set_xlabel("Epoch")
            if column_index == 0:
                axis.set_ylabel("MSE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Training convergence and validation-selected checkpoints",
        fontsize=17,
        y=0.992,
    )
    fig.subplots_adjust(top=0.91, bottom=0.05, hspace=0.48, wspace=0.12)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def channel_figure(path: Path, root: Path, pairs: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    labels = [item[3] for item in PAIR_META]
    x = np.arange(4)
    width = 0.36
    axes[0].bar(
        x - width / 2,
        pairs["selected_stationarity"],
        width,
        label="selected",
        color="#2f6b9a",
    )
    axes[0].bar(
        x + width / 2,
        pairs["excluded_stationarity"],
        width,
        label="excluded",
        color="#bdc3c7",
    )
    axes[0].set_xticks(x, labels)
    axes[0].set_title("Stationarity score")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(
        x - width / 2,
        pairs["selected_importance"],
        width,
        label="selected",
        color="#e67e22",
    )
    axes[1].bar(
        x + width / 2,
        pairs["excluded_importance"],
        width,
        label="excluded",
        color="#bdc3c7",
    )
    axes[1].set_xticks(x, labels)
    axes[1].set_title("Velocity importance score")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    mask_sets: list[set[int]] = []
    for source, target, *_ in PAIR_META:
        mask = load_json(pair_dir(root, source, target) / "channel_mask.json")
        mask_sets.append(set(mask["kept_indices_0based"]))
    overlap = np.eye(4)
    for i in range(4):
        for j in range(4):
            overlap[i, j] = 2 * len(mask_sets[i] & mask_sets[j]) / 128
    image = axes[2].imshow(overlap, vmin=0.55, vmax=1.0, cmap="Blues")
    axes[2].set_xticks(x, labels)
    axes[2].set_yticks(x, labels)
    axes[2].set_title("Pairwise top-64 mask Dice")
    for i in range(4):
        for j in range(4):
            axes[2].text(j, i, f"{overlap[i, j]:.3f}", ha="center", va="center")
    fig.colorbar(image, ax=axes[2], fraction=0.046)
    fig.suptitle("Channel-selection diagnostics", fontsize=17, y=0.98)
    fig.subplots_adjust(top=0.83, bottom=0.14, left=0.055, right=0.94, wspace=0.18)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fmt(value: float) -> str:
    return f"{value:.3f}"


def build_readme(
    root: Path,
    metrics: pd.DataFrame,
    convergence: pd.DataFrame,
    pairs: pd.DataFrame,
    audit: pd.DataFrame,
) -> str:
    primary = pairs[pairs["group"] == "primary"]
    stress = pairs[pairs["group"] == "stress"].iloc[0]
    lines = [
        "# Transfer-aware SUTL 64채널 전체 실험 결과",
        "",
        "## 핵심 결론",
        "",
        (
            "4개 source-target 쌍의 실행이 모두 완료됐다. Transfer fine-tuning은 source-only보다 "
            "3/4 pair에서 개선됐지만, 같은 10-task calibration으로 처음부터 학습한 scratch보다 "
            "좋은 경우는 4일 간격 pair 1개뿐이었다. 따라서 현 설정에서 일관된 전이 이득은 확인되지 않았다."
        ),
        "",
        f"- Primary 3쌍의 평균 `transfer − scratch R²`: **{fmt(primary['transfer_minus_scratch_R2'].mean())}**",
        f"- Primary 3쌍의 평균 `fine-tune − source-only R²`: **{fmt(primary['finetune_minus_source_R2'].mean())}**",
        f"- 208일 stress의 `transfer − scratch R²`: **{fmt(stress['transfer_minus_scratch_R2'])}**",
        f"- Transfer가 scratch를 이긴 pair: **{int((pairs['transfer_minus_scratch_R2'] > 0).sum())}/4**",
        "",
        "![최종 성능 비교](transfer_performance_comparison.png)",
        "",
        "## Pair별 테스트 결과",
        "",
        "| Pair | 역할 | Source-only R² | Transfer R² | Scratch R² | Transfer−Scratch | Fine-tune−Source |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in pairs.itertuples(index=False):
        lines.append(
            f"| {row.pair} | {row.role} | {fmt(row.source_only_R2)} | "
            f"{fmt(row.transfer_R2)} | {fmt(row.scratch_R2)} | "
            f"{fmt(row.transfer_minus_scratch_R2)} | {fmt(row.finetune_minus_source_R2)} |"
        )
    lines.extend(
        [
            "",
            (
                "전체 R²·CC·RMSE 값은 [`full_metrics.csv`](full_metrics.csv)에 저장돼 있다. "
                "`Target full top-64`와 `Target full 96-ch`는 target 전체 train을 사용하므로 "
                "10-task transfer/scratch와 데이터 예산이 같은 대조군이 아니라 성능 상한 참고값이다."
            ),
            "",
            "## 결과 해석",
            "",
            "- **8일 primary:** transfer는 source-only보다 개선됐지만 scratch보다 낮았다.",
            "- **3일 primary:** 같은 양상이 반복됐다. source 초기화의 이득보다 세션 불일치 비용이 컸다.",
            (
                "- **4일 primary:** transfer가 scratch보다 높았지만 source-only보다는 낮았다. "
                "안정적인 source 표현은 유효했으나 소량 fine-tuning이 이미 좋은 모델을 훼손했다."
            ),
            (
                "- **208일 stress:** fine-tuning은 source-only를 개선했지만 scratch보다 낮았다. "
                "장기 간격에서도 현재 초기화가 일관된 이득을 주지는 않았다."
            ),
            "",
            (
                "4일 pair는 stationarity 평균이 가장 높고 유일하게 transfer가 scratch를 이겼다. "
                "흥미로운 가설이지만 표본이 4쌍뿐이고 세션/행동 분포가 얽혀 있으므로 상관관계로 주장할 수 없다."
            ),
            "",
            "## 채널 선택 진단",
            "",
            "![채널 선택 진단](channel_selection_diagnostics.png)",
            "",
            (
                "모든 pair에서 선택 채널의 평균 stationarity와 velocity importance가 제외 채널보다 높았다. "
                "이는 ranking 구현의 내부 sanity check이지 독립적인 성능 검증은 아니다. top-64 집합 둘의 "
                "무작위 기대 Dice는 약 `64/96=0.667`이므로 관측 Dice만으로 생리학적 안정성을 주장하지 않는다."
            ),
            "",
            "## 수렴 및 체크포인트 선택",
            "",
            "![학습 수렴](training_convergence.png)",
            "",
            (
                "모든 12개 stage는 validation 최적 epoch 이후 patience 10에서 종료됐다. 마지막 모델이 아니라 "
                "validation 최적 체크포인트를 테스트에 사용했다. 상세 이력은 "
                "[`training_convergence.csv`](training_convergence.csv)에 있다."
            ),
            "",
            "## 유효성 점검",
            "",
            "| 점검 항목 | 상태 | 근거 |",
            "|---|---|---|",
        ]
    )
    for row in audit.itertuples(index=False):
        evidence = str(row.evidence).replace("|", "/")
        lines.append(f"| {row.check} | {row.status} | {evidence} |")
    lines.extend(
        [
            "",
            "### 확인된 강점",
            "",
            "- 모든 세션을 시간순 train/validation/test로 분리했다.",
            "- 채널 선택은 source train과 target 초반 10 calibration task만 사용했다.",
            "- 원래 target validation/test는 채널 선택 및 학습에 사용하지 않았다.",
            (
                "- Transfer와 scratch는 같은 top-64 mask, 같은 8/2 calibration split, 같은 seed, "
                "같은 학습률·최대 epoch·patience와 동일한 연속 test span을 사용했다."
            ),
            "- 208일 pair는 primary 평균에서 제외하고 stress 결과로 분리했다.",
            "",
            "### 제한점과 주장 범위",
            "",
            "1. **Seed 1개:** seed 42 한 번뿐이므로 초기화 및 batch 순서 불확실성을 추정할 수 없다.",
            "2. **세션쌍 수가 적음:** primary는 3쌍뿐이어서 통계적 유의성 검정을 주장하기 어렵다.",
            "3. **Calibration validation이 2 task:** early stopping 선택의 분산이 클 수 있다.",
            (
                "4. **회귀형 변형:** 원 논문의 분류 label mutual information을 속도 quantile SU로 바꿨으므로 "
                "논문의 동일 알고리즘 재현이 아니라 회귀 적응형 구현이다."
            ),
            (
                "5. **Supervised channel selection:** target calibration의 `vx/vy`를 importance 계산에 사용한다. "
                "완전한 비지도 전이는 아니다."
            ),
            "6. **Target shift 미보정:** 세션별 속도 평균·분산 및 출력 affine shift를 정규화하지 않았다.",
            "7. **Ablation 부재:** stationarity-only, importance-only, random-64, redundancy 제거 비교가 아직 없다.",
            (
                "8. **Full-target 기준선은 비동일 예산:** 96채널/within-session 64채널 모델은 훨씬 많은 target "
                "train 데이터를 사용하므로 transfer 실패의 공정한 직접 대조군은 scratch다."
            ),
            "",
            (
                "따라서 현재 가능한 결론은 **구현은 정상 작동하지만, 한 seed의 4개 pair에서 transfer 초기화가 "
                "scratch 대비 일관된 이득을 주지 않았다**는 것이다. 알고리즘의 일반적 실패나 생리학적 채널 "
                "불안정성을 단정할 수는 없다."
            ),
            "",
            "## 다음 실험 우선순위",
            "",
            "1. 같은 조건에서 최소 5개 seed 반복 후 pair 내 `transfer − scratch` 분포 보고",
            "2. calibration `5/10/20/40 task` learning curve와 validation 최소 4 task 확보",
            "3. source/target velocity 정규화 및 target calibration affine output 보정",
            "4. stationarity/importance/redundancy ablation과 random-64 반복 대조군",
            "5. 4일 pair에서 no-finetune, partial-layer fine-tuning, 더 작은 learning rate 비교",
            "",
            "## 생성 파일",
            "",
            "- [`full_metrics.csv`](full_metrics.csv): 모든 방법의 pair별 R²·CC·RMSE",
            "- [`pair_effects.csv`](pair_effects.csv): transfer 효과와 채널 진단 요약",
            "- [`training_convergence.csv`](training_convergence.csv): 12개 stage 전체 학습 이력",
            "- [`validity_checks.csv`](validity_checks.csv): 자동 유효성 점검 결과",
            "- [`selection_summary.csv`](selection_summary.csv): 채널 선택 요약",
            "- [`pairwise_mask_overlap.csv`](pairwise_mask_overlap.csv): mask 중복도",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default="transfer_sutl_64ch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = OUTPUT_ROOT / args.experiment_name
    metrics, convergence, pairs = collect_results(root)
    audit = validity_audit(root, pairs)
    metrics.to_csv(root / "full_metrics.csv", index=False)
    convergence.to_csv(root / "training_convergence.csv", index=False)
    pairs.to_csv(root / "pair_effects.csv", index=False)
    audit.to_csv(root / "validity_checks.csv", index=False)
    performance_figure(root / "transfer_performance_comparison.png", metrics, pairs)
    convergence_figure(root / "training_convergence.png", convergence)
    channel_figure(root / "channel_selection_diagnostics.png", root, pairs)
    (root / "README.md").write_text(
        build_readme(root, metrics, convergence, pairs, audit),
        encoding="utf-8",
    )
    print(pairs.to_string(index=False))
    print("\nValidity audit")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
