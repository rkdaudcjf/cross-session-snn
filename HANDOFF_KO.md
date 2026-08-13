# 학부연구 인계 요약

작성일: 2026-08-13  
저장소: `cross-session-snn`  
Git 원격: `https://github.com/rkdaudcjf/cross-session-snn.git`

## 현재 상태

- 사용자의 요청으로 정확도 개선 실험을 중단했습니다.
- `train_transfer_adaptation_search.py` 관련 Python 프로세스는 0개입니다.
- 30분 감시 자동화 `sutl-3-seed-30`도 `PAUSED` 상태입니다.
- 중단 시점의 체크포인트와 optimizer 상태는 `outputs/transfer_adaptation_search_seed*/`에 보존되어 있습니다.
- 동일한 명령에 `--resume`을 사용하면 완료된 후보는 건너뛰고 중단된 epoch 다음부터 이어집니다.

## 완료된 핵심 연구

논문의 채널 선택 개념을 다음과 같이 구현했습니다.

1. source 학습 구간과 target calibration 구간에서 채널별 spike-count 분포를 계산합니다.
2. 세션 간 분포가 비슷한 채널에 높은 안정성 점수를 부여합니다.
3. 연속 속도 `vx`, `vy`와 관련성이 큰 채널에 높은 중요도 점수를 부여합니다.
4. 이미 선택된 채널과 정보가 지나치게 겹치는 채널에는 중복 페널티를 적용합니다.
5. `안정성 + 중요도 - 중복도` 순으로 96개 중 64개 채널을 선택합니다.

데이터 누수를 막기 위해 채널 선택과 속도 정규화에는 source train 및 target calibration만 사용했습니다. target validation과 test는 제외했습니다.

## 완료된 20-task·3-seed 전이학습 결과

조건은 target calibration 20 tasks(학습 16, early stopping 4), 속도 z-score, seed 42/43/44, 64채널입니다. Transfer와 Scratch는 동일 seed, calibration, test span, learning rate 및 early stopping 조건을 사용했습니다.

| 세션 차이 | Source-only R2 | Scratch R2 | Transfer R2 | Transfer-Scratch |
|---|---:|---:|---:|---:|
| 8일 | -0.044 ± 0.028 | 0.161 ± 0.092 | 0.140 ± 0.022 | -0.021 ± 0.093 |
| 3일 | -0.030 ± 0.020 | 0.035 ± 0.106 | 0.354 ± 0.062 | +0.319 ± 0.169 |
| 4일 | 0.585 ± 0.029 | -0.096 ± 0.014 | 0.451 ± 0.085 | +0.547 ± 0.077 |

- 총 9회 중 7회에서 Transfer가 Scratch보다 높은 R2를 기록했습니다.
- 3일 및 4일 쌍에서는 세 seed 모두 Transfer가 우세했습니다.
- 8일 쌍에서는 평균적으로 약한 negative transfer가 나타났습니다.
- 채널 mask의 seed 간 Jaccard는 모든 쌍에서 1.0이었습니다.
- 27개 학습 단계 중 25개가 100 epoch 전에 early stopping 됐습니다.

전체 표와 그림은 `outputs/transfer_sutl_norm20_multiseed/README.md`에 있습니다.

## 중단한 정확도 개선 실험

목표는 이전 실험에서 발견한 두 문제를 개선하는 것입니다.

- 4일 쌍: source-only가 fine-tuning보다 좋아 전체 미세조정이 좋은 표현을 손상했을 가능성
- 8일 쌍: 기존 fine-tuning이 Scratch보다 뚜렷하게 좋지 않았음

후보 전략은 다음 7개입니다.

- source-only
- 기존 full fine-tuning, learning rate `1e-3`
- Scratch
- full fine-tuning, learning rate `3e-4`
- full fine-tuning, learning rate `1e-4`
- output head만 fine-tuning, learning rate `1e-3`
- 마지막 hidden block과 output head fine-tuning, learning rate `3e-4`

후보 선택에는 target validation만 사용하며, test는 validation에서 선택된 후보 하나에만 최종 적용하도록 구현했습니다.

중단 시점은 첫 번째 8일 쌍의 `full_lr1e4` 후보입니다.

| seed | 완료 epoch / 100 | 체크포인트 위치 |
|---:|---:|---|
| 42 | 49 | `outputs/transfer_adaptation_search_seed42/.../candidates/full_lr1e4/last_checkpoint.pt` |
| 43 | 53 | `outputs/transfer_adaptation_search_seed43/.../candidates/full_lr1e4/last_checkpoint.pt` |
| 44 | 25 | `outputs/transfer_adaptation_search_seed44/.../candidates/full_lr1e4/last_checkpoint.pt` |

이 실험은 이전 test 결과를 본 뒤 후보군을 설계했으므로 탐색적 실험입니다. 실행 내부의 test 누수는 없지만, 완전히 독립적인 새 세션에서 재검증하기 전에는 확증적 결과로 표현하면 안 됩니다.

## 이동 후 가장 먼저 할 일

1. `docs/MOVE_AND_GIT_GUIDE_KO.md`를 읽습니다.
2. PowerShell에서 `scripts/verify_portable_bundle.ps1`을 실행합니다.
3. `.venv`는 복사하지 않았으므로 `uv sync --dev`로 다시 만듭니다.
4. 정확도 개선 실험을 이어가려면 `scripts/resume_transfer_adaptation_search.ps1`을 실행합니다.

세부 변경 이력과 연구 해석은 `docs/WORK_LOG_20260813_KO.md`에 있습니다.
