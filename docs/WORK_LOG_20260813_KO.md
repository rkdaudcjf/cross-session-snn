# 작업 및 실험 기록

## 1. 논문 기반 분석

`Feature-Selection-Based Transfer Learning for Intracortical Brain-Machine Interface Decoding`의 핵심 아이디어를 학부연구의 5개 SABES 세션에 적용할 수 있는 형태로 분석했습니다. 논문의 목표는 모든 채널을 무조건 쓰는 대신, 날짜가 달라져도 비교적 안정적이며 운동 정보가 많고 서로 중복되지 않는 채널을 선택해 세션 간 전이 성능을 높이는 것입니다.

## 2. 채널 선택 구현

주요 구현 파일:

- `scripts/internal/transfer_selection_core.py`
- `scripts/train_transfer_channel_selection.py`
- `scripts/run_experiment.py`

구현 점수는 다음 세 요소로 구성됩니다.

- 안정성: source와 target calibration의 50 ms spike-count histogram 사이 Jensen-Shannon 거리 사용
- 중요도: source가 정의한 속도 quantile bin과 채널 spike count 사이 symmetrical uncertainty 사용
- 중복도: 선택된 채널들과의 pairwise symmetrical uncertainty 평균 사용

채널 순위는 greedy 방식으로 계산하며 최종적으로 64개를 고정 선택합니다.

## 3. 첫 전이학습 실험

설정: `configs/transfer_sutl.yaml`

- calibration 10 tasks
- seed 42
- source-only, Scratch, Transfer 비교
- primary 8일/3일/4일 쌍과 208일 stress 쌍 분리

첫 결과에서는 4일 쌍만 Transfer가 Scratch보다 좋았고 다른 쌍은 negative transfer가 관찰됐습니다. 이 결과를 통해 세션별 속도 크기 차이와 너무 적은 calibration 데이터가 전이를 방해할 수 있다고 판단했습니다.

## 4. 누수 없는 속도 정규화

`scripts/train_transfer_channel_selection.py`에 다음 기능을 추가했습니다.

- source 속도 평균·표준편차: source train에서만 계산
- target 속도 평균·표준편차: target calibration에서만 계산
- validation/test는 정규화 통계 계산에서 제외
- 모델은 z-score 속도를 학습
- R2, CC, RMSE는 원래 속도 단위로 복원해 저장

기존 프로토콜과 섞이지 않도록 seed별 별도 출력 폴더를 사용했습니다.

## 5. 20-task·3-seed 실험

설정:

- `configs/transfer_sutl_norm20_seed42.yaml`
- `configs/transfer_sutl_norm20_seed43.yaml`
- `configs/transfer_sutl_norm20_seed44.yaml`

프로토콜:

- primary 세션 쌍 3개
- target calibration 20 tasks: 학습 16, early stopping 4
- seed 42, 43, 44
- source pretrain, target full fine-tuning, target Scratch 각각 최대 100 epoch
- learning rate `1e-3`, patience 10
- 동일한 target test span을 사용

결과는 `outputs/transfer_sutl_norm20_multiseed/`에 CSV, README와 PNG로 정리했습니다. 3일과 4일 쌍은 모든 seed에서 전이 효과가 있었고, 8일 쌍은 평균적으로 거의 효과가 없었습니다.

## 6. 정확도 개선 실험 구현

구현 파일:

- `scripts/train_transfer_adaptation_search.py`
- `scripts/summarize_transfer_adaptation_search.py`

데이터 분할 역할을 명확히 분리했습니다.

- calibration 16 tasks: 후보 모델 학습
- calibration 4 tasks: 후보별 early stopping
- original target validation split: 후보 전략 선택
- target test split: 선택된 후보 하나의 최종 성능만 측정

후보 전략은 전체 미세조정의 learning rate 감소, output head-only, last-block fine-tuning, source-only 유지 등을 포함합니다. 기존 완료 모델을 재사용해 source pretraining을 반복하지 않습니다.

## 7. 중단 및 보존

2026-08-13 사용자의 요청으로 정확도 개선 실험을 중단했습니다. 관련 Python 프로세스를 강제 종료하기 전에 각 후보 학습은 매 epoch마다 다음을 저장하고 있었습니다.

- `last_checkpoint.pt`: 모델, optimizer, scheduler, best state, history
- `best_model_in_progress.pt`
- `training_history_in_progress.csv`
- `progress.json`

따라서 이동 후 동일 스크립트와 `--resume`으로 이어갈 수 있습니다. 현재 epoch는 `HANDOFF_KO.md`에 기록했습니다.

## 8. 검증한 항목

- Python `py_compile`
- Ruff 정적 검사
- 채널 선택 self-test
- 실제 데이터 smoke test
- 결과 CSV/JSON 완전성
- 생성 PNG 육안 검사
- test가 학습, 정규화, 채널 선택, early stopping 및 후보 선택에 사용되지 않는지 프로토콜 점검

## 9. 후속 연구 권장 순서

1. 중단된 validation-gated adaptation search 완료
2. 세션별 선택 전략과 test 개선량 분석
3. 가장 좋은 전략을 독립 세션 또는 사전에 고정한 confirmatory split에서 재검증
4. calibration 수 5/10/20에 따른 learning curve 작성
5. 채널 수 32/48/64/80 ablation
6. 가능한 경우 다른 동물 데이터로 일반화 검증

현재 결과만으로는 3개 seed와 3개 세션 쌍에 대한 탐색적 근거이며, 모집단 수준 통계나 임상적 일반화를 주장하면 안 됩니다.
