# Experiment map

## Baseline

- 입력: 96-channel GT-MUA
- 단위: 각 Indy recording을 독립적으로 학습
- 분할: chronological task-window split
- 목적: channel-selected model의 기준 성능 확보

## Channel-count comparison

- 입력: correlation ranking의 top 64 또는 top 32
- ranking source: chronological training entries의 앞 절반
- 모델: 선택 채널 수와 일치하는 구조적 K-input SNN을 새로 학습
- 비교: 같은 세션의 96-channel baseline

## 96-channel fixed-point SNN

- 입력 모델: 각 세션의 validation-best `baseline_96ch` checkpoint
- 입력 spike: binary `0/1` event, 논리 1-bit
- weight: layer별 signed INT8
- decay: unsigned 13-bit `UQ0.13`
- membrane/threshold/reset: signed INT32
- 연산: integer synaptic accumulation과 integer LIF state update
- 출력 변환: 최종 2차원 output membrane만 metric 계산 전에 실수 단위로 복원
- 재학습/fine-tuning: 없음
- 해석 범위: 논문 정밀도의 정수 연산 평가이며 bit-exact 또는 cycle-accurate FPGA 검증은 아님
- 현재 공개 상태: 구현 및 실행 protocol 완료, 새 5세션 수치 결과 미포함

세부 가정과 실행법은
[`MIXED_FIXED_POINT_BASELINE.md`](MIXED_FIXED_POINT_BASELINE.md)에 있습니다.

## 아직 수행해야 할 cross-session 평가

1. 과거 여러 세션만 사용해 stable channel score를 계산한다.
2. channel mask를 고정한다.
3. 미래 세션에서는 mask를 다시 선택하지 않는다.
4. decoder retraining/calibration 허용 여부를 protocol별로 구분한다.
5. zero-calibration 결과와 calibration 허용 결과를 별도로 보고한다.
6. 최종 고정 mask를 FPGA 입력 및 first-layer gating에 연결한다.
