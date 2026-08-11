# Reproduction guide

## 1. 환경 설치

```powershell
uv python install 3.11
uv sync --frozen
```

## 2. 데이터 배치

[`data/README.md`](../data/README.md)의 구조대로 5개 Indy `.mat` 파일을 배치합니다. 데이터 파일은 Git에 추가하지 않습니다.

## 3. Config 검증

YAML 문법, 필수 필드, session 이름, epoch 범위를 검증하고 실제 실행 명령을 출력합니다.

```powershell
uv run python scripts/run_experiment.py --config configs/baseline.yaml --dry-run
uv run python scripts/run_experiment.py --config configs/channel_selection.yaml --dry-run
uv run python scripts/run_experiment.py --config configs/mixed_fixed_point_baseline.yaml --dry-run
```

채널 선택 학습기의 내부 self-test:

```powershell
uv run python scripts/train_channel_selection.py `
  --session indy_20170127_03 `
  --keep-channels 32 `
  --self-test
```

## 4. Baseline 실행

```powershell
uv run python scripts/run_experiment.py --config configs/baseline.yaml
```

결과 위치:

```text
outputs/baseline_96ch/<session>/
```

## 5. 채널 선택 실행

96채널 baseline이 완료된 후 실행합니다.

```powershell
uv run python scripts/run_experiment.py --config configs/channel_selection.yaml
```

결과 위치:

```text
outputs/channel_selection_64_32ch/<session>/top<channels>/
```

## 6. Fixed-point SNN 평가

완료된 96채널 baseline에 논문 보고 수치 폭을 적용하여 integer SNN update를
실행합니다.

```powershell
uv run python scripts/evaluate_mixed_fixed_point_baseline.py --self-test
uv run python scripts/run_experiment.py --config configs/mixed_fixed_point_baseline.yaml
```

결과 위치:

```text
outputs/baseline_96ch_mixed_fixed_point/<session>/
```

입력 1-bit event, weight 8-bit, membrane 32-bit, decay 13-bit가 적용됩니다.
논문에 공개되지 않은 scale 선택은 결과의 `implementation_assumptions`에
기록됩니다. 자세한 내용은
[`MIXED_FIXED_POINT_BASELINE.md`](MIXED_FIXED_POINT_BASELINE.md)를 참고합니다.

## 7. 완료 확인

- `run_summary.json`의 `status`가 `complete`
- `protocol.json`에 session, seed, channel count가 기록됨
- Validation/test metric 파일이 존재함
- Channel mask 길이와 모델 입력 차원이 일치함
- 96채널 비교 결과가 동일한 데이터 split에서 생성됨
- Failure 또는 in-progress 파일을 최종 결과로 사용하지 않음
- Fixed-point 결과의 `smoke`가 `false`이고 potential saturation 횟수를 확인함
- Fixed-point 결과의 `implementation_assumptions`를 함께 공개함

## 8. 결과 공유 단위

```text
YAML config
Git commit SHA
dataset/session
channel count and mask
protocol.json
run_summary.json
representative figure
```
