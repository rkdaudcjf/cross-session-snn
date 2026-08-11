# 논문 정밀도 기반 fixed-point SNN 평가

## 목적

Martis et al. (2024)이 보고한 수치 폭을 96채널 baseline checkpoint에 적용해
시냅스 합, membrane update, threshold와 reset을 정수로 실행하는 평가입니다.

이 문서에서 `mixed fixed-point`는 입력, weight, decay, membrane을 각각의
역할에 맞는 서로 다른 bit width로 표현한다는 뜻입니다. 입력 spike는 원래부터
binary event이므로 일반적인 ANN의 8-bit activation처럼 취급하지 않습니다.

## 적용되는 수치 형식

| 항목 | 구현 | 근거와 범위 |
|---|---|---|
| 입력 MUA spike | 논리 1-bit, 값 `0/1` | 논문의 event-based binary 입력 |
| Linear weight | signed INT8 `[-128, 127]` | 논문 보고값 |
| synaptic sum | signed INT32 누산 | 정수 weight와 spike의 합 |
| membrane potential | signed INT32 | 논문 보고값 |
| decay `beta` | unsigned 13-bit, `UQ0.13` | 13-bit는 논문 보고값, `UQ0.13`은 명시한 구현 가정 |
| threshold/reset | membrane과 같은 INT32 scale | 논문에 Q-format이 없어 명시한 구현 가정 |
| 최종 velocity 출력 | output membrane을 실수 단위로 복원 | 회귀 metric 계산을 위한 경계 변환 |

즉, 입력은 `0/1` spike로 유지되고 **weight 8-bit + membrane 32-bit + decay
13-bit 상태에서 SNN update가 진행**됩니다.

## 정수 update

각 layer는 다음 순서로 실행됩니다.

```text
I[t]       = spike[t] @ weight_int8.T              # INT32 accumulation
decay[t]   = round(mem[t-1] * beta_u13 / 2^13)     # INT64 intermediate
mem[t]     = saturate_int32(decay[t] + I[t] - reset)
spike[t+1] = mem[t] > threshold_int32
```

Hidden layer의 subtract reset timing은 학습 checkpoint가 사용한 snnTorch
`reset_delay=True` 동작을 따릅니다. Output layer는 reset 없이 membrane을
누적합니다. Potential saturation 횟수는 layer별 결과에 기록됩니다.

## 논문에 없어 가정한 항목

논문과 저자 공개 notebook에는 이 프로젝트 checkpoint에 적용할 observer
통계, 모든 fractional-bit 위치, rounding mode, RTL pipeline timing이 없습니다.
재현 가능한 기본값은 다음과 같습니다.

- weight: layer별 power-of-two scale, `scale = 2^-fractional_bits`
- scale 선택: 해당 checkpoint layer weight의 max-abs를 INT8 범위에 맞춤
- decay: `UQ0.13`
- potential: 해당 layer weight와 같은 real-unit scale
- rounding: nearest, half away from zero
- overflow: signed INT32 saturation

이 선택은 [`configs/mixed_fixed_point_baseline.yaml`](../configs/mixed_fixed_point_baseline.yaml),
session별 `fixed_point_run_summary.json`, `layer_quantization.csv`에 저장됩니다.
이 구현은 논문 정밀도를 소프트웨어에서 재현한 정수 연산 평가입니다. RTL의
pipeline과 FPGA 자원 동작까지 재현한 cycle-accurate 또는 bit-exact 검증은
아닙니다.

## 실행

먼저 동일 세션의 완료된 baseline이 필요합니다.

```powershell
uv run python scripts/run_experiment.py --config configs/baseline.yaml
```

### 1. 정수 연산 self-test

```powershell
uv run python scripts/evaluate_mixed_fixed_point_baseline.py --self-test
```

### 2. Config 명령 확인

```powershell
uv run python scripts/run_experiment.py `
  --config configs/mixed_fixed_point_baseline.yaml `
  --dry-run
```

### 3. 한 세션의 짧은 smoke 평가

완료된 baseline checkpoint와 실제 데이터를 사용해 FP32와 fixed-point 경로를
각각 1,000 timestep만 실행합니다. 전체 결과 폴더와 구분하기 위해 별도의
experiment 이름을 사용합니다.

```powershell
uv run python scripts/evaluate_mixed_fixed_point_baseline.py `
  --session indy_20170127_03 `
  --source-experiment-name baseline_96ch `
  --experiment-name baseline_96ch_fixed_point_smoke `
  --weight-scale-mode pow2 `
  --evaluate-fp32 `
  --max-test-steps 1000
```

### 4. 전체 평가

한 세션을 먼저 끝까지 실행한 뒤 문제가 없으면 전체 5세션을 실행합니다.
기본 config는 각 세션의 FP32 경로도 다시 실행하여 동일한 checkpoint와 test
구간에서 fixed-point 결과를 비교합니다.

```powershell
uv run python scripts/run_experiment.py `
  --config configs/mixed_fixed_point_baseline.yaml `
  --session indy_20170127_03

uv run python scripts/run_experiment.py `
  --config configs/mixed_fixed_point_baseline.yaml
```

## 생성 파일

```text
outputs/baseline_96ch_mixed_fixed_point/<session>/
├─ fixed_point_run_summary.json
├─ fixed_point_test_metrics_continuous.csv
├─ fixed_point_test_predictions.npz
├─ layer_quantization.csv
└─ quantized_parameters.npz
```

`quantized_parameters.npz`에는 각 layer의 실제 `np.int8` weight, 13-bit 유효
범위를 담는 `np.uint16` decay, `np.int32` threshold, scale과 fractional-bit가
들어갑니다. 전체 세션 실행이
끝나면 experiment 폴더에 `all_results.csv`, `mean_results.csv`,
`all_results.json`, `MIXED_FIXED_POINT_RESULTS.md`가 생성됩니다.

## 결과 공개 전 확인

- source baseline의 `best_model.pt`와 `run_summary.json`이 5세션 모두 존재하는지
  확인합니다.
- 모든 session의 `fixed_point_run_summary.json`이 `status: complete`이고
  `smoke: false`인지 확인합니다.
- `test_span.evaluated_steps`가 의도한 전체 held-out 구간인지 확인합니다.
- `layer_quantization.csv`에서 weight, decay, threshold, potential saturation을
  확인합니다.
- FP32와 fixed-point가 동일한 test 시작점·길이·state reset 조건을 썼는지
  확인합니다.
- `delta_fixed_point_minus_fp32`가 특정 세션이나 axis에서 비정상적으로 커지지
  않았는지 확인합니다.
- config와 `implementation_assumptions`를 결과와 함께 공개합니다.
- FPGA 합성 전력 또는 bit-exact RTL 결과처럼 표현하지 않습니다.

현재 공개 폴더에는 구현과 실행 절차가 포함되어 있습니다. 5세션 수치 결과는
전체 평가가 끝난 뒤 생성된 summary와 saturation 진단을 검토한 다음 추가합니다.
