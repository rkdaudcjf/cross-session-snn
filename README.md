# Channel Selection for Cross-Session SNN Decoder

96채널 neural recording에서 상관관계 기반으로 64개 또는 32개 채널을 선택하고, 선택 채널 수에 따른 SNN decoder 성능 변화를 평가하는 학부연구 저장소입니다. 장기 목표는 과거 세션에서 정한 고정 channel mask를 미래 세션에 적용하고 FPGA 입력 gating으로 연결하는 것입니다.

> 현재 공개된 결과는 각 세션의 training 구간에서 채널을 선택해 같은 세션의 test 구간에서 평가한 **within-session 결과**입니다. Cross-session zero-calibration과 FPGA 전력 측정 결과는 아직 포함하지 않습니다.

## 현재 결과

5개 Indy 세션 평균이며, 전체 수치는 [`results/summary.csv`](results/summary.csv)에 있습니다.

| 채널 수 | 평균 test R² | 평균 test CC | 96채널 대비 평균 ΔR² |
|---:|---:|---:|---:|
| 96 | 0.6970 | 0.8383 | 0.0000 |
| 64 | 0.6870 | 0.8314 | -0.0100 |
| 32 | 0.6380 | 0.8016 | -0.0589 |

![Indy 20170127 top-32 channel ranking](results/figures/indy_20170127_03_top32_channel_ranking.png)

### Mixed fixed-point SNN 결과

완료된 96채널 baseline 5개를 논문에 보고된 수치 폭으로 변환해 전체 held-out
test 구간에서 정수 SNN 추론을 실행했습니다. 여기서 `mixed fixed-point`는 모든
값을 똑같이 8-bit로 만드는 것이 아니라, 신호의 역할에 따라 서로 다른 정밀도를
사용한다는 뜻입니다.

| 신호 | 적용 정밀도 |
|---|---|
| 입력 MUA spike | binary `0/1` |
| 네 개 Linear layer의 weight | signed 8-bit |
| decay `beta` | unsigned 13-bit |
| membrane, threshold, reset | signed 32-bit |

| 지표 | FP32 평균 | Mixed fixed-point 평균 | Fixed − FP32 |
|---|---:|---:|---:|
| R² | 0.696953 | 0.697146 | +0.000194 |
| CC | 0.838300 | 0.838009 | -0.000290 |
| RMSE | 35.635672 | 35.621248 | -0.014424 |

![Mixed fixed-point minus FP32 performance](results/figures/mixed_fixed_point_performance.png)

총 625,831개 test timestep에서 평가했으며, 5개 세션·4개 layer 모두 weight,
decay, threshold, membrane potential saturation이 0회였습니다. 세션별 최대 절대
변화는 R² 0.001175, CC 0.001475, RMSE 0.066129로 작았습니다. 양자화 후 일부
세션의 지표가 소폭 좋아진 것은 일관된 개선이라기보다 반올림 오차에 따른 변동으로
해석합니다.

최종 2차원 velocity를 실수 단위로 복원하기 전까지 synaptic sum과 LIF state
update는 정수로 계산합니다. 논문에 공개되지 않은 Q-format과 scale 선택은 결과
metadata에 구현 가정으로 함께 저장합니다. 전체 수치와 예측 파형은
[`results/README.md`](results/README.md), 구현 범위와 실행법은
[`docs/MIXED_FIXED_POINT_BASELINE.md`](docs/MIXED_FIXED_POINT_BASELINE.md)를
참고하세요. 이 결과는 정수 연산 emulation이며 FPGA bit-exact, 전력 또는 latency
측정 결과가 아닙니다.

## 저장소 구조

```text
cross-session-snn/
├─ configs/                     # 실제 실행에 사용하는 YAML preset
├─ data/                        # 데이터 배치 안내
├─ docs/                        # 실험 정의, 재현 방법, 참고문헌
├─ results/                     # 요약 결과와 대표 artifact
├─ scripts/
│  ├─ run_experiment.py        # 사용자가 실행하는 진입점
│  ├─ train_baseline.py        # 96채널 baseline 학습기
│  ├─ train_channel_selection.py
│  ├─ evaluate_mixed_fixed_point_baseline.py
│  ├─ summarize_mixed_fixed_point.py
│  ├─ generate_result_figures.py
│  └─ internal/                # 학습·평가기가 공유하는 재현·정수 연산 코드
└─ src/cross_session_snn/      # 공통 데이터·SNN pipeline
```

## 설치

Python 3.11, PyTorch 2.5.1 CPU build, snnTorch 1.0.0을 사용합니다.

```powershell
uv python install 3.11
uv sync --frozen
```

## 데이터

원본 `.mat` 파일은 저장소에 포함하지 않습니다. [`data/README.md`](data/README.md)의 구조대로 다음 세션을 배치합니다.

```text
data/sabes_zenodo/master_mat/
├─ indy_20160622_01.mat
├─ indy_20160630_01.mat
├─ indy_20170124_01.mat
├─ indy_20170127_03.mat
└─ indy_20170131_02.mat
```

## Config 확인

긴 학습 전에 `--dry-run`으로 YAML이 어떤 명령으로 변환되는지 확인합니다.

```powershell
uv run python scripts/run_experiment.py --config configs/baseline.yaml --dry-run
uv run python scripts/run_experiment.py --config configs/channel_selection.yaml --dry-run
uv run python scripts/run_experiment.py --config configs/mixed_fixed_point_baseline.yaml --dry-run
```

특정 세션과 채널 수만 확인할 수도 있습니다.

```powershell
uv run python scripts/run_experiment.py `
  --config configs/channel_selection.yaml `
  --session indy_20170127_03 `
  --channels 32 `
  --dry-run
```

## 학습 실행

먼저 96채널 baseline을 생성합니다.

```powershell
uv run python scripts/run_experiment.py --config configs/baseline.yaml
```

Baseline이 완료된 후 64/32채널 실험을 실행합니다.

```powershell
uv run python scripts/run_experiment.py --config configs/channel_selection.yaml
```

논문 보고 수치 폭을 적용한 fixed-point 정수 평가를 실행합니다.

```powershell
uv run python scripts/run_experiment.py --config configs/mixed_fixed_point_baseline.yaml
```

완료된 run은 건너뛰고, `training.resume: true`이면 기존 checkpoint에서 이어서 실행합니다. 출력은 `outputs/` 아래에 생성되며 Git에는 포함되지 않습니다.

## 설정 파일 역할

- [`pyproject.toml`](pyproject.toml): Python 패키지와 개발 도구 환경
- [`configs/baseline.yaml`](configs/baseline.yaml): baseline session과 학습값
- [`configs/channel_selection.yaml`](configs/channel_selection.yaml): 채널 선택 대상과 학습값
- [`configs/mixed_fixed_point_baseline.yaml`](configs/mixed_fixed_point_baseline.yaml): 논문 정밀도를 적용한 정수 SNN 평가값
- `outputs/**/*.json`: 프로그램이 생성한 protocol과 결과 기록

형식별 자세한 차이는 [`configs/README.md`](configs/README.md)를 참고하세요.

## 결과 해석 시 주의사항

- Channel ranking에는 training 구간만 사용합니다.
- Validation/test 정보를 이용해 mask를 선택하면 데이터 누출입니다.
- 결과 공유 시 config, Git commit, `protocol.json`, `run_summary.json`을 함께 기록합니다.
- 현재 결과를 cross-session 성능이나 FPGA 전력 절감 결과로 표현하지 않습니다.
- Fixed-point 결과에는 논문에 없는 scale 가정이 포함되므로 FPGA bit-exact 결과로 표현하지 않습니다.

재현 절차는 [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md), 참고 논문과 데이터셋은 [`docs/REFERENCES.md`](docs/REFERENCES.md)에 정리했습니다.

## 라이선스

아직 프로젝트 라이선스를 선택하지 않았습니다. 공개 전 코드 소유권, 데이터 이용 조건, 참고 구현의 라이선스를 확인하고 `LICENSE`를 추가해야 합니다.
