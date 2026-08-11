# Experiment configuration

실험값은 YAML에 기록하고 `scripts/run_experiment.py`가 이를 읽어 검증한 뒤 학습기를 실행합니다.

`experiment.name`은 결과 폴더 이름입니다. 기본값은 96채널 기준 실험의 `baseline_96ch`와 64/32채널 비교 실험의 `channel_selection_64_32ch`입니다. 세션 ID는 Zenodo 데이터 파일 이름과 같으며, 세션마다 별도의 실험 이름을 만들지 않습니다.

`mixed_fixed_point_baseline.yaml`은 완료된 FP32 baseline을 입력으로 사용해
integer LIF update를 실행합니다. `experiment.source_name`은 평가할 baseline
결과 폴더를 가리키므로 기본 `baseline_96ch` 이름을 바꾸었다면 이 값도 함께
맞춰야 합니다.

`quantization`의 `input_bits`, `weight_bits`, `potential_bits`, `decay_bits`는 논문
보고값과 다르게 바뀌지 않도록 launcher가 검증합니다. 논문에 없는 scale 선택은
`weight_scale_mode`에 명시하며 기본 `pow2`는 layer별 power-of-two scale을
사용합니다. `evaluate_fp32: true`는 같은 checkpoint와 test 구간을 현재 코드의
FP32 경로로 다시 실행해 fixed-point 결과와 직접 비교합니다.

```powershell
uv run python scripts/run_experiment.py --config configs/baseline.yaml --dry-run
uv run python scripts/run_experiment.py --config configs/baseline.yaml

uv run python scripts/run_experiment.py --config configs/channel_selection.yaml --dry-run
uv run python scripts/run_experiment.py --config configs/channel_selection.yaml

uv run python scripts/run_experiment.py --config configs/mixed_fixed_point_baseline.yaml --dry-run
uv run python scripts/run_experiment.py --config configs/mixed_fixed_point_baseline.yaml
```

특정 실험만 선택할 수도 있습니다.

```powershell
uv run python scripts/run_experiment.py `
  --config configs/channel_selection.yaml `
  --session indy_20170127_03 `
  --channels 32
```

## 파일 형식의 역할

| 형식 | 이 저장소에서의 역할 | 사람이 수정? |
|---|---|---|
| `pyproject.toml` | Python 버전 범위, 패키지 의존성, Ruff와 uv 설정 | 환경을 바꿀 때만 |
| `configs/*.yaml` | session, epoch, channel 수 등 실험 preset | 실험 전에 수정 |
| `outputs/**/*.json` | 실행 코드가 저장한 protocol, progress, 결과 metadata | 직접 수정하지 않음 |
| `.ps1` | PowerShell 명령을 자동화하는 Windows 스크립트 | 사용하지 않음 |

TOML과 YAML은 모두 설정 형식이지만 적용 대상이 다릅니다. `pyproject.toml`은 uv와 Python 도구가 자동으로 읽고, YAML은 이 프로젝트의 `run_experiment.py`가 명시적으로 읽습니다. JSON은 결과 재현 기록이므로 config 입력으로 사용하지 않습니다. PowerShell 스크립트는 설정 파일이 아니라 명령 실행 프로그램이며, OS 종속성과 중복을 줄이기 위해 공개본에서 제거했습니다.
