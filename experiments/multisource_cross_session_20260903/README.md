# Multi-source cross-session test (2026-09-03)

이 디렉터리는 `20260824.pptx` 8–10페이지의 다음 실험을 실행 가능한 형태로 옮긴 것이다.
기존 결과를 덮어쓰지 않으며, 원시 `.mat` 데이터와 모델 checkpoint는 Git에 넣지 않는다.

## 이전 실험과 가장 큰 차이

| 항목 | 이전 cross-session test | 이번 test |
|---|---|---|
| source | source 1개 | source bank 1/3/all-past |
| source 학습량 | 세션별 가용 train 사용 | bank 크기와 무관하게 총 120 task 고정 |
| target calibration | 20 task 고정(16 train + 4 early-stop) | 20/40/80 task learning curve |
| adaptation | full fine-tuning 중심, 사후 후보 탐색 | source-only/head-only/last-block/full을 validation으로 선택 |
| 시간 통제 | 3개 pair 비교 | target보다 과거인 source만 허용하는 frozen bank |
| 평가 단위 | 3 pair × 3 seed | pilot 후 여러 target으로 확장, median ΔR²와 positive-target 비율 보고 |

## 실행

```powershell
uv sync --frozen
uv run python experiments/multisource_cross_session_20260903/run_multisource_pilot.py `
  --source-bank recent3 --validate-only

uv run python experiments/multisource_cross_session_20260903/run_multisource_pilot.py `
  --source-bank recent3 --calibration-tasks 20 --seed 42 --smoke

# 전체 36개 pilot matrix와 실시간 상태 파일 생성
uv run python experiments/multisource_cross_session_20260903/run_pilot_suite.py

# 중단된 suite를 2-worker 동적 대기열로 재개
uv run python experiments/multisource_cross_session_20260903/run_pilot_suite_parallel.py `
  --workers 2 --cpu-threads 3

# 기존 worker 1·2를 유지하면서 worker 3·4 추가
uv run python experiments/multisource_cross_session_20260903/run_pilot_suite_parallel.py `
  --pool-id extra --worker-offset 2 --workers 2 --cpu-threads 2
```

전체 실행 중 현재 테스트와 완료/실패 수는
`outputs/multisource_cross_session_20260903/suite_status.json`, 전체 epoch 로그는
`outputs/multisource_cross_session_20260903/suite.log`에서 확인한다.
병렬 실행에서는 `suite_parallel.log`, `suite_worker_1.log`, `suite_worker_2.log`도 생성된다.
추가 pool은 `suite_status_extra.json`, `suite_worker_3.log`, `suite_worker_4.log`를 사용한다.
중단하려면 output root에 `STOP_PARALLEL_SUITE` 파일을 만들면 현재 테스트가 끝난 뒤 멈춘다.

본 실험은 `single_recent`, `recent3`, `diverse3`, `all_past`를 같은 target·seed·calibration·
held-out test로 비교한다. `--smoke`는 실행 경로 검증용이며 과학적 결과로 해석하지 않는다.

## 전체 실행 순서

1. 네 source bank에 `--validate-only`를 실행해 시간 순서·파일·예산을 감사한다.
2. `recent3`, calibration 20, seed 42 smoke test로 multi-source 경로를 확인한다.
3. pilot target `indy_20170131_02`에서 source bank × calibration 20/40/80 × seed 42/43/44를 실행한다.
4. 추가 Indy 세션을 확보한 뒤 cutoff 이후 target 4–8개로 확장한다.
5. 성공 기준은 single-source 대비 median ΔR² ≥ +0.05이고, target의 70% 이상에서 ΔR² > 0이다.

## 현재 데이터 제약

로컬에는 5개 Indy 세션만 있다. 따라서 마지막 세션 `indy_20170131_02`를 target으로 하는
pilot은 가능하지만, 그 이후의 독립 target 4–8개 평가는 아직 불가능하다. 추가 target을
받기 전에는 pilot 결과를 multi-target 일반화 근거로 표현하지 않는다.
