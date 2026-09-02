# Multi-source cross-session test 계획 및 진행 보고 (2026-09-03)

## 사실 확인

`20260823` 작업 폴더에는 5개 Indy 세션의 within-session 96/64/32채널 결과와
cross-session transfer 결과가 모두 남아 있다. 기존 cross-session 결과는 3개 시간 간격
(8일, 3일, 4일) × seed 3개를 같은 target calibration 20 task 조건으로 비교한 것이다.
이 코드와 요약 결과는 원격 `main`의 `a451abc` 커밋에도 이미 반영되어 있었다.

3-seed 평균 test R²는 다음과 같다.

| pair | Source-only | Scratch | Transfer | Transfer − Scratch |
|---|---:|---:|---:|---:|
| 8-day | -0.044 | 0.161 | 0.140 | -0.021 |
| 3-day | -0.030 | 0.035 | 0.354 | +0.319 |
| 4-day | 0.585 | -0.096 | 0.451 | +0.547 |

따라서 “전이가 항상 우수하다”가 아니라 3·4-day에서는 이득, 8-day에서는 불명확한
세션쌍 의존성이 관찰됐다는 해석이 맞다.

## 이전 test와 이번 test의 차이

이전 실험은 source 한 세션에서 사전학습하고 target 20 task로 적응했다. 이번 실험은
다음 변수를 분리한다.

1. source 수: single recent / recent-3 / diverse-3 / all-past
2. source 데이터 다양성과 총 학습량: source 수가 늘어도 총 source task는 120으로 고정
3. target 데이터 예산: calibration 20 / 40 / 80 task
4. 적응 범위: source-only / head-only / last-block / full fine-tuning
5. 시간 누수: target보다 과거 source만 bank에 포함
6. 모델 선택 누수: target validation으로 후보를 고르고 held-out test는 선택 뒤 한 번 평가

즉 이번 실험의 핵심은 단순히 source 데이터를 늘리는 것이 아니라, 학습량을 고정한 채
source 다양성 자체가 새 세션 적응을 개선하는지 확인하는 것이다.

## 구축한 환경

- 실험 디렉터리: `experiments/multisource_cross_session_20260903/`
- 고정 계획: `plan.yaml`
- 실행기: `run_multisource_pilot.py`
- 환경: Python 3.11.9, `uv sync --frozen`
- 원시 데이터: Git 비추적 junction으로 기존 5개 `.mat` 파일 재사용
- 산출물: `outputs/multisource_cross_session_20260903/`에 checkpoint·전체 로그 저장
- Git 결과: `results/multisource_cross_session_20260903/`에는 공유 가능한 요약만 저장

## 현재 진행 상태

`recent3 → indy_20170131_02`, seed 42 smoke test가 완료됐다. 세 source를 실제로 읽고,
공통 top-64 ranking, multi-source pretraining, 네 adaptation 후보의 validation 선택,
선택 후보와 scratch의 held-out 평가까지 전 경로가 정상 동작했다. validation이 선택한
후보는 `head_only`였다.

추가로 `all_past` bank의 네 source가 모두 target보다 과거이고, 총 120 source task가
30 task씩 균등 배정되는 것도 validator로 확인했다.

Smoke 실행은 source 9 task, target calibration 4 task, 2 epoch, 512-step 평가 제한이다.
따라서 R² 값은 과학적 비교에 사용할 수 없다. 이 결과가 보여 주는 것은 다음 네 가지다.

- 서로 다른 세 source의 물리 채널 이름이 호환된다.
- source별 z-score와 session-balanced pretraining이 실행된다.
- source 보존형 adaptation 후보를 validation으로 선택할 수 있다.
- 선택 후 held-out test를 여는 평가 정책이 코드로 강제된다.

## 다음 실행 순서

1. pilot target에서 `single_recent/recent3/diverse3/all_past × 20/40/80 × seed 42/43/44` 실행
2. 각 조건에서 Transfer − Scratch와 multi-source − single-source의 paired 차이 집계
3. pilot에서 프로토콜을 동결한 후에만 추가 target 세션으로 확장
4. 최종 성공 기준: single-source 대비 median ΔR² ≥ +0.05, positive target ≥ 70%

현재 로컬 데이터는 5세션뿐이므로 마지막 세션을 target으로 한 pilot까지만 가능하다.
PPT의 독립 target 4–8개 확장은 추가 Indy 세션을 확보한 뒤 진행해야 한다.
