# Transfer-aware SUTL channel-selection experiment

이 실험은 Zhang et al.의 SUTL 구조인 `stationarity + importance - redundancy`를
현재 저장소의 연속 `vx/vy` SNN 회귀 문제에 맞게 수정한다. 기존 96채널 baseline과
within-session Pearson channel-selection 결과는 변경하지 않는다.

## 연구 질문

이전 세션과 현재 세션 초반의 소량 calibration 데이터만으로 64개 M1 채널을 고르고,
이전 세션에서 사전학습한 SNN을 현재 세션에서 미세조정했을 때 다음을 확인한다.

1. target 전체 train을 사용한 기존 Pearson top-64와 비교해 어느 정도 성능을 유지하는가?
2. target의 약 30 reach만 사용한 scratch 모델보다 source pretraining이 유리한가?
3. source-only 모델에서 target fine-tuning으로 얼마나 개선되는가?
4. 3일, 4일, 8일 전이와 208일 stress 전이의 차이는 무엇인가?

## 데이터 사용 범위와 누수 방지

각 세션은 기존 공개 코드와 같은 방식으로 3개 연속 reach, 3,876 ms task로 재구성하고
시간순으로 `floor(80%) / floor(10%) / remainder`를 train/validation/test로 분할한다.

- 채널 선택: source train + target train의 첫 10 task만 사용한다.
- 첫 10 target task는 약 30 reach에 해당한다.
- fine-tuning: calibration task 중 앞 8개를 train, 뒤 2개를 early-stopping validation으로 쓴다.
- target의 원래 validation/test는 채널 선택과 학습에 사용하지 않는다.
- 최종 test는 모델이나 threshold 선택에 다시 사용하지 않는다.

## 채널 점수

채널 선택 계산에서만 1 ms binary MUA를 50 ms spike count로 집계한다. 선택 후 SNN에는
선택된 채널의 원래 1 ms binary MUA가 입력된다.

### Stationarity

source와 target calibration의 채널별 50 ms count histogram을 비교한다.

```text
stationarity(c) = 1 - Jensen-Shannon-distance(P_source(c), P_target(c))
```

서로 대응되지 않는 두 세션 trial 벡터에 SU를 직접 적용하는 원 논문의 모호점을 피하고,
두 확률분포의 유사도를 직접 측정한다.

### Continuous-velocity importance

source `vx/vy`의 quantile로 8개 구간을 만든 후 source와 target calibration에 같은 경계를
적용한다. 각 채널 count와 `vx`, `vy` 구간 사이의 symmetrical uncertainty를 계산하고
두 축의 RMS로 합친다.

```text
importance_session(c) = sqrt(mean([SU(count_c, vx_bin)^2,
                                   SU(count_c, vy_bin)^2]))
importance(c) = 0.5 * importance_source(c)
              + 0.5 * importance_target(c)
```

source 표본 수가 많다는 이유로 target 정보가 사라지지 않도록 표본 단위가 아니라 세션
단위로 동일 가중치를 사용한다.

### Redundancy

source와 target calibration에서 채널 쌍의 SU 행렬을 각각 계산하고 동일 가중치로 평균한다.
그 후 고정된 `K`개가 채워질 때까지 greedy selection을 수행한다.

```text
base(c) = 0.5 * percentile(stationarity(c))
        + 0.5 * percentile(importance(c))

greedy(c | selected) = base(c)
                      - 0.25 * mean(SU(c, selected_channels))
```

고정 `K=64`를 사용하는 이유는 기존 5세션 결과에서 64채널의 평균 R2 손실이 약 0.01인
반면 32채널은 약 0.059였기 때문이다. 채널 수 sweep은 64채널 실험이 검증된 후
`80, 64, 48, 32` 순서로 수행한다.

## 학습 단계와 대조군

선택된 64채널로 다음 세 모델을 평가한다.

1. `source_only`: source train/validation으로 학습한 모델을 target test에 바로 적용한다.
2. `target_scratch`: target calibration 8/2 task만으로 새 모델을 학습한다.
3. `transfer_finetune`: source best model을 target calibration으로 fine-tuning한다.

`target_scratch`에는 무작위 초기화 모델에 적합한 독립 학습 설정을 둔다. 초기
파일럿에서는 미세조정에 `1e-4`, scratch에 `1e-3`을 사용했으나, 후속 matched
schedule 실험에서 미세조정도 `1e-3`이 더 나았다. 따라서 현재 기본 설정은 두 모델
모두 최대 100 epoch, 학습률 `1e-3`, patience 10이다. 두 모델 모두 calibration
train/validation과 최종 test 구간은 동일하고, validation 최적 epoch만 비교한다.

추가로 기존 결과에서 다음 기준을 읽어 비교한다.

- target 전체 train으로 학습한 96채널 baseline
- target 전체 train에서 Pearson으로 고른 within-session top-64

따라서 `target_scratch vs transfer_finetune`은 source pretraining 효과를, 기존 top-64와의
비교는 적은 calibration 데이터만 허용했을 때의 성능 비용을 보여준다.

## 세션 쌍

설정 파일 `configs/transfer_sutl.yaml`에는 다음 전이가 들어 있다.

- `indy_20160622_01 -> indy_20160630_01`: primary, 8일
- `indy_20170124_01 -> indy_20170127_03`: primary, 3일
- `indy_20170127_03 -> indy_20170131_02`: primary, 4일
- `indy_20160630_01 -> indy_20170124_01`: 208일 stress test

208일 전이는 primary 평균과 분리해 보고한다.

## 실행 방법

설정과 생성 명령만 확인한다.

```powershell
uv run python scripts/run_experiment.py `
  --config configs/transfer_sutl.yaml `
  --dry-run
```

채널 선택만 네 쌍에 실행한다. SNN 학습은 하지 않는다.

```powershell
uv run python scripts/run_experiment.py `
  --config configs/transfer_sutl.yaml `
  --selection-only
```

한 target에 대해 짧은 smoke run을 실행한다. Smoke 수치는 성능 결과가 아니라 코드 경로
검증용이다.

```powershell
uv run python scripts/run_experiment.py `
  --config configs/transfer_sutl.yaml `
  --session indy_20170127_03 `
  --channels 64 `
  --smoke
```

한 target의 full run을 실행한다. `--session`은 transfer 설정에서 target 필터이다.

```powershell
uv run python scripts/run_experiment.py `
  --config configs/transfer_sutl.yaml `
  --session indy_20170127_03 `
  --channels 64
```

전체 primary/stress run을 순차 실행한다.

```powershell
uv run python scripts/run_experiment.py --config configs/transfer_sutl.yaml
```

현재 mask와 완료된 성능을 요약한다.

```powershell
uv run python scripts/summarize_transfer_sutl.py `
  --experiment-name transfer_sutl_64ch
```

## 결과 파일

각 run은 다음 경로에 저장된다.

```text
outputs/transfer_sutl_64ch/
  <source>_to_<target>/
    top64/
      protocol.json
      channel_mask.json
      channel_ranking.csv
      channel_ranking.png
      channel_redundancy.npy
      source_pretrain/
      target_finetune/
      target_scratch/
      source_only_target_test_metrics_continuous.csv
      target_scratch_test_metrics_continuous.csv
      test_metrics_continuous.csv
      run_summary.json
```

`run_summary.json`의 주 비교 필드는 다음과 같다.

- `source_only_target_test`
- `target_scratch_test`
- `test_continuous`: transfer fine-tune 결과
- `baseline_96`
- `within_session_selection`
- `delta_vs_target_baseline_96`
- `delta_vs_within_session_selection`

## 현재 검증 상태

- 알고리즘 self-test 통과
- YAML validation 및 dry-run 통과
- 네 source-target 쌍의 top-64 mask 생성 완료
- source pretrain, target fine-tune, target scratch, 연속 평가 smoke 경로 통과
- 2017-01-24 → 2017-01-27에 대해 10/10 epoch 파일럿 실행 완료
- 파일럿의 transfer test `R2=-0.034`, scratch `R2=0.001`, 기존 96채널
  `R2=0.731`이었다. 기존 기준선의 최적 epoch가 32~55였으므로 이 결과는
  알고리즘의 최종 성능이 아니라 과소학습 탐지 결과로 취급한다.
- 공정한 scratch 재학습 결과: best epoch 94, `R2=0.383`, `CC=0.636`,
  `RMSE=64.39`
- full source pretrain(최적 epoch 28) + `1e-4/30 epoch` 미세조정:
  `R2=0.020`, `CC=0.209`, `RMSE=81.07`
- 같은 source 모델 + `1e-3/최대 100 epoch` 미세조정은 epoch 18에서
  조기종료했고 `R2=0.101`, `CC=0.302`, `RMSE=77.85`였다.
- 결론: 높은 미세조정 학습률은 개선됐지만 scratch보다 `R2`가 0.282 낮았다.
  이 한 세션쌍에서는 현재 SUTL 회귀 적응이 음의 전이를 보인다. 다른 세션쌍,
  calibration 크기, 회귀 타깃 정규화, 채널 점수 ablation을 실행하기 전에는
  전체 알고리즘이 실패했다고 일반화하지 않는다.
