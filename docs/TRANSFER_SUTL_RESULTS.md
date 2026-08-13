# Transfer-aware channel selection: 실행 결과와 다음 연구 방향

> 전체 4개 pair의 최종 실행 결과, 통합 이미지, 수렴 점검 및 유효성 감사는
> [`outputs/transfer_sutl_64ch/README.md`](../outputs/transfer_sutl_64ch/README.md)에 있다.
> 아래 내용은 전체 실행 전에 작성한 첫 primary pair 중심의 중간 분석이다.

## 한 줄 결론

논문의 `stationarity + importance - redundancy` 개념을 5세션 연속 속도 회귀에
맞게 구현하고 네 개의 시간순 source-target 쌍에서 64채널을 선택했다. 첫 primary
쌍(`2017-01-24 → 2017-01-27`)의 완전 학습 결과에서는 target calibration만으로
처음부터 학습한 모델이 전이 모델보다 좋았다. 따라서 현재 결과는 전이 성공이 아니라
음의 전이를 확인한 중간 결과다.

## 채널 선택 결과

아래 점수는 source train과 target 초반 10 calibration task만 사용했다. target의
원래 validation/test는 점수 계산에 사용하지 않았다.

| source → target | 역할 | 선택/제외 stationarity | 선택/제외 importance | target Pearson top-64와 Dice |
|---|---:|---:|---:|---:|
| 2016-06-22 → 2016-06-30 | 8일 primary | 0.757 / 0.500 | 0.0290 / 0.0113 | 0.797 |
| 2017-01-24 → 2017-01-27 | 3일 primary | 0.813 / 0.536 | 0.0156 / 0.0106 | 0.734 |
| 2017-01-27 → 2017-01-31 | 4일 primary | 0.909 / 0.730 | 0.0156 / 0.0094 | 0.688 |
| 2016-06-30 → 2017-01-24 | 208일 stress | 0.760 / 0.554 | 0.0275 / 0.0143 | 0.656 |

선택 채널의 평균 stationarity와 importance가 제외 채널보다 높은 것은 구현된
ranking이 의도대로 작동한다는 sanity check다. 그러나 top-64 두 집합의 무작위
기대 Dice가 약 `64/96=0.667`이므로, Dice 값만으로 생리학적 안정성을 주장할 수는
없다.

## 실제 학습 결과: 2017-01-24 → 2017-01-27

모든 transfer/scratch 모델은 같은 SUTL top-64, 같은 target calibration 10 task
(학습 8, validation 2), 같은 연속 target test span을 사용했다.

| 모델 | target 사용량 | R2 | CC | RMSE |
|---|---:|---:|---:|---:|
| source-only | 0 task | -0.052 | -0.104 | 83.86 |
| transfer, `1e-4`, 30 epoch | 10 task | 0.020 | 0.209 | 81.07 |
| transfer, `1e-3`, 최대 100 epoch | 10 task | 0.101 | 0.302 | 77.85 |
| scratch, `1e-3`, 최대 100 epoch | 10 task | **0.383** | **0.636** | **64.39** |
| target 전체 train, Pearson top-64 | 전체 train | 0.708 | 0.847 | 44.06 |
| target 전체 train, 96채널 | 전체 train | 0.731 | 0.859 | 42.31 |

높은 미세조정 학습률은 낮은 학습률보다 `R2`를 0.081 개선했지만, 동일한
calibration만 사용한 scratch보다 0.282 낮았다. source-only가 음의 `R2`였다는
점까지 합치면 source와 target의 세션 차이가 크고, 현재 사전학습 가중치가 target
적응을 돕기보다 방해하는 음의 전이로 해석하는 것이 가장 보수적이다.

## 논문 알고리즘과 현재 데이터의 차이

원 논문은 분류 문제이고 채널의 task relevance를 discrete label과의 mutual
information으로 측정한다. 현재 연구는 연속 `vx/vy` 회귀이므로 속도를 source
기준 quantile label로 바꿔 symmetrical uncertainty를 계산했다. 이 수정은 합리적인
출발점이지만 논문의 동일 조건 재현은 아니다.

특히 회귀에서는 세션별 속도 크기·중심·방향 분포가 달라질 수 있다. 분류 label은
그대로여도 되는 반면 절대 속도를 바로 예측하는 모델은 출력 스케일까지 옮겨야 한다.
현재 negative transfer의 일부는 채널보다 이 회귀 target shift에서 생겼을 수 있다.

## 다음 실험 우선순위

### 1. 회귀 target 정규화

source velocity는 source train 평균·표준편차로, target은 calibration 평균·표준편차로
정규화한 뒤 모델 출력을 target 단위로 되돌린다. 추가로 source-only 출력에 target
calibration으로 2차원 affine 보정을 학습하는 저비용 대조군을 둔다. 이 실험이
negative transfer를 가장 직접적으로 진단한다.

### 2. 같은 mask에서 초기화만 비교

현재 `1e-3/최대 100 epoch/patience 10`을 transfer와 scratch에 동일하게 적용하고,
seed를 최소 5개 반복한다. 평가 대상은 `transfer - scratch`의 paired 차이다. 한 번의
seed 결과만으로 전이 효과를 주장하지 않는다.

### 3. calibration learning curve

target calibration을 `5, 10, 20, 40 task`로 늘린다. 각 조건에서 앞 80%는 학습,
뒤 20%는 validation으로 쓰되 validation은 최소 4 task를 권장한다. 현재 2-task
validation은 조기종료 분산이 너무 크다.

### 4. 채널 점수 ablation

같은 학습 프로토콜에서 다음 64채널 mask를 비교한다.

1. stationarity만
2. importance만
3. stationarity + importance, redundancy 없음
4. 전체 SUTL 점수
5. target Pearson top-64
6. random top-64 반복

이 비교가 있어야 성능 차이를 “전이학습”과 “채널 선택” 중 어느 부분에서 얻었는지
분리할 수 있다.

### 5. 5세션 시간 구조

3일·4일·8일 primary 쌍을 먼저 반복하고, 208일 쌍은 별도 stress 결과로 보고한다.
primary 평균과 stress를 섞으면 시간 간격 효과가 가려진다. 이후에는 target보다
과거인 여러 source 세션을 합치는 multi-source pretraining을 추가하되 target 이후
세션을 source로 사용하지 않는다.

## 현재 판단 기준

알고리즘 적용 가능성은 “구현 가능”이다. 그러나 현재 한 primary 쌍의 성능 근거는
“효과 있음”이 아니라 “현 설정에서 negative transfer”다. 다음 단계에서 최소한
세 primary 쌍 × 5 seed × calibration 4수준을 수행하고, paired bootstrap confidence
interval 또는 세션쌍 단위 효과량을 보고해야 학부연구 논문의 주장을 만들 수 있다.
