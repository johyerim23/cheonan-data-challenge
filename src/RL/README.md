# 천안 응급의료 자원배치 RL 작업 정리

이 문서는 현재까지 구현된 강화학습(RL) 환경, 데이터, 정책 시뮬레이터, 학습 결과와 재실행 방법을 팀원에게 공유하기 위한 문서입니다.

## 1. 한눈에 보는 결과

현재 확보된 데이터를 이용해 천안시 응급의료 자원배치 정책 3개를 선택하는 DQN 환경을 구현했습니다.

최종 추천 정책은 다음과 같습니다.

1. 수신면 신규 구급거점 설치
2. 동면 신규 구급거점 설치
3. 서울대정병원 응급 거점 업그레이드

| 지표 | 현행 | 정책 적용 | 개선 |
|---|---:|---:|---:|
| 평균 총 이동시간 | 17.194분 | 15.737분 | 1.457분 |
| 가중 총 이동시간 | 19.299분 | 17.542분 | 1.757분(9.1%) |
| 평균 EMS 이동시간 | 5.600분 | 4.889분 | 0.711분 |
| 평균 병원 이송시간 | 11.594분 | 10.848분 | 0.746분 |
| 최대 EMS 이동시간 | 11.180분 | 10.120분 | 1.060분 |

Random 1,000회, greedy 규칙 기반, DQN 50,000 timestep × 3 seeds를 비교했습니다. 세 DQN 모델은 모두 greedy와 같은 최종 보상 `1.756761`에 도달했고 중복 행동은 없었습니다.

## 2. 현재 모델의 정확한 구조

현재 학습 모델은 GNN이 아니라 `stable-baselines3`의 DQN입니다.

```text
396개 상태값
  → Fully Connected 64 + ReLU
  → Fully Connected 64 + ReLU
  → 54개 행동별 Q값
  → 가장 높은 행동 선택
```

- 행정동 그래프 준비 데이터: 31개 노드, 71개 인접 엣지
- 도로망 그래프: 97,611개 노드, 132,832개 링크
- 현재 DQN은 행정동 인접 엣지를 직접 입력받지 않습니다.
- 도로망은 최단 이동시간 계산과 보정에 사용한 뒤 지역별 숫자로 압축됩니다.

따라서 `data/gnn/gnn_edges.csv`는 향후 GNN 또는 GNN-RL 실험용 데이터이며 현재 DQN 학습에는 직접 사용되지 않습니다.

## 3. RL 문제 정의

### 상태(State)

총 396차원입니다.

```text
31개 지역 × 11개 변수 = 341
54개 행동 사용 여부     = 54
남은 정책 예산          = 1
합계                    = 396
```

지역별 11개 변수:

- EMS 이동시간
- 병원 이송시간
- 대체병원 취약점수
- 교통 혼잡점수
- 취약인구 점수
- 응급발생 위험점수
- 출동 취약 여부
- 이송 취약 여부
- 대체병원 취약 여부
- 교통 취약 여부
- 수요 취약 여부

### 행동(Action)

총 54개입니다.

| 정책 | 행동 수 | 의미 |
|---|---:|---|
| 기존 119센터 구급차 증차 | 16 | 기존 센터에 구급차 1대 추가 |
| 신규 구급거점 설치 | 16 | 기존 거점이 없는 행정동 대표점에 1대 배치 |
| 일반병원 응급거점 업그레이드 | 22 | 응급병원이 아닌 병원을 후보로 전환 |

한 episode에서 중복 없이 3개 정책을 선택합니다.

### 보상(Reward)

```text
Reward
= 정책 적용 전 가중 총 이동시간
 - 정책 적용 후 가중 총 이동시간
 + 기존 센터 증차의 동적 출동시간 개선
```

지역 가중치는 응급발생 위험점수와 취약인구 점수를 동일 비중으로 합산하여 정규화합니다. 정책 비용 데이터가 없으므로 현재 보상에는 설치비·운영비가 포함되지 않습니다.

## 4. 데이터와 시뮬레이션

### 핵심 입력 데이터

| 경로 | 내용 |
|---|---|
| `data/processed/RL/region_features.csv` | 31개 지역 RL 특성 및 이동시간 |
| `data/processed/RL/cheonan_region_db.csv` | 행정동 대표 좌표 |
| `data/processed/RL/cheonan_ems_station_db.csv` | 119센터 16개와 구급차 수 |
| `data/processed/RL/cheonan_emergency_hospital_db.csv` | 응급의료기관 DB |
| `data/processed/RL/cheonan_hospital_db_all.csv` | 병원 업그레이드 후보 DB |
| `data/processed/RL/ambulance_increment_effects.csv` | 센터별 증차 효과 |
| `data/processed/RL/road_network/` | 천안 Node/Link 도로망과 보정 결과 |
| `data/gnn/gnn_nodes.csv` | 행정동 그래프 31개 노드 |
| `data/gnn/gnn_edges.csv` | 공유 경계 기반 71개 엣지 |

### 도로망

국가교통정보센터 Node/Link ZIP에서 천안과 주변 10km를 추출했습니다.

- 노드: 97,611개
- 링크: 132,832개
- 최대 강연결요소: 86,843개 노드
- 자유류 시간: `도로길이(m) × 0.06 / 제한속도(km/h)`
- 관측시간 보정 LOOCV MAE: EMS 1.55분, 병원 2.04분

### 동적 구급차 출동 시뮬레이션

센터 증차는 거리 기반 지표만으로 효과가 나타나지 않으므로 이벤트 기반 큐 시뮬레이터를 추가했습니다.

- 지역별 2019~2023 응급호출 건수로 Poisson 신고 생성
- 가장 빨리 현장에 도착할 수 있는 가용 차량 배차
- 차량 점유: 센터 출발 → 현장 → 지정 병원 → 센터 복귀
- 현행과 증차 시나리오는 같은 seed와 같은 신고를 사용
- 연간 365일 × seeds `42, 123, 2026`으로 센터 16곳 평가

현재 한계:

- 시간대별 실제 신고 데이터가 없어 일중 신고 시각은 균등분포
- 실제 현장 처치시간 데이터가 없어 0분
- 실시간 병상, 차량 위치, 교통속도는 미반영
- 따라서 결과는 실제 관제가 아니라 정책 비교용 모의실험

## 5. 병원 업그레이드 처리

병원을 업그레이드하면 모든 지역에서 다음 항목을 다시 계산합니다.

- 1순위 병원 및 이동시간
- 2순위 대체병원 및 이동시간
- 1·2순위 이동시간 차이
- 대체병원 취약점수
- 이송·대체 취약유형과 복합 취약 라벨

기존 1·2순위의 관측 이동시간은 보존하고 신규 후보만 도로 보정식으로 계산합니다. 임의의 정책 효과 비율은 사용하지 않았습니다.

## 6. 학습 및 기준선 결과

| 방법 | 평가 단위 | 평균 보상 | 가중 총 이동시간 |
|---|---:|---:|---:|
| Random | 1,000 episodes | 0.679502 | 18.638388 |
| Greedy | 1 | 1.756761 | 17.542135 |
| DQN 50k | 3 seeds | 1.756761 | 17.542135 |

DQN 설정:

- timestep: 50,000
- seeds: 42, 123, 2026
- learning rate: `1e-3`
- replay buffer: 50,000
- batch size: 128
- gamma: 0.99
- network: 396 → 64 → 64 → 54
- CPU 학습

모델 파일:

```text
models/RL/cheonan_dqn_refined_50000_seed42.zip
models/RL/cheonan_dqn_refined_50000_seed123.zip
models/RL/cheonan_dqn_refined_50000_seed2026.zip
```

## 7. 주요 코드

| 파일 | 역할 |
|---|---|
| `src/RL/simulator.py` | 정책 적용 및 지역 상태·평가지표 계산 |
| `src/RL/env.py` | Gymnasium RL 환경 |
| `src/RL/policy_actions.py` | 54개 행동 생성 및 정책 적용 |
| `src/RL/dispatch_simulator.py` | 이벤트 기반 구급차 배차·대기 시뮬레이션 |
| `src/RL/road_network.py` | Node/Link 최단경로와 이동시간 보정 |
| `src/RL/evaluate_random.py` | Random 기준선 |
| `src/RL/evaluate_rule.py` | greedy 기준선 |
| `src/RL/train_rl.py` | DQN 학습 |
| `src/RL/evaluate_dqn.py` | 저장 모델 평가 |
| `src/RL/generate_final_reports.py` | 모델 비교와 최종 Before/After 생성 |
| `src/RL/export_dashboard_data.py` | 대시보드용 동일 신고 재생 데이터 생성 |

## 8. 결과 파일

주요 결과는 `results/RL/refined_with_dispatch/`에 있습니다.

| 파일 | 내용 |
|---|---|
| `model_comparison.csv` | Random·greedy·DQN 비교 |
| `dqn_seed_evaluation.csv` | seed별 DQN 결과와 선택 정책 |
| `final_selected_policies.csv` | 최종 추천 정책 3개 |
| `final_before_after_by_region.csv` | 31지역 Before/After |
| `final_policy_summary.json` | 최종 정책 및 전체 지표 요약 |

## 9. 대시보드

### DQN RL 배치 시뮬레이터 (신규)

`dashboard/rl-simulator/천안_RL_배치시뮬레이터.html`

팀에서 전달받은 천안 시뮬레이터의 지도·배차·시간 애니메이션을 유지하고, GNN 가중치와 브라우저 추론을 현재 DQN 정책 결과로 교체한 단일 HTML입니다.

```bash
conda run -n cheonan-ai python dashboard/rl-simulator/export_dqn_trace.py
conda run -n cheonan-ai python dashboard/rl-simulator/build_rl_dashboard.py
```

주요 기능:

- 수신면·동면 신규 구급 거점과 서울대정병원 응급기능 승급 추천안 적용
- seed별 3단계 DQN Q-value 순위와 선택 과정 자동 재생
- Random / Greedy / DQN 성능 비교
- DQN 3개 시드 재현성 및 반복 행동 검증
- 정책 적용 전후 지역별 개선량
- 원본 1일 환자 발생·구급차 배차·이송 애니메이션

주의: RL은 배치 정책을 선택합니다. 움직이는 지도는 선택된 배치를 원본 배차 시뮬레이터로 재생한 것이며 RL 학습 궤적 자체가 아닙니다. 상세 빌드 구조는 `dashboard/rl-simulator/README.md`를 참고합니다.

### RL 정책 재생 대시보드

`data/figures/rl-policy-simulation-dashboard.html`

기능:

- 현행 / RL 추천 시나리오 전환
- 동일한 1일 신고 63건 재생
- 신고 → 출동 → 병원 이송 → 복귀 애니메이션
- 재생·일시정지, 시간 슬라이더, 60~1,800배속
- 이벤트 로그와 진행 중·완료 건수
- 현행 19대 / RL 추천 21대 표시
- 추천지역 정책 효과 데모
- 정책 전체 지표 Before/After

대시보드 데이터는 `data/figures/rl-dashboard-data.js`입니다. 다음 명령으로 다시 생성할 수 있습니다.

```bash
conda run -n cheonan-ai python src/RL/export_dashboard_data.py
```

VS Code Live Server 또는 아래 명령으로 실행합니다.

```bash
conda run -n cheonan-ai python -m http.server 8000 --directory data/figures
```

브라우저에서 `http://localhost:8000/rl-policy-simulation-dashboard.html`을 엽니다.

### 구조 설명 화면

`data/figures/rl-node-edge-explainer.html`

31지역 그래프, 현재 DQN 신경망, 도로망의 차이를 설명합니다.

## 10. 재실행 방법

현재 작업 환경:

- Conda env: `cheonan-ai`
- Python: 3.13.14
- numpy: 2.5.2
- pandas: 2.2.2
- scipy: 1.18.0
- gymnasium: 1.3.0
- stable-baselines3: 2.9.0
- pyproj: 3.7.2

PowerShell에서는 `conda run`의 한글 출력 인코딩 문제가 있으면 환경 Python을 직접 실행할 수 있습니다.

```powershell
$env:PYTHONUTF8='1'
& 'C:\Users\johye\anaconda3\envs\cheonan-ai\python.exe' src/RL/train_rl.py --total-timesteps 50000 --seed 42
```

기준선 평가:

```bash
conda run -n cheonan-ai python src/RL/evaluate_random.py --episodes 1000 --output-dir results/RL/refined_with_dispatch
conda run -n cheonan-ai python src/RL/evaluate_rule.py --output-dir results/RL/refined_with_dispatch
```

DQN 학습:

```bash
conda run -n cheonan-ai python src/RL/train_rl.py --total-timesteps 50000 --seed 42
conda run -n cheonan-ai python src/RL/train_rl.py --total-timesteps 50000 --seed 123
conda run -n cheonan-ai python src/RL/train_rl.py --total-timesteps 50000 --seed 2026
```

DQN 평가:

```bash
conda run -n cheonan-ai python src/RL/evaluate_dqn.py \
  models/RL/cheonan_dqn_refined_50000_seed42.zip \
  models/RL/cheonan_dqn_refined_50000_seed123.zip \
  models/RL/cheonan_dqn_refined_50000_seed2026.zip \
  --refined-actions \
  --output results/RL/refined_with_dispatch/dqn_seed_evaluation.csv
```

최종 보고서 재생성:

```bash
conda run -n cheonan-ai python src/RL/generate_final_reports.py
```

## 11. 검증

관련 테스트:

```bash
conda run -n cheonan-ai python src/RL/test_policy_state_updates.py
conda run -n cheonan-ai python src/RL/test_dispatch_simulator.py
conda run -n cheonan-ai python src/RL/test_action_refinement.py
conda run -n cheonan-ai python src/RL/test_road_network.py
conda run -n cheonan-ai python src/RL/test_simulator.py
```

검증된 항목:

- 54개 행동과 396차원 상태
- 중복 행동 방지
- 정책 적용 후 이동시간 비악화
- 병원 1·2순위와 취약유형 갱신
- 증차 전후 같은 신고 이벤트 사용
- 증차 후 대기·응답시간 비악화
- 도로망 노드·링크 및 보정 오차
- DQN 모델 저장·재로드와 seed 메타데이터
- 원 평가 보상과 최종 정책 재적용 보상 일치
- 대시보드 현행/RL 전환 및 시간축 재생

## 12. 해석 시 주의사항과 다음 단계

현재 결과는 확보된 데이터 범위에서 정책 후보의 상대적인 효과를 비교한 것입니다. 다음 데이터가 확보되면 다시 학습하는 것이 좋습니다.

1. 시간대별 실제 119 신고 및 출동 이력
2. 현장 처치시간과 병원 인계시간
3. 시간대별 도로 링크 속도
4. 실시간 또는 과거 응급실 병상·수용불가 기록
5. 거점 설치비, 차량·인력 운영비, 병원 전환비

그다음 모델 고도화 후보:

- 정책 비용을 포함한 다목적 보상
- 정책 예산 1~10개 민감도 분석
- PPO/A2C 및 조합 최적화 비교
- 31개 행정동 인접 엣지를 사용하는 GNN-RL
- 여러 날짜·seed의 통계적 신뢰구간

대외 발표에서는 `실시간 관제 결과`가 아니라 `공공데이터 기반 정책 시뮬레이션 결과`라고 표현해야 합니다.
