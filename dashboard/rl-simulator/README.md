# 천안 DQN RL 배치 시뮬레이터

사용자가 제공한 `천안_시뮬레이터_소스.zip`과 완성본 `sim_2.html`을 기반으로 만든 독립 실행형 HTML 대시보드입니다.

## 무엇을 유지하고 무엇을 바꿨나

- 유지: MapLibre 지도, 행정동 취약축 계산, 시설 수동 배치, 1일 환자 발생·배차·이송 애니메이션, 경합 로그
- 제거: GNN 가중치, 브라우저 GNN 추론, GNN 검증 지표
- 추가: 학습된 DQN 정책 3개, Random/Greedy/DQN 비교, 다중 시드 검증, 지역별 전후 효과

RL 추천안은 `수신면 신규 구급 거점`, `동면 신규 구급 거점`, `서울대정병원 응급기능 승급`입니다. 지도 애니메이션은 RL 학습 과정이 아니라, RL이 선택한 배치를 원본의 1일 배차 시뮬레이션으로 재생한 것입니다.

## 생성

Anaconda 환경 `cheonan-ai`에서 저장소 루트를 기준으로 실행합니다.

```powershell
conda run -n cheonan-ai python dashboard/rl-simulator/export_dqn_trace.py
conda run -n cheonan-ai python dashboard/rl-simulator/build_rl_dashboard.py
```

출력 파일:

```text
dashboard/rl-simulator/천안_RL_배치시뮬레이터.html
```

HTML은 외부 데이터 파일 없이 열리는 단일 파일입니다. 배경 지도 타일만 인터넷 연결 상태에 따라 달라질 수 있으며, 지도 이외의 표·시뮬레이션 데이터는 내장되어 있습니다.

## 데이터 연결

- `results/RL/refined_with_dispatch/final_policy_summary.json`
- `results/RL/refined_with_dispatch/final_selected_policies.csv`
- `results/RL/refined_with_dispatch/final_before_after_by_region.csv`
- `results/RL/refined_with_dispatch/model_comparison.csv`
- `results/RL/refined_with_dispatch/dqn_seed_evaluation.csv`

대시보드 수치는 빌드할 때 위 파일에서 다시 읽으므로 RL 결과가 바뀌면 빌드 명령만 재실행하면 됩니다.

`export_dqn_trace.py`는 저장된 세 DQN 모델의 `q_net`을 직접 실행해 각 단계 54개 행동의 Q-value,
선택 행동, 즉시·누적 보상과 단계별 성능을 `dqn_trace.json`으로 내보냅니다. 지도 패널의 seed·단계 전환과
선택 과정 재생은 이 실제 평가 trace를 사용합니다.
