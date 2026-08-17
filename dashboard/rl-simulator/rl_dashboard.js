/* RL dashboard layer.
 * The original source keeps its map, dispatch simulation, and time animation.
 * This layer replaces the GNN surrogate-model UI with evaluated DQN outputs.
 */
const RL = __RL_DATA_JSON__;

const _sourcePaneC = paneC;
const _sourcePaneB = paneB;

function rlPct(before, after) {
  return before ? (before - after) / before * 100 : 0;
}

function rlPolicyLabel(p) {
  return p.policy === 'ambulance_new_base' ? '신규 구급 거점' : '응급기능 승급';
}

function paneA() {
  const b = RL.summary.baseline_metrics;
  const f = RL.summary.final_metrics;
  const improvement = rlPct(b.weighted_total_time, f.weighted_total_time);
  const impacted = RL.regions.filter(r => r.total_improvement_min > 0.001)
    .sort((a, b) => b.total_improvement_min - a.total_improvement_min);
  const policies = RL.policies.map((p, i) => `
    <div class="rl-policy">
      <span class="rl-step">${i + 1}</span>
      <div><b>${rlPolicyLabel(p)}</b><div>${p.target_name}</div></div>
    </div>`).join('');
  const comparisons = RL.comparison.map(r => {
    const width = Math.max(4, r.reward_mean / RL.max_reward * 100);
    return `<tr><td>${r.label}</td><td><div class="rl-bar"><i style="width:${width}%"></i></div></td>
      <td><b>${r.reward_mean.toFixed(4)}</b></td><td>${r.weighted_total_time.toFixed(3)}분</td></tr>`;
  }).join('');
  const regionRows = impacted.map(r => `<tr>
    <td>${r.region_name}</td>
    <td>${r.before_ems_duration_min.toFixed(2)} → <b>${r.after_ems_duration_min.toFixed(2)}</b></td>
    <td>${r.before_hospital_duration_min.toFixed(2)} → <b>${r.after_hospital_duration_min.toFixed(2)}</b></td>
    <td style="color:var(--s2);font-weight:650">−${r.total_improvement_min.toFixed(2)}분</td>
  </tr>`).join('');
  const seeds = RL.seeds.map(s => `<tr><td>${s.seed}</td><td>${s.total_timesteps.toLocaleString()}</td>
    <td>${s.reward.toFixed(6)}</td><td>${s.weighted_total_time.toFixed(6)}</td>
    <td>${s.has_repeated_action ? '있음' : '없음'}</td></tr>`).join('');

  return `<div class="grid">
    <div class="card c3"><div class="klabel">최종 선택 정책</div>
      <div class="hero">${RL.policies.length}<small>개</small></div>
      <p class="note">예산 3개 안에서 DQN이 선택한 조합.</p></div>
    <div class="card c3"><div class="klabel">가중 총 이동시간</div>
      <div class="hero">−${improvement.toFixed(1)}<small>%</small></div>
      <p class="note">${b.weighted_total_time.toFixed(3)} → <b>${f.weighted_total_time.toFixed(3)}분</b>.</p></div>
    <div class="card c3"><div class="klabel">DQN 평가 보상</div>
      <div class="hero">${RL.summary.expected_reward.toFixed(4)}</div>
      <p class="note">50,000 스텝 학습 후 결정론적 평가.</p></div>
    <div class="card c3"><div class="klabel">다중 시드 재현</div>
      <div class="hero">${RL.seeds.length}/${RL.seeds.length}</div>
      <p class="note">seed ${RL.seeds.map(s => s.seed).join(' · ')}가 동일 최종 성능.</p></div>

    <div class="card c5"><h2>DQN 최종 추천안</h2>
      <p class="note">행동 순서는 시드마다 달라도 최종 정책 집합은 같았다.</p>
      <div class="rl-policies">${policies}</div>
      <button class="mbtn on" onclick="showTab('C');setTimeout(preset,80)">지도에 추천안 적용</button></div>
    <div class="card c7"><h2>정책 탐색 방법 비교</h2>
      <p class="note">동일 환경·동일 정책 예산에서 평가했다. 막대는 누적 개선 보상이며 클수록 좋다.</p>
      <div style="overflow:auto"><table><thead><tr><th>방법</th><th style="text-align:left">상대 보상</th>
        <th>평균 보상</th><th>가중 총시간</th></tr></thead><tbody>${comparisons}</tbody></table></div>
      <div class="warnbox">Random은 1,000회 평균이다. Greedy와 DQN이 같은 최종값에 도달했다는 사실은
        이 데이터·예산에서 최적 조합이 안정적이었다는 근거이지, 모든 상황에서 DQN이 우월하다는 뜻은 아니다.</div></div>

    <div class="card c7"><h2>개선이 발생한 행정동</h2>
      <p class="note">RL 환경의 정책 효과 계산 결과. 0.001분 이하 변화는 제외했다.</p>
      <div style="overflow:auto;max-height:330px"><table><thead><tr><th>행정동</th><th>출동 전→후</th>
        <th>이송 전→후</th><th>총 개선</th></tr></thead><tbody>${regionRows}</tbody></table></div></div>
    <div class="card c5"><h2>RL 모델 구조와 행동 공간</h2>
      <dl class="kv">
        <dt>알고리즘</dt><dd>DQN</dd><dt>상태 벡터</dt><dd>${RL.model.state_dim}차원</dd>
        <dt>신경망</dt><dd>${RL.model.network.join(' → ')}</dd><dt>행동 수</dt><dd>${RL.model.action_dim}개</dd>
        <dt>정책 예산</dt><dd>${RL.summary.policy_budget}개</dd>
        <dt>행동 구성</dt><dd>기존 거점 증차 16 · 신규 거점 16 · 병원 승급 22</dd>
      </dl>
      <div class="warnbox"><b>지도 애니메이션의 역할.</b> DQN은 배치 정책을 선택하고, 지도에서는 그 정책을 원본
        배차 시뮬레이션 엔진에 적용해 하루 흐름을 재생한다. 지도에서 수동 배치를 바꿔도 DQN을 재학습하지는 않는다.</div></div>

    <div class="card c12"><h2>DQN 다중 시드 셀프 검증</h2>
      <p class="note">반복 행동 없이 세 모델이 같은 보상과 가중 총시간을 재현했다.</p>
      <div style="overflow:auto"><table><thead><tr><th>seed</th><th>학습 스텝</th><th>보상</th>
        <th>가중 총시간</th><th>반복 행동</th></tr></thead><tbody>${seeds}</tbody></table></div></div>
  </div>`;
}

function rlMapCard() {
  return `<div class="card c8">
    <h2>DQN 정책 모델 — <span class="pill">실제 Q-network 단계별 추론</span></h2>
    <p class="note">학습된 DQN에 RL 환경의 396차원 상태를 입력하고 <b>54개 행동의 Q-value를 실제로 계산</b>한 결과다.
      seed와 정책 단계를 바꾸면 해당 시점의 행동 순위, 선택 정책, 즉시 보상과 누적 보상을 볼 수 있다.</p>
    <div id="rlKpi" class="kpirow"></div>
    <div class="ctl" style="margin:0 0 10px">
      <label class="chk">모델 seed
        <select id="dqnSeed" onchange="selectDqnSeed(+this.value)">
          ${RL.traces.map(t => `<option value="${t.seed}"${t.seed === 42 ? ' selected' : ''}>${t.seed}</option>`).join('')}
        </select></label>
      <span class="dqn-step-buttons">
        ${[0, 1, 2].map(i => `<button class="mbtn" id="dqnStep${i}" onclick="selectDqnStep(${i})">${i + 1}단계</button>`).join('')}
      </span>
      <button class="mbtn" id="dqnReplay" onclick="replayDqnTrace()">▶ 선택 과정 재생</button>
      <button class="mbtn on" onclick="preset()">RL 추천안 적용</button>
    </div>
    <div class="dqn-layout">
      <div>
        <div class="klabel" style="margin-bottom:7px">현재 상태의 상위 Q-value 행동</div>
        <div id="cDqn"></div>
      </div>
      <div id="dqnDecision" class="dqn-decision"></div>
    </div>
    <div class="warnbox"><b>Q-value를 읽는 법.</b> Q-value는 분 단위 예측값이 아니라 현재 행동 이후 기대되는
      누적 보상의 상대 점수다. 같은 단계 안에서 행동 순위를 비교해야 하며 서로 다른 seed의 절대값을 직접 비교하면 안 된다.</div>
    <p class="note" style="margin:8px 0 0">이 패널은 Python에서 Stable-Baselines3 DQN의 q_net을 실행해 저장한
      결정론적 평가 trace다. 지도에서 임의 배치를 바꿔도 Q-value를 새로 계산하지 않으며,
      아래 구급차 애니메이션은 선택된 최종 배치를 원본 1일 배차 시뮬레이터로 재생한다.</p>
  </div>`;
}

paneC = function () {
  let html = _sourcePaneC().replace('정책 A 재현', 'RL 추천안 적용');
  const start = html.indexOf('<div class="card c8">\n    <h2>GNN 대체모델');
  const end = html.indexOf('  <div class="card c4">\n    <h2>축별 취약 판정', start);
  if (start >= 0 && end > start) html = html.slice(0, start) + rlMapCard() + '\n' + html.slice(end);
  return html;
};

paneB = function () {
  return _sourcePaneB().replaceAll('GNN', 'RL 정책 평가');
};

function activeHospitals() {
  const list = M.hospitals.filter(h => !OFF.hosp.has(h.id)).map(h => Object.assign({}, h, {neu: 0}));
  PLACED.filter(p => p.kind === 'hosp').forEach((p, i) => {
    const g = p.grade || '지역응급의료센터', c = HGRADE[g];
    list.push({id: 'NEW_H' + i, lon: p.lon, lat: p.lat,
      name: p.name || ('[신설] ' + g + ' ' + (i + 1)), etype: g, inside: true, beds: 6,
      cardiac: c.cardiac, stroke: c.stroke, trauma: c.trauma, neu: 1});
  });
  return list;
}

function preset() {
  PLACED = [
    {kind: 'ems', lon: RL.locations['수신면'].lon, lat: RL.locations['수신면'].lat, amb: 1},
    {kind: 'ems', lon: RL.locations['동면'].lon, lat: RL.locations['동면'].lat, amb: 1},
    {kind: 'hosp', lon: RL.locations['서울대정병원'].lon, lat: RL.locations['서울대정병원'].lat,
      grade: '지역응급의료센터', name: '[RL 승급] 서울대정병원'}
  ];
  OFF = {ems: new Set(), hosp: new Set()};
  refresh(true);
}

let DQN_SEED = 42;
let DQN_STEP = 0;
let DQN_TIMER = null;

function currentDqnTrace() {
  return RL.traces.find(t => t.seed === DQN_SEED) || RL.traces[0];
}

function selectDqnSeed(seed) {
  DQN_SEED = seed;
  DQN_STEP = 0;
  drawDqnPanel();
}

function selectDqnStep(step) {
  DQN_STEP = Math.max(0, Math.min(2, step));
  drawDqnPanel();
}

function replayDqnTrace() {
  if (DQN_TIMER) clearInterval(DQN_TIMER);
  DQN_STEP = 0;
  drawDqnPanel();
  const button = document.getElementById('dqnReplay');
  if (button) button.textContent = '재생 중…';
  DQN_TIMER = setInterval(() => {
    DQN_STEP += 1;
    drawDqnPanel();
    if (DQN_STEP >= 2) {
      clearInterval(DQN_TIMER);
      DQN_TIMER = null;
      const done = document.getElementById('dqnReplay');
      if (done) done.textContent = '▶ 다시 재생';
    }
  }, 900);
}

function drawDqnPanel() {
  const box = document.getElementById('rlKpi');
  const chart = document.getElementById('cDqn');
  const decision = document.getElementById('dqnDecision');
  if (!box || !chart || !decision) return;
  const trace = currentDqnTrace();
  const step = trace.steps[DQN_STEP];
  const ranked = step.ranked_actions.slice(0, 8);
  const qMin = Math.min(...ranked.map(a => a.q_value));
  const qMax = Math.max(...ranked.map(a => a.q_value));
  const weightedBefore = trace.baseline_metrics.weighted_total_time;
  const weightedNow = step.metrics.weighted_total_time;
  const seedControl = document.getElementById('dqnSeed');
  if (seedControl) seedControl.value = String(DQN_SEED);
  [0, 1, 2].forEach(i => {
    const button = document.getElementById('dqnStep' + i);
    if (button) button.classList.toggle('on', i === DQN_STEP);
  });

  box.innerHTML = `<div>선택 행동 <b>#${step.selected_action_index} ${step.selected_policy.target_name}</b></div>
    <div>선택 Q <b>${step.selected_q_value.toFixed(4)}</b></div>
    <div>즉시 보상 <b class="hi">+${step.reward.toFixed(4)}</b></div>
    <div>누적 보상 <b>${step.cumulative_reward.toFixed(4)}</b></div>
    <div>가중 총시간 <b>${weightedBefore.toFixed(2)} → ${weightedNow.toFixed(2)}분</b></div>`;

  chart.innerHTML = ranked.map(action => {
    const width = 18 + (action.q_value - qMin) / Math.max(qMax - qMin, 1e-9) * 82;
    return `<div class="q-row${action.selected ? ' selected' : ''}">
      <div class="q-rank">${action.rank}</div>
      <div class="q-name"><b>${action.target_name}</b><small>${action.policy_label}${action.used_before_step ? ' · 이미 선택됨' : ''}</small></div>
      <div class="q-track"><i style="width:${width}%"></i></div>
      <div class="q-value">${action.q_value.toFixed(4)}</div>
    </div>`;
  }).join('');

  const chosen = trace.steps.slice(0, DQN_STEP + 1).map((item, i) => `
    <div class="dqn-path-item${i === DQN_STEP ? ' current' : ''}">
      <span>${i + 1}</span><div><b>${item.selected_policy.target_name}</b>
      <small>${item.selected_policy_label} · +${item.reward.toFixed(4)}</small></div>
    </div>`).join('');
  decision.innerHTML = `<div class="klabel">seed ${trace.seed} · ${DQN_STEP + 1}단계 의사결정</div>
    <div class="dqn-choice"><small>DQN 선택</small><strong>${step.selected_policy.target_name}</strong>
      <span>${step.selected_policy_label}</span></div>
    <dl class="kv">
      <dt>행동 번호</dt><dd>#${step.selected_action_index} / 53</dd>
      <dt>2위와 Q 차이</dt><dd>${step.q_gap_to_second.toFixed(4)}</dd>
      <dt>이동시간 보상</dt><dd>+${step.reward_components.travel_time_reward.toFixed(4)}</dd>
      <dt>증차 보너스</dt><dd>+${step.reward_components.dispatch_reward.toFixed(4)}</dd>
    </dl><div class="dqn-path">${chosen}</div>`;
}

// render() still invokes the source dashboard drawing hooks. Route the first hook
// to the RL view and disable the obsolete GNN validation charts.
function drawRlDashboard() { drawDqnPanel(); }
drawSweep = drawRlDashboard;
drawDecomp = function () {};
drawEdge = function () {};
drawRank = function () {};
drawLoad = function () {};
drawCurve = function () {};
drawLeak = function () {};
fillTable = function () {};
drawGnnPanel = drawDqnPanel;
