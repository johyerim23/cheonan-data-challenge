/* ---------- 공통 ---------- */
const TT=document.getElementById('tt');
function tip(e,html){TT.innerHTML=html;TT.style.opacity=1;
  let x=e.clientX+14,y=e.clientY+14;
  if(x+290>innerWidth)x=e.clientX-296; if(y+130>innerHeight)y=e.clientY-130;
  TT.style.left=x+'px';TT.style.top=y+'px';}
function untip(){TT.style.opacity=0;}
function toggleTheme(){const r=document.documentElement;
  const d=r.getAttribute('data-theme')==='dark';
  r.setAttribute('data-theme',d?'light':'dark');
  document.getElementById('themeBtn').textContent=d?'다크':'라이트';
  render();}
function showTab(t){
  ['A','B','C'].forEach(k=>{
    document.getElementById('pane'+k).classList.toggle('hidden',t!==k);
    document.getElementById('tab'+k).setAttribute('aria-selected',t===k);});
  if(t==='B')drawGantt(); if(t==='C'&&MAP)setTimeout(()=>{MAP.resize();sizeCanvas();drawOverlay();},60);
}
const NS='http://www.w3.org/2000/svg';
function el(t,a={},p){const n=document.createElementNS(NS,t);
  for(const k in a)n.setAttribute(k,a[k]); if(p)p.appendChild(n); return n;}
function svg(w,h){const s=document.createElementNS(NS,'svg');
  s.setAttribute('viewBox',`0 0 ${w} ${h}`);s.setAttribute('role','img');return s;}
const fmt=(v,d=2)=>v.toFixed(d);
function css(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim();}

/* ---------- 대시보드 ---------- */
function paneA(){
  const m=D.model, gnn=m.gnn, gbm=m.gbm;
  const s1=D.sweep.filter(s=>s.amb===1).sort((a,b)=>a.pen-b.pen);
  const best=s1[0].dT10, worst=s1[s1.length-1].dT10;
  return `
<div class="grid">
  <div class="card c3">
    <div class="klabel">정책 A 효과 (T10 초과 감소)</div>
    <div class="hero">−0.2 ~ −16.4<small> %p</small></div>
    <p class="note">신설 거점 입지 가정 하나로 이만큼 흔들린다. 경험 중앙값(4.8분) 기준 <b>−7.7%p</b>.</p>
  </div>
  <div class="card c3">
    <div class="klabel">증차 단독 +2대</div>
    <div class="hero">−6.5<small> %p</small></div>
    <p class="note">불당·청당 2대만으로 안전센터 3개소 신설(−7.7%p)과 사실상 동급. 부지·건축 불필요.</p>
  </div>
  <div class="card c3">
    <div class="klabel">GNN 분류 AUC</div>
    <div class="hero">${fmt(gnn.auc,3)}</div>
    <p class="note">그래프 없는 GBM 베이스라인 <b>${fmt(gbm.auc,3)}</b> 대비 +${fmt(gnn.auc-gbm.auc,3)}.</p>
  </div>
  <div class="card c3">
    <div class="klabel">정책 순위 Spearman ρ</div>
    <div class="hero">${fmt(gnn.rho,3)} <small>vs ${fmt(gbm.rho,3)}</small></div>
    <p class="note">GBM이 미세 우위. <b>순위 지표에서 그래프는 기여하지 않는다.</b></p>
  </div>

  <div class="card c7">
    <h2>신설 거점 입지 패널티 스윕</h2>
    <p class="note">x = 신설 안전센터가 관할 지역에 도달하는 시간(분). 0분 = "거점이 동 대표점에 정확히 있다"는 기존 가정.
      바닥의 눈금은 <b>관내 동명 거점을 가진 8개 지역의 실측값</b>이다 — 0.65분(입장면)부터 10.12분(직산읍)까지 퍼져 있어 단일 값으로 못 정한다.</p>
    <div class="legend"><span><i style="background:var(--s1)"></i>신설 1대</span><span><i style="background:var(--s2)"></i>신설 2대</span></div>
    <div id="cSweep"></div>
  </div>
  <div class="card c5">
    <h2>응답시간 3분할</h2>
    <p class="note">응답 = 60초 + 대기 + d1(거점→지역) + intra(지역 내부).
      <b>대기는 9,488건 전부 0</b> — 경합은 대기가 아니라 차선배차로 흡수된다.</p>
    <div class="legend">
      <span><i style="background:var(--q2)"></i>구조적 하한</span>
      <span><i style="background:var(--s2)"></i>차선배차 손실</span>
      <span><i style="background:var(--text-muted)"></i>대기(=0)</span></div>
    <div id="cDecomp"></div>
  </div>

  <div class="card c5">
    <h2>엣지 규칙 — 후보 거점 수가 윈도보다 중요하다</h2>
    <p class="note">막대 = y&gt;0인데 들어오는 엣지가 없는 비율(그래프가 구조적으로 설명 불가한 몫).
      최근접 거점만 쓰면 <b>파급 차단</b>(최근접이 만차라 2순위로 넘어간 선행 케이스)이 누락된다.</p>
    <div id="cEdge"></div>
  </div>
  <div class="card c7">
    <h2>정책 순위 재현 — 시뮬 vs GNN</h2>
    <p class="note">test 100 시나리오의 평균 손실. 대각선에 붙을수록 대체모델이 시뮬을 재현한다.
      색이 진할수록 고부하. <b>전체 ρ 0.98은 상당 부분 부하가 만든 쉬운 신호다</b> — 오른쪽 막대 참조.</p>
    <div class="legend" id="rankLegend"></div>
    <div id="cRank"></div>
  </div>

  <div class="card c4">
    <h2>부하 내부 순위 정확도</h2>
    <p class="note">같은 DAILY 안에서만 순위를 재면 ρ가 떨어진다. 경합이 실제로 많은 고부하에서 가장 나쁘다.</p>
    <div id="cLoad"></div>
    <div class="warnbox">전체 ρ 0.981 → 부하 내부 평균 <b>0.876</b>. DAILY=300에서 0.727.</div>
  </div>
  <div class="card c4">
    <h2>학습 곡선 (val AUC)</h2>
    <p class="note">22 에폭에서도 상승 중이다. <b>수렴시키지 않은 수치</b>이므로 더 오를 여지가 있다.</p>
    <div id="cCurve"></div>
  </div>
  <div class="card c4">
    <h2>단일 변수 AUC — 누수 점검</h2>
    <p class="note"><code>n_free</code>는 전역 가용 대수인데 단독 0.731이다. 큐 상태 요약값이라 입력에서 제외했다.
      <code>min_amb</code>는 배치 정책의 일부라 유지.</p>
    <div id="cLeak"></div>
  </div>

  <div class="card c12">
    <h2>정책 시나리오 21종 <span class="pill">표 보기</span></h2>
    <p class="note">ΔT10 = 현행 대비 응답기준(도심 10분/읍면 15분) 초과율 변화. 이송완료는 신고→병원 도착.</p>
    <div style="overflow:auto;max-height:420px">
      <table id="tScen"><thead><tr>
        <th>시나리오</th><th>평균 응답</th><th>p90</th><th>T10 초과</th><th>ΔT10</th>
        <th>이송완료</th><th>구조적 하한</th><th>차선배차 손실</th><th>손실 0 비율</th>
      </tr></thead><tbody></tbody></table>
    </div>
  </div>
</div>`;
}

function drawSweep(){
  const box=document.getElementById('cSweep'); if(!box)return; box.innerHTML='';
  const W=680,H=300,L=52,R=16,T=14,B=52;
  const s=svg(W,H); box.appendChild(s);
  const xs=[0,10.5], ys=[-18,1];
  const X=v=>L+(v-xs[0])/(xs[1]-xs[0])*(W-L-R);
  const Y=v=>T+(ys[1]-v)/(ys[1]-ys[0])*(H-T-B);
  for(let g=-18;g<=0;g+=3){
    el('line',{x1:L,x2:W-R,y1:Y(g),y2:Y(g),class:g===0?'zl':'gl'},s);
    el('text',{x:L-8,y:Y(g)+4,'text-anchor':'end',class:'ax'},s).textContent=g+'%p';
  }
  for(let g=0;g<=10;g+=2){
    el('text',{x:X(g),y:H-B+18,'text-anchor':'middle',class:'ax'},s).textContent=g+'분';
  }
  el('text',{x:(L+W-R)/2,y:H-6,'text-anchor':'middle',class:'axt'},s).textContent='신설 거점 입지 패널티 (분)';
  // 경험 분포 rug
  D.own_eta.forEach(v=>{
    el('line',{x1:X(v),x2:X(v),y1:H-B+2,y2:H-B+9,stroke:css('--text-muted'),'stroke-width':2,
      'stroke-linecap':'round',opacity:.85},s);
  });
  el('text',{x:X(10.12),y:H-B-4,'text-anchor':'end',class:'ax'},s).textContent='실측 관내거점 ETA 분포 ↓';
  const series=[{k:1,c:css('--s1'),n:'신설 1대'},{k:2,c:css('--s2'),n:'신설 2대'}];
  series.forEach(sr=>{
    const pts=D.sweep.filter(d=>d.amb===sr.k).sort((a,b)=>a.pen-b.pen);
    const path=pts.map((p,i)=>(i?'L':'M')+X(p.pen)+' '+Y(p.dT10)).join(' ');
    el('path',{d:path,fill:'none',stroke:sr.c,'stroke-width':2,'stroke-linejoin':'round'},s);
    pts.forEach(p=>{
      const c=el('circle',{cx:X(p.pen),cy:Y(p.dT10),r:4.5,fill:sr.c,stroke:css('--surface-1'),
        'stroke-width':2,style:'cursor:pointer'},s);
      c.addEventListener('mousemove',e=>tip(e,
        `<b>패널티 ${fmt(p.pen,1)}분 · ${sr.n}</b><br>ΔT10 <b>${fmt(p.dT10,1)}%p</b><br>
         평균 응답 ${fmt(p.resp,2)}분 · p90 ${fmt(p.p90,2)}<br>이송완료 ${fmt(p.total,2)}분<br>
         하한 ${fmt(p.floor,2)} · 차선배차 손실 ${fmt(p.loss,2)}분`));
      c.addEventListener('mouseleave',untip);
    });
    const f=pts[0];
    el('text',{x:X(f.pen)+9,y:Y(f.dT10)+(sr.k===1?-9:15),class:'axt',fill:sr.c,
      style:'font-weight:600'},s).textContent=sr.n;
  });
  // 중앙값 안내선
  el('line',{x1:X(4.8),x2:X(4.8),y1:T,y2:H-B,stroke:css('--text-muted'),'stroke-width':1,
    'stroke-dasharray':'3 4',opacity:.7},s);
  el('text',{x:X(4.8)+5,y:T+12,class:'ax'},s).textContent='경험 중앙값 4.8분';
}

function drawDecomp(){
  const box=document.getElementById('cDecomp'); if(!box)return; box.innerHTML='';
  const rows=[{n:'현행',floor:9.83,wait:0,loss:0.79},
              {n:'A 신설(4.8·1대)',floor:9.13,wait:0,loss:0.93},
              {n:'A 신설(4.8·2대)',floor:9.13,wait:0,loss:0.64},
              {n:'증차 단독 +4대',floor:9.83,wait:0,loss:0.33}];
  const W=460,H=210,L=118,R=48,T=8,B=34, bh=26, gap=(H-T-B-rows.length*bh)/(rows.length-1);
  const s=svg(W,H); box.appendChild(s);
  const max=11, X=v=>L+v/max*(W-L-R);
  for(let g=0;g<=10;g+=2){
    el('line',{x1:X(g),x2:X(g),y1:T,y2:H-B,class:'gl'},s);
    el('text',{x:X(g),y:H-B+16,'text-anchor':'middle',class:'ax'},s).textContent=g;
  }
  el('text',{x:(L+W-R)/2,y:H-4,'text-anchor':'middle',class:'axt'},s).textContent='평균 응답시간 (분)';
  rows.forEach((r,i)=>{
    const y=T+i*(bh+gap);
    el('text',{x:L-9,y:y+bh/2+4,'text-anchor':'end',class:'axt'},s).textContent=r.n;
    const segs=[{v:r.floor,c:css('--q2'),n:'구조적 하한 (60초+min_eta+intra)'},
                {v:r.loss,c:css('--s2'),n:'차선배차 손실 (d1−min_eta)'}];
    let x0=0;
    segs.forEach((sg,k)=>{
      const w=X(sg.v)-X(0);
      const rct=el('rect',{x:X(x0)+(k?2:0),y:y,width:Math.max(w-(k?2:0),1),height:bh,fill:sg.c,
        rx:k===segs.length-1?4:0,style:'cursor:pointer'},s);
      rct.addEventListener('mousemove',e=>tip(e,`<b>${r.n}</b><br>${sg.n}<br><b>${fmt(sg.v,2)}분</b>
        (전체 ${fmt(r.floor+r.loss,2)}분의 ${fmt(sg.v/(r.floor+r.loss)*100,0)}%)`));
      rct.addEventListener('mouseleave',untip);
      x0+=sg.v;
    });
    el('text',{x:X(x0)+7,y:y+bh/2+4,class:'axt'},s).textContent=fmt(x0,2);
  });
}

function drawEdge(){
  const box=document.getElementById('cEdge'); if(!box)return; box.innerHTML='';
  const R2=D.edges, W=460,H=230,L=104,Rr=96,T=8,B=34,bh=26;
  const gap=(H-T-B-R2.length*bh)/(R2.length-1);
  const s=svg(W,H); box.appendChild(s);
  const max=16, X=v=>L+v/max*(W-L-Rr);
  for(let g=0;g<=16;g+=4){el('line',{x1:X(g),x2:X(g),y1:T,y2:H-B,class:'gl'},s);
    el('text',{x:X(g),y:H-B+16,'text-anchor':'middle',class:'ax'},s).textContent=g+'%';}
  el('text',{x:(L+W-Rr)/2,y:H-4,'text-anchor':'middle',class:'axt'},s).textContent='y>0 인데 in-degree 0 (%)';
  el('text',{x:W,y:T-1,'text-anchor':'end',class:'ax'},s).textContent='엣지 밀도';
  R2.forEach((r,i)=>{
    const y=T+i*(bh+gap), chosen=r.rule==='top-3 · 60분';
    el('text',{x:L-9,y:y+bh/2+4,'text-anchor':'end',class:'axt',
      style:chosen?'font-weight:650':''},s).textContent=r.rule;
    const rct=el('rect',{x:L,y:y,width:Math.max(X(r.miss)-L,2),height:bh,rx:4,
      fill:chosen?css('--good'):css('--s2'),opacity:chosen?1:.9,style:'cursor:pointer'},s);
    rct.addEventListener('mousemove',e=>tip(e,`<b>${r.rule}</b><br>미설명 <b>${fmt(r.miss,2)}%</b><br>
      엣지/노드 ${fmt(r.epn,2)} · degree max ${r.dmax}${chosen?'<br><b>채택</b>':''}`));
    rct.addEventListener('mouseleave',untip);
    el('text',{x:X(r.miss)+7,y:y+bh/2+4,class:'axt',
      style:chosen?'font-weight:650':''},s).textContent=fmt(r.miss,2)+'%';
    el('text',{x:W,y:y+bh/2+4,'text-anchor':'end',class:'ax'},s).textContent=fmt(r.epn,1)+' 엣지/노드';
  });
}

function drawRank(){
  const box=document.getElementById('cRank'); if(!box)return; box.innerHTML='';
  const W=680,H=330,L=54,R=16,T=12,B=48;
  const s=svg(W,H); box.appendChild(s);
  const vals=D.rank.flatMap(r=>[r.sim,r.gnn]);
  const lo=0, hi=Math.ceil(Math.max(...vals)*1.05*10)/10;
  const X=v=>L+(v-lo)/(hi-lo)*(W-L-R), Y=v=>T+(hi-v)/(hi-lo)*(H-T-B);
  const step=hi>4?2:1;
  for(let g=0;g<=hi;g+=step){
    el('line',{x1:L,x2:W-R,y1:Y(g),y2:Y(g),class:'gl'},s);
    el('line',{x1:X(g),x2:X(g),y1:T,y2:H-B,class:'gl'},s);
    el('text',{x:L-8,y:Y(g)+4,'text-anchor':'end',class:'ax'},s).textContent=g;
    el('text',{x:X(g),y:H-B+17,'text-anchor':'middle',class:'ax'},s).textContent=g;
  }
  el('line',{x1:X(lo),y1:Y(lo),x2:X(hi),y2:Y(hi),stroke:css('--text-muted'),'stroke-width':1.5,
    'stroke-dasharray':'5 4',opacity:.8},s);
  el('text',{x:X(hi*0.66)+8,y:Y(hi*0.66)+15,class:'ax'},s).textContent='완전 일치선';
  el('text',{x:(L+W-R)/2,y:H-6,'text-anchor':'middle',class:'axt'},s).textContent='시뮬레이터 평균 손실 (분)';
  el('text',{x:14,y:(T+H-B)/2,'text-anchor':'middle',class:'axt',
    transform:`rotate(-90 14 ${(T+H-B)/2})`},s).textContent='GNN 예측 (분)';
  const ramp={90:css('--q1'),150:css('--q2'),200:css('--q3'),300:css('--q4')};
  D.rank.forEach(r=>{
    const c=el('circle',{cx:X(r.sim),cy:Y(r.gnn),r:5,fill:ramp[r.daily],stroke:css('--surface-1'),
      'stroke-width':2,style:'cursor:pointer'},s);
    c.addEventListener('mousemove',e=>tip(e,`<b>시나리오 #${r.scen}</b> · DAILY ${r.daily}<br>
      시뮬 <b>${fmt(r.sim,3)}분</b> → GNN <b>${fmt(r.gnn,3)}분</b><br>오차 ${fmt(r.gnn-r.sim,3)}분`));
    c.addEventListener('mouseleave',untip);
  });
  const lg=document.getElementById('rankLegend');
  lg.innerHTML=[90,150,200,300].map(d=>`<span><i style="background:${ramp[d]}"></i>DAILY ${d}</span>`).join('');
}

function drawLoad(){
  const box=document.getElementById('cLoad'); if(!box)return; box.innerHTML='';
  const W=380,H=190,L=60,R=44,T=8,B=32,bh=24;
  const rows=D.byload, gap=(H-T-B-rows.length*bh)/(rows.length-1);
  const s=svg(W,H); box.appendChild(s);
  const X=v=>L+v*(W-L-R);
  [0,.25,.5,.75,1].forEach(g=>{el('line',{x1:X(g),x2:X(g),y1:T,y2:H-B,class:'gl'},s);
    el('text',{x:X(g),y:H-B+16,'text-anchor':'middle',class:'ax'},s).textContent=g;});
  el('text',{x:(L+W-R)/2,y:H-3,'text-anchor':'middle',class:'axt'},s).textContent='Spearman ρ (부하 내부)';
  const ramp={90:css('--q1'),150:css('--q2'),200:css('--q3'),300:css('--q4')};
  rows.forEach((r,i)=>{
    const y=T+i*(bh+gap);
    el('text',{x:L-9,y:y+bh/2+4,'text-anchor':'end',class:'axt'},s).textContent='DAILY '+r.daily;
    const rc=el('rect',{x:L,y:y,width:Math.max(X(r.rho)-L,2),height:bh,rx:4,fill:ramp[r.daily],
      style:'cursor:pointer'},s);
    rc.addEventListener('mousemove',e=>tip(e,`<b>DAILY ${r.daily}</b><br>ρ = <b>${fmt(r.rho,4)}</b><br>
      시나리오 ${r.n}개`));
    rc.addEventListener('mouseleave',untip);
    el('text',{x:X(r.rho)+7,y:y+bh/2+4,class:'axt'},s).textContent=fmt(r.rho,3);
  });
}

function drawCurve(){
  const box=document.getElementById('cCurve'); if(!box)return; box.innerHTML='';
  const W=380,H=190,L=48,R=16,T=12,B=34;
  const s=svg(W,H); box.appendChild(s);
  const c=D.curve, X=v=>L+(v-1)/(c.length-1)*(W-L-R), Y=v=>T+(0.9-v)/(0.9-0.55)*(H-T-B);
  [0.6,0.7,0.8,0.9].forEach(g=>{el('line',{x1:L,x2:W-R,y1:Y(g),y2:Y(g),class:'gl'},s);
    el('text',{x:L-7,y:Y(g)+4,'text-anchor':'end',class:'ax'},s).textContent=g.toFixed(2);});
  [1,5,10,15,20].forEach(g=>el('text',{x:X(g),y:H-B+16,'text-anchor':'middle',class:'ax'},s).textContent=g);
  el('text',{x:(L+W-R)/2,y:H-3,'text-anchor':'middle',class:'axt'},s).textContent='에폭';
  // GBM 베이스라인
  el('line',{x1:L,x2:W-R,y1:Y(D.model.gbm.auc),y2:Y(D.model.gbm.auc),stroke:css('--s2'),
    'stroke-width':1.5,'stroke-dasharray':'4 4'},s);
  el('text',{x:W-R,y:Y(D.model.gbm.auc)-6,'text-anchor':'end',class:'ax',fill:css('--s2')},s)
    .textContent='GBM '+fmt(D.model.gbm.auc,3);
  el('path',{d:c.map((p,i)=>(i?'L':'M')+X(p.ep)+' '+Y(p.auc)).join(' '),fill:'none',
    stroke:css('--s1'),'stroke-width':2},s);
  c.forEach(p=>{const dot=el('circle',{cx:X(p.ep),cy:Y(p.auc),r:3.5,fill:css('--s1'),
      stroke:css('--surface-1'),'stroke-width':1.5,style:'cursor:pointer'},s);
    dot.addEventListener('mousemove',e=>tip(e,`<b>ep ${p.ep}</b><br>val AUC <b>${fmt(p.auc,4)}</b><br>
      loss ${fmt(p.loss,4)} · 조건부 MAE ${fmt(p.mae,2)}분`));
    dot.addEventListener('mouseleave',untip);});
}

function drawLeak(){
  const box=document.getElementById('cLeak'); if(!box)return; box.innerHTML='';
  const rows=D.single_auc, W=380,H=196,L=76,R=40,T=6,B=30,bh=19;
  const gap=(H-T-B-rows.length*bh)/(rows.length-1);
  const s=svg(W,H); box.appendChild(s);
  const X=v=>L+(v-0.5)/0.3*(W-L-R);
  [0.5,0.6,0.7,0.8].forEach(g=>{el('line',{x1:X(g),x2:X(g),y1:T,y2:H-B,class:g===0.5?'zl':'gl'},s);
    el('text',{x:X(g),y:H-B+15,'text-anchor':'middle',class:'ax'},s).textContent=g.toFixed(1);});
  el('text',{x:(L+W-R)/2,y:H-2,'text-anchor':'middle',class:'axt'},s).textContent='단독 AUC';
  rows.forEach((r,i)=>{
    const y=T+i*(bh+gap), drop=r.f==='n_free'||r.f==='free_ratio';
    el('text',{x:L-8,y:y+bh/2+4,'text-anchor':'end',class:'axt'},s).textContent=r.f;
    const rc=el('rect',{x:L,y:y,width:Math.max(X(r.a)-L,2),height:bh,rx:4,
      fill:drop?css('--s8'):css('--s1'),style:'cursor:pointer'},s);
    rc.addEventListener('mousemove',e=>tip(e,`<b>${r.f}</b><br>AUC <b>${fmt(r.a,3)}</b>
      ${drop?'<br><b>입력에서 제외</b> — 큐 상태 요약값':''}`));
    rc.addEventListener('mouseleave',untip);
    el('text',{x:X(r.a)+6,y:y+bh/2+4,class:'axt'},s).textContent=fmt(r.a,3);
  });
}

function fillTable(){
  const tb=document.querySelector('#tScen tbody'); if(!tb)return; tb.innerHTML='';
  D.scenarios.forEach(r=>{
    const tr=document.createElement('tr');
    const d=r['ΔT10'];
    tr.innerHTML=`<td>${r.scen}</td><td>${fmt(r.resp,2)}</td><td>${fmt(r.p90,2)}</td>
      <td>${fmt(r.T10*100,1)}%</td>
      <td style="color:${d<-5?'var(--good)':d<-0.5?'var(--text-primary)':'var(--text-muted)'}">${d>0?'+':''}${fmt(d,1)}%p</td>
      <td>${fmt(r.total,2)}</td><td>${fmt(r.floor,2)}</td><td>${fmt(r.loss,2)}</td>
      <td>${fmt(r.zero_loss*100,1)}%</td>`;
    tb.appendChild(tr);
  });
}

/* ---------- 경합 구조 탐색 ---------- */
function paneB(){
  const st=G.stations.map(s=>`<option value="${s.sid}">${s.name}</option>`).join('');
  return `
<div class="grid">
  <div class="card c12">
    <h2>구급차 점유와 차선배차 — DAILY ${G.daily} 시나리오, ${G.day+1}일차</h2>
    <p class="note">가로축은 하루 24시간, 세로는 구급차 19대. 막대는 <b>출동~복귀까지의 점유 구간</b>(평균 48분)이고,
      점은 신고 시각이다. <b>주황색 점 = 최근접 거점이 만차라 더 먼 거점이 배차된 건</b>.
      점을 클릭하면 그 시점에 최근접 거점을 점유하고 있던 케이스가 표시된다 — 이것이 <code>precedes</code> 엣지의 실체다.</p>
    <div class="ctl">
      <label class="chk"><input type="checkbox" id="onlyLoss"> 차선배차만 보기</label>
      <select id="fStation"><option value="">전체 거점</option>${st}</select>
      <span class="pill" id="statPill"></span>
    </div>
    <div class="legend">
      <span><i style="background:var(--s1)"></i>최근접 거점 배차 (y = 0)</span>
      <span><i style="background:var(--s2)"></i>차선 배차 (y &gt; 0)</span>
      <span><i style="background:var(--text-muted);opacity:.35"></i>점유 구간</span>
      <span><i style="background:var(--s8)"></i>선택 건을 막은 케이스</span>
    </div>
    <div id="cGantt"></div>
  </div>
  <div class="card c5">
    <h2>선택한 건</h2>
    <p class="note">점을 클릭하면 손실이 어떻게 생겼는지 분해해서 보여준다.</p>
    <div id="detail" style="color:var(--text-muted);font-size:12.5px">아직 선택된 건이 없다.</div>
  </div>
  <div class="card c7">
    <h2>이 화면이 보여주는 것</h2>
    <p class="note" style="margin-bottom:8px">
      <b>1. 대기는 0인데 경합은 있다.</b> 모든 구급차가 동시에 나가 있는 순간은 한 번도 없다.
      그런데도 최근접 거점은 자주 만차이고, 그때마다 더 먼 거점이 대신 간다. 손실은 대기가 아니라 거리로 나타난다.<br><br>
      <b>2. 차단자는 40분 전 환자다.</b> 점유가 평균 48분이라 한 번 출동하면 그 거점은 한동안 비어 있지 않다.
      이 시간 구조가 <code>precedes</code> 엣지의 근거이고, GNN이 배워야 하는 유일한 부분이다.<br><br>
      <b>3. 파급 차단이 존재한다.</b> 최근접이 만차라 2순위 거점으로 넘어간 환자가, 다시 그 2순위 거점을 점유해
      다음 환자를 막는다. 그래서 엣지를 최근접 거점만으로 만들면 12.5%를 놓친다.
    </p>
    <div class="warnbox">이 시나리오는 학습 데이터용 <b>고부하 조건(DAILY ${G.daily})</b>이다.
      현재 천안 추정치(DAILY 90)에서는 차선배차 비율이 17.6%로 훨씬 낮다.</div>
  </div>
</div>`;
}

let SEL=null;
function drawGantt(){
  const box=document.getElementById('cGantt'); if(!box)return; box.innerHTML='';
  const onlyLoss=document.getElementById('onlyLoss').checked;
  const fS=document.getElementById('fStation').value;
  const uids=[...new Set(G.busy.map(b=>b.uid))].sort();
  const W=1380,L=104,R=20,T=18,rowH=19,B=40;
  const H=T+uids.length*rowH+B;
  const s=svg(W,H); box.appendChild(s);
  const day0=G.day*86400-10800, span=86400+10800;
  const X=t=>L+Math.max(0,Math.min(1,(t-day0)/span))*(W-L-R);
  const Yr=u=>T+uids.indexOf(u)*rowH;
  for(let h=-3;h<=24;h+=3){
    const x=X(G.day*86400+h*3600);
    el('line',{x1:x,x2:x,y1:T-4,y2:H-B+4,class:h===0?'zl':'gl'},s);
    el('text',{x:x,y:H-B+20,'text-anchor':'middle',class:'ax'},s)
      .textContent=(h<0?'전날 '+(24+h):h)+'시';
  }
  el('rect',{x:L,y:T-4,width:X(G.day*86400)-L,height:H-B-T+8,fill:css('--text-muted'),opacity:.05},s);
  el('text',{x:L+4,y:T+8,class:'ax'},s).textContent='리드인 (차단자 추적용)';
  uids.forEach(u=>{
    el('text',{x:L-8,y:Yr(u)+13,'text-anchor':'end',class:'ax'},s).textContent=u;
    el('line',{x1:L,x2:W-R,y1:Yr(u)+rowH-1,y2:Yr(u)+rowH-1,stroke:css('--line'),'stroke-width':.5},s);
  });
  const byI={}; G.events.forEach(e=>byI[e.i]=e);
  const busyByI={}; G.busy.forEach(b=>busyByI[b.i]=b);
  // 점유 막대
  G.busy.forEach(b=>{
    const e=byI[b.i]; if(!e)return;
    if(fS && e.sid!==fS && e.min_sid!==fS)return;
    const isBlk=SEL && SEL.blockers.includes(b.i);
    el('rect',{x:X(b.t0),y:Yr(b.uid)+3,width:Math.max(X(b.t1)-X(b.t0),1.5),height:rowH-8,rx:2,
      fill:isBlk?css('--s8'):css('--text-muted'),opacity:isBlk?.75:.22},s);
  });
  // 신고 점
  let nLoss=0,nTot=0;
  G.events.forEach(e=>{
    if(fS && e.sid!==fS && e.min_sid!==fS)return;
    const loss=e.y>0.01; if(!e.lead){nTot++; if(loss)nLoss++;}
    if(onlyLoss && !loss)return;
    const b=busyByI[e.i]; if(!b)return;
    const c=el('circle',{cx:X(e.t),cy:Yr(b.uid)+rowH/2-1,r:loss?4.6:3.2,
      fill:loss?css('--s2'):css('--s1'),stroke:css('--surface-1'),'stroke-width':1.4,
      opacity:e.lead?.4:1,style:'cursor:pointer'},s);
    if(SEL&&SEL.i===e.i){c.setAttribute('r',7);c.setAttribute('stroke',css('--text-primary'));
      c.setAttribute('stroke-width',2.2);}
    c.addEventListener('mousemove',ev=>tip(ev,
      `<b>${e.region}</b> · ${hhmm(e.t)}<br>최근접 ${sname(e.min_sid)} (${fmt(e.me,2)}분)<br>
       실제 배차 ${sname(e.sid)} (${fmt(e.d1,2)}분)<br>
       손실 <b style="color:${loss?css('--s2'):'inherit'}">${fmt(e.y,2)}분</b>
       ${e.blockers.length?'<br>차단자 '+e.blockers.length+'건 — 클릭':''}`));
    c.addEventListener('mouseleave',untip);
    c.addEventListener('click',()=>{SEL=e;drawGantt();detail(e);});
  });
  // 차단 화살표
  if(SEL){
    const sb=busyByI[SEL.i];
    SEL.blockers.forEach(bi=>{
      const bb=busyByI[bi]; if(!bb||!sb)return;
      const x=X(SEL.t), y0=Yr(bb.uid)+rowH/2-1, y1=Yr(sb.uid)+rowH/2-1;
      el('path',{d:`M${x} ${y0} C ${x-34} ${y0}, ${x-34} ${y1}, ${x} ${y1}`,fill:'none',
        stroke:css('--s8'),'stroke-width':2,opacity:.9},s);
      el('circle',{cx:x,cy:y0,r:3.5,fill:css('--s8')},s);
    });
  }
  document.getElementById('statPill').textContent=
    `본 하루 ${nTot}건 중 차선배차 ${nLoss}건 (${fmt(nLoss/Math.max(nTot,1)*100,1)}%)`;
  document.getElementById('statPill').className='pill '+(nLoss/Math.max(nTot,1)>0.3?'bad':'');
}
function sname(sid){const s=G.stations.find(x=>x.sid===sid);return s?s.name.replace('119안전센터',''):sid;}
function hhmm(t){const s=Math.floor(t%86400);return String(Math.floor(s/3600)).padStart(2,'0')+':'+
  String(Math.floor(s%3600/60)).padStart(2,'0');}
function detail(e){
  const bl=e.blockers.map(i=>{const b=G.events.find(x=>x.i===i);
    return b?`<li>${b.region} (${hhmm(b.t)} 출동, ${sname(b.sid)})</li>`:`<li>창 밖 케이스 #${i}</li>`;}).join('');
  document.getElementById('detail').innerHTML=`
    <div style="color:var(--text-primary);font-weight:640;font-size:14px">${e.region} · ${hhmm(e.t)} · ${e.sev}</div>
    <dl class="kv">
      <dt>최근접 거점</dt><dd>${sname(e.min_sid)} — ${fmt(e.me,2)}분</dd>
      <dt>실제 배차</dt><dd>${sname(e.sid)} (${e.uid}) — ${fmt(e.d1,2)}분</dd>
      <dt>차선배차 손실 y</dt><dd style="color:${e.y>0.01?css('--s2'):'inherit'};font-weight:650">${fmt(e.y,2)}분</dd>
      <dt>대기시간</dt><dd>0.00분 <span style="color:var(--text-muted)">(가용 차량은 있었다)</span></dd>
    </dl>
    ${e.blockers.length?`<div style="margin-top:10px"><b style="font-size:12px">이 시점에 ${sname(e.min_sid)}을 점유 중이던 케이스</b>
      <ul style="margin:5px 0 0 16px;padding:0">${bl}</ul></div>`
    :'<div style="margin-top:10px;color:var(--text-muted)">최근접 거점이 비어 있었다 — 손실 없음.</div>'}`;
}

/* ---------- 부트 ---------- */
function render(){
  document.getElementById('paneC').innerHTML=paneC();
  document.getElementById('paneA').innerHTML=paneA();
  document.getElementById('paneB').innerHTML=paneB();
  drawSweep();drawDecomp();drawEdge();drawRank();drawLoad();drawCurve();drawLeak();fillTable();
  bootMap();
  document.getElementById('onlyLoss').addEventListener('change',drawGantt);
  document.getElementById('fStation').addEventListener('change',()=>{SEL=null;drawGantt();
    document.getElementById('detail').innerHTML='아직 선택된 건이 없다.';});
  if(!document.getElementById('paneB').classList.contains('hidden'))drawGantt();
}
render();
