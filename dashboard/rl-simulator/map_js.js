/* ========== 지도(MapLibre) · 유형 재분류 · 실시간 이송 애니메이션 ========== */
const AX=M.cfg.axes, AXK=AX.map(a=>a.k);
const AXLAB=Object.fromEntries(AX.map(a=>[a.k,a.label]));
const AXSHORT=Object.fromEntries(AX.map(a=>[a.k,a.short]));
const RG=Object.fromEntries(M.regions.map(r=>[r.name,r]));
let PLACED=[], OFF={ems:new Set(),hosp:new Set()};
let MODE='pan', SELR=null, PCT=0.70, GATE=0.667, PEN=4.8, PENH=0, DAILY=M.sim.daily, SEED=7;
const HGRADE={'권역응급의료센터':{stroke:1,cardiac:1,trauma:1},
              '지역응급의료센터':{stroke:1,cardiac:1,trauma:0},
              '지역응급의료기관':{stroke:0,cardiac:0,trauma:0}};
let MAP=null, OVC=null, SIM=null, PLAY=false, TNOW=0, SPEED=180, RAF=null, LASTTS=0;

const hav=(a,b)=>Math.hypot((a[0]-b[0])*88.9,(a[1]-b[1])*111.0);
function qlin(vals,p){const v=vals.filter(x=>x!==null&&x!==undefined&&!isNaN(x)).sort((a,b)=>a-b);
  if(!v.length)return NaN; const i=(v.length-1)*p,lo=Math.floor(i),hi=Math.ceil(i);
  return lo===hi?v[lo]:v[lo]+(v[hi]-v[lo])*(i-lo);}
function rng(seed){let s=seed>>>0||1;return()=>{s^=s<<13;s>>>=0;s^=s>>>17;s^=s<<5;s>>>=0;return s/4294967296;};}
function gauss(R){let u=0,v=0;while(!u)u=R();while(!v)v=R();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
function pick(R,arr,p){let x=R(),c=0;for(let i=0;i<arr.length;i++){c+=p[i];if(x<c)return arr[i];}return arr[arr.length-1];}

/* ---------- 축 원값 + 분류 (typology.py 와 동일 · 기준 31/31 재현) ---------- */
function activeStations(){
  const l=M.stations.filter(s=>!OFF.ems.has(s.id)).map(s=>({id:s.id,lon:s.lon,lat:s.lat,amb:s.amb,name:s.name,neu:0}));
  PLACED.filter(p=>p.kind==='ems').forEach((p,i)=>l.push({id:'NEW_E'+i,lon:p.lon,lat:p.lat,amb:p.amb||1,
    name:'[신설] 안전센터 '+(i+1),neu:1}));
  return l;}
function activeHospitals(){
  const l=M.hospitals.filter(h=>!OFF.hosp.has(h.id)).map(h=>Object.assign({},h,{neu:0}));
  PLACED.filter(p=>p.kind==='hosp').forEach((p,i)=>{
    const g=p.grade||'지역응급의료센터', c=HGRADE[g];
    l.push({id:'NEW_H'+i,lon:p.lon,lat:p.lat,name:'[신설] '+g+' '+(i+1),etype:g,inside:true,beds:6,
      cardiac:c.cardiac,stroke:c.stroke,trauma:c.trauma,neu:1});});
  return l;}
function etaER(st,r){
  if(st.neu)return Math.max(hav([st.lon,st.lat],[r.lon,r.lat])*M.cfg.detour/M.cfg.new_station_kmh*60,PEN);
  return (st.id in r.er)?r.er[st.id]:hav([st.lon,st.lat],[r.lon,r.lat])*M.cfg.detour/M.cfg.new_station_kmh*60;}
function etaRH(r,h){
  // ★ 4.8분 패널티는 119 거점 실측값이다. 병원에 전용할 근거가 없어 별도 값(기본 0)을 쓴다
  if(h.neu)return Math.max(hav([r.lon,r.lat],[h.lon,h.lat])*M.cfg.detour/M.cfg.new_hospital_kmh*60,PENH);
  return (h.id in r.rh)?r.rh[h.id]:hav([r.lon,r.lat],[h.lon,h.lat])*M.cfg.detour/M.cfg.new_hospital_kmh*60;}
function axisValues(){
  const ST=activeStations(), HO=activeHospitals(), out={};
  M.regions.forEach(r=>{
    const er=ST.map(s=>etaER(s,r)), rh=HO.map(h=>etaRH(r,h)).sort((a,b)=>a-b);
    out[r.name]={dispatch:er.length?Math.min.apply(null,er):null, transport:rh.length?rh[0]:null,
      backup:rh.length>1?rh[1]-rh[0]:null, traffic:r.traffic, demand:r.demand};});
  return out;}
function classify(V,pct,gpct){
  pct=pct===undefined?PCT:pct; gpct=gpct===undefined?GATE:gpct;
  const names=Object.keys(V),cuts={},Z={};
  AXK.forEach(k=>{
    cuts[k]=qlin(names.map(n=>V[n][k]),pct);
    const v=names.map(n=>V[n][k]).filter(x=>x!==null);
    const mu=v.reduce((a,b)=>a+b,0)/v.length, sd=Math.sqrt(v.reduce((a,b)=>a+(b-mu)*(b-mu),0)/v.length);
    names.forEach(n=>{(Z[n]=Z[n]||{})[k]=V[n][k]===null?null:(sd===0?0:(V[n][k]-mu)/sd);});});
  const sev={}; names.forEach(n=>sev[n]=AXK.reduce((a,k)=>a+Math.max(Z[n][k]===null?0:Z[n][k],0),0));
  const gate=qlin(names.map(n=>sev[n]),gpct), res={};
  names.forEach(n=>{
    const hit=AXK.filter(k=>V[n][k]!==null&&V[n][k]>=cuts[k]);
    const zz=AXK.filter(k=>Z[n][k]!==null);
    const lead=zz.length?zz.reduce((a,k)=>Z[n][k]>Z[n][a]?k:a,zz[0]):null;
    let t;
    if(!hit.length)t=M.cfg.none_label;
    else if(hit.length===1)t=AXLAB[hit[0]];
    else if(hit.length>=M.cfg.multi_min_flags)t=M.cfg.multi_label;
    else if(hit.length===M.cfg.gate_at_flags)t=sev[n]>=gate?M.cfg.multi_label:AXLAB[lead];
    else t=AXLAB[lead];
    res[n]={vtype:t,hit:hit,lead:lead,sev:sev[n],z:Z[n],v:V[n]};});
  return {res:res,cuts:cuts,gate:gate};}
const BASE=classify(axisValues());
const SHORT=t=>t===M.cfg.none_label?'안정':t===M.cfg.multi_label?'복합':(AX.find(a=>a.label===t)||{short:'?'}).short;

/* ---------- 하루치 시뮬 (real_build.py 규칙) ---------- */
function runSim(){
  const S=M.sim, R=rng(SEED*7919+Math.round(DAILY)*31), ST=activeStations(), HO=activeHospitals();
  const wsum=M.regions.reduce((a,r)=>a+r.w,0), cases=[];
  const HRS=[]; for(let i=0;i<24;i++)HRS.push(i);
  M.regions.forEach(r=>{
    const lam=DAILY*r.w/wsum, n=Math.max(0,Math.floor(lam+gauss(R)*Math.sqrt(Math.max(lam,0.5))));
    for(let i=0;i<n;i++){
      const hr=pick(R,HRS,S.hrp), th=R()*2*Math.PI, dd=r.rmax*Math.sqrt(R());
      cases.push({r:r.name,t:hr*3600+R()*3600,sev:pick(R,S.sev,S.sevp),
        olon:r.lon+dd*Math.cos(th)/88.9, olat:r.lat+dd*Math.sin(th)/111.0,
        intra:dd*S.detour/r.ikmh*3600, scene:S.scene[0]+R()*(S.scene[1]-S.scene[0]),
        rolls:Array.from({length:8},()=>R())});}});
  cases.sort((a,b)=>a.t-b.t);
  const units=[], byId={};
  ST.forEach(s=>{byId[s.id]=s;for(let k=0;k<s.amb;k++)
    units.push({uid:s.id+'-'+(k+1),sid:s.id,name:s.name,lon:s.lon,lat:s.lat,free:0});});
  const ev=[];
  cases.forEach((c,ci)=>{
    const r=RG[c.r];
    const free=units.filter(u=>u.free<=c.t), pool=free.length?free:units;
    const cost=u=>etaER(byId[u.sid],r)*60+Math.max(0,u.free-c.t);
    const u=pool.reduce((a,b)=>cost(b)<cost(a)?b:a);
    const wait=Math.max(0,u.free-c.t), d1=etaER(byId[u.sid],r)*60;
    const ord=ST.map(s=>({sid:s.id,eta:etaER(s,r)*60,amb:s.amb})).sort((a,b)=>a.eta-b.eta);
    const cand3=ord.slice(0,3);
    while(cand3.length<3&&cand3.length)cand3.push(cand3[cand3.length-1]);
    const minEta=ord[0].eta;
    const tDisp=c.t+S.prep_sec+wait, tScene=tDisp+d1+c.intra, tDep=tScene+c.scene;
    let cand=HO.filter(h=>c.sev==='other'||h[c.sev]===1);
    if(!cand.length)cand=HO.slice();
    cand.sort((a,b)=>etaRH(r,a)-etaRH(r,b));
    let ch=null,rej=0;
    for(let k=0;k<cand.length;k++){if(c.rolls[Math.min(k,7)]<S.reject){rej++;continue;}ch=cand[k];break;}
    ch=ch||cand[0];
    const near=HO.reduce((a,b)=>etaRH(r,b)<etaRH(r,a)?b:a);   // 기능 무시 최근접
    let why='최근접';
    if(ch!==near){
      const capable=(c.sev==='other')||near[c.sev]===1;
      why=!capable?('중증도 미대응 ('+({cardiac:'심근경색 재관류',stroke:'뇌졸중',trauma:'중증외상'}[c.sev])+')')
          :(rej>0?'수용거부 '+rej+'회':'기타');}
    const hc=cand.slice(0,3).map(h=>({id:h.id,eta:etaRH(r,h)*60,
      cardiac:h.cardiac,stroke:h.stroke,trauma:h.trauma,beds:h.beds,inside:h.inside?1:0}));
    while(hc.length&&hc.length<3)hc.push(hc[hc.length-1]);
    const d2=etaRH(r,ch)*60, tHosp=tDep+d2;
    const back=hav([ch.lon,ch.lat],[u.lon,u.lat])*S.detour/S.back_kmh*3600;
    const tFree=tHosp+S.handover+back;
    u.free=tFree;
    ev.push({i:ci,r:c.r,sev:c.sev,olon:c.olon,olat:c.olat,uid:u.uid,uname:u.name,
      ublon:u.lon,ublat:u.lat,hlon:ch.lon,hlat:ch.lat,hname:ch.name,hout:!ch.inside,rej:rej,
      why:why,nearName:near.name,nearEta:+etaRH(r,near).toFixed(2),chEta:+etaRH(r,ch).toFixed(2),
      nlon:near.lon,nlat:near.lat,
      t:c.t,tDisp:tDisp,tScene:tScene,tDep:tDep,tHosp:tHosp,tFree:tFree,
      resp:tScene-c.t,loss:(d1-minEta)/60,lim:(r.rural?15:10)*60,
      intra:c.intra,cand:cand3,c1e:cand3[0].eta,c2e:cand3[1]?cand3[1].eta:cand3[0].eta,
      minAmb:cand3[0].amb, hcand:hc});});
  return {ev:ev,units:units.length};}
function simStats(t){
  if(!SIM)return {n:0,total:0,act:0,resp:0,over:0,loss:0,out:0};
  const done=SIM.ev.filter(e=>e.tScene<=t), act=SIM.ev.filter(e=>e.t<=t&&e.tFree>t);
  const rs=done.map(e=>e.resp);
  return {n:done.length,total:SIM.ev.length,act:act.length,
    resp:rs.length?rs.reduce((a,b)=>a+b,0)/rs.length/60:0,
    over:done.length?done.filter(e=>e.resp>e.lim).length/done.length:0,
    loss:done.length?done.filter(e=>e.loss>0.01).length/done.length:0,
    out:done.length?done.filter(e=>e.hout).length/done.length:0,
    byp:done.length?done.filter(e=>e.why!=='최근접').length/done.length:0};}

/* ---------- 패널 ---------- */
function paneC(){
  return `
<div class="grid">
  <div class="card c8">
    <h2>지도에 거점을 놓고, 하루를 돌려 본다</h2>
    <p class="note">지도를 클릭해 <b>119안전센터</b>·<b>응급의료기관</b>을 놓으면 5개 축을 다시 계산해
      31개 행정동 유형을 즉시 재분류한다. 기존 시설을 클릭하면 <b>없앤 경우</b>도 볼 수 있다.
      <b>▶ 재생</b>을 누르면 실측 시간대 분포로 환자가 발생하고 구급차가 출동·이송하는 하루가 돌아간다 —
      배치를 바꾸면 그 자리에서 다시 계산해 다시 돌린다.</p>
    <div class="ctl">
      <button class="mbtn" id="mEms" onclick="setMode('ems')">＋ 안전센터</button>
      <select id="newAmb" title="신설 안전센터 구급차 대수">
        <option value="1">1대</option><option value="2">2대</option><option value="3">3대</option></select>
      <button class="mbtn" id="mHosp" onclick="setMode('hosp')">＋ 응급의료기관</button>
      <select id="newGrade" title="신설 병원 등급 — 중증 대응 가능 범위가 달라진다">
        <option value="권역응급의료센터">권역센터 (심·뇌·외상)</option>
        <option value="지역응급의료센터" selected>지역센터 (심·뇌)</option>
        <option value="지역응급의료기관">지역기관 (중증 불가)</option></select>
      <button class="mbtn" onclick="preset()">정책 A 재현</button>
      <button class="mbtn" onclick="resetAll()">초기화</button>
      <span style="flex:1"></span>
      <label class="chk" title="119 거점 실측 분포(0.65~10.12분)에서 온 값">거점 입지 패널티
        <input type="range" id="pen" min="0" max="100" step="5" value="48" style="width:82px"><b id="penLab">4.8분</b></label>
      <label class="chk" title="병원 입지 패널티는 근거가 없어 기본 0이다">병원 패널티
        <input type="range" id="penh" min="0" max="60" step="5" value="0" style="width:66px"><b id="penhLab">0.0분</b></label>
      <label class="chk">취약 임계 상위
        <input type="range" id="pct" min="60" max="80" step="5" value="70" style="width:86px"><b id="pctLab">30%</b></label>
    </div>
    <div class="ctl" style="background:var(--surface-2);border-radius:9px;padding:8px 10px">
      <button class="mbtn on" id="playBtn" onclick="togglePlay()">▶ 재생</button>
      <input type="range" id="tSlider" min="0" max="86340" step="60" value="0" style="flex:1;min-width:150px">
      <b id="clock" style="font-variant-numeric:tabular-nums;min-width:48px">00:00</b>
      <select id="spd" onchange="SPEED=+this.value">
        <option value="60">×60</option><option value="180" selected>×180</option>
        <option value="600">×600</option><option value="1800">×1800</option></select>
      <label class="chk">일 발생 <input type="range" id="daily" min="40" max="300" step="10" value="90"
        style="width:90px"><b id="dailyLab">90건</b></label>
    </div>
    <div id="kpi" class="kpirow"></div>
    <div id="mapWrap" style="position:relative;height:600px;border-radius:10px;overflow:hidden;border:1px solid var(--line)">
      <div id="mlmap" style="position:absolute;inset:0"></div>
      <canvas id="ovc" style="position:absolute;inset:0;pointer-events:none"></canvas>
      <div id="maptip" class="maptip"></div>
    </div>
    <div class="legend" style="margin-top:9px">
      <span><i style="background:var(--m0)"></i>해당 축 0개</span><span><i style="background:var(--m1)"></i>1개</span>
      <span><i style="background:var(--m2)"></i>2개</span><span><i style="background:var(--m3)"></i>3개+</span>
      <span><i style="background:transparent;border:2px solid var(--s2)"></i>유형이 바뀐 동</span>
      <span><i style="background:var(--s1);border-radius:50%"></i>안전센터</span>
      <span><i style="background:var(--s8);border-radius:50%"></i>응급의료기관</span>
      <span><i style="background:var(--warn);border-radius:50%"></i>발생 환자</span>
      <span><i style="background:var(--s3);border-radius:2px"></i>구급차 — 현장으로</span>
      <span><i style="background:var(--s7);border-radius:2px"></i>구급차 — 환자 이송</span>
      <span><i style="background:var(--s7);border-radius:50%;box-shadow:0 0 0 2px var(--s7) inset"></i>병원 인계 중</span>
      <span><i style="background:var(--text-muted);border-radius:2px"></i>거점 복귀</span>
      <span style="color:var(--s2)">✕ 지나친 최근접 병원</span>
    </div>
    <p class="note" style="margin:8px 0 0">기존 시설 ETA는 <b>Kakao 실측 라우팅</b>, 새로 놓은 시설은
      <b>직선거리×1.3 근사 + 입지 패널티 하한</b>이다. 발생은 119 구급상황 <b>실측 시간대 분포</b>(19시 최다·04시 최소)와
      읍면 실측 건수를 따르고, <b>일 발생 건수는 가정값</b>이다. 경로선은 직선 근사이므로 도로 형상이 아니다.</p>
  </div>

  <div class="card c4">
    <h2>유형 분포</h2>
    <p class="note">배치 전 → 후.</p>
    <div id="cDist"></div>
    <div id="changeBox" style="margin-top:12px"></div>
  </div>

  <div class="card c4" style="grid-column:span 4">
    <h2>이송 로그 <span class="pill">왜 저 병원으로 갔나</span></h2>
    <p class="note">최근 도착 7건. <b>최근접 병원을 지나친 경우 사유를 표시</b>한다 —
      중증도 미대응(그 병원이 심·뇌·외상을 못 받음) 또는 수용거부(18% 확률).</p>
    <div id="tlog"></div>
  </div>

  <div class="card c8">
    <h2>GNN 대체모델 — <span class="pill">지금 이 배치에서 실제로 추론한다</span></h2>
    <p class="note">브라우저 안에서 학습된 GNN을 돌린다. 배치를 바꾸면 그래프(환자·거점 노드, precedes·can_reach 엣지)를
      다시 만들고 <b>케이스마다 차선배차 손실 y = d1 − min_eta 를 예측</b>한 뒤 시뮬 실제값과 비교한다.
      대각선에 붙을수록 대체모델이 시뮬을 재현한 것이다. 주황 = 실제로 차선배차가 일어난 건.</p>
    <div id="gnnKpi" class="kpirow"></div>
    <div class="ctl" style="margin:0 0 8px">
      <label class="chk"><input type="checkbox" id="calSw" onchange="CAL=this.checked;drawGnnPanel();">
        평균 편향 보정 (×0.885)</label>
      <span class="note" style="margin:0">test 451,508건에서 평균 예측이 실제의 <b>1.13배</b>였다.
        보정은 평균만 맞출 뿐 아래 파란 열은 그대로다.</span>
    </div>
    <div id="cGnn"></div>
    <div class="warnbox" style="margin-top:10px"><b>파란 열을 읽는 법.</b>
      왼쪽 파란 띠는 <b>시뮬 실제 손실이 정확히 0</b>인 케이스다 — 최근접 거점의 구급차가 그대로 출동했다는 뜻이지
      <b>빨리 도착했다는 뜻이 아니다</b>. 이 열에는 현장 도착 9.4분(부성1동)짜리도 들어 있다.
      GNN이 이 열에 큰 값을 얹는 것은 시뮬 오류가 아니라 <b>ŷ = P(차선배차) × 크기</b> 라는 곱 형태 때문이다 —
      P가 0이 아니면 실제 0에도 예측이 새어 나간다. test에서 <b>예측 총량의 35%가 실제 0인 케이스에 얹혀 있다</b>.</div>
    <p class="note" style="margin:8px 0 0">가중치는 500 시나리오 · 222만 건으로 학습한 것을 그대로 내장했다.
      <b>이 타깃은 시뮬레이터의 결정론적 함수</b>이므로 "응급 지연 예측"이 아니라 <b>시뮬을 학습 가능한 형태로
      대체할 수 있는가의 검증</b>이다. 시뮬 자체가 밀리초 단위라 속도 이득은 없다.</p>
  </div>
  <div class="card c4">
    <h2>축별 취약 판정 (요약)</h2>
    <p class="note">아래 5개 소지도와 같은 기준.</p>
    <div id="cutSummary" class="note" style="margin:0"></div>
  </div>

  <div class="card c12">
    <h2>축별 취약 판정 <span class="pill" id="cutPill"></span></h2>
    <p class="note">거점을 놓으면 <b>출동</b>이, 병원을 놓으면 <b>이송·대체</b>가 움직인다.
      교통 축은 <b>31곳 중 19곳만 유효</b>(면 10곳 + 문성·일봉 미산출) — 회색은 자료 없음이지 안전이 아니다.</p>
    <div id="cFacets" style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px"></div>
  </div>

  <div class="card c12">
    <h2>행정동 31곳 <span class="pill" id="selPill">행을 클릭하면 지도가 이동한다</span></h2>
    <div style="overflow:auto;max-height:340px"><table id="tReg"><thead><tr>
      <th>행정동</th><th>구분</th><th>인구</th><th>유형(현재)</th><th>유형(배치 후)</th><th>해당 축</th>
      <th>출동(분)</th><th>이송(분)</th><th>대체(분)</th><th>교통</th><th>수요</th><th>심각도</th>
    </tr></thead><tbody></tbody></table></div>
  </div>
</div>`;}

function setMode(m){MODE=MODE===m?'pan':m;
  ['mEms','mHosp'].forEach(id=>{const b=document.getElementById(id);if(b)b.classList.remove('on');});
  if(MODE!=='pan'){const b=document.getElementById(MODE==='ems'?'mEms':'mHosp');if(b)b.classList.add('on');}
  if(MAP)MAP.getCanvas().style.cursor=MODE==='pan'?'':'crosshair';}
function preset(){PLACED=['수신면','동면','부성1동'].map(n=>({kind:'ems',lon:RG[n].lon,lat:RG[n].lat,amb:1}));
  OFF={ems:new Set(),hosp:new Set()};refresh(true);}
function resetAll(){PLACED=[];OFF={ems:new Set(),hosp:new Set()};SELR=null;setMode('pan');refresh(true);}
function togglePlay(){PLAY=!PLAY;
  const b=document.getElementById('playBtn');
  b.textContent=PLAY?'❚❚ 일시정지':'▶ 재생'; b.classList.toggle('on',!PLAY);
  if(PLAY){LASTTS=performance.now();loop();}else if(RAF)cancelAnimationFrame(RAF);}

/* ---------- MapLibre ---------- */
function basemapStyle(){
  const dark=document.documentElement.getAttribute('data-theme')==='dark';
  const u=dark?'dark_all':'light_all';
  return {version:8,sources:{bg:{type:'raster',
      tiles:['a','b','c'].map(s=>'https://'+s+'.basemaps.cartocdn.com/'+u+'/{z}/{x}/{y}@2x.png'),
      tileSize:256,attribution:'© OpenStreetMap · © CARTO'}},
    layers:[{id:'bg',type:'raster',source:'bg'}]};}
function dongFC(C){
  return {type:'FeatureCollection',features:M.geo.features.map(f=>{
    const n=f.properties.name,r=C.res[n],b=BASE.res[n];
    return {type:'Feature',geometry:f.geometry,properties:{name:n,nflag:r.hit.length,
      changed:r.vtype!==b.vtype?1:0,label:SHORT(r.vtype)}};})};}
function facFC(){
  const fs=[];
  activeStations().forEach(s=>fs.push({type:'Feature',properties:{kind:'ems',id:s.id,name:s.name,
    amb:s.amb,neu:s.neu,off:0},geometry:{type:'Point',coordinates:[s.lon,s.lat]}}));
  M.stations.filter(s=>OFF.ems.has(s.id)).forEach(s=>fs.push({type:'Feature',
    properties:{kind:'ems',id:s.id,name:s.name,amb:s.amb,neu:0,off:1},
    geometry:{type:'Point',coordinates:[s.lon,s.lat]}}));
  activeHospitals().filter(h=>h.inside).forEach(h=>fs.push({type:'Feature',properties:{kind:'hosp',id:h.id,
    name:h.name,etype:h.etype,neu:h.neu,off:0},geometry:{type:'Point',coordinates:[h.lon,h.lat]}}));
  M.hospitals.filter(h=>OFF.hosp.has(h.id)&&h.inside).forEach(h=>fs.push({type:'Feature',
    properties:{kind:'hosp',id:h.id,name:h.name,etype:h.etype,neu:0,off:1},
    geometry:{type:'Point',coordinates:[h.lon,h.lat]}}));
  return {type:'FeatureCollection',features:fs};}
function initMap(){
  const C=classify(axisValues());
  MAP=new maplibregl.Map({container:'mlmap',style:basemapStyle(),center:[127.14,36.83],zoom:10.1});
  MAP.addControl(new maplibregl.NavigationControl({showCompass:false}),'top-right');
  MAP.on('load',function(){
    MAP.addSource('dong',{type:'geojson',data:dongFC(C)});
    MAP.addLayer({id:'dong-fill',type:'fill',source:'dong',paint:{'fill-color':
      ['case',['==',['get','nflag'],0],css('--m0'),['==',['get','nflag'],1],css('--m1'),
              ['==',['get','nflag'],2],css('--m2'),css('--m3')],'fill-opacity':0.55}});
    MAP.addLayer({id:'dong-line',type:'line',source:'dong',paint:{
      'line-color':['case',['==',['get','changed'],1],css('--s2'),css('--surface-1')],
      'line-width':['case',['==',['get','changed'],1],3,1]}});
    MAP.addSource('fac',{type:'geojson',data:facFC()});
    MAP.addLayer({id:'fac-hosp',type:'circle',source:'fac',filter:['==',['get','kind'],'hosp'],
      paint:{'circle-radius':['case',['==',['get','neu'],1],9,6],
        'circle-color':['case',['==',['get','off'],1],'rgba(0,0,0,0)',css('--s8')],
        'circle-stroke-color':['case',['==',['get','neu'],1],'#ffffff',css('--s8')],
        'circle-stroke-width':2}});
    MAP.addLayer({id:'fac-ems',type:'circle',source:'fac',filter:['==',['get','kind'],'ems'],
      paint:{'circle-radius':['case',['==',['get','neu'],1],9,5.5],
        'circle-color':['case',['==',['get','off'],1],'rgba(0,0,0,0)',css('--s1')],
        'circle-stroke-color':['case',['==',['get','neu'],1],'#ffffff',css('--s1')],
        'circle-stroke-width':2}});
    ['fac-ems','fac-hosp'].forEach(function(L){
      MAP.on('click',L,function(e){
        e.preventDefault();
        const p=e.features[0].properties, c=e.features[0].geometry.coordinates;
        if(String(p.neu)==='1'){PLACED=PLACED.filter(x=>Math.abs(x.lon-c[0])>1e-9||Math.abs(x.lat-c[1])>1e-9);}
        else{const S=p.kind==='ems'?OFF.ems:OFF.hosp; S.has(p.id)?S.delete(p.id):S.add(p.id);}
        refresh(true);});
      MAP.on('mousemove',L,function(e){
        MAP.getCanvas().style.cursor='pointer';
        const p=e.features[0].properties;
        showMapTip(e,'<b>'+p.name+'</b><br>'+(p.kind==='ems'?'구급차 '+p.amb+'대':p.etype)+'<br>'+
          (String(p.off)==='1'?'<b style="color:var(--s2)">제외됨</b> — 클릭해 복구':
           String(p.neu)==='1'?'클릭하면 제거':'클릭하면 없앤 경우를 본다'));});
      MAP.on('mouseleave',L,function(){MAP.getCanvas().style.cursor=MODE==='pan'?'':'crosshair';hideMapTip();});
    });
    MAP.on('mousemove','dong-fill',function(e){
      if(MAP.queryRenderedFeatures(e.point,{layers:['fac-ems','fac-hosp']}).length)return;
      const n=e.features[0].properties.name, C2=classify(axisValues());
      showMapTip(e,regTip(n,C2.res[n],BASE.res[n],C2.cuts));});
    MAP.on('mouseleave','dong-fill',hideMapTip);
    MAP.on('click','dong-fill',function(e){
      if(e.defaultPrevented||MODE!=='pan')return;
      SELR=e.features[0].properties.name;refresh(false);});
    MAP.on('click',function(e){
      if(e.defaultPrevented||MODE==='pan')return;
      PLACED.push({kind:MODE,lon:+e.lngLat.lng.toFixed(5),lat:+e.lngLat.lat.toFixed(5),
        amb:+(document.getElementById('newAmb')||{value:1}).value,
        grade:(document.getElementById('newGrade')||{value:'지역응급의료센터'}).value});
      refresh(true);});
    sizeCanvas(); refresh(true);});
  MAP.on('move',drawOverlay);
  MAP.on('resize',function(){sizeCanvas();drawOverlay();});}
function showMapTip(e,html){
  const el=document.getElementById('maptip'); if(!el)return;
  el.innerHTML=html; el.style.opacity=1;
  const w=document.getElementById('mapWrap').getBoundingClientRect();
  let x=e.point.x+14,y=e.point.y+14;
  if(x+280>w.width)x=Math.max(4,e.point.x-286);
  if(y+170>w.height)y=Math.max(4,e.point.y-170);
  el.style.left=x+'px'; el.style.top=y+'px';}
function hideMapTip(){const el=document.getElementById('maptip');if(el)el.style.opacity=0;}
function sizeCanvas(){
  const w=document.getElementById('mapWrap'); if(!w||!OVC)return;
  const dpr=window.devicePixelRatio||1;
  OVC.width=w.clientWidth*dpr; OVC.height=w.clientHeight*dpr;
  OVC.style.width=w.clientWidth+'px'; OVC.style.height=w.clientHeight+'px';
  OVC.getContext('2d').setTransform(dpr,0,0,dpr,0,0);}

/* ---------- 애니메이션 ---------- */
function lerp(a,b,u){return [a[0]+(b[0]-a[0])*u,a[1]+(b[1]-a[1])*u];}
function drawOverlay(){
  if(!OVC||!MAP||!SIM)return;
  const dpr=window.devicePixelRatio||1;
  const g=OVC.getContext('2d'), W=OVC.width/dpr, H=OVC.height/dpr;
  g.clearRect(0,0,W,H);
  const t=TNOW, P=function(c){const p=MAP.project(c);return [p.x,p.y];};
  const MUT=css('--text-muted');
  SIM.ev.forEach(function(e){
    if(e.t>t||e.tFree<t)return;
    if(t<e.tDep){
      const o=P([e.olon,e.olat]), ph=(t-e.t)%45/45;
      g.beginPath();g.arc(o[0],o[1],9+ph*10,0,7);
      g.strokeStyle=css('--warn');g.globalAlpha=Math.max(0,0.5-ph*0.5);g.lineWidth=2;g.stroke();g.globalAlpha=1;
      g.beginPath();g.arc(o[0],o[1],4.5,0,7);g.fillStyle=css('--warn');g.fill();
      g.strokeStyle='#fff';g.lineWidth=1.4;g.stroke();}
    var pos=null,col=null,tgt=null;
    if(t>=e.tDisp&&t<e.tScene){pos=lerp([e.ublon,e.ublat],[e.olon,e.olat],(t-e.tDisp)/(e.tScene-e.tDisp));
      col=css('--s3');tgt=[e.olon,e.olat];}
    else if(t>=e.tScene&&t<e.tDep){pos=[e.olon,e.olat];col=css('--s3');}
    else if(t>=e.tDep&&t<e.tHosp){pos=lerp([e.olon,e.olat],[e.hlon,e.hlat],(t-e.tDep)/(e.tHosp-e.tDep));
      col=css('--s7');tgt=[e.hlon,e.hlat];
      if(e.why!=='최근접'){          // 지나친 최근접 병원에 X 표시
        const n=P([e.nlon,e.nlat]);
        g.strokeStyle=css('--s2');g.lineWidth=2;g.globalAlpha=.9;
        g.beginPath();g.moveTo(n[0]-5,n[1]-5);g.lineTo(n[0]+5,n[1]+5);
        g.moveTo(n[0]+5,n[1]-5);g.lineTo(n[0]-5,n[1]+5);g.stroke();g.globalAlpha=1;}}
    else if(t>=e.tHosp&&t<e.tHosp+M.sim.handover){pos=[e.hlon,e.hlat];col=css('--s7');}
    else if(t>=e.tHosp+M.sim.handover&&t<e.tFree){
      pos=lerp([e.hlon,e.hlat],[e.ublon,e.ublat],(t-e.tHosp-M.sim.handover)/Math.max(1,e.tFree-e.tHosp-M.sim.handover));
      col=MUT;}
    if(!pos)return;
    const q=P(pos);
    if(tgt){const z=P(tgt);g.beginPath();g.moveTo(q[0],q[1]);g.lineTo(z[0],z[1]);
      g.strokeStyle=col;g.globalAlpha=.4;g.lineWidth=1.5;g.setLineDash([4,4]);g.stroke();
      g.setLineDash([]);g.globalAlpha=1;}
    if(e.loss>0.5&&col===css('--s3')){g.beginPath();g.arc(q[0],q[1],9.5,0,7);
      g.strokeStyle=css('--s2');g.lineWidth=2;g.stroke();}
    if(t>=e.tHosp&&t<e.tHosp+M.sim.handover){    // 병원 인계 중
      g.beginPath();g.arc(q[0],q[1],11,0,7);g.strokeStyle=css('--s7');
      g.globalAlpha=.75;g.lineWidth=2;g.setLineDash([3,3]);g.stroke();
      g.setLineDash([]);g.globalAlpha=1;}
    const w=col===MUT?5:8;                       // 구급차는 사각형 — 시설(원)과 구분
    g.beginPath();
    if(g.roundRect)g.roundRect(q[0]-w/2,q[1]-w/2,w,w,2); else g.rect(q[0]-w/2,q[1]-w/2,w,w);
    g.fillStyle=col;g.fill();g.strokeStyle='#fff';g.lineWidth=1.6;g.stroke();});}
function loop(){
  RAF=requestAnimationFrame(loop);
  const now=performance.now(), dt=(now-LASTTS)/1000; LASTTS=now;
  if(!PLAY)return;
  TNOW=(TNOW+dt*SPEED)%86400;
  const sl=document.getElementById('tSlider'); if(sl)sl.value=Math.floor(TNOW);
  tickClock(); drawOverlay();}
function tickClock(){
  const c=document.getElementById('clock'); if(!c)return;
  c.textContent=String(Math.floor(TNOW/3600)).padStart(2,'0')+':'+String(Math.floor(TNOW%3600/60)).padStart(2,'0');
  const s=simStats(TNOW);
  document.getElementById('kpi').innerHTML=
    '<div><span class="klabel">발생</span><b>'+s.n+'</b> / '+s.total+'건</div>'+
    '<div><span class="klabel">출동 중</span><b>'+s.act+'</b>대</div>'+
    '<div><span class="klabel">평균 응답</span><b>'+s.resp.toFixed(2)+'</b>분</div>'+
    '<div><span class="klabel">기준 초과</span><b class="'+(s.over>0.4?'hi':'')+'">'+(s.over*100).toFixed(1)+'</b>%</div>'+
    '<div><span class="klabel">차선 배차</span><b class="'+(s.loss>0.3?'hi':'')+'">'+(s.loss*100).toFixed(1)+'</b>%</div>'+
    '<div><span class="klabel">관외 이송</span><b>'+(s.out*100).toFixed(1)+'</b>%</div>'+
    '<div><span class="klabel">최근접 병원 우회</span><b class="'+(s.byp>0.3?'hi':'')+'">'+
      (s.byp*100).toFixed(1)+'</b>%</div>';
  transLog();}
function transLog(){
  const box=document.getElementById('tlog'); if(!box||!SIM)return;
  const done=SIM.ev.filter(e=>e.tHosp<=TNOW).sort((a,b)=>b.tHosp-a.tHosp).slice(0,7);
  if(!done.length){box.innerHTML='<div class="note" style="margin:0">아직 도착한 이송이 없다.</div>';return;}
  box.innerHTML=done.map(function(e){
    const byp=e.why!=='최근접';
    return '<div style="font-size:12px;padding:5px 0;border-bottom:1px solid var(--line)">'+
      '<b>'+hhmm2(e.tHosp)+'</b> '+e.r+' · '+({cardiac:'심장',stroke:'뇌',trauma:'외상',other:'기타'}[e.sev])+
      '<div style="color:var(--text-secondary);margin-top:1px">→ '+e.hname.slice(0,22)+
      ' <span style="opacity:.65">'+e.chEta.toFixed(1)+'분</span></div>'+
      (byp?'<div style="color:var(--s2);font-size:11px;margin-top:1px">최근접 '+e.nearName.slice(0,16)+
        '('+e.nearEta.toFixed(1)+'분) 우회 — '+e.why+'</div>':'')+
      '</div>';}).join('');}
function hhmm2(t){return String(Math.floor(t/3600)%24).padStart(2,'0')+':'+String(Math.floor(t%3600/60)).padStart(2,'0');}

/* ---------- 갱신 ---------- */
function refresh(resim){
  const V=axisValues(), C=classify(V);
  if(resim||!SIM)SIM=runSim();
  if(MAP&&MAP.getSource&&MAP.getSource('dong')){
    MAP.getSource('dong').setData(dongFC(C));
    MAP.getSource('fac').setData(facFC());}
  const changed=M.regions.filter(r=>C.res[r.name].vtype!==BASE.res[r.name].vtype).map(function(r){
    const a=BASE.res[r.name],b=C.res[r.name];
    const same=AXK.every(function(k){const x=b.v[k],y=a.v[k];
      return (x===null&&y===null)||(x!==null&&y!==null&&Math.abs(x-y)<1e-9);});
    return [r.name,a.vtype,b.vtype,same];});
  drawDist(C,changed); drawFacets(V,C); fillReg(V,C); tickClock(); drawOverlay();
  try{drawGnnPanel();}catch(err){const b=document.getElementById('cGnn');
    if(b)b.innerHTML='<div class="note">GNN 추론 실패: '+err.message+'</div>';}
  const cs=document.getElementById('cutSummary');
  if(cs)cs.innerHTML=AXK.map(k=>'<div style="padding:2px 0"><b>'+AXSHORT[k]+'</b> 기준 '+
    C.cuts[k].toFixed(2)+' · 취약 '+Object.keys(C.res).filter(n=>C.res[n].hit.indexOf(k)>=0).length+
    '곳</div>').join('');
  document.getElementById('cutPill').textContent=AXK.map(k=>AXSHORT[k]+' '+C.cuts[k].toFixed(2)).join('  ·  ');
  document.getElementById('selPill').textContent=SELR?SELR+' 선택됨':'행을 클릭하면 지도가 이동한다';}
function regTip(n,r,b,cuts){
  const R=RG[n];
  const rows=AXK.map(function(k){const v=r.v[k],f=r.hit.indexOf(k)>=0;
    return '<tr><td style="text-align:left;padding:1px 6px 1px 0;border:0">'+AXSHORT[k]+'</td>'+
      '<td style="padding:1px 0;border:0">'+(v===null?'<span style="opacity:.5">자료없음</span>':v.toFixed(2))+'</td>'+
      '<td style="padding:1px 0 1px 8px;border:0;opacity:.6">기준 '+cuts[k].toFixed(2)+'</td>'+
      '<td style="padding:1px 0 1px 6px;border:0">'+(f?'<b style="color:var(--s2)">해당</b>':'')+'</td></tr>';}).join('');
  return '<b>'+n+'</b> · '+R.rtype+' · 인구 '+R.pop.toLocaleString()+'<br>'+
    '<b style="color:'+(r.vtype!==b.vtype?css('--s2'):'inherit')+'">'+r.vtype+'</b>'+
    (r.vtype!==b.vtype?'<br><span style="opacity:.7">전: '+b.vtype+'</span>':'')+
    '<table style="margin-top:5px;font-size:11.5px">'+rows+'</table>'+
    '<span style="opacity:.7">심각도 '+r.sev.toFixed(2)+' · 수요 '+R.src+'</span>';}

/* ---------- 분포 · 축별 소지도 · 표 (SVG) ---------- */
function makeProj(W,H,pad){
  let x0=1e9,x1=-1e9,y0=1e9,y1=-1e9;
  M.geo.features.forEach(f=>walk(f.geometry.coordinates,function(c){
    x0=Math.min(x0,c[0]);x1=Math.max(x1,c[0]);y0=Math.min(y0,c[1]);y1=Math.max(y1,c[1]);}));
  const kx=Math.cos((y0+y1)/2*Math.PI/180);
  const s=Math.min((W-2*pad)/((x1-x0)*kx),(H-2*pad)/(y1-y0));
  const ox=pad+((W-2*pad)-(x1-x0)*kx*s)/2, oy=pad+((H-2*pad)-(y1-y0)*s)/2;
  return {to:c=>[ox+(c[0]-x0)*kx*s, oy+(y1-c[1])*s]};}
function walk(c,f){if(typeof c[0]==='number')f(c);else c.forEach(x=>walk(x,f));}
function pathOf(geom,P){
  const rings=geom.type==='Polygon'?geom.coordinates:geom.coordinates.reduce((a,b)=>a.concat(b),[]);
  return rings.map(r=>r.map(function(c,i){const p=P.to(c);
    return (i?'L':'M')+p[0].toFixed(1)+' '+p[1].toFixed(1);}).join(' ')+'Z').join(' ');}
function drawDist(C,changed){
  const box=document.getElementById('cDist');box.innerHTML='';
  const labs=AX.map(a=>a.label).concat([M.cfg.multi_label,M.cfg.none_label]);
  const cnt=l=>Object.keys(C.res).filter(n=>C.res[n].vtype===l).length;
  const cnt0=l=>Object.keys(BASE.res).filter(n=>BASE.res[n].vtype===l).length;
  const W=380,H=200,L=118,R=44,T=6,B=26,bh=17,gap=(H-T-B-labs.length*bh)/(labs.length-1);
  const s=svg(W,H);box.appendChild(s);
  const max=Math.max.apply(null,[12].concat(labs.map(l=>Math.max(cnt(l),cnt0(l)))));
  const X=v=>L+v/max*(W-L-R);
  [0,5,10,15].filter(g=>g<=max).forEach(function(g){
    el('line',{x1:X(g),x2:X(g),y1:T,y2:H-B,class:'gl'},s);
    el('text',{x:X(g),y:H-B+14,'text-anchor':'middle',class:'ax'},s).textContent=g;});
  el('text',{x:(L+W-R)/2,y:H-2,'text-anchor':'middle',class:'axt'},s).textContent='해당 행정동 수';
  labs.forEach(function(l,i){
    const y=T+i*(bh+gap),a=cnt0(l),b=cnt(l);
    el('text',{x:L-8,y:y+bh/2+4,'text-anchor':'end',class:'ax'},s).textContent=l.replace('형','');
    el('rect',{x:L,y:y+1,width:Math.max(X(a)-L,1),height:bh-2,rx:3,fill:css('--text-muted'),opacity:.28},s);
    const rc=el('rect',{x:L,y:y+3,width:Math.max(X(b)-L,1),height:bh-6,rx:3,
      fill:b>a?css('--s2'):b<a?css('--s1'):css('--text-muted'),opacity:b===a?.65:1,style:'cursor:pointer'},s);
    rc.addEventListener('mousemove',e=>tip(e,'<b>'+l+'</b><br>배치 전 '+a+'곳 → 후 <b>'+b+'곳</b>'+
      (b!==a?' ('+(b>a?'+':'')+(b-a)+')':'')));
    rc.addEventListener('mouseleave',untip);
    el('text',{x:X(Math.max(a,b))+6,y:y+bh/2+4,class:'ax'},s).textContent=a===b?a:a+'→'+b;});
  const cb=document.getElementById('changeBox');
  if(!changed.length){cb.innerHTML='<div class="note" style="margin:0">유형이 바뀐 동이 없다. '+
    (PLACED.length?'위치를 옮기거나 패널티를 낮춰 보라.':'지도를 클릭해 거점을 놓아 보라.')+'</div>';return;}
  cb.innerHTML='<div class="klabel" style="margin-bottom:5px">유형이 바뀐 동 '+changed.length+'곳</div>'+
    changed.map(function(c){return '<div style="font-size:12.5px;padding:4px 0;border-bottom:1px solid var(--line)">'+
      '<b>'+c[0]+'</b> <span style="color:var(--text-muted)">'+c[1]+'</span> '+
      '<span style="color:var(--s2)">→ '+c[2]+'</span>'+
      (c[3]?'<div style="font-size:11px;color:var(--warn);margin-top:1px">자기 값은 그대로 — 상대 기준이 움직여서 바뀐 것</div>':'')+
      '</div>';}).join('');}
function drawFacets(V,C){
  const box=document.getElementById('cFacets');box.innerHTML='';
  AX.forEach(function(a){
    const d=document.createElement('div');
    const nOK=M.regions.filter(r=>V[r.name][a.k]!==null).length;
    const nHit=Object.keys(C.res).filter(n=>C.res[n].hit.indexOf(a.k)>=0).length;
    d.innerHTML='<div class="klabel">'+a.short+' · '+a.label.replace('형','')+'</div>'+
      '<div class="note" style="margin:2px 0 5px">'+a.desc+'<br><b>'+nHit+'곳 취약</b> · 유효 '+nOK+
      '/31 · 기준 '+C.cuts[a.k].toFixed(2)+a.unit+'</div>';
    const W=260,H=190,P2=makeProj(W,H,6),s=svg(W,H);d.appendChild(s);
    M.geo.features.forEach(function(f){
      const n=f.properties.name,r=C.res[n],v=V[n][a.k];
      const fill=v===null?css('--surface-2'):(r.hit.indexOf(a.k)>=0?css('--m3'):css('--m0'));
      const p=el('path',{d:pathOf(f.geometry,P2),fill:fill,stroke:css('--surface-1'),'stroke-width':.8,
        style:'cursor:pointer'},s);
      p.addEventListener('mousemove',e=>tip(e,'<b>'+n+'</b> — '+a.label+'<br>'+
        (v===null?'<b>자료 없음</b> (판정 제외)':'값 <b>'+v.toFixed(2)+a.unit+'</b> · 기준 '+C.cuts[a.k].toFixed(2)+
         (r.hit.indexOf(a.k)>=0?'<br><b style="color:var(--s2)">취약 해당</b>':''))));
      p.addEventListener('mouseleave',untip);});
    box.appendChild(d);});}
function fillReg(V,C){
  const tb=document.querySelector('#tReg tbody');tb.innerHTML='';
  M.regions.slice().sort((a,b)=>C.res[b.name].sev-C.res[a.name].sev).forEach(function(r){
    const c=C.res[r.name],b=BASE.res[r.name],ch=c.vtype!==b.vtype;
    const f=function(k){const v=c.v[k];return v===null?'<span style="opacity:.45">—</span>':
      '<span style="'+(c.hit.indexOf(k)>=0?'color:var(--s2);font-weight:650':'')+'">'+v.toFixed(2)+'</span>';};
    const tr=document.createElement('tr');
    if(SELR===r.name)tr.style.background='var(--surface-2)';
    tr.innerHTML='<td>'+r.name+'</td><td>'+r.rtype+'</td><td>'+r.pop.toLocaleString()+'</td>'+
      '<td style="color:var(--text-muted)">'+b.vtype+'</td>'+
      '<td style="'+(ch?'color:var(--s2);font-weight:650':'')+'">'+c.vtype+(ch?' ★':'')+'</td>'+
      '<td style="text-align:left">'+(c.hit.map(k=>AXSHORT[k]).join('·')||'—')+'</td>'+
      '<td>'+f('dispatch')+'</td><td>'+f('transport')+'</td><td>'+f('backup')+'</td>'+
      '<td>'+f('traffic')+'</td><td>'+f('demand')+'</td><td>'+c.sev.toFixed(2)+'</td>';
    tr.style.cursor='pointer';
    tr.addEventListener('click',function(){SELR=r.name;
      if(MAP)MAP.flyTo({center:[r.lon,r.lat],zoom:12,duration:700});
      refresh(false);});
    tb.appendChild(tr);});}
function bootMap(){
  OVC=document.getElementById('ovc');
  if(MAP){try{MAP.remove();}catch(e){}MAP=null;}
  refresh(true);          // ★ 지도 로드와 무관하게 먼저 계산·표시한다
  try{initMap();}catch(e){
    document.getElementById('mlmap').innerHTML=
      '<div style="padding:18px;color:var(--text-muted);font-size:13px">지도를 불러오지 못했다. '+
      '표와 축별 소지도는 정상 동작한다.</div>';}
  document.getElementById('tSlider').addEventListener('input',function(e){
    TNOW=+e.target.value;tickClock();drawOverlay();});
  document.getElementById('daily').addEventListener('input',function(e){
    DAILY=+e.target.value;document.getElementById('dailyLab').textContent=DAILY+'건';refresh(true);});
  document.getElementById('pen').addEventListener('input',function(e){
    PEN=+e.target.value/10;document.getElementById('penLab').textContent=PEN.toFixed(1)+'분';refresh(true);});
  document.getElementById('penh').addEventListener('input',function(e){
    PENH=+e.target.value/10;document.getElementById('penhLab').textContent=PENH.toFixed(1)+'분';refresh(true);});
  document.getElementById('pct').addEventListener('input',function(e){
    PCT=+e.target.value/100;
    document.getElementById('pctLab').textContent=Math.round((1-PCT)*100)+'%';refresh(false);});}
