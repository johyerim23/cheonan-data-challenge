/* ================= GNN 대체모델 — 브라우저 추론 =================
   PyG TransformerConv(heads=1, concat=True, beta=False, root_weight=True, edge_dim=D)
   를 그대로 옮긴 것. 학습된 가중치(GW)는 파일에 내장돼 있다.

   forward:
     q = lin_query(x_dst); k = lin_key(x_src); v = lin_value(x_src)
     e = lin_edge(edge_attr)                       (bias 없음)
     k_j += e ;  alpha = softmax_i( (q_i·k_j)/sqrt(C) )
     out_i = Σ_j alpha_ij (v_j + e_ij)  +  lin_skip(x_dst)
   HeteroConv(aggr='sum') → 두 관계의 출력을 더한다.
*/
const GNN_ON = typeof GW !== 'undefined' && GW && GW.weights;

function T(name){const w=GW.weights[name];return w?{s:w.shape,d:w.data}:null;}
function matvec(W,x){                       // W:[out,in] row-major, x:[in] → [out]
  const [o,i]=W.s, y=new Float64Array(o);
  for(let a=0;a<o;a++){let s=0;const base=a*i;for(let b=0;b<i;b++)s+=W.d[base+b]*x[b];y[a]=s;}
  return y;}
function lin(pre,x,useBias){
  const W=T(pre+'.weight'); if(!W)return null;
  const y=matvec(W,x); const B=useBias===false?null:T(pre+'.bias');
  if(B)for(let a=0;a<y.length;a++)y[a]+=B.d[a];
  return y;}
function relu(v){const o=new Float64Array(v.length);for(let i=0;i<v.length;i++)o[i]=v[i]>0?v[i]:0;return o;}
function znorm(vec,st){const o=new Float64Array(vec.length);
  for(let i=0;i<vec.length;i++)o[i]=(vec[i]-st.mean[i])/st.std[i];return o;}

/* 한 관계에 대한 TransformerConv */
function tconv(prefix,Xsrc,Xdst,ei,ea,hid){
  const N=Xdst.length, C=hid;
  const q=Xdst.map(x=>lin(prefix+'.lin_query',x));
  const k=Xsrc.map(x=>lin(prefix+'.lin_key',x));
  const v=Xsrc.map(x=>lin(prefix+'.lin_value',x));
  const E=ei[0].length, sc=1/Math.sqrt(C);
  const eE=new Array(E), logit=new Float64Array(E);
  for(let e=0;e<E;e++){
    const j=ei[0][e], i=ei[1][e];
    const em=lin(prefix+'.lin_edge',ea[e],false);       // bias 없음
    eE[e]=em;
    let s=0; for(let c=0;c<C;c++) s+=q[i][c]*(k[j][c]+(em?em[c]:0));
    logit[e]=s*sc;}
  // 목적 노드별 softmax
  const mx=new Float64Array(N).fill(-Infinity), sum=new Float64Array(N);
  for(let e=0;e<E;e++){const i=ei[1][e]; if(logit[e]>mx[i])mx[i]=logit[e];}
  const ex=new Float64Array(E);
  for(let e=0;e<E;e++){const i=ei[1][e]; ex[e]=Math.exp(logit[e]-mx[i]); sum[i]+=ex[e];}
  const out=Array.from({length:N},()=>new Float64Array(C));
  for(let e=0;e<E;e++){
    const j=ei[0][e], i=ei[1][e], a=ex[e]/sum[i], em=eE[e];
    for(let c=0;c<C;c++) out[i][c]+=a*(v[j][c]+(em?em[c]:0));}
  for(let i=0;i<N;i++){const sk=lin(prefix+'.lin_skip',Xdst[i]);
    for(let c=0;c<C;c++) out[i][c]+=sk[c];}
  return out;}

/* 현재 SIM 으로 그래프를 만들고 y = d1 − min_eta (초) 를 예측한다 */
function gnnPredict(){
  if(!GNN_ON||!SIM||!SIM.ev.length)return null;
  const hid=GW.arch.hid, LUT=GW.occupy_lut.lut, DEF=GW.occupy_lut.default;
  const ev=SIM.ev.slice().sort((a,b)=>a.t-b.t);
  // --- 노드 피처 (build_graphs.py 와 동일 순서) ---
  const SEVK=['cardiac','stroke','trauma','other'];
  const XP=ev.map(e=>{const h=(e.t%86400)/3600;
    const base=[h,Math.sin(2*Math.PI*h/24),Math.cos(2*Math.PI*h/24),
      e.intra,e.c1e,e.c2e,e.c2e-e.c1e,e.minAmb,RG[e.r].rural?1:0,DAILY];
    const v=GW.arch.din_p>10?base.concat(SEVK.map(k=>e.sev===k?1:0)):base;
    return znorm(v,GW.norm.p);});
  // 거점 노드
  const sids=[]; const sidx={};
  ev.forEach(e=>e.cand.forEach(c=>{if(!(c.sid in sidx)){sidx[c.sid]=sids.length;sids.push(c);}}));
  const XS=sids.map(c=>znorm([c.amb],GW.norm.s));
  // --- precedes: 후보 거점 공유 + 0 < Δt < 60분, 과거→미래 ---
  const W=3600, src=[],dst=[],eat=[], byS={};
  for(let i=0;i<ev.length;i++){
    const seen=new Set();
    ev[i].cand.forEach(c=>{const L=byS[c.sid]||(byS[c.sid]=[]);
      while(L.length&&ev[i].t-ev[L[0]].t>=W)L.shift();
      L.forEach(j=>seen.add(j));});
    seen.delete(i);
    Array.from(seen).sort((a,b)=>a-b).forEach(j=>{
      src.push(j);dst.push(i);
      eat.push(znorm([ev[i].t-ev[j].t, LUT[ev[j].r]!==undefined?LUT[ev[j].r]:DEF],GW.norm.ep));});
    ev[i].cand.forEach(c=>byS[c.sid].push(i));}
  // --- 병원 노드 + serves 엣지 (arch.din_h>0 일 때만) ---
  const useH=GW.arch.din_h>0&&ev[0].hcand&&ev[0].hcand.length;
  const hids=[], hidx={}, XH=[], hsrc=[], hdst=[], hea=[];
  if(useH){
    ev.forEach(e=>e.hcand.forEach(h=>{if(!(h.id in hidx)){hidx[h.id]=hids.length;hids.push(h);
      XH.push(znorm([h.cardiac,h.stroke,h.trauma,h.beds,h.inside],GW.norm.h));}}));
    ev.forEach((e,i)=>e.hcand.forEach(h=>{
      hsrc.push(hidx[h.id]);hdst.push(i);hea.push(znorm([h.eta],GW.norm.eh));}));}
  // --- can_reach: top-3 거점 → 환자 ---
  const csrc=[],cdst=[],cea=[];
  ev.forEach((e,i)=>e.cand.forEach(c=>{
    csrc.push(sidx[c.sid]);cdst.push(i);cea.push(znorm([c.eta],GW.norm.ec));}));
  // --- forward ---
  const P0=XP.map(x=>relu(lin('lin_p',x))), S0=XS.map(x=>relu(lin('lin_s',x)));
  const H0=useH?XH.map(x=>relu(lin('lin_h',x))):[];
  const step=(Xp,Xs,Xh,pre)=>{
    const a=tconv(pre+'.convs.<station___can_reach___patient>',Xs,Xp,[csrc,cdst],cea,hid);
    const b=tconv(pre+'.convs.<patient___precedes___patient>',Xp,Xp,[src,dst],eat,hid);
    const c=useH?tconv(pre+'.convs.<hospital___serves___patient>',Xh,Xp,[hsrc,hdst],hea,hid):null;
    return a.map((v,i)=>relu(v.map((x,k)=>x+b[i][k]+(c?c[i][k]:0))));};
  const H1=step(P0,S0,H0,'c1'), H2=step(H1,S0,H0,'c2');
  const out=H1.map((h1,i)=>{
    const z=new Float64Array(hid*2);
    for(let c=0;c<hid;c++){z[c]=h1[c];z[hid+c]=H2[i][c];}
    const hh=relu(lin('head.0',z));
    const p=1/(1+Math.exp(-lin('clf',hh)[0]));
    const mu=Math.min(lin('reg',hh)[0],8);
    return {i:ev[i].i,p:p,yhat:(CAL?CAL_A:1)*p*Math.expm1(mu),y:ev[i].loss*60,d1:ev[i].c1e+ev[i].loss*60};});
  return out;}

/* ---------- 성능 요약 ---------- */
function gnnStats(pred){
  if(!pred||!pred.length)return null;
  const pos=pred.filter(o=>o.y>0.6), neg=pred.filter(o=>o.y<=0.6);
  let auc=null;
  if(pos.length&&neg.length){
    const all=pred.slice().sort((a,b)=>a.p-b.p);
    const rank={}; all.forEach((o,i)=>rank[o.i]=i+1);
    const rsum=pos.reduce((a,o)=>a+rank[o.i],0);
    auc=(rsum-pos.length*(pos.length+1)/2)/(pos.length*neg.length);}
  const mae=pos.length?pos.reduce((a,o)=>a+Math.abs(o.yhat-o.y),0)/pos.length/60:null;
  const simMean=pred.reduce((a,o)=>a+o.y,0)/pred.length/60;
  const gnnMean=pred.reduce((a,o)=>a+o.yhat,0)/pred.length/60;
  const tot=pred.reduce((a,o)=>a+o.yhat,0);
  const onZero=neg.reduce((a,o)=>a+o.yhat,0);
  return {n:pred.length,nPos:pos.length,auc:auc,maePos:mae,simMean:simMean,gnnMean:gnnMean,
    zeroShare:tot>0?onZero/tot:0, zeroBig:neg.filter(o=>o.yhat>60).length,
    d1max:Math.max.apply(null,neg.map(o=>o.d1))/60};}

function drawGnnPanel(){
  const box=document.getElementById('cGnn'); if(!box)return;
  if(!GNN_ON){box.innerHTML='<div class="note" style="margin:0">가중치가 내장돼 있지 않다.</div>';return;}
  const pred=gnnPredict(); PRED=pred;
  const s=gnnStats(pred);
  document.getElementById('gnnKpi').innerHTML=
    '<div><span class="klabel">이 배치 · 시뮬 평균 손실</span><b>'+s.simMean.toFixed(2)+'</b>분</div>'+
    '<div><span class="klabel">GNN 예측 평균</span><b>'+s.gnnMean.toFixed(2)+'</b>분</div>'+
    '<div><span class="klabel">분류 AUC</span><b>'+(s.auc===null?'—':s.auc.toFixed(3))+'</b></div>'+
    '<div><span class="klabel">조건부 MAE</span><b>'+(s.maePos===null?'—':s.maePos.toFixed(2))+'</b>분</div>'+
    '<div><span class="klabel">차선배차 실제</span><b>'+s.nPos+'</b> / '+s.n+'건</div>'+
    '<div><span class="klabel">실제 0에 얹힌 예측</span><b class="'+(s.zeroShare>0.3?'hi':'')+'">'+
      (s.zeroShare*100).toFixed(0)+'</b>% <span style="font-size:11px;color:var(--text-muted)">('+
      s.zeroBig+'건이 1분 초과)</span></div>';
  // 산점도
  box.innerHTML='';
  const W=430,H=280,L=52,R=14,T=12,B=42, sv=svg(W,H); box.appendChild(sv);
  const mx=Math.max(2,...pred.map(o=>Math.max(o.y,o.yhat)))/60;
  const X=v=>L+v/mx*(W-L-R), Y=v=>T+(mx-v)/mx*(H-T-B);
  const stp=mx>12?5:mx>6?2:1;
  for(let g=0;g<=mx;g+=stp){
    el('line',{x1:L,x2:W-R,y1:Y(g),y2:Y(g),class:'gl'},sv);
    el('line',{x1:X(g),x2:X(g),y1:T,y2:H-B,class:'gl'},sv);
    el('text',{x:L-7,y:Y(g)+4,'text-anchor':'end',class:'ax'},sv).textContent=g;
    el('text',{x:X(g),y:H-B+15,'text-anchor':'middle',class:'ax'},sv).textContent=g;}
  el('line',{x1:X(0),y1:Y(0),x2:X(mx),y2:Y(mx),stroke:css('--text-muted'),'stroke-width':1.4,
    'stroke-dasharray':'5 4',opacity:.8},sv);
  el('text',{x:(L+W-R)/2,y:H-4,'text-anchor':'middle',class:'axt'},sv).textContent='시뮬 실제 손실 (분)';
  el('text',{x:13,y:(T+H-B)/2,'text-anchor':'middle',class:'axt',
    transform:'rotate(-90 13 '+((T+H-B)/2)+')'},sv).textContent='GNN 예측 (분)';
  // 실제 0 인 케이스는 x=0 한 점에 겹친다 — 밴드로 표시하고 지터를 준다
  const bw=Math.max(14,(W-L-R)*0.055);
  el('rect',{x:L-3,y:T,width:bw,height:H-T-B,fill:css('--s1'),opacity:.07},sv);
  el('text',{x:L+bw/2,y:T+11,'text-anchor':'middle',class:'ax',fill:css('--s1')},sv).textContent='실제 0';
  let jz=0;
  pred.forEach(o=>{
    const jit=o.y<=0.6?((jz++%9)-4)*(bw/11):0;
    const c=el('circle',{cx:X(o.y/60)+jit,cy:Y(o.yhat/60),r:4,
      fill:o.y>0.6?css('--s2'):css('--s1'),opacity:.75,stroke:css('--surface-1'),'stroke-width':1,
      style:'cursor:pointer'},sv);
    const e=SIM.ev.find(x=>x.i===o.i);
    c.addEventListener('mousemove',ev2=>tip(ev2,'<b>'+(e?e.r:'')+'</b> '+(e?hhmm2(e.t):'')+
      '<br>실제 <b>'+(o.y/60).toFixed(2)+'분</b> · GNN <b>'+(o.yhat/60).toFixed(2)+'분</b>'+
      '<br>P(차선배차) '+(o.p*100).toFixed(0)+'%'+
      '<br><span style="opacity:.75">현장 도착 '+(o.d1/60).toFixed(2)+'분'+
      (o.y<=0.6?' — 최근접 거점이 갔는데도 이만큼 걸린다':'')+'</span>'));
    c.addEventListener('mouseleave',untip);});
}
let PRED=null, CAL=false;
const CAL_A=0.885;   // test 451,508건에서 평균 편향을 1.000 으로 맞추는 계수
