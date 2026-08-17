# -*- coding: utf-8 -*-
"""GNN 학습 데이터 생성 — 랜덤 배치 시나리오 × 다중 케이스 시드 × 부하 수준.

설계 요점
  · 케이스 시드를 시나리오마다 바꾼다. 배치만 바꾸면 환자 집합이 train/test에 공유돼
    시나리오 단위 분할을 해도 t_call·intra·지역이 그대로 새어나간다.
  · 점유시간(occupy_sec)과 '같은 거점 직전 출동과의 Δt'를 함께 기록한다.
    precedes 엣지 윈도를 60분으로 둘지 데이터로 정해야 하기 때문.
"""
import math, random, json, time
import numpy as np, pandas as pd

src = open("case_log_build.py", encoding="utf-8").read().split("SEV, SEV_P")[0]
exec(src)   # REGIONS, R, EMS, HOSP, ETA_ER, ETA_RH, WEIGHT, HRP, km, THRESH ... 정의됨

def km(a,b): return math.hypot((a[0]-b[0])*88.9,(a[1]-b[1])*111.0)

SEV, SEV_P = ["cardiac","stroke","trauma","other"], [0.12,0.10,0.18,0.60]
WSUM = sum(WEIGHT.values())
OWN_ETA = [0.65, 3.12, 4.07, 4.12, 5.47, 7.47, 8.85, 10.12]   # 신설 입지 패널티 경험분포(분)

def gen_cases(seed, daily, days=30):
    rng = random.Random(seed)
    C=[]
    for day in range(days):
        for r in REGIONS:
            lam = daily * WEIGHT[r["name"]]/WSUM
            n = max(0, int(rng.gauss(lam, math.sqrt(max(lam,.5)))))
            for _ in range(n):
                hr = rng.choices(range(24), HRP)[0]
                th = rng.uniform(0,2*math.pi); dd = r["Rmax_km"]*math.sqrt(rng.random())
                C.append(dict(day=day, region=r["name"],
                    t_call=day*86400 + hr*3600 + rng.uniform(0,3600),
                    sev=rng.choices(SEV,SEV_P)[0],
                    intra=dd*1.3/r["intra_kmh"]*3600,
                    rolls=[rng.random() for _ in range(8)], scene=rng.uniform(240,480)))
    C.sort(key=lambda c:c["t_call"])
    return C

def simulate(cases, extra_amb=None, new_ems=None, penalty_min=0.0, extra_hosp=None, off_hosp=None):
    ems=[dict(e) for e in EMS]
    if extra_amb:
        for sid,v in extra_amb.items():
            for e in ems:
                if e["ems_id"]==sid: e["amb"]=max(0, e["amb"]+v)
    NEWSET={}
    if new_ems:
        for j,(host,n) in enumerate(new_ems):
            r=R[host]; sid=f"NEW{j+1:02d}"
            ems.append(dict(ems_id=sid, name=f"[신설]{host}", lon=r["lon"], lat=r["lat"], amb=n))
            NEWSET[sid]=host
    ems=[e for e in ems if e["amb"]>0]
    hosp=[dict(h) for h in HOSP if h['hospital_id'] not in (off_hosp or set())]+(extra_hosp or [])
    P=penalty_min*60

    def eta_er(sid, rn, elon, elat):
        v=ETA_ER.get((sid,rn))
        if v is not None: return v
        r=R[rn]; a=km((elon,elat),(r["lon"],r["lat"]))*1.3/40*3600
        return max(a,P) if sid in NEWSET else a
    def eta_rh(rn,h):
        v=ETA_RH.get((rn,h["hospital_id"]))
        if v is not None: return v
        r=R[rn]; return km((r["lon"],r["lat"]),(h["lon"],h["lat"]))*1.3/50*3600

    STA={e["ems_id"]:(e["lon"],e["lat"]) for e in ems}
    AMB={e["ems_id"]:e["amb"] for e in ems}
    # 지역별 거점 ETA 정렬 (1·2순위) — 사전계산
    ORD={}
    for rn in R:
        v=sorted(((eta_er(s,rn,lo,la),s) for s,(lo,la) in STA.items()), key=lambda x:x[0])
        ORD[rn]=v
    units=[]
    for e in ems:
        for k in range(e["amb"]):
            units.append(dict(sid=e["ems_id"], blon=e["lon"], blat=e["lat"], free_at=0.0))
    last_disp={}          # 거점별 직전 출동 시각
    out=[]
    for c in cases:
        r=R[c["region"]]; rn=r["name"]
        free=[u for u in units if u["free_at"]<=c["t_call"]] or units
        u=min(free,key=lambda x: eta_er(x["sid"],rn,x["blon"],x["blat"])+max(0,x["free_at"]-c["t_call"]))
        wait=max(0.0,u["free_at"]-c["t_call"])
        t_disp=c["t_call"]+60+wait
        d1=eta_er(u["sid"],rn,u["blon"],u["blat"])
        t_scene=t_disp+d1+c["intra"]; t_dep=t_scene+c["scene"]
        need=None if c["sev"]=="other" else c["sev"]
        cand=sorted([h for h in hosp if need is None or h[need]==1],key=lambda h: eta_rh(rn,h)) or \
             sorted(hosp,key=lambda h: eta_rh(rn,h))
        hc=[(h["hospital_id"], eta_rh(rn,h)) for h in cand[:3]]
        while len(hc)<3: hc.append(hc[-1])
        chosen=None
        for k,h in enumerate(cand):
            if c["rolls"][min(k,7)]<0.18: continue
            chosen=h; break
        chosen=chosen or cand[0]
        t_hosp=t_dep+eta_rh(rn,chosen)
        back=km((chosen["lon"],chosen["lat"]),(u["blon"],u["blat"]))*1.3/55*3600
        new_free=t_hosp+900+back
        occupy=new_free-t_disp
        u["free_at"]=new_free
        me,me_s = ORD[rn][0]
        e2,s2 = ORD[rn][1] if len(ORD[rn])>1 else (me,me_s)
        e3,s3 = ORD[rn][2] if len(ORD[rn])>2 else (e2,s2)
        dtp = c["t_call"]-last_disp.get(me_s, -1e9)
        last_disp[u["sid"]]=t_disp
        out.append((c["t_call"], rn, r["rural"], c["intra"], me, e2, e3, me_s, s2, s3, AMB.get(me_s,0),
                    d1-me, u["sid"], occupy, min(dtp,1e5), len(free), len(units),
                    c["sev"], hc[0][0], hc[1][0], hc[2][0],
                    round(hc[0][1],1), round(hc[1][1],1), round(hc[2][1],1)))
    return out

COLS=["t_call","region","rural","intra_sec","min_eta","eta2","eta3","min_sid","sid2","sid3","min_amb",
      "y","assigned_sid","occupy_sec","dt_prev_same_sid","n_free","n_units",
      "sev","hid1","hid2","hid3","heta1","heta2","heta3"]

def sample_deploy(rng):
    """랜덤 배치: 기존 거점 증감 + 신설 0~3개소"""
    ids=[e["ems_id"] for e in EMS]
    extra={}
    for s in ids:
        u=rng.random()
        if u<0.12: extra[s]=1
        elif u<0.16: extra[s]=2
        elif u<0.19: extra[s]=-1      # 감차도 넣는다 (경합 신호 강화)
    k=rng.choice([0,0,1,2,3])
    hosts=rng.sample([r["name"] for r in REGIONS], k)
    new=[(h, rng.choice([1,1,2])) for h in hosts]
    pen=rng.choice(OWN_ETA)
    # ★ 병원도 흔든다 — 고정이면 모델이 병원 집합을 외우고, 화면에서 신설/제거를 못 다룬다
    inside=[h["hospital_id"] for h in HOSP if h["inside_cheonan"]]
    noff=rng.choice([0,0,0,1,1,2])
    off=set(rng.sample(inside, min(noff,max(0,len(inside)-2))))
    GR={"권역응급의료센터":dict(stroke=1,cardiac=1,trauma=1),
        "지역응급의료센터":dict(stroke=1,cardiac=1,trauma=0),
        "지역응급의료기관":dict(stroke=0,cardiac=0,trauma=0)}
    exh=[]
    if rng.random()<0.45:
        hn=rng.choice([r["name"] for r in REGIONS]); g=rng.choice(list(GR))
        exh=[dict(hospital_id="NEWH1", name="[신설]"+g, etype=g, lon=R[hn]["lon"], lat=R[hn]["lat"],
                  inside_cheonan=True, beds=6, cap_source={}, **GR[g])]
    return extra, new, pen, off, exh

def main(n_scen=500, seeds=(11,22,33,44,55,66,77,88,99,1010), loads=(90,90,150,200,300)):
    rng=random.Random(20260816)
    cache={}
    rows=[]; meta=[]
    t0=time.time()
    for i in range(n_scen):
        # ★ 부하를 순환으로 뽑으면 seed와 교락된다(10과 5의 gcd). 독립 추출로 바꾼다
        seed=seeds[i%len(seeds)]; load=rng.choice(loads)
        key=(seed,load)
        if key not in cache: cache[key]=gen_cases(seed,load)
        cases=cache[key]
        extra,new,pen,off,exh = sample_deploy(rng)
        rec=simulate(cases, extra_amb=extra, new_ems=new, penalty_min=pen, extra_hosp=exh, off_hosp=off)
        a=pd.DataFrame(rec, columns=COLS)
        a["scen"]=i; a["case_seed"]=seed; a["daily"]=load
        rows.append(a)
        meta.append(dict(scen=i, case_seed=seed, daily=load, n=len(cases),
            extra_amb=json.dumps(extra), new_ems=json.dumps(new), penalty_min=pen,
            off_hosp=json.dumps(sorted(off)), new_hosp=json.dumps([{k:h[k] for k in
              ("hospital_id","etype","lon","lat","cardiac","stroke","trauma","beds")} for h in exh]),
            n_units=int(a.n_units.iloc[0]),
            mean_resp=float((60+a.y+a.min_eta+a.intra_sec).mean()/60),
            y_pos=float((a.y>1e-9).mean())))
        if (i+1)%100==0: print(f"  {i+1}/{n_scen} · {time.time()-t0:.0f}s · 누적 {sum(len(x) for x in rows):,}행")
    D=pd.concat(rows,ignore_index=True)
    M=pd.DataFrame(meta)
    for c in ["t_call","intra_sec","min_eta","eta2","eta3","y","occupy_sec","dt_prev_same_sid","heta1","heta2","heta3"]:
        D[c]=D[c].astype("float32")
    for c in ["min_amb","n_free","n_units","scen","case_seed","daily"]:
        D[c]=D[c].astype("int32")
    for c in ["region","min_sid","sid2","sid3","assigned_sid","sev","hid1","hid2","hid3"]: D[c]=D[c].astype("category")
    D.to_parquet("out/gnn_dataset.parquet", compression="zstd", index=False)
    M.to_csv("out/gnn_scenarios.csv", index=False, encoding="utf-8-sig")
    json.dump([dict(hospital_id=h["hospital_id"], etype=h["etype"], lon=h["lon"], lat=h["lat"],
                    cardiac=h["cardiac"], stroke=h["stroke"], trauma=h["trauma"], beds=h["beds"],
                    inside=bool(h["inside_cheonan"])) for h in HOSP],
              open("out/hospitals.json","w"), ensure_ascii=False)
    print(f"\n총 {len(D):,}행 · {time.time()-t0:.0f}초 · 비영 {(D.y>1e-9).sum():,}건 ({(D.y>1e-9).mean()*100:.1f}%)")
    return D, M

if __name__=="__main__":
    D,M=main()
    import os
    print("parquet:", os.path.getsize("out/gnn_dataset.parquet")/1e6, "MB")
    print("\n=== 부하별 비영 비율 ===")
    print(D.groupby("daily").agg(n=("y","size"), pos=("y",lambda s:(s>1e-9).mean()),
        ymean=("y",lambda s: s[s>1e-9].mean()/60 if (s>1e-9).any() else 0)).round(3))
    print("\n=== 점유시간 (분) ===")
    print((D.occupy_sec/60).describe([.5,.9,.99]).round(1).to_string())
    print("\n=== 같은 거점 직전 출동과의 Δt (분), y>0 케이스만 ===")
    pos=D[D.y>1e-9]; neg=D[D.y<=1e-9]
    print("y>0 :", (pos.dt_prev_same_sid/60).describe([.5,.9,.95,.99]).round(1).to_string())
    print("y=0 중앙값:", round(float((neg.dt_prev_same_sid/60).median()),1), "분")
    for w in (15,30,60,90,120):
        print(f"  Δt<{w}분 이 포착하는 y>0 비율: {(pos.dt_prev_same_sid<w*60).mean()*100:.1f}%")
