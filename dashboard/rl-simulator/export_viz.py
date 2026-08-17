# -*- coding: utf-8 -*-
"""HTML 대시보드/탐색기용 데이터 추출."""
import json, math, random, pandas as pd, numpy as np

src = open("gen_dataset.py", encoding="utf-8").read().split("def sample_deploy")[0]
exec(src)

# ---------- 1. 그래프 탐색기용: 한 시나리오를 상세 로깅으로 재실행 ----------
def detailed(cases, extra_amb=None, day=3):
    ems = [dict(e) for e in EMS]
    if extra_amb:
        for sid, v in extra_amb.items():
            for e in ems:
                if e["ems_id"] == sid: e["amb"] += v
    def eta_er(sid, rn, elon, elat):
        v = ETA_ER.get((sid, rn))
        if v is not None: return v
        r = R[rn]; return km((elon,elat),(r["lon"],r["lat"]))*1.3/40*3600
    def eta_rh(rn, h):
        v = ETA_RH.get((rn, h["hospital_id"]))
        if v is not None: return v
        r = R[rn]; return km((r["lon"],r["lat"]),(h["lon"],h["lat"]))*1.3/50*3600
    STA = {e["ems_id"]:(e["lon"],e["lat"]) for e in ems}
    ORD = {rn: sorted(((eta_er(s,rn,lo,la), s) for s,(lo,la) in STA.items()), key=lambda x:x[0]) for rn in R}
    units = []
    for e in ems:
        for k in range(e["amb"]):
            units.append(dict(uid=f'{e["ems_id"]}-{k+1}', sid=e["ems_id"],
                              blon=e["lon"], blat=e["lat"], free_at=0.0, last_case=None))
    out, busy = [], []
    for i, c in enumerate(cases):
        r = R[c["region"]]; rn = r["name"]
        free = [u for u in units if u["free_at"] <= c["t_call"]] or units
        u = min(free, key=lambda x: eta_er(x["sid"],rn,x["blon"],x["blat"]) + max(0,x["free_at"]-c["t_call"]))
        t_disp = c["t_call"] + 60 + max(0.0, u["free_at"]-c["t_call"])
        d1 = eta_er(u["sid"], rn, u["blon"], u["blat"])
        me, me_s = ORD[rn][0]
        # 차단자: 최근접 거점 소속 유닛 중 t_call 시점에 점유 중인 케이스
        blockers = [x["last_case"] for x in units
                    if x["sid"] == me_s and x["free_at"] > c["t_call"] and x["last_case"] is not None]
        t_scene = t_disp + d1 + c["intra"]; t_dep = t_scene + c["scene"]
        need = None if c["sev"]=="other" else c["sev"]
        cand = sorted([h for h in hospL if need is None or h[need]==1], key=lambda h: eta_rh(rn,h)) or \
               sorted(hospL, key=lambda h: eta_rh(rn,h))
        chosen = None
        for k,h in enumerate(cand):
            if c["rolls"][min(k,7)] < 0.18: continue
            chosen = h; break
        chosen = chosen or cand[0]
        t_hosp = t_dep + eta_rh(rn, chosen)
        nf = t_hosp + 900 + km((chosen["lon"],chosen["lat"]),(u["blon"],u["blat"]))*1.3/55*3600
        u["free_at"] = nf; u["last_case"] = i
        if day*86400-10800 <= c["t_call"] < (day+1)*86400:
            out.append(dict(i=i, t=round(c["t_call"],1), region=rn, sev=c["sev"],
                            lead=1 if c["t_call"] < day*86400 else 0,
                            uid=u["uid"], sid=u["sid"], min_sid=me_s,
                            y=round((d1-me)/60,2), d1=round(d1/60,2), me=round(me/60,2),
                            blockers=[b for b in blockers][:3]))
            busy.append(dict(i=i, uid=u["uid"], t0=round(t_disp,1), t1=round(nf,1)))
    return out, busy, [e["ems_id"] for e in ems], {e["ems_id"]: e["name"] for e in ems}

hospL = HOSP
cases = gen_cases(99, 200)
ev, busy, sids, snames = detailed(cases, day=3)
keep = {e["i"] for e in ev}
busy = [b for b in busy if b["i"] in keep]
# 차단자가 창 밖일 수 있으므로 유지, 프런트에서 필터
graph = dict(events=ev, busy=busy, stations=[{"sid":s,"name":snames[s]} for s in sids],
             day=3, daily=200)
json.dump(graph, open("out/viz_graph.json","w"), ensure_ascii=False, separators=(",",":"))
print(f"탐색기: day3 {len(ev)}건 · 유닛 {len(set(e['uid'] for e in ev))} · 차단 존재 {sum(1 for e in ev if e['blockers'])}건")

# ---------- 2. 대시보드 데이터 ----------
sm = pd.read_csv("out/scenario_matrix.csv")
pr = pd.read_csv("out/gnn_policy_rank.csv")
M = pd.read_csv("out/gnn_scenarios.csv").set_index("scen")
pr["daily"] = M.loc[pr.scen, "daily"].values
gnn = json.load(open("out/gnn_result.json")); gbm = json.load(open("out/gbm_baseline.json"))
from scipy.stats import spearmanr
byload = [dict(daily=int(d), n=len(g), rho=float(spearmanr(g.sim,g.gnn).statistic))
          for d,g in pr.groupby("daily") if len(g) >= 5]
import re
curve = []
for l in open("out/gnn_train.log", encoding="utf-8"):
    m = re.search(r"ep\s*(\d+) loss ([\d.]+) . val AUC ([\d.]+) . 조건부MAE ([\d.]+)", l)
    if m: curve.append(dict(ep=int(m[1]), loss=float(m[2]), auc=float(m[3]), mae=float(m[4])))

dash = dict(
    scenarios=sm.to_dict("records"),
    sweep=[dict(pen=float(r.scen.split("패널티")[1].split("분")[0]),
                amb=int(r.scen.split("×")[1].replace("대","")),
                dT10=float(r["ΔT10"]), resp=float(r.resp), p90=float(r.p90),
                total=float(r.total), floor=float(r.floor), loss=float(r.loss))
           for _, r in sm.iterrows() if r.scen.startswith("A 패널티")],
    own_eta=[0.65,3.12,4.07,4.12,5.47,7.47,8.85,10.12],
    edges=[dict(rule="top-1 · 60분", epn=1.28, miss=12.47, dmax=12),
           dict(rule="top-1 · 90분", epn=1.91, miss=8.19, dmax=15),
           dict(rule="top-2 · 60분", epn=3.26, miss=2.51, dmax=19),
           dict(rule="top-3 · 60분", epn=5.73, miss=0.37, dmax=31),
           dict(rule="top-3 · 90분", epn=9.23, miss=0.10, dmax=39)],
    model=dict(gnn=dict(auc=gnn["test"]["auc"], mae=gnn["test"]["mae_pos"], rho=gnn["spearman"]),
               gbm=dict(auc=gbm["auc"], mae=gbm["mae_pos"], rho=gbm["spearman"])),
    byload=byload, rank=pr.to_dict("records"), curve=curve,
    single_auc=[dict(f="n_free",a=0.731),dict(f="free_ratio",a=0.750),dict(f="daily",a=0.674),
                dict(f="min_amb",a=0.624),dict(f="eta2",a=0.558),dict(f="min_eta",a=0.522),
                dict(f="t_call",a=0.501)],
    load=[dict(daily=90,n=505954,pos=0.176,ym=3.29),dict(daily=150,n=425308,pos=0.289,ym=3.67),
          dict(daily=200,n=446045,pos=0.389,ym=4.27),dict(daily=300,n=847116,pos=0.551,ym=6.08)],
)
json.dump(dash, open("out/viz_dash.json","w"), ensure_ascii=False, separators=(",",":"))
print("대시보드 데이터 저장:", {k: (len(v) if isinstance(v,(list,dict)) else v) for k,v in dash.items()})
