# -*- coding: utf-8 -*-
"""케이스 로그 재생성 — real_build.py 의 시뮬레이션부를 그대로 복제하고
   응답시간 3분할(구조적 하한 / 대기 / 차선배차 손실)을 케이스 단위로 기록한다.

   real_build.py 와의 차이는 출력뿐이다. 난수 호출 순서·순번을 건드리지 않았으므로
   CASES 는 비트 단위로 동일하게 재생된다. (검증: summary 가 표의 값과 일치해야 함)
"""
import json, math, random, glob, os
import pandas as pd, geopandas as gpd

SRC = "/mnt/user-data/uploads/비타민 최종 프로젝트"
def rd(p, **k):
    for e in ("utf-8-sig","cp949","euc-kr"):
        try: return pd.read_csv(os.path.join(SRC,p), encoding=e, low_memory=False, **k)
        except Exception as ex: last=ex
    raise last

random.seed(20260812)
SIM_HOURS, DAYS = 24, 30
THRESH = {"T8":(8,12), "T10":(10,15), "T12":(12,18), "T15":(15,20)}
DEFAULT_T = "T10"
RESP_LIMIT = {"urban": 10*60, "rural": 15*60}
DEST_LIMIT = {"cardiac": 60*60, "stroke": 60*60, "trauma": 60*60, "other": 90*60}

# ---------- 1. 경계 + 지역 ----------
geo = gpd.read_file(os.path.join(SRC,"analysis/data/raw/cheonan_hjd_ver20260401.geojson"))
geo["name"] = geo["adm_nm"].str.split().str[-1] if "adm_nm" in geo else geo.iloc[:,0]
reg = rd("cheonan_region_db.csv")
eta = rd("analysis/outputs/eta_by_dong.csv")
vul = rd("취약인구_읍면동_지수.csv")
trf = rd("교통_읍면동_지수.csv")
assert set(reg.region_name) == set(geo.name), set(reg.region_name) ^ set(geo.name)

m5179 = geo.to_crs(5179)
geo["area_km2"] = (m5179.geometry.area/1e6).round(2)
geo["geometry"] = m5179.geometry.simplify(30).to_crs(4326)

E = eta.set_index("dong"); V = vul.set_index("행정동"); T = trf.set_index("행정동")
REGIONS=[]
for _,r in reg.iterrows():
    n=r.region_name; g=geo[geo.name==n].iloc[0]; e=E.loc[n]
    rural = r.region_type in ("읍","면")
    REGIONS.append(dict(
        adm_cd=str(e.adm_cd2), name=n, gu=r.gu_name, rtype=r.region_type, rural=bool(rural),
        lon=float(r.longitude), lat=float(r.latitude), area_km2=float(g.area_km2),
        pop=int(r.population), elderly=int(r.elderly_population),
        elder_ratio=round(float(r.elderly_population_ratio),4),
        eta_nearest=round(float(e.eta_min),2), eta_2nd=round(float(e.eta_min_2nd),2),
        eta_regional=round(float(e.eta_min_regional),2),
        eta_penalty=round(float(e.eta_penalty_regional),2),
        nearest_hospital=str(e.nearest_hospital), nearest_regional=str(e.nearest_regional),
        expected_cases=round(float(e.expected_cases),1),
        vul_index=round(float(V.loc[n,"취약인구지수_V2"]),4) if n in V.index else None,
        traffic=round(float(T.loc[n,"교통혼잡도"]),4) if n in T.index and pd.notna(T.loc[n,"교통혼잡도"]) else None,
    ))
R = {r["name"]:r for r in REGIONS}

# ---------- 2. EMS 거점 ----------
ems = rd("cheonan_ems_station_db.csv")
EMS = [dict(ems_id=r.station_id, name=r.station_name, parent=r.parent_fire_station,
            stype=r.station_type, lon=float(r.longitude), lat=float(r.latitude),
            amb=int(r.ambulance_count)) for _,r in ems.iterrows()]
print(f"EMS {len(EMS)}개소 · 구급차 {sum(e['amb'] for e in EMS)}대")

# ---------- 3. 병원 + 중증 대응 ----------
hos = rd("cheonan_emergency_hospital_db.csv")
cap = rd("analysis/data/raw/hospital_capabilities.csv").set_index("hpid")
CAPMAP = {"stroke":["MKioskTy1","MKioskTy2"], "cardiac":["MKioskTy3"], "trauma":["MKioskTy4","MKioskTy5"]}
GRADE  = {"권역응급의료센터":dict(stroke=1,cardiac=1,trauma=1),
          "지역응급의료센터":dict(stroke=1,cardiac=1,trauma=0),
          "지역응급의료기관":dict(stroke=0,cardiac=0,trauma=0),
          "응급실운영신고기관":dict(stroke=0,cardiac=0,trauma=0)}
HOSP=[]
for _,r in hos.iterrows():
    g = GRADE.get(r.emergency_type, GRADE["지역응급의료기관"])
    c, src = {}, {}
    for k, cols in CAPMAP.items():
        v = None
        if r.hospital_id in cap.index:
            vals=[cap.loc[r.hospital_id,cc] for cc in cols if cc in cap.columns]
            if any(str(x)=="Y" for x in vals): v,s=1,"MKioskTy"
            elif all(str(x)=="불가능" for x in vals) and vals: v,s=0,"MKioskTy"
        if v is None: v,s = g[k], "등급규칙"
        c[k], src[k] = v, s
    HOSP.append(dict(hospital_id=r.hospital_id, name=r.hospital_name, etype=r.emergency_type,
        lon=float(r.longitude), lat=float(r.latitude), inside_cheonan=bool(r.inside_cheonan),
        beds=int(r.available_er_beds) if pd.notna(r.available_er_beds) else 3,
        cap_source=src, **c))
print(f"병원 {len(HOSP)}개소 (천안내 {sum(h['inside_cheonan'] for h in HOSP)}) · "
      f"중증대응 심{sum(h['cardiac'] for h in HOSP)} 뇌{sum(h['stroke'] for h in HOSP)} 외상{sum(h['trauma'] for h in HOSP)}")

# ---------- 4. 실측 ETA 행렬 (Kakao 라우팅) ----------
e2r = rd("ems_to_region_accessibility.csv"); r2h = rd("region_to_hospital_accessibility.csv")
ETA_ER = {(r.station_id, r.region_name): float(r.duration_sec) for _,r in e2r.iterrows()}
ETA_RH = {(r.region_name, r.hospital_id): float(r.duration_sec) for _,r in r2h.iterrows()}
DIST_ER = {(r.station_id, r.region_name): float(r.distance_m) for _,r in e2r.iterrows()}
print(f"ETA 행렬: 거점→지역 {len(ETA_ER)} · 지역→병원 {len(ETA_RH)}")

# ---------- 5. 실측 발생 분포 ----------
d119 = pd.concat([rd(f.replace(SRC+"/","")) for f in sorted(glob.glob(f"{SRC}/analysis/data/interim/cheonan_119_*.csv"))],
                 ignore_index=True)
EMB = {r["name"] for r in REGIONS if r["rural"]}
d2 = d119.dropna(subset=["EMD_NM"])
obs = d2[d2.EMD_NM.isin(EMB)].EMD_NM.value_counts().to_dict()
urban_total = len(d2) - sum(obs.values())
upop = sum(r["pop"] for r in REGIONS if not r["rural"])
WEIGHT, SRC_KIND = {}, {}
for r in REGIONS:
    if r["rural"]:
        WEIGHT[r["name"]] = obs.get(r["name"], 0); SRC_KIND[r["name"]]="실측(읍면 1:1)"
    else:
        WEIGHT[r["name"]] = urban_total * r["pop"]/upop; SRC_KIND[r["name"]]="법정동→인구비례 배분"
HR = d119["MDLCR_DSCSN_BGNG_HR"].dropna().astype(int).value_counts().sort_index()
HRP = (HR/HR.sum()).reindex(range(24)).fillna(0).tolist()
DAILY = 90
print(f"발생 가중: 읍면 실측 {sum(obs.values()):,}건 · 도심 {urban_total:,}건 인구비례 배분")

# ---------- 6. 시뮬레이션 ----------
for r in REGIONS:
    r["Rmax_km"] = math.sqrt(r["area_km2"]/math.pi)
    r["intra_kmh"] = 35 if r["rural"] else 25

SEV, SEV_P = ["cardiac","stroke","trauma","other"], [0.12,0.10,0.18,0.60]
wsum = sum(WEIGHT.values())
CASES=[]
for day in range(DAYS):
    for r in REGIONS:
        lam = DAILY * WEIGHT[r["name"]]/wsum
        n = max(0, int(random.gauss(lam, math.sqrt(max(lam,.5)))))
        for _ in range(n):
            hr = random.choices(range(24), HRP)[0]
            th = random.uniform(0, 2*math.pi); dd = r["Rmax_km"]*math.sqrt(random.random())
            dlon = dd*math.cos(th)/88.9; dlat = dd*math.sin(th)/111.0
            CASES.append(dict(day=day, region=r["name"],
                t_call=day*86400 + hr*3600 + random.uniform(0,3600),
                sev=random.choices(SEV,SEV_P)[0],
                olon=round(r["lon"]+dlon,5), olat=round(r["lat"]+dlat,5),
                intra=dd*1.3/r["intra_kmh"]*3600,
                rolls=[random.random() for _ in range(8)], scene=random.uniform(240,480)))
CASES.sort(key=lambda c:c["t_call"])
for i,c in enumerate(CASES): c["idx"]=i
print(f"공통 케이스 {len(CASES):,}건 / {DAYS}일 (일 {len(CASES)/DAYS:.0f}건)")


def km(a,b): return math.hypot((a[0]-b[0])*88.9,(a[1]-b[1])*111.0)

def scenario(name, new_ems=None, extra_amb=None, extra_hosp=None):
    ems=[dict(e) for e in EMS]
    if extra_amb:
        for k,v in extra_amb.items():
            for e in ems:
                if k in e["name"]: e["amb"]+=v
    if new_ems:
        for j,(host,n) in enumerate(new_ems):
            r=R[host]; ems.append(dict(ems_id=f"NEW{j+1:02d}", name=f"[신설]{host}119안전센터",
                parent="신설", stype="119안전센터", lon=r["lon"], lat=r["lat"], amb=n, is_new=True))
    hosp=[dict(h) for h in HOSP]+(extra_hosp or [])

    def eta_er(sid, rn, elon, elat):
        v=ETA_ER.get((sid,rn))
        if v is not None: return v
        r=R[rn]; return km((elon,elat),(r["lon"],r["lat"]))*1.3/40*3600
    def eta_rh(rn, h):
        v=ETA_RH.get((rn,h["hospital_id"]))
        if v is not None: return v
        r=R[rn]; return km((r["lon"],r["lat"]),(h["lon"],h["lat"]))*1.3/50*3600

    # ★ 추가: 가용성을 무시한 최근접 거점 ETA (구조적 하한). 거점 집합은 시나리오마다 다르다.
    STA = {e["ems_id"]:(e["lon"],e["lat"],e["name"]) for e in ems}
    MINETA={}
    for rn in R:
        best=min(((eta_er(sid, rn, lo, la), sid) for sid,(lo,la,_) in STA.items()), key=lambda x:x[0])
        MINETA[rn]=best

    units=[]
    for e in ems:
        for k in range(e["amb"]):
            units.append(dict(amb_id=f'{e["ems_id"]}-{k+1}', base=e["ems_id"], sid=e["ems_id"],
                lon=e["lon"], lat=e["lat"], blon=e["lon"], blat=e["lat"], free_at=0.0))

    events=[]
    for c in CASES:
        r=R[c["region"]]; cid=f'{name}-C{c["idx"]+1:05d}'; sev=c["sev"]
        free=[u for u in units if u["free_at"]<=c["t_call"]] or units
        u=min(free,key=lambda x: eta_er(x["sid"],r["name"],x["blon"],x["blat"])
                                 + max(0,x["free_at"]-c["t_call"]))
        wait = max(0.0, u["free_at"]-c["t_call"])          # ★ 갱신 전에 캡처
        all_busy = not any(x["free_at"]<=c["t_call"] for x in units)
        t_disp=c["t_call"]+60+wait
        d1=eta_er(u["sid"],r["name"],u["blon"],u["blat"])
        t_scene=t_disp+d1+c["intra"]
        t_dep=t_scene+c["scene"]
        need=None if sev=="other" else sev
        cand=sorted([h for h in hosp if need is None or h[need]==1], key=lambda h: eta_rh(r["name"],h))
        if not cand: cand=sorted(hosp,key=lambda h: eta_rh(r["name"],h))
        rej,chosen=[],None
        for k,h in enumerate(cand):
            if c["rolls"][min(k,7)] < 0.18: rej.append(h["hospital_id"]); continue
            chosen=h; break
        chosen=chosen or cand[0]
        d2_=eta_rh(r["name"],chosen); t_hosp=t_dep+d2_
        back=km((chosen["lon"],chosen["lat"]),(u["blon"],u["blat"]))*1.3/55*3600
        u["free_at"]=t_hosp+900+back
        tot=t_hosp-c["t_call"]
        lim=RESP_LIMIT["rural" if r["rural"] else "urban"]
        me, me_sid = MINETA[r["name"]]
        resp = t_scene-c["t_call"]
        events.append(dict(run_id=name, case_id=cid, day=c["day"], adm_cd=r["adm_cd"], region=r["name"],
            rtype=r["rtype"], rural=r["rural"], pop=r["pop"],
            origin_lon=c["olon"], origin_lat=c["olat"], severity=sev,
            t_call=round(c["t_call"],1), t_dispatch=round(t_disp,1), t_scene=round(t_scene,1),
            t_depart=round(t_dep,1), t_hospital=round(t_hosp,1),
            amb_id=u["amb_id"], amb_base_id=u["base"], hospital_id=chosen["hospital_id"],
            hospital_outside=not chosen["inside_cheonan"], n_rejected=len(rej),
            rejected_hosp_ids="|".join(rej),
            response_sec=round(resp,1), total_sec=round(tot,1),
            resp_ok=bool(resp<=lim), golden_ok=bool(tot<=DEST_LIMIT[sev]),
            # ★ 3분할
            prep_sec=60.0,
            wait_sec=round(wait,1),                     # 대기: 전 구급차 출동 중
            d1_sec=round(d1,1),                         # 배정 거점 → 동 대표점
            intra_sec=round(c["intra"],1),              # 동 내부 이동
            min_eta_sec=round(me,1),                    # 가용성 무시 최근접 거점
            min_eta_station=me_sid,
            floor_sec=round(60.0+me+c["intra"],1),      # 구조적 하한
            dispatch_loss_sec=round(d1-me,1),           # 차선 배차 손실
            all_busy=bool(all_busy),
            resp_limit_sec=lim))
        # 항등식 검증: response = 60 + wait + d1 + intra
        assert abs(resp - (60.0+wait+d1+c["intra"])) < 1e-6
    return events, ems

NEWH=[dict(hospital_id="H900", name="[신설]북부응급의료센터", etype="지역응급의료센터",
           lon=R["성환읍"]["lon"], lat=R["성환읍"]["lat"], inside_cheonan=True, beds=6,
           stroke=1, cardiac=1, trauma=0, cap_source={})]
NEWE=[("수신면",1),("동면",1),("부성1동",1)]
SCEN={}
SCEN["baseline"]=scenario("baseline")
SCEN["policy_A"]=scenario("policy_A", new_ems=NEWE)
SCEN["policy_B"]=scenario("policy_B", extra_hosp=NEWH)
SCEN["policy_C"]=scenario("policy_C", new_ems=NEWE, extra_hosp=NEWH)

# ---------- 7. 출력 ----------
os.makedirs("out",exist_ok=True)
allev=[e for sc in SCEN for e in SCEN[sc][0]]
df=pd.DataFrame(allev)
df.to_csv("out/case_log.csv", index=False, encoding="utf-8-sig")
print(f"\ncase_log.csv: {len(df):,}행 ({len(SCEN)} 시나리오 × {len(CASES):,}건)")

summary={}
for sc,(ev,_) in SCEN.items():
    n=len(ev); rs=sorted(e["response_sec"] for e in ev)
    d=dict(n=n, days=DAYS, resp=round(sum(rs)/n/60,3), p90=round(rs[int(n*.9)-1]/60,3),
        fail=round(sum(0 if e["golden_ok"] else 1 for e in ev)/n,5),
        out=round(sum(1 if e["hospital_outside"] else 0 for e in ev)/n,5))
    for tk,(tu,tr) in THRESH.items():
        lim=lambda e:(tr if R[e["region"]]["rural"] else tu)*60
        d["respfail_"+tk]=round(sum(1 for e in ev if e["response_sec"]>lim(e))/n,5)
        d["expo_"+tk]=round(sum(max(0.0,e["response_sec"]-lim(e)) for e in ev)/60)
    # 3분할 평균(분)
    d["floor_min"]=round(sum(e["floor_sec"] for e in ev)/n/60,3)
    d["wait_min"]=round(sum(e["wait_sec"] for e in ev)/n/60,3)
    d["loss_min"]=round(sum(e["dispatch_loss_sec"] for e in ev)/n/60,3)
    d["wait_share"]=round(sum(e["wait_sec"] for e in ev)/sum(e["response_sec"] for e in ev),4)
    d["allbusy_rate"]=round(sum(1 for e in ev if e["all_busy"])/n,4)
    summary[sc]=d
json.dump(summary, open("out/summary_check.json","w"), ensure_ascii=False, indent=1)

print("\n=== 재현 검증 (표 값과 비교) ===")
print(f"{'시나리오':10s} {'평균':>7s} {'p90':>7s} {'T10초과':>8s} | {'하한':>6s} {'대기':>6s} {'차선손실':>7s}")
for sc,v in summary.items():
    print(f"{sc:10s} {v['resp']:6.2f}분 {v['p90']:6.2f}분 {v['respfail_T10']*100:7.1f}% | "
          f"{v['floor_min']:5.2f} {v['wait_min']:5.2f} {v['loss_min']:6.2f}분  (대기비중 {v['wait_share']*100:.1f}%, 전대출동 {v['allbusy_rate']*100:.1f}%)")
