# -*- coding: utf-8 -*-
"""인터랙티브 지도용 데이터 — 경계 · 5축 원값 · 거점/병원 ETA 행렬"""
import json, os, math
import pandas as pd, geopandas as gpd, numpy as np

SRC = "/mnt/user-data/uploads/비타민 최종 프로젝트"
def rd(p):
    for e in ("utf-8-sig","cp949","euc-kr"):
        try: return pd.read_csv(f"{SRC}/{p}", encoding=e, low_memory=False)
        except Exception as ex: last = ex
    raise last

geo = gpd.read_file(f"{SRC}/analysis/data/raw/cheonan_hjd_ver20260401.geojson")
geo["name"] = geo["adm_nm"].str.split().str[-1]
m = geo.to_crs(5179)
geo["area_km2"] = (m.geometry.area/1e6).round(2)
geo["geometry"] = m.geometry.simplify(60).to_crs(4326)

reg = rd("cheonan_region_db.csv")
trf = rd("교통_읍면동_지수.csv").set_index("행정동")
vul = rd("취약인구_읍면동_지수.csv").set_index("행정동")
e2r = rd("ems_to_region_accessibility.csv")
r2h = rd("region_to_hospital_accessibility.csv")
ems = rd("cheonan_ems_station_db.csv")
hos = rd("cheonan_emergency_hospital_db.csv")

# ETA 행렬 (분)
e2r["min"] = e2r.duration_sec/60
r2h["min"] = r2h.duration_sec/60
ER = {}   # region -> {station_id: min}
for _, r in e2r.iterrows(): ER.setdefault(r.region_name, {})[r.station_id] = round(float(r["min"]), 3)
RH = {}   # region -> {hospital_id: min}
for _, r in r2h.iterrows(): RH.setdefault(r.region_name, {})[r.hospital_id] = round(float(r["min"]), 3)

regions, feats = [], []
for _, r in reg.iterrows():
    n = r.region_name
    g = geo[geo.name == n].iloc[0]
    gj = json.loads(gpd.GeoSeries([g.geometry], crs=4326).to_json())["features"][0]["geometry"]
    def rnd(c):
        if isinstance(c[0], (int, float)): return [round(c[0],5), round(c[1],5)]
        return [rnd(x) for x in c]
    gj["coordinates"] = rnd(gj["coordinates"])
    feats.append(dict(type="Feature", properties=dict(name=n), geometry=gj))
    regions.append(dict(
        name=n, gu=r.gu_name, rtype=r.region_type, pop=int(r.population),
        elderly=int(r.elderly_population), lon=round(float(r.longitude),5), lat=round(float(r.latitude),5),
        area=float(g.area_km2),
        traffic=(round(float(trf.loc[n,"교통혼잡도"]),4)
                 if n in trf.index and pd.notna(trf.loc[n,"교통혼잡도"]) else None),
        demand=(round(float(vul.loc[n,"취약인구지수_V2"]),4) if n in vul.index else None),
        er=ER.get(n, {}), rh=RH.get(n, {}),
    ))

# ---- 애니메이션 시뮬용 파라미터 (real_build.py 와 동일 규칙) ----
import glob, math
d119 = pd.concat([rd(f.replace(SRC+"/","")) for f in sorted(glob.glob(f"{SRC}/analysis/data/interim/cheonan_119_*.csv"))],
                 ignore_index=True)
EMB = {r["name"] for r in regions if r["rtype"] in ("읍","면")}
d2 = d119.dropna(subset=["EMD_NM"])
obs = d2[d2.EMD_NM.isin(EMB)].EMD_NM.value_counts().to_dict()
urban_total = len(d2) - sum(obs.values())
upop = sum(r["pop"] for r in regions if r["rtype"] not in ("읍","면"))
for r in regions:
    rural = r["rtype"] in ("읍","면")
    r["rural"] = rural
    r["w"] = round(float(obs.get(r["name"],0) if rural else urban_total*r["pop"]/upop), 2)
    r["src"] = "실측(읍면 1:1)" if rural else "법정동→인구비례 배분"
    r["obs5y"] = int(obs.get(r["name"],0)) if rural else None
    r["rmax"] = round(math.sqrt(r["area"]/math.pi), 4)
    r["ikmh"] = 35 if rural else 25
HR = d119["MDLCR_DSCSN_BGNG_HR"].dropna().astype(int).value_counts().sort_index()
HRP = [round(float(x),5) for x in (HR/HR.sum()).reindex(range(24)).fillna(0)]

cap = rd("analysis/data/raw/hospital_capabilities.csv").set_index("hpid")
CAPMAP = {"stroke":["MKioskTy1","MKioskTy2"], "cardiac":["MKioskTy3"], "trauma":["MKioskTy4","MKioskTy5"]}
GRADE  = {"권역응급의료센터":dict(stroke=1,cardiac=1,trauma=1),
          "지역응급의료센터":dict(stroke=1,cardiac=1,trauma=0),
          "지역응급의료기관":dict(stroke=0,cardiac=0,trauma=0),
          "응급실운영신고기관":dict(stroke=0,cardiac=0,trauma=0)}
def caps(hid, etype):
    g = GRADE.get(etype, GRADE["지역응급의료기관"]); c = {}
    for k, cols in CAPMAP.items():
        v = None
        if hid in cap.index:
            vals=[cap.loc[hid,cc] for cc in cols if cc in cap.columns]
            if any(str(x)=="Y" for x in vals): v=1
            elif vals and all(str(x)=="불가능" for x in vals): v=0
        c[k] = g[k] if v is None else v
    return c

stations = [dict(id=r.station_id, name=r.station_name, type=r.station_type,
                 lon=round(float(r.longitude),5), lat=round(float(r.latitude),5),
                 amb=int(r.ambulance_count)) for _, r in ems.iterrows()]
hospitals = [dict(id=r.hospital_id, name=r.hospital_name, etype=r.emergency_type,
                  lon=round(float(r.longitude),5), lat=round(float(r.latitude),5),
                  inside=bool(r.inside_cheonan),
                  beds=int(r.available_er_beds) if pd.notna(r.available_er_beds) else 3,
                  **caps(r.hospital_id, r.emergency_type))
             for _, r in hos.iterrows()]

out = dict(geo=dict(type="FeatureCollection", features=feats),
           regions=regions, stations=stations, hospitals=hospitals,
           sim=dict(hrp=HRP, daily=90, sev=["cardiac","stroke","trauma","other"],
                    sevp=[0.12,0.10,0.18,0.60], reject=0.18, prep_sec=60,
                    scene=[240,480], handover=900, back_kmh=55, detour=1.3,
                    obs_total=int(len(d2)), obs_rural=int(sum(obs.values()))),
           cfg=dict(relative_pct=0.70, severity_gate_pct=0.667,
                    multi_min_flags=3, gate_at_flags=2,
                    axes=[dict(k="dispatch", label="구급차 출동 취약형", short="출동", unit="분",
                               desc="119 거점 → 환자 도착이 늦다", policy="구급차 증차·전진배치 (정책①)"),
                          dict(k="transport", label="응급실 이송 취약형", short="이송", unit="분",
                               desc="환자 → 응급실 이동이 멀다", policy="기존 병원 응급기능 강화 (정책②)"),
                          dict(k="backup", label="대체병원 부족형", short="대체", unit="분",
                               desc="1순위가 못 받으면 2순위가 너무 멀다", policy="응급환자 수용 네트워크 (정책③)"),
                          dict(k="traffic", label="교통장애형", short="교통", unit="지수",
                               desc="혼잡·불법주정차로 이동 지연", policy="긴급차량 우선신호·주정차 단속 (정책④)"),
                          dict(k="demand", label="응급수요 고위험형", short="수요", unit="지수",
                               desc="고령·영유아 비율이 높아 발생이 잦다", policy="고령층 응급안전망·사전예방")],
                    none_label="상대적 안정형", multi_label="복합 취약형",
                    multi_policy="복합 집중투자 — 주도 축부터", none_policy="추가 개입 불필요",
                    new_station_kmh=40, new_hospital_kmh=50, detour=1.3))
json.dump(out, open("out/viz_map.json","w"), ensure_ascii=False, separators=(",",":"))
print("regions", len(regions), "· stations", len(stations), "· hospitals", len(hospitals),
      "·", round(os.path.getsize("out/viz_map.json")/1024,1), "KB")
print("교통 결측:", sum(1 for r in regions if r["traffic"] is None), "곳")
