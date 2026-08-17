# -*- coding: utf-8 -*-
"""시나리오별 이종 그래프 구성.

  Patient  ─ precedes ─→ Patient      같은 min_sid 공유 · 0 < Δt < W · 과거→미래 단방향
  Station  ─ can_reach ─→ Patient      k=5 최근접 · edge_attr = eta_sec

누수 방지
  · n_free 는 입력에서 제외 (전역이라도 큐 상태 요약값. GBM 단독 AUC 0.731)
  · occupy_sec 는 노드 피처가 아니라 '지역별 평균 lookup' 으로만 사용하며,
    그 lookup 은 **train 분할에서만** 계산한다
  · precedes 는 단방향. 양방향이면 미래가 과거에 영향을 준다
"""
import numpy as np, pandas as pd, argparse, os, json

PAT_FEAT = ["t_call_h","sin_h","cos_h","intra_sec","min_eta","eta2","gap21","min_amb","rural","daily"]

SEVK=["cardiac","stroke","trauma","other"]

def build_scenario(g, occ_lut, occ_default, W, kmax=None, topk=3, hlut=None):
    """g: 한 시나리오의 DataFrame. topk: 후보 거점 공유 범위(1=최근접만).
       top-1 만 쓰면 파급 차단(최근접이 만차라 2순위로 넘어간 선행 케이스)이 누락돼
       y>0 의 8.2%가 in-degree 0 이 된다. top-3 에서 0.33%."""
    from collections import deque
    g = g.sort_values("t_call", kind="mergesort").reset_index(drop=True)
    t = g.t_call.to_numpy(np.float64)
    CAND = [c for c in ["min_sid","sid2","sid3"][:topk] if c in g.columns]
    S = [tuple(dict.fromkeys(x)) for x in g[CAND].astype(str).to_numpy()]
    reg = g.region.astype(str).to_numpy()
    # --- precedes: 후보 거점 집합이 겹치고 0 < Δt < W, past→future (단방향) ---
    src, dst, eatt = [], [], []
    by_sid = {}
    for i in range(len(g)):
        seen = set()
        for s in S[i]:
            lst = by_sid.setdefault(s, deque())
            while lst and t[i] - t[lst[0]] >= W: lst.popleft()
            seen.update(lst)
        seen.discard(i)
        prev = sorted(seen)
        if kmax is not None: prev = prev[-kmax:]
        for j in prev:
            src.append(j); dst.append(i)
            eatt.append((t[i]-t[j], occ_lut.get(reg[j], occ_default)))
        for s in S[i]: by_sid[s].append(i)
    prec = np.array([src, dst], dtype=np.int64) if src else np.zeros((2,0), np.int64)
    pattr = np.array(eatt, dtype=np.float32) if eatt else np.zeros((0,2), np.float32)
    # --- 노드 피처 ---
    h = (t % 86400) / 3600.0
    sev = g.sev.astype(str).to_numpy()
    S1H = np.stack([(sev == k).astype(float) for k in SEVK], 1)      # ★ 중증도 원핫
    X = np.column_stack([
        h, np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24),
        g.intra_sec, g.min_eta, g.eta2, g.eta2 - g.min_eta,
        g.min_amb, g.rural.astype(float), g.daily, S1H,
    ]).astype(np.float32)
    y = g.y.to_numpy(np.float32)
    # 거점 노드: 이 시나리오에 등장하는 후보 거점 전체
    allsid = pd.unique(np.concatenate([g[c].astype(str).to_numpy() for c in CAND]))
    smap = {s_: i for i, s_ in enumerate(allsid)}
    cand_sid = np.stack([g[c].astype(str).map(smap).to_numpy() for c in CAND], 1).astype(np.int64)
    cand_eta = np.stack([g[c].to_numpy() for c in ["min_eta","eta2","eta3"][:len(CAND)]], 1).astype(np.float32)
    # 거점별 보유 대수: 그 거점이 최근접인 행의 min_amb 에서 복원
    amb = np.zeros(len(allsid), np.float32)
    for s_, v in g.groupby(g.min_sid.astype(str), observed=True).min_amb.first().items():
        if s_ in smap: amb[smap[s_]] = v
    # ★ 병원 노드 — 이 시나리오에 등장하는 후보 병원 전체
    HC = [c for c in ["hid1","hid2","hid3"] if c in g.columns]
    if HC and hlut is not None:
        allh = pd.unique(np.concatenate([g[c].astype(str).to_numpy() for c in HC]))
        hmap = {h_: i for i, h_ in enumerate(allh)}
        hos_id = np.stack([g[c].astype(str).map(hmap).to_numpy() for c in HC], 1).astype(np.int64)
        hos_eta = np.stack([g[c].to_numpy() for c in ["heta1","heta2","heta3"][:len(HC)]], 1).astype(np.float32)
        DEFH = dict(cardiac=0, stroke=0, trauma=0, beds=3, inside=1)
        hx = np.array([[float(hlut.get(h_, DEFH)[k]) for k in ("cardiac","stroke","trauma","beds","inside")]
                       for h_ in allh], dtype=np.float32)
    else:
        hos_id = np.zeros((len(g), 0), np.int64); hos_eta = np.zeros((len(g), 0), np.float32)
        hx = np.zeros((0, 5), np.float32); allh = np.array([], dtype=object)
    return dict(x=X, y=y, prec=prec, prec_attr=pattr,
                cand_sid=cand_sid, cand_eta=cand_eta,
                sid_amb=amb, sid_names=np.array(allsid, dtype=object),
                hos_id=hos_id, hos_eta=hos_eta, hos_x=hx, hos_names=np.array(allh, dtype=object),
                region=reg.astype(object), n=len(g))

def main(a):
    D = pd.read_parquet(a.data)
    TRAIN_SEEDS = [int(s) for s in a.train_seeds.split(",")]
    tr = D[D.case_seed.isin(TRAIN_SEEDS)]
    # ★ 점유시간 lookup 은 train 에서만 — test 점유시간이 피처로 새면 안 된다
    occ_lut = (tr.groupby("region", observed=True).occupy_sec.mean()).to_dict()
    occ_default = float(tr.occupy_sec.mean())
    json.dump({"lut": {k: float(v) for k, v in occ_lut.items()}, "default": occ_default,
               "train_seeds": TRAIN_SEEDS},
              open(os.path.join(a.out, "occupy_lut.json"), "w"), ensure_ascii=False, indent=1)

    # 병원 정적 피처 (신설 NEWH1 은 시나리오 메타에서)
    HL = {h["hospital_id"]: h for h in json.load(open("out/hospitals.json"))}
    SM = pd.read_csv("out/gnn_scenarios.csv").set_index("scen")
    stats = []
    os.makedirs(a.out, exist_ok=True)
    for W in [w*60 for w in a.windows]:
        deg, ne, npat, nz0, nyp = [], 0, 0, 0, 0
        for sc, g in D.groupby("scen", sort=True):
            hl = dict(HL)
            try:
                for nh in json.loads(SM.loc[sc, "new_hosp"]):
                    hl[nh["hospital_id"]] = dict(cardiac=nh["cardiac"], stroke=nh["stroke"],
                                                 trauma=nh["trauma"], beds=nh["beds"], inside=1)
            except Exception: pass
            b = build_scenario(g, occ_lut, occ_default, W, a.kmax, a.topk, hl)
            ne += b["prec"].shape[1]; npat += b["n"]
            d = np.bincount(b["prec"][1], minlength=b["n"]) if b["prec"].shape[1] else np.zeros(b["n"],int)
            deg.append(d)
            yp = b["y"] > 0
            nz0 += int((yp & (d == 0)).sum()); nyp += int(yp.sum())
            if a.save and W == a.windows[0]*60 and sc < a.save:
                np.savez_compressed(os.path.join(a.out, f"g_{sc:04d}.npz"), **b)
        d = np.concatenate(deg)
        stats.append(dict(window_min=W//60, edges=ne, patients=npat,
                          edges_per_node=ne/npat, deg_mean=float(d.mean()),
                          deg_p50=float(np.percentile(d,50)), deg_p90=float(np.percentile(d,90)),
                          deg_p99=float(np.percentile(d,99)), deg_max=int(d.max()),
                          topk=a.topk, ypos_indeg0=nz0/max(nyp,1), mem_MB=ne*24/1e6))
        print(f"W={W//60:3d}분 · 엣지 {ne:>10,} ({ne/npat:5.2f}/노드) · "
              f"degree 평균{d.mean():5.2f} p90 {np.percentile(d,90):4.0f} p99 {np.percentile(d,99):4.0f} "
              f"max {d.max():4d} · y>0 중 in-deg0 {nz0/max(nyp,1)*100:5.2f}% · {ne*24/1e6:6.1f}MB")
    pd.DataFrame(stats).to_csv(os.path.join(a.out, "edge_stats.csv"), index=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="out/gnn_dataset.parquet")
    p.add_argument("--out", default="out/graphs")
    p.add_argument("--windows", type=int, nargs="+", default=[60])
    p.add_argument("--topk", type=int, default=3, help="후보 거점 공유 범위")
    p.add_argument("--kmax", type=int, default=None, help="precedes 선행자 상한 (최근 k개)")
    p.add_argument("--train-seeds", default="11,22,33,44,55,66")
    p.add_argument("--save", type=int, default=0, help="앞 N개 시나리오를 npz로 저장")
    main(p.parse_args())
