# -*- coding: utf-8 -*-
"""2단(분류+회귀) 이종 GNN — 차선배차 손실 y = d1 − min_eta 의 대체모델.

  Patient ─precedes→ Patient   (과거→미래 단방향, edge_attr=[Δt, 선행지역 평균점유])
  Station ─can_reach→ Patient   (top-3, edge_attr=[eta])
  aggr='sum'  — 경합은 '몇 명이 겹쳤나'의 문제라 mean 이면 정보가 사라진다

분할: case_seed 그룹. 시나리오 단위로 나누면 환자 집합이 공유돼 누수.
"""
import numpy as np, pandas as pd, torch, glob, json, argparse, time
import torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HeteroConv, TransformerConv
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

def load(path, meta, variant="hosp"):
    d = np.load(path, allow_pickle=True)
    n = int(d["n"]); cs = d["cand_sid"]; ce = d["cand_eta"]
    h = HeteroData()
    X = torch.tensor(d["x"])
    if variant == "base": X = X[:, :10]        # 중증도 원핫 제거
    h["patient"].x = X
    h["patient"].y = torch.tensor(d["y"])
    h["station"].x = torch.tensor(np.column_stack([d["sid_amb"]]).astype(np.float32))
    h["patient","precedes","patient"].edge_index = torch.tensor(d["prec"])
    h["patient","precedes","patient"].edge_attr = torch.tensor(d["prec_attr"])
    k = cs.shape[1]
    src = torch.tensor(cs.T.reshape(-1))
    dst = torch.tensor(np.tile(np.arange(n), k))
    h["station","can_reach","patient"].edge_index = torch.stack([src, dst])
    h["station","can_reach","patient"].edge_attr = torch.tensor(ce.T.reshape(-1,1))
    if variant == "hosp" and "hos_x" in d.files and d["hos_x"].shape[0] > 0:
        hi = d["hos_id"]; he = d["hos_eta"]; kh = hi.shape[1]
        h["hospital"].x = torch.tensor(d["hos_x"])
        h["hospital","serves","patient"].edge_index = torch.stack([
            torch.tensor(hi.T.reshape(-1)), torch.tensor(np.tile(np.arange(n), kh))])
        h["hospital","serves","patient"].edge_attr = torch.tensor(he.T.reshape(-1,1))
    h.scen = int(meta["scen"]); h.case_seed = int(meta["case_seed"]); h.daily = int(meta["daily"])
    return h

class DelayNet(nn.Module):
    def __init__(self, din_p, din_s, hid=64, din_h=0):
        super().__init__()
        self.use_h = din_h > 0
        self.lin_p = nn.Linear(din_p, hid); self.lin_s = nn.Linear(din_s, hid)
        if self.use_h: self.lin_h = nn.Linear(din_h, hid)
        rel = {("station","can_reach","patient"): TransformerConv((hid,hid), hid, edge_dim=1),
               ("patient","precedes","patient"):  TransformerConv((hid,hid), hid, edge_dim=2)}
        if self.use_h:
            rel[("hospital","serves","patient")] = TransformerConv((hid,hid), hid, edge_dim=1)
        mk = lambda: HeteroConv({k: v.__class__((hid,hid), hid, edge_dim=v.edge_dim) for k,v in rel.items()},
                                aggr="sum")
        self.c1, self.c2 = mk(), mk()
        self.head = nn.Sequential(nn.Linear(hid*2, hid), nn.ReLU())
        self.clf = nn.Linear(hid, 1); self.reg = nn.Linear(hid, 1)
    def forward(self, b):
        x = {"patient": F.relu(self.lin_p(b["patient"].x)),
             "station": F.relu(self.lin_s(b["station"].x))}
        if self.use_h: x["hospital"] = F.relu(self.lin_h(b["hospital"].x))
        ei, ea = b.edge_index_dict, b.edge_attr_dict
        h1 = self.c1(x, ei, edge_attr_dict=ea); h1 = {k: F.relu(v) for k, v in h1.items()}
        h1["station"] = x["station"]
        if self.use_h: h1["hospital"] = x["hospital"]
        h2 = self.c2(h1, ei, edge_attr_dict=ea); h2 = {k: F.relu(v) for k, v in h2.items()}
        z = self.head(torch.cat([h1["patient"], h2["patient"]], 1))
        return self.clf(z).squeeze(-1), self.reg(z).squeeze(-1)

def evaluate(model, loader, dev):
    model.eval(); P, Yb, Ym, S = [], [], [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(dev); lo, mu = model(b)
            y = b["patient"].y
            P.append(torch.sigmoid(lo).cpu()); Yb.append((y > 0).float().cpu())
            Ym.append(y.cpu()); S.append((torch.sigmoid(lo)*torch.expm1(mu.clamp(max=8))).cpu())
    p = torch.cat(P).numpy(); yb = torch.cat(Yb).numpy(); ym = torch.cat(Ym).numpy(); yh = torch.cat(S).numpy()
    m = ym > 0
    return dict(auc=roc_auc_score(yb, p), mae_pos=float(np.abs(yh[m]-ym[m]).mean()/60),
                mae_all=float(np.abs(yh-ym).mean()/60))

def main(a):
    M = pd.read_csv("out/gnn_scenarios.csv").set_index("scen")
    files = sorted(glob.glob("out/graphs/g_*.npz"))
    TR = [11,22,33,44,55,66]; VA = [77,88]; TE = [99,1010]
    tr, va, te = [], [], []
    for f in files:
        sc = int(f.split("_")[-1].split(".")[0]); mt = M.loc[sc]
        if a.limit and len(tr)+len(va)+len(te) >= a.limit: break
        g = load(f, dict(scen=sc, case_seed=mt.case_seed, daily=mt.daily), a.variant)
        (tr if mt.case_seed in TR else va if mt.case_seed in VA else te).append(g)
    print(f"그래프 train {len(tr)} / val {len(va)} / test {len(te)}")
    # ★ 표준화 통계는 train 에서만
    XP = torch.cat([g["patient"].x for g in tr]); XS = torch.cat([g["station"].x for g in tr])
    EP = torch.cat([g["patient","precedes","patient"].edge_attr for g in tr])
    EC = torch.cat([g["station","can_reach","patient"].edge_attr for g in tr])
    st = {"p": (XP.mean(0), XP.std(0)+1e-6), "s": (XS.mean(0), XS.std(0)+1e-6),
          "ep": (EP.mean(0), EP.std(0)+1e-6), "ec": (EC.mean(0), EC.std(0)+1e-6)}
    if a.variant == "hosp":
        XH = torch.cat([g["hospital"].x for g in tr])
        EH = torch.cat([g["hospital","serves","patient"].edge_attr for g in tr])
        st["h"] = (XH.mean(0), XH.std(0)+1e-6); st["eh"] = (EH.mean(0), EH.std(0)+1e-6)
    for g in tr+va+te:
        g["patient"].x = (g["patient"].x - st["p"][0]) / st["p"][1]
        g["station"].x = (g["station"].x - st["s"][0]) / st["s"][1]
        e = g["patient","precedes","patient"]; e.edge_attr = (e.edge_attr - st["ep"][0]) / st["ep"][1]
        e2 = g["station","can_reach","patient"]; e2.edge_attr = (e2.edge_attr - st["ec"][0]) / st["ec"][1]
        if a.variant == "hosp":
            g["hospital"].x = (g["hospital"].x - st["h"][0]) / st["h"][1]
            e3 = g["hospital","serves","patient"]; e3.edge_attr = (e3.edge_attr - st["eh"][0]) / st["eh"][1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dh = int(tr[0]["hospital"].x.shape[1]) if a.variant == "hosp" else 0
    model = DelayNet(tr[0]["patient"].x.shape[1], tr[0]["station"].x.shape[1], 64, dh).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    L_tr = DataLoader(tr, batch_size=a.bs, shuffle=True)
    L_va = DataLoader(va, batch_size=a.bs); L_te = DataLoader(te, batch_size=a.bs)
    best, bstate = -1, None
    for ep in range(a.epochs):
        model.train(); tot = 0; t0 = time.time()
        for b in L_tr:
            b = b.to(dev); opt.zero_grad()
            lo, mu = model(b); y = b["patient"].y; pos = y > 0
            loss = F.binary_cross_entropy_with_logits(lo, pos.float())
            if pos.any(): loss = loss + F.mse_loss(mu[pos], torch.log1p(y[pos]))
            loss.backward(); opt.step(); tot += float(loss.detach())
        v = evaluate(model, L_va, dev)
        print(f"  ep{ep+1:2d} loss {tot/len(L_tr):.4f} · val AUC {v['auc']:.4f} · 조건부MAE {v['mae_pos']:.2f}분 · {time.time()-t0:.0f}s")
        if v["auc"] > best: best, bstate = v["auc"], {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(bstate)
    t = evaluate(model, L_te, dev)
    print(f"\n=== TEST (case_seed {TE}) ===\n  AUC {t['auc']:.4f} · 조건부 MAE {t['mae_pos']:.2f}분 · 전체 MAE {t['mae_all']:.2f}분")

    # --- 정책 순위 일치 (시나리오 평균 y) ---
    model.eval(); rows = []
    with torch.no_grad():
        for g in te:
            b = g.to(dev); lo, mu = model(b)
            yh = (torch.sigmoid(lo)*torch.expm1(mu.clamp(max=8))).cpu().numpy()
            rows.append((g.scen, float(g["patient"].y.mean())/60, float(yh.mean())/60))
    r = pd.DataFrame(rows, columns=["scen","sim","gnn"])
    rho = spearmanr(r.sim, r.gnn)
    print(f"  정책 순위 Spearman ρ={rho.statistic:.4f} (p={rho.pvalue:.1e}, n={len(r)} 시나리오)")
    r.to_csv(f"out/gnn_policy_rank_{a.variant}.csv", index=False)
    json.dump(dict(test=t, spearman=float(rho.statistic), n_test_scen=len(r)),
              open(f"out/gnn_result_{a.variant}.json","w"), indent=1)

    # ---- 가중치 + 정규화 통계 내보내기 (브라우저 추론용) ----
    torch.save(bstate, f"out/gnn_weights_{a.variant}.pt")
    def arr(t): return [round(float(x),6) for x in t.detach().cpu().reshape(-1)]
    W={k:{"shape":list(v.shape),"data":arr(v)} for k,v in model.state_dict().items()}
    norm={k:{"mean":arr(st[k][0]),"std":arr(st[k][1])} for k in st}
    lut=json.load(open("out/graphs/occupy_lut.json"))
    json.dump(dict(weights=W, norm=norm, occupy_lut=lut,
                   arch=dict(hid=64, din_p=int(tr[0]["patient"].x.shape[1]),
                             din_s=int(tr[0]["station"].x.shape[1]), din_h=dh, heads=1,
                             variant=a.variant, edge_dim=dict(can_reach=1, precedes=2, serves=1)),
                   test=t, spearman=float(rho.statistic)),
              open(f"out/gnn_weights_{a.variant}.json","w"), separators=(",",":"))
    import os
    print("가중치 저장:", round(os.path.getsize("out/gnn_weights.json")/1024,1),"KB ·",
          sum(v.numel() for v in bstate.values()), "파라미터")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--variant", default="hosp", choices=["base","sev","hosp"])
    main(p.parse_args())
