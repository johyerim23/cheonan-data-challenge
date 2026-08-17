"""Build the standalone RL dashboard from the supplied simulator HTML.

The map/dispatch simulator is retained. The embedded GNN weights and browser
inference are removed, then the evaluated DQN result layer is injected.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE / "base_simulator.html"
LAYER = HERE / "rl_dashboard.js"
OUTPUT = HERE / "천안_RL_배치시뮬레이터.html"
RESULTS = ROOT / "results" / "RL" / "refined_with_dispatch"


def records(path: Path) -> list[dict]:
    frame = pd.read_csv(path)
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def load_payload() -> dict:
    summary = json.loads((RESULTS / "final_policy_summary.json").read_text(encoding="utf-8"))
    policies = records(RESULTS / "final_selected_policies.csv")
    regions = records(RESULTS / "final_before_after_by_region.csv")
    comparison = records(RESULTS / "model_comparison.csv")
    seeds = records(RESULTS / "dqn_seed_evaluation.csv")
    trace_path = HERE / "dqn_trace.json"
    if not trace_path.exists():
        raise FileNotFoundError(
            "dqn_trace.json이 없습니다. 먼저 export_dqn_trace.py를 실행하세요."
        )
    traces = json.loads(trace_path.read_text(encoding="utf-8"))["traces"]

    labels = {
        "random_1000": "Random 1,000회",
        "rule_greedy": "Rule-based Greedy",
        "dqn_50000_multi_seed": "DQN 50k · 3 seeds",
    }
    comparison = [
        {
            "method": row["method"],
            "label": labels.get(row["method"], row["method"]),
            "reward_mean": row["reward_mean"],
            "reward_std": row["reward_std"],
            "weighted_total_time": row["weighted_total_time"],
        }
        for row in comparison
    ]
    seeds = [
        {
            "seed": row["seed"],
            "total_timesteps": row["total_timesteps"],
            "reward": row["reward"],
            "weighted_total_time": row["weighted_total_time"],
            "has_repeated_action": bool(row["has_repeated_action"]),
        }
        for row in seeds
    ]
    return {
        "summary": summary,
        "policies": policies,
        "regions": regions,
        "comparison": comparison,
        "seeds": seeds,
        "traces": traces,
        "max_reward": max(row["reward_mean"] for row in comparison),
        "model": {"algorithm": "DQN", "state_dim": 396, "network": [396, 64, 64, 54], "action_dim": 54},
        "locations": {
            "수신면": {"lat": 36.725209, "lon": 127.285805},
            "동면": {"lat": 36.781065, "lon": 127.367161},
            "서울대정병원": {"lat": 36.811595, "lon": 127.108008},
        },
    }


def build() -> Path:
    html = BASE.read_text(encoding="utf-8")
    payload = json.dumps(load_payload(), ensure_ascii=False, separators=(",", ":"))
    layer = LAYER.read_text(encoding="utf-8").replace("__RL_DATA_JSON__", payload)

    # Drop the very large GNN weight object and its browser inference code.
    lines = html.splitlines(keepends=True)
    lines = [line for line in lines if not line.startswith("const GW=")]
    html = "".join(lines)
    gnn_start = html.find("/* ================= GNN 대체모델 — 브라우저 추론")
    render_call = html.rfind("render();")
    if gnn_start < 0 or render_call < 0 or render_call <= gnn_start:
        raise RuntimeError("Could not locate the GNN block or final render call in base HTML")
    html = html[:gnn_start] + layer + "\n\n" + html[render_call:]

    replacements = {
        "천안 응급의료 — 배치 시뮬레이터 & 검증 결과": "천안 응급의료 — DQN RL 배치 시뮬레이터",
        "천안 응급의료 — 배치 시뮬레이터 &amp; 검증 결과": "천안 응급의료 — DQN RL 배치 시뮬레이터",
        "행정동 31곳 · 거점 16 · 응급의료기관 18 · 정책 21종 · 2026-08-16":
            "행정동 31곳 · DQN 50,000 steps · 54 actions · 정책 예산 3",
        "지도 · 배치 시뮬레이터": "지도 · RL 배치 시뮬레이터",
        ">검증 대시보드<": ">RL 검증 대시보드<",
        "출처: <code>viz/case_log/</code>, <code>gnn/</code> · 시뮬레이터 재현 검증 통과(10.62 / 15.30 / 46.6%) ·\n  타깃 y = d1 − min_eta (차선배차 손실) · 이 화면의 모든 수치는 시뮬레이션 산출물이며 실측 관측이 아니다.":
            "RL 결과: <code>results/RL/refined_with_dispatch/</code> · 모델: DQN 50,000 steps × 3 seeds ·\n  지도 애니메이션은 원본 배차 시뮬레이터를 유지하며, 표시 수치는 정책 평가용 시뮬레이션 산출물이다.",
    }
    for old, new in replacements.items():
        html = html.replace(old, new)

    extra_css = """
<style>
.rl-policies{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px;margin:10px 0 12px}
.rl-policy{display:flex;align-items:center;gap:9px;padding:10px;border:1px solid var(--line);border-radius:9px;background:var(--surface-2)}
.rl-policy div div{font-size:12px;color:var(--text-secondary);margin-top:2px}.rl-step{display:grid;place-items:center;width:25px;height:25px;border-radius:50%;background:var(--s1);color:#fff;font-weight:700;flex:0 0 auto}
.rl-bar{height:9px;min-width:130px;background:var(--surface-2);border-radius:99px;overflow:hidden}.rl-bar i{display:block;height:100%;background:linear-gradient(90deg,var(--s1),var(--s2));border-radius:inherit}
.dqn-layout{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(230px,.75fr);gap:16px;align-items:start;margin:8px 0 12px}
.q-row{display:grid;grid-template-columns:24px minmax(120px,1fr) minmax(100px,1.45fr) 58px;gap:8px;align-items:center;padding:6px 8px;border-radius:7px;font-size:12px}
.q-row:nth-child(odd){background:var(--surface-2)}.q-row.selected{box-shadow:inset 3px 0 0 var(--s2);background:color-mix(in srgb,var(--s2) 12%,var(--surface-1))}
.q-rank{text-align:center;color:var(--text-muted);font-variant-numeric:tabular-nums}.q-name{min-width:0}.q-name b{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.q-name small{display:block;color:var(--text-muted);margin-top:1px}
.q-track{height:10px;background:var(--surface-2);border-radius:99px;overflow:hidden}.q-track i{display:block;height:100%;border-radius:inherit;background:var(--s1)}.q-row.selected .q-track i{background:var(--s2)}.q-value{text-align:right;font-variant-numeric:tabular-nums;font-weight:650}
.dqn-decision{padding:12px;border:1px solid var(--line);border-radius:10px;background:var(--surface-2)}.dqn-choice{margin:9px 0 12px;padding:12px;border-radius:8px;background:var(--surface-1);border-left:3px solid var(--s2)}.dqn-choice small,.dqn-choice span{display:block;color:var(--text-muted);font-size:11px}.dqn-choice strong{display:block;font-size:19px;margin:3px 0}
.dqn-path{display:flex;flex-direction:column;gap:5px;margin-top:10px}.dqn-path-item{display:flex;align-items:center;gap:7px;opacity:.62}.dqn-path-item.current{opacity:1}.dqn-path-item>span{display:grid;place-items:center;width:21px;height:21px;border-radius:50%;background:var(--s1);color:#fff;font-size:11px;font-weight:700}.dqn-path-item small{display:block;color:var(--text-muted)}
.dqn-step-buttons{display:flex;gap:4px}
@media(max-width:850px){.dqn-layout{grid-template-columns:1fr}.rl-policies{grid-template-columns:1fr}.rl-bar{min-width:70px}.q-row{grid-template-columns:22px minmax(105px,1fr) minmax(70px,1fr) 52px}}
</style>
"""
    html = html.replace("</head>", extra_css + "</head>")
    OUTPUT.write_text(html, encoding="utf-8")
    return OUTPUT


if __name__ == "__main__":
    output = build()
    print(output)
