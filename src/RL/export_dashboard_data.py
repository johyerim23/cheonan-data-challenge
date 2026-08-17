"""Export deterministic current-vs-RL playback data for the HTML dashboard."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dispatch_simulator import AmbulanceDispatchSimulator
from policy_actions import apply_policy
from simulator import EmergencyPolicySimulator


DEFAULT_POLICIES = "results/RL/refined_with_dispatch/final_selected_policies.csv"


def _hospital_lookup(sim):
    emergency = sim.emergency_hospitals.rename(
        columns={"hospital_id": "id", "hospital_name": "name"}
    )[["id", "name", "latitude", "longitude"]]
    candidates = sim.hospital_candidates.rename(columns={"institution_id": "id"})[
        ["id", "name", "latitude", "longitude"]
    ]
    return pd.concat([emergency, candidates], ignore_index=True).drop_duplicates("id").set_index("id")


def _scenario(sim, dispatch, events, extra_base_regions=()):
    regions = sim.state.set_index("region_id")
    region_db = sim.regions.set_index("region_id")
    hospitals = _hospital_lookup(sim)

    sources = []
    for _, station in sim.current_ems.iterrows():
        for number in range(1, int(station["ambulance_count"]) + 1):
            sources.append(
                {
                    "source_id": station["station_id"],
                    "vehicle_id": f"{station['station_id']}-{number}",
                    "name": station["station_name"],
                    "latitude": float(station["latitude"]),
                    "longitude": float(station["longitude"]),
                    "is_new": False,
                }
            )
    for region_id in extra_base_regions:
        region = region_db.loc[region_id]
        sources.append(
            {
                "source_id": f"NEW_{region_id}",
                "vehicle_id": f"NEW_{region_id}-1",
                "name": f"{region['region_name']} 신규 구급거점",
                "latitude": float(region["latitude"]),
                "longitude": float(region["longitude"]),
                "is_new": True,
            }
        )

    available = np.zeros(len(sources), dtype=float)
    records = []
    for event_index, event in enumerate(events, start=1):
        region = regions.loc[event.region_id]
        target = region_db.loc[event.region_id]
        dispatch_times = []
        for source in sources:
            if not source["is_new"] and source["source_id"] in dispatch.dispatch_time.index:
                dispatch_time = float(
                    dispatch.dispatch_time.loc[source["source_id"], event.region_id]
                )
            else:
                dispatch_time = sim.estimate_ems_time(
                    source["latitude"], source["longitude"],
                    target["latitude"], target["longitude"],
                )
            dispatch_times.append(dispatch_time)
        dispatch_times = np.asarray(dispatch_times)
        arrival_options = np.maximum(event.occurred_at_min, available) + dispatch_times
        vehicle_index = int(arrival_options.argmin())
        source = sources[vehicle_index]
        wait = max(0.0, available[vehicle_index] - event.occurred_at_min)
        depart_at = event.occurred_at_min + wait
        scene_at = depart_at + dispatch_times[vehicle_index]
        hospital_id = region["hospital_id"]
        hospital = hospitals.loc[hospital_id]
        hospital_time = float(region["hospital_duration_min"])
        hospital_at = scene_at + hospital_time
        return_time = sim.estimate_ems_time(
            hospital["latitude"], hospital["longitude"],
            source["latitude"], source["longitude"],
        )
        available[vehicle_index] = hospital_at + return_time
        records.append(
            {
                "event_id": event_index,
                "occurred_at_min": round(event.occurred_at_min, 4),
                "depart_at_min": round(depart_at, 4),
                "scene_at_min": round(scene_at, 4),
                "hospital_at_min": round(hospital_at, 4),
                "complete_at_min": round(available[vehicle_index], 4),
                "wait_time_min": round(wait, 4),
                "response_time_min": round(wait + dispatch_times[vehicle_index], 4),
                "hospital_time_min": round(hospital_time, 4),
                "region_id": event.region_id,
                "region_name": region["region_name"],
                "region_latitude": round(float(target["latitude"]), 7),
                "region_longitude": round(float(target["longitude"]), 7),
                "source_id": source["source_id"],
                "source_name": source["name"],
                "source_latitude": round(source["latitude"], 7),
                "source_longitude": round(source["longitude"], 7),
                "vehicle_id": source["vehicle_id"],
                "hospital_id": hospital_id,
                "hospital_name": hospital["name"],
                "hospital_latitude": round(float(hospital["latitude"]), 7),
                "hospital_longitude": round(float(hospital["longitude"]), 7),
            }
        )

    frame = pd.DataFrame(records)
    metrics = {
        "event_count": int(len(frame)),
        "vehicle_count": int(len(sources)),
        "mean_response_time_min": float(frame["response_time_min"].mean()),
        "p90_response_time_min": float(frame["response_time_min"].quantile(0.90)),
        "mean_total_to_hospital_min": float(
            (frame["response_time_min"] + frame["hospital_time_min"]).mean()
        ),
        "waited_event_count": int((frame["wait_time_min"] > 0).sum()),
        "waited_event_rate": float((frame["wait_time_min"] > 0).mean()),
    }
    return records, metrics, sources


def build_dashboard_data(data_dir, policies_path, seed=42, days=1):
    dispatch = AmbulanceDispatchSimulator(data_dir=data_dir)
    events = dispatch.generate_events(days=days, seed=seed)

    current_sim = EmergencyPolicySimulator(data_dir=data_dir)
    current_records, current_metrics, current_sources = _scenario(
        current_sim, dispatch, events
    )

    policies = pd.read_csv(policies_path).to_dict("records")
    rl_sim = EmergencyPolicySimulator(data_dir=data_dir)
    extra_bases = []
    for policy in policies:
        apply_policy(rl_sim, policy)
        if policy["policy"] == "ambulance_new_base":
            extra_bases.append(policy["target_id"])
    rl_records, rl_metrics, rl_sources = _scenario(
        rl_sim, dispatch, events, extra_base_regions=extra_bases
    )
    current_policy_metrics = current_sim.calculate_metrics()
    rl_policy_metrics = rl_sim.calculate_metrics()

    regions = rl_sim.regions[["region_id", "region_name", "latitude", "longitude"]]
    before = current_sim.state.set_index("region_id")
    after = rl_sim.state.set_index("region_id")
    region_rows = []
    for row in regions.to_dict("records"):
        region_id = row["region_id"]
        row.update(
            {
                "latitude": round(float(row["latitude"]), 7),
                "longitude": round(float(row["longitude"]), 7),
                "before_total_min": round(
                    float(before.loc[region_id, "ems_duration_min"])
                    + float(before.loc[region_id, "hospital_duration_min"]), 4
                ),
                "after_total_min": round(
                    float(after.loc[region_id, "ems_duration_min"])
                    + float(after.loc[region_id, "hospital_duration_min"]), 4
                ),
                "occurrence_score": round(
                    float(before.loc[region_id, "occurrence_score_01"]), 4
                ),
                "vulnerability_types": before.loc[region_id, "vulnerability_types"],
            }
        )
        region_rows.append(row)

    hospital_ids = set(before["hospital_id"]) | set(after["hospital_id"])
    hospitals = _hospital_lookup(rl_sim).loc[list(hospital_ids)].reset_index().to_dict("records")
    for hospital in hospitals:
        hospital["latitude"] = round(float(hospital["latitude"]), 7)
        hospital["longitude"] = round(float(hospital["longitude"]), 7)

    gnn_edges_path = Path("data/gnn/gnn_edges.csv")
    region_edges = []
    if gnn_edges_path.exists():
        name_to_id = regions.set_index("region_name")["region_id"].to_dict()
        for edge in pd.read_csv(gnn_edges_path).to_dict("records"):
            region_edges.append(
                {
                    "source": name_to_id[edge["source_name"]],
                    "target": name_to_id[edge["target_name"]],
                    "shared_boundary_m": round(float(edge["shared_boundary_m"]), 1),
                }
            )

    return {
        "metadata": {
            "seed": seed,
            "days": days,
            "horizon_min": days * 1440,
            "event_generation": "지역별 2019-2023 응급호출 연간 건수 기반 Poisson; 일중 시각 균등",
            "route_geometry": "시설과 지역 대표점 사이 근사 곡선; 시간은 보정 이동시간 사용",
            "on_scene_time_min": 0,
        },
        "policies": policies,
        "policy_outcome": {
            "before": current_policy_metrics,
            "after": rl_policy_metrics,
            "weighted_total_improvement_min": (
                current_policy_metrics["weighted_total_time"]
                - rl_policy_metrics["weighted_total_time"]
            ),
        },
        "regions": region_rows,
        "region_edges": region_edges,
        "hospitals": hospitals,
        "scenarios": {
            "current": {
                "label": "현행",
                "metrics": current_metrics,
                "sources": current_sources,
                "events": current_records,
            },
            "rl": {
                "label": "RL 추천",
                "metrics": rl_metrics,
                "sources": rl_sources,
                "events": rl_records,
            },
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/RL")
    parser.add_argument("--policies", default=DEFAULT_POLICIES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--output", default="data/figures/rl-dashboard-data.js")
    args = parser.parse_args()
    data = build_dashboard_data(args.data_dir, args.policies, args.seed, args.days)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "window.RL_DASHBOARD_DATA = "
        + json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "current": data["scenarios"]["current"]["metrics"],
        "rl": data["scenarios"]["rl"]["metrics"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
