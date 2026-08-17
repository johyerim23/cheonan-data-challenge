"""Evaluate one additional ambulance at every existing Cheonan EMS station."""

import argparse
import json
from pathlib import Path

import pandas as pd

from dispatch_simulator import AmbulanceDispatchSimulator


def parse_args():
    parser = argparse.ArgumentParser(description="기존센터 구급차 증차 효과를 평가합니다.")
    parser.add_argument("--data-dir", default="data/processed/RL")
    parser.add_argument("--output-dir", default="results/RL/dispatch")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument(
        "--effects-output",
        default="data/processed/RL/ambulance_increment_effects.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    simulator = AmbulanceDispatchSimulator(data_dir=args.data_dir)
    rows = []
    baseline_rows = []

    for seed in args.seeds:
        _, baseline = simulator.simulate(days=args.days, seed=seed)
        baseline_rows.append({"seed": seed, **baseline})
        for station_id, station in simulator.stations.iterrows():
            _, incremented = simulator.simulate(
                days=args.days,
                seed=seed,
                increments={station_id: 1},
            )
            rows.append(
                {
                    "seed": seed,
                    "station_id": station_id,
                    "station_name": station["station_name"],
                    "baseline_mean_wait_time_min": baseline["mean_wait_time_min"],
                    "incremented_mean_wait_time_min": incremented[
                        "mean_wait_time_min"
                    ],
                    "mean_wait_improvement_min": baseline["mean_wait_time_min"]
                    - incremented["mean_wait_time_min"],
                    "p95_wait_improvement_min": baseline["p95_wait_time_min"]
                    - incremented["p95_wait_time_min"],
                    "mean_response_improvement_min": baseline[
                        "mean_response_time_min"
                    ]
                    - incremented["mean_response_time_min"],
                    "waited_event_reduction": baseline["waited_event_count"]
                    - incremented["waited_event_count"],
                    "event_count": baseline["event_count"],
                }
            )

    detail = pd.DataFrame(rows)
    baseline_detail = pd.DataFrame(baseline_rows)
    summary = (
        detail.groupby(["station_id", "station_name"], as_index=False)
        .agg(
            mean_wait_improvement_min=("mean_wait_improvement_min", "mean"),
            mean_wait_improvement_std=("mean_wait_improvement_min", "std"),
            p95_wait_improvement_min=("p95_wait_improvement_min", "mean"),
            mean_response_improvement_min=("mean_response_improvement_min", "mean"),
            waited_event_reduction=("waited_event_reduction", "mean"),
        )
        .sort_values("mean_wait_improvement_min", ascending=False)
        .reset_index(drop=True)
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(
        output_dir / "ambulance_increment_by_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    baseline_detail.to_csv(
        output_dir / "dispatch_baseline_by_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        output_dir / "ambulance_increment_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    effects_path = Path(args.effects_output)
    effects_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(effects_path, index=False, encoding="utf-8-sig")
    metadata = {
        "days": args.days,
        "seeds": args.seeds,
        "regional_demand_source": "data/processed/RL/region_features.csv: emergency_calls",
        "arrival_process": "region-specific Poisson arrivals with uniform rate over the simulated horizon",
        "busy_time_components": "dispatch_to_scene + region_to_assigned_hospital + hospital_to_station_return",
        "on_scene_time_min": 0.0,
        "comparison_design": "common random numbers: baseline and each increment share the same seed-specific incident events",
    }
    (output_dir / "dispatch_simulation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
