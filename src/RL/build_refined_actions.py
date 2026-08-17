"""Export the refined ambulance mapping and canonical action catalog."""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from policy_actions import build_policy_actions, map_ems_stations_to_regions
from simulator import EmergencyPolicySimulator


def parse_args():
    parser = argparse.ArgumentParser(description="정제된 RL Action 후보를 생성합니다.")
    parser.add_argument("--data-dir", default="data/processed/RL")
    parser.add_argument("--output-dir", default="results/RL")
    return parser.parse_args()


def main():
    args = parse_args()
    sim = EmergencyPolicySimulator(data_dir=args.data_dir)
    mapping = map_ems_stations_to_regions(sim)
    actions = pd.DataFrame(build_policy_actions(sim, refined_ambulance_actions=True))
    counts = Counter(actions["policy"])

    summary = {
        "station_count": len(mapping),
        "existing_region_count": mapping["region_id"].nunique(),
        "address_exact_count": int((mapping["match_method"] == "address_exact").sum()),
        "nearest_coordinate_count": int(
            (mapping["match_method"] == "nearest_coordinate").sum()
        ),
        "ambulance_existing_actions": counts["ambulance_existing"],
        "ambulance_new_base_actions": counts["ambulance_new_base"],
        "hospital_upgrade_actions": counts["hospital_upgrade"],
        "total_actions": len(actions),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = output_dir / "ems_station_region_mapping.csv"
    actions_path = output_dir / "refined_action_catalog.csv"
    summary_path = output_dir / "refined_action_summary.json"
    mapping.to_csv(mapping_path, index=False, encoding="utf-8-sig")
    actions.to_csv(actions_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(mapping.to_string(index=False))
    print("\n", json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n매칭 결과: {mapping_path}")
    print(f"Action 목록: {actions_path}")
    print(f"요약 결과: {summary_path}")


if __name__ == "__main__":
    main()
