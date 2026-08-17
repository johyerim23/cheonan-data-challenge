"""Assertions for dashboard playback data integrity."""

from export_dashboard_data import build_dashboard_data


def run_tests():
    data = build_dashboard_data(
        data_dir="data/processed/RL",
        policies_path="results/RL/refined_with_dispatch/final_selected_policies.csv",
        seed=42,
        days=1,
    )
    current = data["scenarios"]["current"]
    rl = data["scenarios"]["rl"]

    assert len(data["regions"]) == 31
    assert len(data["region_edges"]) == 71
    assert len(data["policies"]) == 3
    assert len(current["events"]) == len(rl["events"]) == 63
    assert current["metrics"]["vehicle_count"] == 19
    assert rl["metrics"]["vehicle_count"] == 21
    assert [event["occurred_at_min"] for event in current["events"]] == [
        event["occurred_at_min"] for event in rl["events"]
    ]
    assert all(
        event["occurred_at_min"]
        <= event["depart_at_min"]
        <= event["scene_at_min"]
        <= event["hospital_at_min"]
        <= event["complete_at_min"]
        for scenario in data["scenarios"].values()
        for event in scenario["events"]
    )
    assert data["policy_outcome"]["weighted_total_improvement_min"] > 0
    assert rl["metrics"]["mean_total_to_hospital_min"] <= current["metrics"][
        "mean_total_to_hospital_min"
    ]
    print("Dashboard export tests passed")


if __name__ == "__main__":
    run_tests()
