"""Assertions for policy-sensitive state and reward updates."""

import numpy as np

from env import CheonanEmergencyEnv
from simulator import EmergencyPolicySimulator


def run_tests():
    sim = EmergencyPolicySimulator()
    baseline = sim.state.copy(deep=True)
    candidate_id = sim.hospital_candidates.iloc[0]["institution_id"]
    sim.upgrade_hospital(candidate_id)

    assert (sim.state["hospital_duration_min"] <= baseline["hospital_duration_min"] + 1e-12).all()
    assert (
        sim.state["nearest_alternative_time_min"]
        <= baseline["nearest_alternative_time_min"] + 1e-12
    ).all()
    assert (sim.state["alternative_score_01"] <= baseline["alternative_score_01"] + 1e-12).all()
    assert np.allclose(
        sim.state["alternative_gap_min"],
        (
            sim.state["nearest_alternative_time_min"]
            - sim.state["hospital_duration_min"]
        ).clip(lower=0),
    )
    type_cols = [
        "type_dispatch", "type_transfer", "type_alternative",
        "type_traffic", "type_demand",
    ]
    assert (
        sim.state["vulnerability_type_count"] == sim.state[type_cols].sum(axis=1)
    ).all()
    assert (sim.state["is_complex_type"] == (sim.state["vulnerability_type_count"] >= 2)).all()

    saved = sim.snapshot()
    station_id = sim.ems_stations.iloc[0]["station_id"]
    before_count = int(sim.current_ems.loc[
        sim.current_ems["station_id"] == station_id, "ambulance_count"
    ].iloc[0])
    sim.add_ambulance_existing_center(station_id)
    sim.restore(saved)
    after_restore_count = int(sim.current_ems.loc[
        sim.current_ems["station_id"] == station_id, "ambulance_count"
    ].iloc[0])
    assert after_restore_count == before_count
    assert sim.upgraded_hospital_ids == [candidate_id]

    sim.reset()
    assert sim.upgraded_hospital_ids == []
    assert sim.state.equals(sim.base_state)

    env = CheonanEmergencyEnv()
    env.reset(seed=42)
    existing_index = next(
        i for i, action in enumerate(env.actions)
        if action["policy"] == "ambulance_existing"
        and env.ambulance_increment_effects.get(action["target_id"], 0.0) > 0
    )
    _, reward, _, _, info = env.step(existing_index)
    expected = env.ambulance_increment_effects[env.actions[existing_index]["target_id"]]
    assert np.isclose(reward, expected)
    assert np.isclose(info["reward_components"]["dispatch_reward"], expected)
    assert np.isclose(info["metrics"]["dispatch_response_improvement_min"], expected)
    env.reset()
    assert env.cumulative_dispatch_improvement == 0.0

    print("Policy state/reward update tests passed")


if __name__ == "__main__":
    run_tests()
