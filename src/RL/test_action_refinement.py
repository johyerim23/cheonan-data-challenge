from collections import Counter

from env import CheonanEmergencyEnv
from policy_actions import build_policy_actions, map_ems_stations_to_regions
from simulator import EmergencyPolicySimulator


DATA_DIR = "data/processed/RL"


def run_tests():
    sim = EmergencyPolicySimulator(DATA_DIR)
    mapping = map_ems_stations_to_regions(sim)
    actions = build_policy_actions(sim, refined_ambulance_actions=True)
    counts = Counter(action["policy"] for action in actions)

    assert len(mapping) == len(sim.ems_stations) == 16
    assert mapping["station_id"].is_unique
    assert set(mapping["region_id"]).issubset(set(sim.regions["region_id"]))
    assert mapping["region_id"].nunique() == 15
    assert (mapping["match_method"] == "address_exact").sum() == 10
    assert (mapping["match_method"] == "nearest_coordinate").sum() == 6

    ems008 = mapping.set_index("station_id").loc["EMS008"]
    assert ems008["region_name"] == "직산읍"
    assert ems008["match_method"] == "address_exact"

    assert counts == {
        "ambulance_existing": 16,
        "ambulance_new_base": 16,
        "hospital_upgrade": 22,
    }
    assert len(actions) == 54

    existing_regions = set(mapping["region_id"])
    new_base_regions = {
        action["target_id"]
        for action in actions
        if action["policy"] == "ambulance_new_base"
    }
    assert existing_regions.isdisjoint(new_base_regions)
    assert len(existing_regions | new_base_regions) == 31

    env = CheonanEmergencyEnv(DATA_DIR, policy_budget=3)
    state, _ = env.reset(seed=42)
    assert env.action_space.n == 54
    assert state.shape == (396,)

    existing_index = next(
        i for i, action in enumerate(env.actions)
        if action["policy"] == "ambulance_existing"
    )
    station_id = env.actions[existing_index]["target_id"]
    before_count = int(
        env.sim.current_ems.loc[
            env.sim.current_ems["station_id"] == station_id,
            "ambulance_count",
        ].iloc[0]
    )
    _, reward, _, _, info = env.step(existing_index)
    after_count = int(
        env.sim.current_ems.loc[
            env.sim.current_ems["station_id"] == station_id,
            "ambulance_count",
        ].iloc[0]
    )
    assert after_count == before_count + 1
    expected_reward = env.ambulance_increment_effects.get(station_id, 0.0)
    assert abs(reward - expected_reward) < 1e-9
    assert abs(info["reward_components"]["dispatch_reward"] - expected_reward) < 1e-9

    legacy_env = CheonanEmergencyEnv(
        DATA_DIR,
        policy_budget=3,
        refined_ambulance_actions=False,
    )
    legacy_state, _ = legacy_env.reset(seed=42)
    assert legacy_env.action_space.n == 53
    assert legacy_state.shape == (395,)

    print("Action refinement tests passed")
    print(dict(counts))
    print({"refined_state": state.shape, "legacy_state": legacy_state.shape})


if __name__ == "__main__":
    run_tests()
