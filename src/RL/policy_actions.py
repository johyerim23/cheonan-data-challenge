"""Shared policy action catalog, application, and reward helpers."""

from pathlib import Path

import pandas as pd


def map_ems_stations_to_regions(sim):
    """Map EMS stations to regions using address first, then nearest centroid."""

    rows = []
    regions = sim.regions.copy()

    for _, station in sim.ems_stations.iterrows():
        address = str(station.get("address", ""))
        address_matches = regions[
            regions["region_name"].map(lambda name: str(name) in address)
        ]

        if len(address_matches) == 1:
            matched = address_matches.iloc[0]
            method = "address_exact"
        else:
            distances = regions.apply(
                lambda region: sim.haversine(
                    station["latitude"],
                    station["longitude"],
                    region["latitude"],
                    region["longitude"],
                ),
                axis=1,
            )
            matched = regions.loc[distances.idxmin()]
            method = "nearest_coordinate"

        distance_km = sim.haversine(
            station["latitude"],
            station["longitude"],
            matched["latitude"],
            matched["longitude"],
        )
        rows.append(
            {
                "station_id": station["station_id"],
                "station_name": station["station_name"],
                "region_id": matched["region_id"],
                "region_name": matched["region_name"],
                "match_method": method,
                "centroid_distance_km": distance_km,
            }
        )

    return pd.DataFrame(rows)


def build_policy_actions(sim, refined_ambulance_actions=True):
    """Build a single canonical action catalog for training and evaluation."""

    actions = []

    if refined_ambulance_actions:
        station_regions = map_ems_stations_to_regions(sim)
        existing_region_ids = set(station_regions["region_id"])

        for _, station in station_regions.iterrows():
            actions.append(
                {
                    "policy": "ambulance_existing",
                    "target_id": station["station_id"],
                    "target_name": station["station_name"],
                    "target_region_id": station["region_id"],
                    "target_region_name": station["region_name"],
                }
            )

        new_base_regions = sim.state[
            ~sim.state["region_id"].isin(existing_region_ids)
        ]
    else:
        new_base_regions = sim.state

    for _, region in new_base_regions.iterrows():
        actions.append(
            {
                "policy": "ambulance_new_base",
                "target_id": region["region_id"],
                "target_name": region["region_name"],
                "target_region_id": region["region_id"],
                "target_region_name": region["region_name"],
            }
        )

    for _, hospital in sim.hospital_candidates.iterrows():
        actions.append(
            {
                "policy": "hospital_upgrade",
                "target_id": hospital["institution_id"],
                "target_name": hospital["name"],
                "target_region_id": None,
                "target_region_name": None,
            }
        )

    return actions


def apply_policy(sim, policy):
    if policy["policy"] == "ambulance_existing":
        sim.add_ambulance_existing_center(policy["target_id"])
    elif policy["policy"] == "ambulance_new_base":
        sim.add_new_ambulance_base(policy["target_id"])
    elif policy["policy"] == "hospital_upgrade":
        sim.upgrade_hospital(policy["target_id"])
    else:
        raise ValueError(f"지원하지 않는 정책: {policy['policy']}")


def load_ambulance_increment_effects(data_dir):
    path = Path(data_dir) / "ambulance_increment_effects.csv"
    if not path.exists():
        return {}
    effects = pd.read_csv(path)
    required = {"station_id", "mean_response_improvement_min"}
    missing = required - set(effects.columns)
    if missing:
        raise ValueError(f"증차 효과 파일 필수 컬럼 누락: {sorted(missing)}")
    return effects.set_index("station_id")[
        "mean_response_improvement_min"
    ].astype(float).to_dict()


def policy_reward_bonus(policy, ambulance_increment_effects):
    if policy["policy"] != "ambulance_existing":
        return 0.0
    return float(ambulance_increment_effects.get(policy["target_id"], 0.0))
