import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio

from road_network import CheonanRoadNetwork


NETWORK_DIR = Path("data/processed/RL/road_network")
RESULT_DIR = Path("results/RL/road_network")


def run_tests():
    metadata = json.loads(
        (NETWORK_DIR / "road_network_metadata.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (RESULT_DIR / "road_calibration_summary.json").read_text(encoding="utf-8")
    )
    network = CheonanRoadNetwork(NETWORK_DIR)

    assert len(network.nodes) == metadata["node_count"] == 97_611
    assert len(network.links) == metadata["link_count"] == 132_832
    assert pyogrio.read_info(
        NETWORK_DIR / "cheonan_road_network.gpkg", layer="nodes"
    )["features"] == len(network.nodes)
    assert pyogrio.read_info(
        NETWORK_DIR / "cheonan_road_network.gpkg", layer="links"
    )["features"] == len(network.links)

    node_ids = set(network.nodes["NODE_ID"])
    assert network.nodes["NODE_ID"].is_unique
    assert network.links["LINK_ID"].is_unique
    assert set(network.links["F_NODE"]).issubset(node_ids)
    assert set(network.links["T_NODE"]).issubset(node_ids)
    expected_time = network.links["LENGTH"] * 0.06 / network.links["MAX_SPD"]
    assert np.allclose(expected_time, network.links["FREE_FLOW_MIN"])
    assert (network.links["FREE_FLOW_MIN"] > 0).all()

    assert network.routing_node_count == summary["routing_node_count"]
    assert network.routing_node_count / len(network.nodes) > 0.85
    assert summary["ems_reachable"] == 31
    assert summary["hospital_reachable"] == 31
    assert summary["ems_slope"] >= 0 and summary["ems_intercept"] >= 0
    assert summary["hospital_slope"] >= 0 and summary["hospital_intercept"] >= 0
    assert summary["ems_leave_one_out"]["mae_min"] < summary["ems_raw"]["mae_min"]
    assert (
        summary["hospital_leave_one_out"]["mae_min"]
        < summary["hospital_raw"]["mae_min"]
    )

    ems_routes = pd.read_csv(RESULT_DIR / "ems_route_calibration.csv")
    hospital_routes = pd.read_csv(RESULT_DIR / "hospital_route_calibration.csv")
    mapping = pd.read_csv(RESULT_DIR / "facility_node_mapping.csv")
    assert len(ems_routes) == len(hospital_routes) == 31
    assert ems_routes["reachable"].all() and hospital_routes["reachable"].all()
    assert len(mapping) == summary["mapped_facility_count"]
    assert set(mapping["routing_node_id"].astype(str)).issubset(node_ids)

    first = ems_routes.iloc[0]
    repeated = network.shortest_path_minutes(
        36.799873,
        127.165347,
        36.812482,
        127.152453,
    )
    repeated_again = network.shortest_path_minutes(
        36.799873,
        127.165347,
        36.812482,
        127.152453,
    )
    assert repeated["reachable"]
    assert repeated["raw_time_min"] == repeated_again["raw_time_min"]
    assert np.isfinite(first["raw_time_min"])

    print("Road network tests passed")
    print(
        {
            "nodes": len(network.nodes),
            "links": len(network.links),
            "routing_nodes": network.routing_node_count,
            "mapped_facilities": len(mapping),
            "ems_loocv_mae": summary["ems_leave_one_out"]["mae_min"],
            "hospital_loocv_mae": summary["hospital_leave_one_out"]["mae_min"],
        }
    )


if __name__ == "__main__":
    run_tests()
