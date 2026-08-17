"""Calibrate road-network route times against the existing observations."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from road_network import CheonanRoadNetwork


def error_metrics(frame, prediction_column):
    error = frame[prediction_column] - frame["observed_min"]
    return {
        "mae_min": float(error.abs().mean()),
        "rmse_min": float(np.sqrt(np.mean(error**2))),
        "median_error_min": float(error.median()),
    }


def leave_one_out_metrics(frame):
    raw = frame["raw_time_min"].to_numpy()
    observed = frame["observed_min"].to_numpy()
    predictions = np.empty(len(frame))
    for held_out in range(len(frame)):
        train_mask = np.arange(len(frame)) != held_out
        design = np.column_stack(
            [raw[train_mask], np.ones(train_mask.sum())]
        )
        slope, intercept = np.linalg.lstsq(
            design,
            observed[train_mask],
            rcond=None,
        )[0]
        predictions[held_out] = raw[held_out] * slope + intercept
    error = predictions - observed
    return {
        "mae_min": float(np.mean(np.abs(error))),
        "rmse_min": float(np.sqrt(np.mean(error**2))),
    }


def build_snap_mapping(network, frame, entity_type, id_column, name_column):
    rows = []
    for _, row in frame.dropna(subset=["latitude", "longitude"]).iterrows():
        snapped = network.snap(row["latitude"], row["longitude"])
        rows.append(
            {
                "entity_type": entity_type,
                "entity_id": row[id_column],
                "entity_name": row[name_column],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "routing_node_id": snapped["node_id"],
                "snap_distance_m": snapped["distance_m"],
            }
        )
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="도로망 이동시간을 관측값으로 보정합니다.")
    parser.add_argument("--data-dir", default="data/processed/RL")
    parser.add_argument(
        "--network-dir",
        default="data/processed/RL/road_network",
    )
    parser.add_argument("--output-dir", default="results/RL/road_network")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    network = CheonanRoadNetwork(args.network_dir)
    region_features = pd.read_csv(data_dir / "region_features.csv")
    ems_stations = pd.read_csv(data_dir / "cheonan_ems_station_db.csv")
    emergency_hospitals = pd.read_csv(
        data_dir / "cheonan_emergency_hospital_db.csv"
    )
    regions = pd.read_csv(data_dir / "cheonan_region_db.csv")
    hospitals_all = pd.read_csv(data_dir / "cheonan_hospital_db_all.csv")
    calibration = network.calibrate(
        region_features=region_features,
        ems_stations=ems_stations,
        emergency_hospitals=emergency_hospitals,
        regions=regions,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ems = calibration["ems_routes"]
    hospital = calibration["hospital_routes"]
    ems.to_csv(output_dir / "ems_route_calibration.csv", index=False, encoding="utf-8-sig")
    hospital.to_csv(
        output_dir / "hospital_route_calibration.csv",
        index=False,
        encoding="utf-8-sig",
    )
    snap_mapping = pd.concat(
        [
            build_snap_mapping(
                network, regions, "region", "region_id", "region_name"
            ),
            build_snap_mapping(
                network, ems_stations, "ems_station", "station_id", "station_name"
            ),
            build_snap_mapping(
                network, hospitals_all, "hospital", "institution_id", "name"
            ),
            build_snap_mapping(
                network,
                emergency_hospitals,
                "emergency_hospital",
                "hospital_id",
                "hospital_name",
            ),
        ],
        ignore_index=True,
    )
    snap_mapping.to_csv(
        output_dir / "facility_node_mapping.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "node_count": len(network.nodes),
        "link_count": len(network.links),
        "strong_component_count": network.strong_component_count,
        "routing_node_count": network.routing_node_count,
        "routing_node_share": network.routing_node_count / len(network.nodes),
        "connector_min_per_km": network.connector_min_per_km,
        "ems_slope": calibration["ems_slope"],
        "ems_intercept": calibration["ems_intercept"],
        "hospital_slope": calibration["hospital_slope"],
        "hospital_intercept": calibration["hospital_intercept"],
        "ems_raw": error_metrics(ems, "raw_time_min"),
        "ems_calibrated": error_metrics(ems, "calibrated_time_min"),
        "hospital_raw": error_metrics(hospital, "raw_time_min"),
        "hospital_calibrated": error_metrics(hospital, "calibrated_time_min"),
        "ems_leave_one_out": leave_one_out_metrics(ems),
        "hospital_leave_one_out": leave_one_out_metrics(hospital),
        "ems_reachable": int(ems["reachable"].sum()),
        "hospital_reachable": int(hospital["reachable"].sum()),
        "mapped_facility_count": len(snap_mapping),
        "max_snap_distance_m": float(snap_mapping["snap_distance_m"].max()),
        "p95_snap_distance_m": float(
            snap_mapping["snap_distance_m"].quantile(0.95)
        ),
    }
    (output_dir / "road_calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
