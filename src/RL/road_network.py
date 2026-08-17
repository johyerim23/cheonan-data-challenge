"""Routing utilities for the extracted Cheonan Node/Link graph."""

from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree


class CheonanRoadNetwork:
    def __init__(self, network_dir="data/processed/RL/road_network"):
        network_dir = Path(network_dir)
        self.nodes = pd.read_csv(
            network_dir / "cheonan_nodes.csv.gz",
            dtype={"NODE_ID": str},
        )
        self.links = pd.read_csv(
            network_dir / "cheonan_links.csv.gz",
            dtype={"LINK_ID": str, "F_NODE": str, "T_NODE": str},
        )
        self.node_ids = self.nodes["NODE_ID"].to_numpy()
        self.node_index = {
            node_id: index for index, node_id in enumerate(self.node_ids)
        }
        self.transformer = Transformer.from_crs(
            "EPSG:4326",
            "EPSG:5186",
            always_xy=True,
        )
        edge_min = (
            self.links.groupby(["F_NODE", "T_NODE"], as_index=False)[
                "FREE_FLOW_MIN"
            ]
            .min()
        )
        rows = edge_min["F_NODE"].map(self.node_index).to_numpy()
        cols = edge_min["T_NODE"].map(self.node_index).to_numpy()
        weights = edge_min["FREE_FLOW_MIN"].to_numpy(dtype=float)
        self.graph = csr_matrix(
            (weights, (rows, cols)),
            shape=(len(self.nodes), len(self.nodes)),
        )

        component_count, labels = connected_components(
            self.graph,
            directed=True,
            connection="strong",
        )
        component_sizes = np.bincount(labels)
        largest_label = int(component_sizes.argmax())
        self.routing_node_indices = np.flatnonzero(labels == largest_label)
        self.routing_node_count = len(self.routing_node_indices)
        self.strong_component_count = int(component_count)
        routing_coordinates = self.nodes.loc[
            self.routing_node_indices,
            ["X_5186", "Y_5186"],
        ].to_numpy()
        self.node_tree = cKDTree(routing_coordinates)

        # Median free-flow minutes/km, derived only from link speed limits.
        self.connector_min_per_km = float(
            np.median(self.links["FREE_FLOW_MIN"] / (self.links["LENGTH"] / 1000.0))
        )

    def snap(self, latitude, longitude):
        x, y = self.transformer.transform(longitude, latitude)
        distance_m, routing_index = self.node_tree.query([x, y])
        index = int(self.routing_node_indices[int(routing_index)])
        return {
            "node_id": self.node_ids[index],
            "node_index": index,
            "distance_m": float(distance_m),
        }

    def shortest_path_minutes(
        self,
        source_latitude,
        source_longitude,
        target_latitude,
        target_longitude,
        calibration_slope=1.0,
        calibration_intercept=0.0,
    ):
        source = self.snap(source_latitude, source_longitude)
        target = self.snap(target_latitude, target_longitude)
        network_time = float(
            dijkstra(
                self.graph,
                directed=True,
                indices=source["node_index"],
                return_predecessors=False,
            )[target["node_index"]]
        )
        connector_km = (source["distance_m"] + target["distance_m"]) / 1000.0
        connector_time = connector_km * self.connector_min_per_km
        raw_time = network_time + connector_time
        return {
            "source_node_id": source["node_id"],
            "target_node_id": target["node_id"],
            "source_snap_m": source["distance_m"],
            "target_snap_m": target["distance_m"],
            "network_time_min": network_time,
            "connector_time_min": connector_time,
            "raw_time_min": raw_time,
            "calibrated_time_min": (
                raw_time * calibration_slope + calibration_intercept
            ),
            "reachable": bool(np.isfinite(network_time)),
        }

    def calibrate(self, region_features, ems_stations, emergency_hospitals, regions):
        station_map = ems_stations.set_index("station_id")
        hospital_map = emergency_hospitals.set_index("hospital_id")
        region_map = regions.set_index("region_id")
        ems_rows = []
        hospital_rows = []

        for _, row in region_features.iterrows():
            region = region_map.loc[row["region_id"]]
            station = station_map.loc[row["station_id"]]
            hospital = hospital_map.loc[row["hospital_id"]]

            ems_route = self.shortest_path_minutes(
                station["latitude"],
                station["longitude"],
                region["latitude"],
                region["longitude"],
            )
            hospital_route = self.shortest_path_minutes(
                region["latitude"],
                region["longitude"],
                hospital["latitude"],
                hospital["longitude"],
            )
            ems_rows.append(
                {
                    "region_id": row["region_id"],
                    "observed_min": row["ems_duration_min"],
                    **ems_route,
                }
            )
            hospital_rows.append(
                {
                    "region_id": row["region_id"],
                    "observed_min": row["hospital_duration_min"],
                    **hospital_route,
                }
            )

        ems = pd.DataFrame(ems_rows)
        hospital = pd.DataFrame(hospital_rows)
        if not ems["reachable"].all() or not hospital["reachable"].all():
            raise ValueError("관측 구간 중 도로망에서 도달 불가능한 경로가 있습니다.")

        ems_slope, ems_intercept = np.linalg.lstsq(
            np.column_stack([ems["raw_time_min"], np.ones(len(ems))]),
            ems["observed_min"],
            rcond=None,
        )[0]
        hospital_slope, hospital_intercept = np.linalg.lstsq(
            np.column_stack(
                [hospital["raw_time_min"], np.ones(len(hospital))]
            ),
            hospital["observed_min"],
            rcond=None,
        )[0]
        if min(ems_slope, ems_intercept, hospital_slope, hospital_intercept) < 0:
            raise ValueError("도로망 보정 계수는 음수가 될 수 없습니다.")
        ems["calibrated_time_min"] = (
            ems["raw_time_min"] * ems_slope + ems_intercept
        )
        hospital["calibrated_time_min"] = (
            hospital["raw_time_min"] * hospital_slope + hospital_intercept
        )
        return {
            "ems_slope": float(ems_slope),
            "ems_intercept": float(ems_intercept),
            "hospital_slope": float(hospital_slope),
            "hospital_intercept": float(hospital_intercept),
            "ems_routes": ems,
            "hospital_routes": hospital,
        }
