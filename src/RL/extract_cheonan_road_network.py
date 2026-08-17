"""Extract a self-contained Cheonan road graph from the national Node/Link ZIP."""

import argparse
import json
from pathlib import Path

import pandas as pd
import pyogrio
from pyproj import Transformer


SOURCE_CRS = "EPSG:5186"
TARGET_CRS = "EPSG:4326"
DEFAULT_BUFFER_KM = 10.0
COORDINATE_FILES = [
    "cheonan_region_db.csv",
    "cheonan_ems_station_db.csv",
    "cheonan_hospital_db_all.csv",
    "cheonan_emergency_hospital_db.csv",
]


def coordinate_bounds(data_dir, buffer_km):
    frames = [pd.read_csv(Path(data_dir) / name) for name in COORDINATE_FILES]
    min_lon = min(frame["longitude"].min() for frame in frames)
    min_lat = min(frame["latitude"].min() for frame in frames)
    max_lon = max(frame["longitude"].max() for frame in frames)
    max_lat = max(frame["latitude"].max() for frame in frames)

    transformer = Transformer.from_crs(TARGET_CRS, SOURCE_CRS, always_xy=True)
    min_x, min_y = transformer.transform(min_lon, min_lat)
    max_x, max_y = transformer.transform(max_lon, max_lat)
    buffer_m = buffer_km * 1_000.0
    return (
        min_x - buffer_m,
        min_y - buffer_m,
        max_x + buffer_m,
        max_y + buffer_m,
    ), (min_lon, min_lat, max_lon, max_lat)


def extract_network(zip_path, data_dir, output_dir, buffer_km=DEFAULT_BUFFER_KM):
    zip_path = Path(zip_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bbox, source_coordinate_bounds = coordinate_bounds(data_dir, buffer_km)
    vsi_root = "/vsizip/" + zip_path.as_posix()

    link_columns = [
        "LINK_ID",
        "F_NODE",
        "T_NODE",
        "LANES",
        "ROAD_RANK",
        "ROAD_TYPE",
        "ROAD_NO",
        "ROAD_NAME",
        "ROAD_USE",
        "MULTI_LINK",
        "CONNECT",
        "MAX_SPD",
        "LENGTH",
        "UPDATEDATE",
    ]
    node_columns = [
        "NODE_ID",
        "NODE_TYPE",
        "NODE_NAME",
        "TURN_P",
        "UPDATEDATE",
    ]

    links = pyogrio.read_dataframe(
        f"{vsi_root}/MOCT_LINK.shp",
        bbox=bbox,
        columns=link_columns,
    )
    nodes = pyogrio.read_dataframe(
        f"{vsi_root}/MOCT_NODE.shp",
        bbox=bbox,
        columns=node_columns,
    )

    raw_link_count = len(links)
    valid_node_ids = set(nodes["NODE_ID"])
    links = links[
        links["F_NODE"].isin(valid_node_ids)
        & links["T_NODE"].isin(valid_node_ids)
    ].copy()
    used_node_ids = set(links["F_NODE"]) | set(links["T_NODE"])
    nodes = nodes[nodes["NODE_ID"].isin(used_node_ids)].copy()

    links["FREE_FLOW_MIN"] = links["LENGTH"] * 0.06 / links["MAX_SPD"]
    nodes["X_5186"] = nodes.geometry.x
    nodes["Y_5186"] = nodes.geometry.y
    nodes_wgs84 = nodes.to_crs(TARGET_CRS)
    nodes["LONGITUDE"] = nodes_wgs84.geometry.x
    nodes["LATITUDE"] = nodes_wgs84.geometry.y

    nodes = nodes.sort_values("NODE_ID").reset_index(drop=True)
    links = links.sort_values("LINK_ID").reset_index(drop=True)

    node_csv_columns = [
        "NODE_ID",
        "NODE_TYPE",
        "NODE_NAME",
        "TURN_P",
        "UPDATEDATE",
        "X_5186",
        "Y_5186",
        "LONGITUDE",
        "LATITUDE",
    ]
    link_csv_columns = link_columns + ["FREE_FLOW_MIN"]
    nodes[node_csv_columns].to_csv(
        output_dir / "cheonan_nodes.csv.gz",
        index=False,
        encoding="utf-8-sig",
        compression="gzip",
    )
    links[link_csv_columns].to_csv(
        output_dir / "cheonan_links.csv.gz",
        index=False,
        encoding="utf-8-sig",
        compression="gzip",
    )

    gpkg_path = output_dir / "cheonan_road_network.gpkg"
    pyogrio.write_dataframe(nodes, gpkg_path, layer="nodes", driver="GPKG")
    pyogrio.write_dataframe(
        links,
        gpkg_path,
        layer="links",
        driver="GPKG",
        append=True,
    )

    metadata = {
        "source_file": zip_path.name,
        "source_crs": SOURCE_CRS,
        "output_crs": SOURCE_CRS,
        "coordinate_data_bounds_wgs84": source_coordinate_bounds,
        "extraction_bbox_epsg5186": bbox,
        "buffer_km": buffer_km,
        "raw_spatial_link_count": raw_link_count,
        "dropped_boundary_link_count": raw_link_count - len(links),
        "node_count": len(nodes),
        "link_count": len(links),
        "free_flow_formula": "LENGTH_m * 0.06 / MAX_SPD_kmh",
    }
    (output_dir / "road_network_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return nodes, links, metadata


def parse_args():
    parser = argparse.ArgumentParser(description="천안 도로망 Node/Link를 추출합니다.")
    parser.add_argument("zip_path")
    parser.add_argument("--data-dir", default="data/processed/RL")
    parser.add_argument(
        "--output-dir",
        default="data/processed/RL/road_network",
    )
    parser.add_argument("--buffer-km", type=float, default=DEFAULT_BUFFER_KM)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.buffer_km < 0:
        raise ValueError("buffer-km는 0 이상이어야 합니다.")
    _, _, metadata = extract_network(
        zip_path=args.zip_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        buffer_km=args.buffer_km,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
