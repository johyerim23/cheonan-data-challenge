from pathlib import Path

import geopandas as gpd
import pandas as pd


# ============================================================
# 경로 설정
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT_DIR / "data" / "raw"
GNN_DIR = ROOT_DIR / "data" / "gnn"

GNN_DIR.mkdir(parents=True, exist_ok=True)

NODE_OUTPUT = GNN_DIR / "gnn_nodes.csv"
EDGE_OUTPUT = GNN_DIR / "gnn_edges.csv"


# ============================================================
# 천안시 31개 읍면동
# ============================================================

DONGNAM = [
    "목천읍", "풍세면", "광덕면", "북면",
    "성남면", "수신면", "병천면", "동면",
    "중앙동", "문성동", "원성1동", "원성2동",
    "봉명동", "일봉동", "신방동", "청룡동",
    "신안동",
]

SEOBUK = [
    "성환읍", "성거읍", "직산읍", "입장면",
    "성정1동", "성정2동",
    "쌍용1동", "쌍용2동", "쌍용3동",
    "백석동", "불당1동", "불당2동",
    "부성1동", "부성2동",
]

TARGET_DONGS = DONGNAM + SEOBUK


# SHP와 우리 데이터의 행정동명 통일
REPLACEMENTS = {
    "원성제1동": "원성1동",
    "원성제2동": "원성2동",
    "성정제1동": "성정1동",
    "성정제2동": "성정2동",
    "쌍용제1동": "쌍용1동",
    "쌍용제2동": "쌍용2동",
    "쌍용제3동": "쌍용3동",
    "불당제1동": "불당1동",
    "불당제2동": "불당2동",
    "부성제1동": "부성1동",
    "부성제2동": "부성2동",
}


# ============================================================
# SHP 파일 찾기
# ============================================================

def find_boundary_file():

    candidates = list(
        RAW_DIR.rglob("bnd_dong*.shp")
    )

    if not candidates:
        candidates = list(
            RAW_DIR.rglob("*.shp")
        )

    if not candidates:
        raise FileNotFoundError(
            "data/raw 아래에서 SHP 파일을 찾지 못했습니다."
        )

    print("사용 경계 파일:", candidates[0])

    return candidates[0]


# ============================================================
# 행정동 경계 불러오기
# ============================================================

def load_boundary():

    shp_path = find_boundary_file()
    gdf = gpd.read_file(shp_path)

    print("SHP 컬럼:", gdf.columns.tolist())


    # 행정동 이름 컬럼 찾기
    name_candidates = [
        "adm_nm",
        "ADM_NM",
        "행정동",
        "name",
    ]

    name_col = None

    for col in name_candidates:
        if col in gdf.columns:
            name_col = col
            break

    if name_col is None:
        raise ValueError(
            "행정동 이름 컬럼을 찾지 못했습니다."
        )


    # ========================================================
    # 1. ADM_CD 기준 천안시만 필터링
    # 동남구: 34011...
    # 서북구: 34012...
    # ========================================================

    gdf["ADM_CD"] = gdf["ADM_CD"].astype(str)

    gdf = gdf[
        gdf["ADM_CD"].str.startswith(
            ("34011", "34012")
        )
    ].copy()

    print("천안시 코드 필터 후 행 수:", len(gdf))


    # ========================================================
    # 2. 행정동 이름 정리
    # ========================================================

    gdf["행정동"] = (
        gdf[name_col]
        .astype(str)
        .str.strip()
        .replace(REPLACEMENTS)
    )


    # ========================================================
    # 3. 최종 31개만 선택
    # ========================================================

    gdf = gdf[
        gdf["행정동"].isin(TARGET_DONGS)
    ].copy()


    # 같은 행정동 geometry가 여러 개면 결합
    gdf = gdf.dissolve(
        by="행정동",
        as_index=False,
    )


    print("추출된 행정동 수:", len(gdf))

    print("\n추출된 행정동:")
    print(
        sorted(
            gdf["행정동"].tolist()
        )
    )


    if len(gdf) != 31:

        missing = sorted(
            set(TARGET_DONGS)
            - set(gdf["행정동"])
        )

        raise ValueError(
            f"행정동이 31개가 아닙니다.\n"
            f"현재: {len(gdf)}개\n"
            f"누락: {missing}"
        )


    return gdf


# ============================================================
# Node 생성
# ============================================================

def build_nodes(boundary):

    # 좌표를 위도/경도로 변환
    boundary_wgs = boundary.to_crs(epsg=4326)


    # polygon 내부 대표 좌표
    centers = boundary_wgs.geometry.representative_point()


    nodes = pd.DataFrame({
        "행정동": boundary_wgs["행정동"],
        "latitude": centers.y,
        "longitude": centers.x,
    })


    # 행정동마다 고정 node_id 부여
    node_order = {
        dong: i
        for i, dong in enumerate(TARGET_DONGS)
    }

    nodes["node_id"] = (
        nodes["행정동"]
        .map(node_order)
    )


    # 구 정보
    nodes["구"] = nodes["행정동"].apply(
        lambda x:
        "동남구"
        if x in DONGNAM
        else "서북구"
    )


    nodes = (
        nodes[
            [
                "node_id",
                "행정동",
                "구",
                "latitude",
                "longitude",
            ]
        ]
        .sort_values("node_id")
        .reset_index(drop=True)
    )


    nodes.to_csv(
        NODE_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )


    print("\nNode 저장 완료")
    print(NODE_OUTPUT)

    return nodes


# ============================================================
# Edge 생성
# ============================================================

def build_edges(boundary, nodes):

    # 경계 길이를 meter 단위로 계산하기 위해 좌표계 변경
    boundary_m = boundary.to_crs(epsg=5186)


    geometry = dict(
        zip(
            boundary_m["행정동"],
            boundary_m.geometry,
        )
    )


    node_map = dict(
        zip(
            nodes["행정동"],
            nodes["node_id"],
        )
    )


    edges = []

    names = nodes["행정동"].tolist()


    # 모든 행정동 쌍 비교
    for i in range(len(names)):

        for j in range(i + 1, len(names)):

            dong_a = names[i]
            dong_b = names[j]

            geom_a = geometry[dong_a]
            geom_b = geometry[dong_b]


            # 두 행정동이 공유하는 경계
            shared_boundary = (
                geom_a.boundary
                .intersection(geom_b.boundary)
            )

            shared_length = shared_boundary.length


            # 선 형태로 실제 경계를 공유하는 경우
            if shared_length > 1:

                edges.append({
                    "source": node_map[dong_a],
                    "target": node_map[dong_b],
                    "source_name": dong_a,
                    "target_name": dong_b,
                    "shared_boundary_m": shared_length,
                })


    edges = pd.DataFrame(edges)


    edges.to_csv(
        EDGE_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )


    print("\nEdge 저장 완료")
    print(EDGE_OUTPUT)

    return edges


# ============================================================
# 그래프 검증
# ============================================================

def validate_graph(nodes, edges):

    connected_nodes = (
        set(edges["source"])
        | set(edges["target"])
    )


    isolated_nodes = (
        set(nodes["node_id"])
        - connected_nodes
    )


    print()
    print("==============================")
    print("GNN Graph 생성 결과")
    print("==============================")

    print("Node 수:", len(nodes))
    print("무방향 Edge 수:", len(edges))

    print(
        "평균 Degree:",
        round(
            2 * len(edges) / len(nodes),
            2,
        )
    )


    if isolated_nodes:

        isolated_names = nodes[
            nodes["node_id"].isin(isolated_nodes)
        ]["행정동"].tolist()

        print(
            "고립 Node:",
            isolated_names
        )

    else:

        print("고립 Node: 없음")


# ============================================================
# 실행
# ============================================================

def main():

    boundary = load_boundary()

    nodes = build_nodes(
        boundary
    )

    edges = build_edges(
        boundary,
        nodes,
    )

    validate_graph(
        nodes,
        edges,
    )


if __name__ == "__main__":
    main()