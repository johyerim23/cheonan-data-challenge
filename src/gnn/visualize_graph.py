from pathlib import Path
from adjustText import adjust_text

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 경로
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT_DIR / "data" / "raw"
GNN_DIR = ROOT_DIR / "data" / "gnn"
FIGURE_DIR = ROOT_DIR / "figures" / "gnn"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

NODE_PATH = GNN_DIR / "gnn_nodes.csv"
EDGE_PATH = GNN_DIR / "gnn_edges.csv"

OUTPUT_PATH = FIGURE_DIR / "gnn_graph_map.png"


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
# 경계 SHP 찾기
# ============================================================

shp_candidates = list(
    RAW_DIR.rglob("bnd_dong_00_2025_2Q.shp")
)

if not shp_candidates:
    raise FileNotFoundError(
        "행정동 경계 SHP 파일을 찾지 못했습니다."
    )

SHP_PATH = shp_candidates[0]


# ============================================================
# 데이터 로드
# ============================================================

nodes = pd.read_csv(NODE_PATH)
edges = pd.read_csv(EDGE_PATH)

gdf = gpd.read_file(SHP_PATH)


# ============================================================
# 천안시 행정동만 필터링
# ============================================================

gdf["ADM_CD"] = gdf["ADM_CD"].astype(str)

gdf = gdf[
    gdf["ADM_CD"].str.startswith(
        ("34011", "34012")
    )
].copy()

print("천안시 코드 필터 후 행 수:", len(gdf))


# 그 다음 행정동명 추출
gdf["행정동"] = (
    gdf["ADM_NM"]
    .astype(str)
    .str.strip()
    .replace(REPLACEMENTS)
)

# 우리가 사용할 31개만 남김
gdf = gdf[
    gdf["행정동"].isin(TARGET_DONGS)
].copy()


gdf = gdf.dissolve(
    by="행정동",
    as_index=False,
)

print("최종 행정동 수:", len(gdf))

# ============================================================
# 좌표계 통일
# ============================================================

# 지도용 좌표계
gdf = gdf.to_crs(epsg=5186)

# 기존 위경도 Node를 GeoDataFrame으로 변환
node_gdf = gpd.GeoDataFrame(
    nodes,
    geometry=gpd.points_from_xy(
        nodes["longitude"],
        nodes["latitude"],
    ),
    crs="EPSG:4326",
)

node_gdf = node_gdf.to_crs(epsg=5186)


# node_id -> 좌표
pos = {
    row["node_id"]: (
        row.geometry.x,
        row.geometry.y,
    )
    for _, row in node_gdf.iterrows()
}


# ============================================================
# 시각화
# ============================================================

fig, ax = plt.subplots(
    figsize=(6, 6)
)


# 1. 행정동 Polygon 배경
gdf.plot(
    ax=ax,
    facecolor="#f4f6f8",
    edgecolor="#9aa0a6",
    linewidth=0.8,
)


# 2. Edge 그리기
for _, row in edges.iterrows():

    source = row["source"]
    target = row["target"]

    x1, y1 = pos[source]
    x2, y2 = pos[target]

    ax.plot(
        [x1, x2],
        [y1, y2],
        color="#666666",
        linewidth=0.6,
        alpha=0.45,
        zorder=2,
    )


# 3. Node 그리기
ax.scatter(
    node_gdf.geometry.x,
    node_gdf.geometry.y,
    s=35,
    color="#C6EA82",
    edgecolor="white",
    linewidth=0.7,
    zorder=3,
)


# 4. 행정동 이름
texts = []

for _, row in node_gdf.iterrows():

    text = ax.text(
        row.geometry.x,
        row.geometry.y,
        row["행정동"],
        fontsize=6,
        fontfamily="Malgun Gothic",
        ha="center",
        va="center",
        zorder=4,
    )

    texts.append(text)


adjust_text(
    texts,
    ax=ax,
    arrowprops=dict(
        arrowstyle="-",
        lw=0.4,
        alpha=0.4,
    ),
)


# ============================================================
# 제목 및 스타일
# ============================================================

ax.set_title(
    "천안시 행정동 기반 GNN Graph",
    fontsize=10,
    fontfamily="Malgun Gothic",
    pad=15,
)

ax.axis("off")

plt.tight_layout()


# ============================================================
# 저장
# ============================================================

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("저장 완료:")
print(OUTPUT_PATH)