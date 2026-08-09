"""
천안시 교통혼잡도 HTML 대시보드

입력
----
data/analysis/교통혼잡도_도로구간.csv
data/analysis/교통혼잡도_시간대요약.csv
data/raw/sgis/hadmarea.geojson

출력
----
outputs/maps/traffic_dashboard.html

주의
----
도로구간 데이터에 위도·경도 컬럼이 있으면 지도에 마커를 표시한다.
좌표 컬럼이 없으면 행정동 경계 지도와 교통 통계 대시보드만 생성한다.
"""

from pathlib import Path
import json

import folium
import geopandas as gpd
import pandas as pd
from branca.colormap import linear
from folium.plugins import Fullscreen, MiniMap, MarkerCluster


# ============================================================
# 1. 경로
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[3]

ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"
RAW_DIR = ROOT_DIR / "data" / "raw"

TRAFFIC_PATH = ANALYSIS_DIR / "교통혼잡도_도로구간.csv"
TIME_PATH = ANALYSIS_DIR / "교통혼잡도_시간대요약.csv"

OUTPUT_DIR = ROOT_DIR / "outputs" / "maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "traffic_dashboard.html"

GEOJSON_CANDIDATES = [
    RAW_DIR / "sgis" / "hadmarea.geojson",
    RAW_DIR / "hadmarea.geojson",
    ROOT_DIR
    / "scripts"
    / "data_collection"
    / "boundary"
    / "hadmarea.geojson",
]


# ============================================================
# 2. 공통 함수
# ============================================================

def find_geojson() -> Path:
    """
    SGIS 행정동 경계 파일을 찾는다.
    """
    for path in GEOJSON_CANDIDATES:
        if path.exists():
            return path

    searched = "\n".join(str(path) for path in GEOJSON_CANDIDATES)

    raise FileNotFoundError(
        "행정동 GeoJSON 파일을 찾지 못했습니다.\n"
        f"확인 경로:\n{searched}"
    )


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """
    후보 목록 중 실제 데이터프레임에 존재하는 첫 번째 컬럼을 반환한다.
    """
    normalized_columns = {
        str(column).replace(" ", "").lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        normalized_candidate = (
            candidate.replace(" ", "").lower()
        )

        if normalized_candidate in normalized_columns:
            return normalized_columns[normalized_candidate]

    return None


def format_number(
    value,
    decimals: int = 2,
    suffix: str = "",
) -> str:
    """
    숫자를 HTML 표시용 문자열로 변환한다.
    """
    if pd.isna(value):
        return "-"

    try:
        number = float(value)

        if decimals == 0:
            return f"{number:,.0f}{suffix}"

        return f"{number:,.{decimals}f}{suffix}"

    except (TypeError, ValueError):
        return str(value)


# ============================================================
# 3. 데이터 불러오기
# ============================================================

def load_traffic_data() -> pd.DataFrame:
    if not TRAFFIC_PATH.exists():
        raise FileNotFoundError(
            f"교통혼잡도 분석 파일이 없습니다:\n{TRAFFIC_PATH}\n\n"
            "먼저 traffic_analysis.py를 실행해야 합니다."
        )

    df = pd.read_csv(
        TRAFFIC_PATH,
        encoding="utf-8-sig",
    )

    df.columns = df.columns.astype(str).str.strip()

    congestion_column = find_column(
        df,
        [
            "교통혼잡도",
            "혼잡도",
            "congestion_index",
        ],
    )

    speed_column = find_column(
        df,
        [
            "평균속도",
            "속도",
            "average_speed",
        ],
    )

    volume_column = find_column(
        df,
        [
            "교통량",
            "평균교통량",
            "average_volume",
        ],
    )

    if congestion_column is None:
        raise ValueError(
            "교통혼잡도 컬럼을 찾지 못했습니다.\n"
            f"현재 컬럼: {df.columns.tolist()}"
        )

    rename_map = {
        congestion_column: "교통혼잡도",
    }

    if speed_column is not None:
        rename_map[speed_column] = "평균속도"

    if volume_column is not None:
        rename_map[volume_column] = "교통량"

    df = df.rename(columns=rename_map)

    numeric_columns = [
        "교통혼잡도",
        "평균속도",
        "교통량",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    df = df.dropna(
        subset=["교통혼잡도"]
    ).copy()

    df["혼잡순위"] = (
        df["교통혼잡도"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    df["도로구간명"] = make_road_section_name(df)

    return df


def load_time_data() -> pd.DataFrame:
    if not TIME_PATH.exists():
        print(
            "[경고] 시간대 요약 파일이 없어 "
            "시간대 그래프를 제외합니다."
        )

        return pd.DataFrame()

    df = pd.read_csv(
        TIME_PATH,
        encoding="utf-8-sig",
    )

    df.columns = df.columns.astype(str).str.strip()

    time_column = find_column(
        df,
        [
            "혼잡시간",
            "시간대",
            "시간",
            "hour",
        ],
    )

    congestion_column = find_column(
        df,
        [
            "평균교통혼잡도",
            "평균혼잡도",
            "교통혼잡도",
            "혼잡도",
        ],
    )

    if time_column is None or congestion_column is None:
        print(
            "[경고] 시간대 요약 파일의 컬럼을 인식하지 못했습니다."
        )

        return pd.DataFrame()

    df = df.rename(
        columns={
            time_column: "혼잡시간",
            congestion_column: "평균교통혼잡도",
        }
    )

    df["혼잡시간"] = pd.to_numeric(
        df["혼잡시간"],
        errors="coerce",
    )

    df["평균교통혼잡도"] = pd.to_numeric(
        df["평균교통혼잡도"],
        errors="coerce",
    )

    return (
        df
        .dropna(
            subset=[
                "혼잡시간",
                "평균교통혼잡도",
            ]
        )
        .sort_values("혼잡시간")
        .reset_index(drop=True)
    )


def make_road_section_name(
    df: pd.DataFrame,
) -> pd.Series:
    """
    데이터 컬럼에 따라 사람이 읽기 쉬운 도로구간명을 만든다.
    """
    road_column = find_column(
        df,
        [
            "도로명",
            "road_name",
        ],
    )

    start_column = find_column(
        df,
        [
            "시점명",
            "시점",
            "시작지점",
            "from_node",
        ],
    )

    end_column = find_column(
        df,
        [
            "종점명",
            "종점",
            "종료지점",
            "to_node",
        ],
    )

    section_column = find_column(
        df,
        [
            "도로구간명",
            "구간명",
            "링크명",
            "section_name",
        ],
    )

    if section_column is not None:
        return df[section_column].fillna("도로구간")

    names = pd.Series(
        ["도로구간"] * len(df),
        index=df.index,
        dtype="object",
    )

    if road_column is not None:
        names = df[road_column].fillna("").astype(str)

    if start_column is not None and end_column is not None:
        names = (
            names
            + " | "
            + df[start_column].fillna("").astype(str)
            + " → "
            + df[end_column].fillna("").astype(str)
        )

    return names.str.strip(" |")


def load_boundary() -> gpd.GeoDataFrame:
    path = find_geojson()

    print(f"[GeoJSON] {path}")

    geo = gpd.read_file(path)

    # SGIS 좌표가 투영좌표인데 CRS 정보가 없거나
    # 잘못 기록된 경우를 보정한다.
    if geo.crs is None:
        geo = geo.set_crs(
            epsg=5179,
            allow_override=True,
        )

    else:
        bounds = geo.total_bounds

        # 경도 범위가 아닌 큰 좌표값이면 투영좌표로 판단
        if abs(bounds[0]) > 180 or abs(bounds[2]) > 180:
            geo = geo.set_crs(
                epsg=5179,
                allow_override=True,
            )

    geo = geo.to_crs(epsg=4326)

    return geo


# ============================================================
# 4. 좌표 컬럼 확인
# ============================================================

def find_coordinate_columns(
    df: pd.DataFrame,
) -> tuple[str | None, str | None]:
    latitude_column = find_column(
        df,
        [
            "위도",
            "lat",
            "latitude",
            "중심위도",
            "시점위도",
        ],
    )

    longitude_column = find_column(
        df,
        [
            "경도",
            "lon",
            "lng",
            "longitude",
            "중심경도",
            "시점경도",
        ],
    )

    return latitude_column, longitude_column


# ============================================================
# 5. 대시보드 HTML
# ============================================================

def create_header(
    traffic: pd.DataFrame,
) -> str:
    top_row = traffic.loc[
        traffic["교통혼잡도"].idxmax()
    ]

    average_congestion = traffic[
        "교통혼잡도"
    ].mean()

    average_speed = (
        traffic["평균속도"].mean()
        if "평균속도" in traffic.columns
        else None
    )

    average_volume = (
        traffic["교통량"].mean()
        if "교통량" in traffic.columns
        else None
    )

    return f"""
    <div id="traffic-header">
        <div class="dashboard-title">
            천안시 교통혼잡도 대시보드
        </div>

        <div class="dashboard-subtitle">
            교통혼잡도 =
            정규화 교통량 ×
            (1 − 정규화 평균속도)
        </div>

        <div class="metric-grid">
            <div class="metric-card-custom">
                <div class="metric-label">분석 도로구간</div>
                <div class="metric-value">
                    {len(traffic):,}개
                </div>
            </div>

            <div class="metric-card-custom">
                <div class="metric-label">최고 혼잡 구간</div>
                <div class="metric-value-small">
                    {top_row["도로구간명"]}
                </div>
            </div>

            <div class="metric-card-custom">
                <div class="metric-label">평균 혼잡도</div>
                <div class="metric-value">
                    {average_congestion:.3f}
                </div>
            </div>

            <div class="metric-card-custom">
                <div class="metric-label">평균속도</div>
                <div class="metric-value">
                    {
                        format_number(
                            average_speed,
                            decimals=1,
                            suffix=" km/h",
                        )
                    }
                </div>
            </div>

            <div class="metric-card-custom">
                <div class="metric-label">평균 교통량</div>
                <div class="metric-value">
                    {
                        format_number(
                            average_volume,
                            decimals=0,
                            suffix="대",
                        )
                    }
                </div>
            </div>
        </div>
    </div>
    """


def create_ranking_panel(
    traffic: pd.DataFrame,
    top_n: int = 20,
) -> str:
    top = (
        traffic
        .sort_values(
            "교통혼잡도",
            ascending=False,
        )
        .head(top_n)
    )

    rows = []

    for _, row in top.iterrows():
        rows.append(
            f"""
            <tr>
                <td>{int(row["혼잡순위"])}</td>
                <td class="road-name">
                    {row["도로구간명"]}
                </td>
                <td>
                    {format_number(row.get("평균속도"), 1)}
                </td>
                <td>
                    {format_number(row.get("교통량"), 0)}
                </td>
                <td class="score">
                    {row["교통혼잡도"]:.3f}
                </td>
            </tr>
            """
        )

    return f"""
    <div id="ranking-panel" class="dashboard-panel">
        <div class="panel-title">
            교통혼잡도 상위 {top_n}개 구간
        </div>

        <div class="table-wrapper">
            <table>
                <thead>
                    <tr>
                        <th>순위</th>
                        <th>도로구간</th>
                        <th>속도</th>
                        <th>교통량</th>
                        <th>혼잡도</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
    </div>
    """


def create_time_chart(
    time_df: pd.DataFrame,
) -> str:
    if time_df.empty:
        return """
        <div id="time-panel" class="dashboard-panel">
            <div class="panel-title">혼잡시간대 분석</div>
            <div class="empty-message">
                시간대별 요약 데이터가 없습니다.
            </div>
        </div>
        """

    labels = [
        f"{int(value)}시"
        for value in time_df["혼잡시간"]
    ]

    values = [
        round(float(value), 4)
        for value in time_df["평균교통혼잡도"]
    ]

    max_value = max(values) if values else 1

    bars = []

    for label, value in zip(labels, values):
        width = (
            value / max_value * 100
            if max_value > 0
            else 0
        )

        bars.append(
            f"""
            <div class="time-row">
                <div class="time-label">{label}</div>

                <div class="bar-track">
                    <div
                        class="bar-fill"
                        style="width: {width:.1f}%;">
                    </div>
                </div>

                <div class="bar-value">
                    {value:.3f}
                </div>
            </div>
            """
        )

    return f"""
    <div id="time-panel" class="dashboard-panel">
        <div class="panel-title">
            혼잡시간대별 평균 혼잡도
        </div>

        <div class="time-chart">
            {''.join(bars)}
        </div>
    </div>
    """


def create_scatter_panel(
    traffic: pd.DataFrame,
) -> str:
    if (
        "평균속도" not in traffic.columns
        or "교통량" not in traffic.columns
    ):
        return ""

    scatter_df = (
        traffic[
            [
                "평균속도",
                "교통량",
                "교통혼잡도",
                "도로구간명",
            ]
        ]
        .dropna()
        .copy()
    )

    points = scatter_df.to_dict("records")

    points_json = json.dumps(
        points,
        ensure_ascii=False,
    ).replace("</", "<\\/")

    return f"""
    <div id="scatter-panel" class="dashboard-panel">
        <div class="panel-title">
            평균속도와 교통량의 관계
        </div>

        <canvas
            id="traffic-scatter"
            width="500"
            height="270"
            aria-label="평균속도와 교통량 산점도">
        </canvas>

        <div class="chart-note">
            원의 크기가 클수록 교통혼잡도가 높은 도로구간입니다.
        </div>
    </div>

    <script>
        window.TRAFFIC_SCATTER_DATA = {points_json};
    </script>
    """


def create_notice(
    has_coordinates: bool,
) -> str:
    if has_coordinates:
        return ""

    return """
    <div id="coordinate-notice">
        현재 도로구간 데이터에는 위도·경도 또는 도로 선형 정보가 없어
        지도 위에 개별 도로구간을 표시하지 않았습니다.
        좌표가 추가되면 마커가 자동 생성됩니다.
    </div>
    """


def create_styles() -> str:
    return """
    <style>
        body {
            font-family:
                "Malgun Gothic",
                "Apple SD Gothic Neo",
                sans-serif;
        }

        #traffic-header {
            position: fixed;
            top: 12px;
            left: 45px;
            right: 45px;
            z-index: 9999;
            padding: 15px 18px;
            background: rgba(255, 255, 255, 0.96);
            border-radius: 14px;
            box-shadow: 0 3px 14px rgba(0, 0, 0, 0.22);
        }

        .dashboard-title {
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .dashboard-subtitle {
            color: #555;
            font-size: 13px;
            margin-bottom: 12px;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(130px, 1fr));
            gap: 9px;
        }

        .metric-card-custom {
            min-height: 59px;
            padding: 9px 11px;
            background: #f4f6f8;
            border-radius: 9px;
        }

        .metric-label {
            color: #666;
            font-size: 11px;
            margin-bottom: 4px;
        }

        .metric-value {
            font-size: 18px;
            font-weight: 700;
        }

        .metric-value-small {
            font-size: 12px;
            font-weight: 700;
            line-height: 1.4;
        }

        .dashboard-panel {
            position: fixed;
            z-index: 9998;
            background: rgba(255, 255, 255, 0.96);
            border-radius: 12px;
            padding: 13px;
            box-shadow: 0 3px 14px rgba(0, 0, 0, 0.2);
        }

        .panel-title {
            font-size: 15px;
            font-weight: 700;
            margin-bottom: 9px;
        }

        #ranking-panel {
            left: 45px;
            bottom: 30px;
            width: 510px;
            max-height: 390px;
        }

        #time-panel {
            right: 45px;
            top: 175px;
            width: 310px;
        }

        #scatter-panel {
            right: 45px;
            bottom: 30px;
            width: 410px;
        }

        .table-wrapper {
            max-height: 330px;
            overflow-y: auto;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
        }

        thead {
            position: sticky;
            top: 0;
            background: #eef1f4;
        }

        th,
        td {
            padding: 5px 6px;
            border-bottom: 1px solid #ddd;
            text-align: right;
        }

        th:nth-child(1),
        td:nth-child(1) {
            text-align: center;
        }

        th:nth-child(2),
        td:nth-child(2) {
            text-align: left;
        }

        .road-name {
            max-width: 260px;
            line-height: 1.35;
        }

        .score {
            font-weight: 700;
        }

        .time-row {
            display: grid;
            grid-template-columns: 35px 1fr 44px;
            gap: 7px;
            align-items: center;
            margin-bottom: 8px;
            font-size: 11px;
        }

        .bar-track {
            height: 10px;
            background: #e9ecef;
            border-radius: 5px;
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            background: #d95f0e;
            border-radius: 5px;
        }

        .bar-value {
            text-align: right;
            font-weight: 600;
        }

        #traffic-scatter {
            width: 100%;
            height: 230px;
        }

        .chart-note {
            color: #666;
            font-size: 10px;
            margin-top: 5px;
        }

        #coordinate-notice {
            position: fixed;
            z-index: 9997;
            top: 175px;
            left: 45px;
            width: 360px;
            padding: 10px 13px;
            border-radius: 9px;
            background: rgba(255, 248, 220, 0.96);
            border: 1px solid #e0c36d;
            font-size: 12px;
            line-height: 1.5;
        }

        .empty-message {
            color: #777;
            font-size: 12px;
        }

        @media (max-width: 1100px) {
            .metric-grid {
                grid-template-columns: repeat(3, 1fr);
            }

            #scatter-panel {
                display: none;
            }

            #ranking-panel {
                width: 440px;
            }
        }
    </style>
    """


def create_scatter_script() -> str:
    return """
    <script>
        document.addEventListener("DOMContentLoaded", function () {
            const canvas = document.getElementById("traffic-scatter");

            if (!canvas || !window.TRAFFIC_SCATTER_DATA) {
                return;
            }

            const context = canvas.getContext("2d");
            const data = window.TRAFFIC_SCATTER_DATA;

            if (data.length === 0) {
                return;
            }

            const width = canvas.width;
            const height = canvas.height;

            const padding = {
                left: 50,
                right: 15,
                top: 15,
                bottom: 38
            };

            const speeds = data.map(item => Number(item["평균속도"]));
            const volumes = data.map(item => Number(item["교통량"]));
            const congestion = data.map(
                item => Number(item["교통혼잡도"])
            );

            const minSpeed = Math.min(...speeds);
            const maxSpeed = Math.max(...speeds);
            const minVolume = Math.min(...volumes);
            const maxVolume = Math.max(...volumes);
            const maxCongestion = Math.max(...congestion);

            function scaleX(value) {
                const range = maxSpeed - minSpeed || 1;

                return padding.left
                    + (value - minSpeed) / range
                    * (width - padding.left - padding.right);
            }

            function scaleY(value) {
                const range = maxVolume - minVolume || 1;

                return height
                    - padding.bottom
                    - (value - minVolume) / range
                    * (height - padding.top - padding.bottom);
            }

            context.clearRect(0, 0, width, height);

            context.strokeStyle = "#d6d6d6";
            context.lineWidth = 1;

            for (let i = 0; i <= 5; i += 1) {
                const x = padding.left
                    + i / 5
                    * (width - padding.left - padding.right);

                const y = padding.top
                    + i / 5
                    * (height - padding.top - padding.bottom);

                context.beginPath();
                context.moveTo(x, padding.top);
                context.lineTo(x, height - padding.bottom);
                context.stroke();

                context.beginPath();
                context.moveTo(padding.left, y);
                context.lineTo(width - padding.right, y);
                context.stroke();
            }

            context.strokeStyle = "#333";
            context.beginPath();
            context.moveTo(padding.left, padding.top);
            context.lineTo(padding.left, height - padding.bottom);
            context.lineTo(width - padding.right, height - padding.bottom);
            context.stroke();

            data.forEach(function (item) {
                const x = scaleX(Number(item["평균속도"]));
                const y = scaleY(Number(item["교통량"]));

                const score = Number(item["교통혼잡도"]);
                const radius = 3 + (
                    maxCongestion > 0
                        ? score / maxCongestion * 7
                        : 0
                );

                context.beginPath();
                context.arc(x, y, radius, 0, Math.PI * 2);
                context.fillStyle = "rgba(217, 95, 14, 0.58)";
                context.fill();
                context.strokeStyle = "rgba(160, 55, 5, 0.85)";
                context.stroke();
            });

            context.fillStyle = "#333";
            context.font = "12px Malgun Gothic";

            context.textAlign = "center";
            context.fillText(
                "평균속도 (km/h)",
                width / 2,
                height - 8
            );

            context.save();
            context.translate(14, height / 2);
            context.rotate(-Math.PI / 2);
            context.fillText("교통량 (대)", 0, 0);
            context.restore();
        });
    </script>
    """


# ============================================================
# 6. 지도 생성
# ============================================================

def add_boundary_layer(
    map_object: folium.Map,
    boundary: gpd.GeoDataFrame,
) -> None:
    """
    천안시 행정동 경계를 지도에 표시한다.
    """
    folium.GeoJson(
        boundary.to_json(),
        name="천안시 행정동 경계",
        style_function=lambda _: {
            "fillColor": "#f4f4f4",
            "fillOpacity": 0.2,
            "color": "#555555",
            "weight": 1,
        },
        highlight_function=lambda _: {
            "fillOpacity": 0.35,
            "weight": 2,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=(
                ["adm_nm"]
                if "adm_nm" in boundary.columns
                else []
            ),
            aliases=(
                ["행정구역:"]
                if "adm_nm" in boundary.columns
                else []
            ),
        ),
    ).add_to(map_object)


def add_traffic_markers(
    map_object: folium.Map,
    traffic: pd.DataFrame,
    latitude_column: str,
    longitude_column: str,
) -> None:
    """
    도로구간 위도·경도가 있는 경우 혼잡도 마커를 표시한다.
    """
    marker_df = traffic.copy()

    marker_df[latitude_column] = pd.to_numeric(
        marker_df[latitude_column],
        errors="coerce",
    )

    marker_df[longitude_column] = pd.to_numeric(
        marker_df[longitude_column],
        errors="coerce",
    )

    marker_df = marker_df.dropna(
        subset=[
            latitude_column,
            longitude_column,
            "교통혼잡도",
        ]
    )

    if marker_df.empty:
        return

    min_score = marker_df["교통혼잡도"].min()
    max_score = marker_df["교통혼잡도"].max()

    if min_score == max_score:
        max_score = min_score + 0.001

    colormap = linear.YlOrRd_09.scale(
        min_score,
        max_score,
    )

    colormap.caption = "도로구간 교통혼잡도"
    colormap.add_to(map_object)

    marker_cluster = MarkerCluster(
        name="도로구간 혼잡도",
    ).add_to(map_object)

    max_congestion = marker_df[
        "교통혼잡도"
    ].max()

    for _, row in marker_df.iterrows():
        score = row["교통혼잡도"]

        radius = (
            5
            + (
                score / max_congestion * 10
                if max_congestion > 0
                else 0
            )
        )

        popup_html = f"""
        <div style="
            width: 290px;
            font-family: Malgun Gothic;
            line-height: 1.6;
        ">
            <div style="
                font-size: 15px;
                font-weight: 700;
                margin-bottom: 7px;
            ">
                {row["도로구간명"]}
            </div>

            <div>
                <b>혼잡 순위:</b>
                {int(row["혼잡순위"])}위
            </div>

            <div>
                <b>교통혼잡도:</b>
                {row["교통혼잡도"]:.3f}
            </div>

            <div>
                <b>평균속도:</b>
                {format_number(row.get("평균속도"), 1, " km/h")}
            </div>

            <div>
                <b>교통량:</b>
                {format_number(row.get("교통량"), 0, "대")}
            </div>
        </div>
        """

        folium.CircleMarker(
            location=[
                row[latitude_column],
                row[longitude_column],
            ],
            radius=radius,
            color=colormap(score),
            fill=True,
            fill_color=colormap(score),
            fill_opacity=0.75,
            weight=1,
            tooltip=(
                f'{row["도로구간명"]} · '
                f'혼잡도 {score:.3f}'
            ),
            popup=folium.Popup(
                popup_html,
                max_width=340,
            ),
        ).add_to(marker_cluster)


def create_dashboard() -> Path:
    traffic = load_traffic_data()
    time_df = load_time_data()
    boundary = load_boundary()

    bounds = boundary.total_bounds

    center = [
        (bounds[1] + bounds[3]) / 2,
        (bounds[0] + bounds[2]) / 2,
    ]

    traffic_map = folium.Map(
        location=center,
        zoom_start=11,
        tiles=None,
        control_scale=True,
    )

    folium.TileLayer(
        tiles="CartoDB positron",
        name="기본 지도",
        show=True,
    ).add_to(traffic_map)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        show=False,
    ).add_to(traffic_map)

    add_boundary_layer(
        traffic_map,
        boundary,
    )

    latitude_column, longitude_column = (
        find_coordinate_columns(traffic)
    )

    has_coordinates = (
        latitude_column is not None
        and longitude_column is not None
    )

    if has_coordinates:
        add_traffic_markers(
            traffic_map,
            traffic,
            latitude_column,
            longitude_column,
        )

        print(
            "[지도] 도로구간 위도·경도 마커를 추가했습니다."
        )

    else:
        print(
            "[지도] 좌표 컬럼이 없어 도로구간 마커는 "
            "표시하지 않습니다."
        )

    traffic_map.fit_bounds(
        [
            [bounds[1], bounds[0]],
            [bounds[3], bounds[2]],
        ]
    )

    Fullscreen(
        position="topright",
        title="전체 화면",
        title_cancel="전체 화면 종료",
        force_separate_button=True,
    ).add_to(traffic_map)

    MiniMap(
        toggle_display=True,
        position="bottomright",
    ).add_to(traffic_map)

    folium.LayerControl(
        collapsed=False,
        position="topright",
    ).add_to(traffic_map)

    root = traffic_map.get_root()

    root.header.add_child(
        folium.Element(
            create_styles()
        )
    )

    root.html.add_child(
        folium.Element(
            create_header(traffic)
        )
    )

    root.html.add_child(
        folium.Element(
            create_ranking_panel(traffic)
        )
    )

    root.html.add_child(
        folium.Element(
            create_time_chart(time_df)
        )
    )

    root.html.add_child(
        folium.Element(
            create_scatter_panel(traffic)
        )
    )

    root.html.add_child(
        folium.Element(
            create_notice(has_coordinates)
        )
    )

    root.html.add_child(
        folium.Element(
            create_scatter_script()
        )
    )

    traffic_map.save(OUTPUT_PATH)

    return OUTPUT_PATH


# ============================================================
# 7. 실행
# ============================================================

if __name__ == "__main__":
    result_path = create_dashboard()

    print()
    print("=" * 60)
    print("교통혼잡도 HTML 대시보드 생성 완료")
    print(f"저장 위치: {result_path}")
    print("=" * 60)