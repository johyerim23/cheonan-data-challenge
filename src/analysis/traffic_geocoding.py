from __future__ import annotations

from pathlib import Path

import os
import re
import time

import pandas as pd
import requests
from difflib import SequenceMatcher
from dotenv import load_dotenv


# ============================================================
# 경로
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    ROOT_DIR
    / "data"
    / "processed"
)

MAPPING_DIR = (
    ROOT_DIR
    / "data"
    / "mapping"
)

MAPPING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# 입력
# ------------------------------------------------------------

ROAD_INPUT = (
    PROCESSED_DIR
    / "교통_도로구간_소통통계.csv"
)

SMART_INPUT = (
    PROCESSED_DIR
    / "교통_스마트교차로_일별요약.csv"
)

PARKING_INPUT = (
    PROCESSED_DIR
    / "교통_불법주정차_기초.csv"
)


# ------------------------------------------------------------
# 출력
# ------------------------------------------------------------

# TMAP이 자동 생성하는 원본
SMART_AUTO_OUTPUT = (
    MAPPING_DIR
    / "스마트교차로_TMAP자동매핑.csv"
)

# 사람이 검토/수정하는 최종본
SMART_OUTPUT = (
    MAPPING_DIR
    / "스마트교차로_행정동매핑.csv"
)

ROAD_POINT_OUTPUT = (
    MAPPING_DIR
    / "도로지점_TMAP매핑.csv"
)

ROAD_OUTPUT = (
    MAPPING_DIR
    / "도로구간_행정동매핑.csv"
)

PARKING_PLACE_OUTPUT = (
    MAPPING_DIR
    / "불법주정차_장소_TMAP매핑.csv"
)

PARKING_MAPPING_OUTPUT = (
    MAPPING_DIR
    / "불법주정차_행정동매핑.csv"
)


# ============================================================
# 천안시 행정동
# ============================================================

DONGNAM = [
    "목천읍",
    "풍세면",
    "광덕면",
    "북면",
    "성남면",
    "수신면",
    "병천면",
    "동면",
    "중앙동",
    "문성동",
    "원성1동",
    "원성2동",
    "봉명동",
    "일봉동",
    "신방동",
    "청룡동",
    "신안동",
]

SEOBUK = [
    "성환읍",
    "성거읍",
    "직산읍",
    "입장면",
    "성정1동",
    "성정2동",
    "쌍용1동",
    "쌍용2동",
    "쌍용3동",
    "백석동",
    "불당1동",
    "불당2동",
    "부성1동",
    "부성2동",
]

TARGET_DONGS = (
    DONGNAM
    + SEOBUK
)


# ============================================================
# TMAP 설정
# ============================================================

load_dotenv(
    ROOT_DIR / ".env"
)


TMAP_APP_KEY = os.getenv(
    "TMAP_APP_KEY"
)


if not TMAP_APP_KEY:

    raise RuntimeError(
        "TMAP_APP_KEY가 없습니다.\n"
        "프로젝트 루트의 .env 파일에\n"
        "TMAP_APP_KEY=발급받은키\n"
        "형태로 저장하세요."
    )


POI_URL = (
    "https://apis.openapi.sk.com"
    "/tmap/pois"
)

REVERSE_URL = (
    "https://apis.openapi.sk.com"
    "/tmap/geo/reversegeocoding"
)


HEADERS = {
    "appKey": TMAP_APP_KEY,
    "Accept": "application/json",
}


# ============================================================
# 공통
# ============================================================

def clean_text(
    value,
) -> str:

    if pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def api_get(
    url: str,
    params: dict,
) -> dict | None:

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=15,
        )
        # 검색 결과 없음
        if response.status_code == 204:
            return None

        response.raise_for_status()

        time.sleep(0.1)

        try:
            return response.json()

        except ValueError:
            print(
                f"[JSON 오류] "
                f"status={response.status_code}, "
                f"text={response.text[:100]}"
            )
            return None

    except requests.RequestException as e:

        print(
            f"[API 오류] {e}"
        )

        return None


# ============================================================
# TMAP POI 응답 파싱
# ============================================================

def get_poi_list(
    data: dict | None,
) -> list:

    if not data:
        return []


    search_info = data.get(
        "searchPoiInfo",
        {}
    )

    pois = search_info.get(
        "pois",
        {}
    )

    poi = pois.get(
        "poi",
        []
    )


    if isinstance(
        poi,
        dict,
    ):
        return [poi]


    if isinstance(
        poi,
        list,
    ):
        return poi


    return []


def make_poi_address(
    poi: dict,
) -> str:
    """
    TMAP POI 응답의 주소 조합
    """

    parts = [
        poi.get("upperAddrName", ""),
        poi.get("middleAddrName", ""),
        poi.get("lowerAddrName", ""),
        poi.get("detailAddrName", ""),
    ]

    return " ".join(
        str(x).strip()
        for x in parts
        if x
    )


# ============================================================
# 1. 장소 검색
# ============================================================

def search_place(
    keyword: str,
    original_name: str | None = None,
) -> dict | None:

    data = api_get(
        POI_URL,
        {
            "version": "1",
            "searchKeyword": keyword,
            "searchType": "all",
            "searchtypCd": "A",
            "resCoordType": "WGS84GEO",
            "count": 20,
            "page": 1,
        },
    )

    pois = get_poi_list(
        data
    )

    if not pois:
        return None


    candidates = []


    for poi in pois:

        address = make_poi_address(
            poi
        )

        # 천안시 결과만
        if "천안" not in address:
            continue


        poi_name = clean_text(
            poi.get(
                "name",
                ""
            )
        )


        if original_name:

            similarity = name_similarity(
                original_name,
                poi_name,
            )

        else:

            similarity = 1.0


        candidates.append(
            (
                similarity,
                poi,
            )
        )


    if not candidates:
        return None


    candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )


    best_score, best_poi = (
        candidates[0]
    )


    # 이름이 너무 다르면 자동 채택 X
    if (
        original_name
        and best_score < 0.70
    ):

        return None


    best_poi[
        "_similarity"
    ] = best_score


    return best_poi


# ============================================================
# 2. 좌표 → 행정동
# ============================================================

def reverse_geocode(
    lon,
    lat,
) -> dict | None:
    """
    TMAP Reverse Geocoding

    addressType=A01
    → 행정동 주소 요청
    """

    data = api_get(
        REVERSE_URL,
        {
            "version": "1",

            "lat": lat,
            "lon": lon,

            "coordType":
                "WGS84GEO",

            "addressType":
                "A01",

            "newAddressExtend":
                "Y",
        },
    )


    if not data:
        return None


    info = data.get(
        "addressInfo",
        {}
    )


    if not info:
        return None


    dong = clean_text(
        info.get(
            "adminDong",
            ""
        )
    )


    return {
        "행정동": dong,

        "행정동코드":
            info.get(
                "adminDongCode",
                ""
            ),

        "법정동":
            info.get(
                "legalDong",
                ""
            ),

        "구":
            info.get(
                "gu_gun",
                ""
            ),

        "시도":
            info.get(
                "city_do",
                ""
            ),

        "전체주소":
            info.get(
                "fullAddress",
                ""
            ),
    }


# ============================================================
# 3. 장소명 → 좌표 → 행정동
# ============================================================

def geocode_keyword(
    keyword: str,
    original_name: str | None = None,
) -> dict:

    poi = search_place(
        keyword,
        original_name=original_name,
    )


    # --------------------------------------------------------
    # POI 검색 실패
    # --------------------------------------------------------

    if poi is None:

        return {
            "검색어":
                keyword,

            "검색성공":
                False,

            "TMAP장소명":
                "",

            "TMAP주소":
                "",

            "lon":
                pd.NA,

            "lat":
                pd.NA,

            "행정동":
                "",

            "행정동코드":
                "",

            "구":
                "",

            "행정동유효":
                False,
            "이름유사도": 
                pd.NA,
        }


    # TMAP POI 응답에서
    # noorLon / noorLat 사용
    lon = poi.get(
        "noorLon"
    )

    lat = poi.get(
        "noorLat"
    )


    if not lon or not lat:

        lon = poi.get(
            "frontLon"
        )

        lat = poi.get(
            "frontLat"
        )


    # --------------------------------------------------------
    # 좌표 없음
    # --------------------------------------------------------

    if not lon or not lat:

        return {
            "검색어":
                keyword,

            "검색성공":
                True,

            "TMAP장소명":
                poi.get(
                    "name",
                    ""
                ),

            "TMAP주소":
                make_poi_address(
                    poi
                ),

            "lon":
                pd.NA,

            "lat":
                pd.NA,

            "행정동":
                "",

            "행정동코드":
                "",

            "구":
                "",

            "행정동유효":
                False,

            "이름유사도": 
                pd.NA,
        }


    region = reverse_geocode(
        lon,
        lat,
    )


    if region:

        dong = region[
            "행정동"
        ]

        code = region[
            "행정동코드"
        ]

        gu = region[
            "구"
        ]

    else:

        dong = ""
        code = ""
        gu = ""


    return {
        "검색어":
            keyword,

        "검색성공":
            True,

        "TMAP장소명":
            poi.get(
                "name",
                ""
            ),

        "TMAP주소":
            make_poi_address(
                poi
            ),

        "lon":
            lon,

        "lat":
            lat,

        "행정동":
            dong,

        "행정동코드":
            code,

        "구":
            gu,

        "행정동유효":
            dong in TARGET_DONGS,

        "이름유사도":
            poi.get(
                "_similarity",
                pd.NA,
            ),
    }


# ============================================================
# 테스트
# ============================================================

def test_tmap():
    """
    전체 호출 전에 TMAP API가
    정상 동작하는지 1건 확인
    """

    print(
        "\n[TMAP API 테스트]"
    )


    result = geocode_keyword(
        "천안시청"
    )


    print(
        result
    )


    if not result[
        "검색성공"
    ]:

        raise RuntimeError(
            "TMAP POI 검색 테스트에 실패했습니다."
        )


    if not result[
        "행정동"
    ]:

        raise RuntimeError(
            "TMAP Reverse Geocoding 테스트에 실패했습니다."
        )


    print(
        "\nTMAP 연결 정상"
    )


# ============================================================
# 4. 스마트교차로
# ============================================================
INTERSECTION_ALIASES = {
    # 오매핑 교정
    "삼용": "삼룡사거리",

    # 검색 실패 항목
    "굴울(변전소)": "굴울 변전소",
    "방아다리(이마트)": "방아다리 이마트",
    "번영로(갤러리아)": "갤러리아 센터시티",
    "불당현대": "불당 현대아파트",
    "삼성전관": "삼성SDI 천안",
    "성정중(통계청)": "성정중학교",
    "수자인입구": "직산 한양수자인",
    "쌍용아이파크": "쌍용 아이파크",
    "새말(한라APT입구)": "새말 한라아파트",
    "이수(3산업)": "천안 제3산업단지",
    "청당동입구": "청당동",
}

def clean_intersection_name(
    name: str,
) -> str:
    """
    스마트교차로 원본의
    끝자리 3/4/5 제거

    예:
    구성3 → 구성
    방죽안5 → 방죽안
    """
    name = clean_text(
        name
    )

    name = re.sub(
        r"[345]$",
        "",
        name,
    )

    return name

def search_intersection(
    name: str,
) -> dict:

    # 별칭이 있으면 별칭 우선 사용
    search_name = INTERSECTION_ALIASES.get(
        name,
        name,
    )

    queries = [
        f"천안 {search_name} 교차로",
        f"천안 {search_name} 사거리",
        f"천안 {search_name} 삼거리",
        f"천안 {search_name}",
    ]

    last_result = None

    for query in queries:

        result = geocode_keyword(
            query,
            original_name=search_name,
        )

        last_result = result

        if result[
            "행정동유효"
        ]:
            return result

    return last_result

def geocode_smart():

    smart = pd.read_csv(
        SMART_INPUT,
        encoding="utf-8-sig",
    )


    names = (
        smart[
            "교차로명"
        ]
        .dropna()
        .drop_duplicates()
        .tolist()
    )


    records = []


    print(
        "\n=============================="
    )

    print(
        "스마트교차로 TMAP 매핑"
    )

    print(
        "=============================="
    )


    for i, original_name in enumerate(
        names,
        start=1,
    ):

        name = clean_intersection_name(
            original_name
        )


        # ----------------------------------------------------
        # 검색어 1
        # ----------------------------------------------------

        result = search_intersection(
            name
        )


        result[
            "교차로명"
        ] = original_name


        result[
            "검색용교차로명"
        ] = name

        similarity = result.get(
            "이름유사도",
            pd.NA,
        )

        poi_name = str(
            result.get(
                "TMAP장소명",
                ""
            )
        )

        intersection_words = [
            "사거리",
            "삼거리",
            "교차로",
            "오거리",
            "육거리",
            "지하차도",
            "고가차도",
        ]


        if not result[
            "행정동유효"
        ]:

            mapping_status = "검색실패"


        elif any(
            word in poi_name
            for word in intersection_words
        ):

            mapping_status = "자동확정"


        elif (
            pd.notna(similarity)
            and similarity >= 0.85
        ):

            mapping_status = "검토필요"


        else:

            mapping_status = "검토필요"


        result[
            "매핑상태"
        ] = mapping_status
        records.append(
            result
        )


        status = (
            result["행정동"]
            if result[
                "행정동유효"
            ]
            else "검토필요"
        )


        print(
            f"[{i:02d}/{len(names)}] "
            f"{original_name} "
            f"→ {status}"
        )


    result_df = (
        pd.DataFrame(
            records
        )
    )


    # traffic_analysis.py가 기대하는
    # 교차로명 + 행정동 형식 그대로 저장
    result_df.to_csv(
        SMART_AUTO_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )


    return result_df

def create_smart_manual_file(
    auto_df: pd.DataFrame,
):

    final_df = auto_df.copy()

    final_df[
        "수동확정행정동"
    ] = ""

    # TMAP에서 유효한 행정동을 찾은 경우
    # 자동확정 + 검토필요 모두 우선 사용
    final_df[
        "최종행정동"
    ] = ""

    valid_mask = (
        final_df[
            "행정동"
        ].isin(
            TARGET_DONGS
        )
    )

    final_df.loc[
        valid_mask,
        "최종행정동",
    ] = final_df.loc[
        valid_mask,
        "행정동",
    ]

    final_df.to_csv(
        SMART_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\n최종 매핑 파일 생성: {SMART_OUTPUT}"
    )

    print(
        f"- 행정동 사용 가능: "
        f"{valid_mask.sum()}개"
    )

    print(
        f"- 미매핑: "
        f"{(~valid_mask).sum()}개"
    )

    return final_df

# ============================================================
# 5. 도로구간
# ============================================================

def split_segment(
    value: str,
) -> tuple[str, str]:

    value = clean_text(
        value
    )


    if "->" not in value:

        return (
            value,
            value,
        )


    start, end = value.split(
        "->",
        1,
    )


    return (
        start.strip(),
        end.strip(),
    )

def search_road_point(
    point: str,
) -> dict:

    # 원본 우선
    result = geocode_keyword(
        f"천안 {point}",
        original_name=point,
    )

    if result["행정동유효"]:
        return result

    # 동측/서측 같은 방향 표현 제거 후 재검색
    base_point = re.sub(
        r"(동측|서측|남측|북측)$",
        "",
        point,
    ).strip()

    if base_point != point:

        retry = geocode_keyword(
            f"천안 {base_point}",
            original_name=base_point,
        )

        if retry["행정동유효"]:
            return retry

    return result

def geocode_road():

    road = pd.read_csv(
        ROAD_INPUT,
        encoding="utf-8-sig",
    )


    # --------------------------------------------------------
    # 모든 시작/종료 지점 추출
    # --------------------------------------------------------

    points = []


    for segment in road[
        "구간명"
    ]:

        start, end = split_segment(
            segment
        )

        points.extend(
            [
                start,
                end,
            ]
        )


    points = (
        pd.Series(
            points
        )
        .dropna()
        .drop_duplicates()
        .tolist()
    )


    records = []


    print(
        "\n=============================="
    )

    print(
        "도로 지점 TMAP 매핑"
    )

    print(
        "=============================="
    )


    for i, point in enumerate(
        points,
        start=1,
    ):

        result = search_road_point(
            point
        )

        result[
            "지점명"
        ] = point


        records.append(
            result
        )


        status = (
            result["행정동"]
            if result[
                "행정동유효"
            ]
            else "검토필요"
        )


        print(
            f"[{i:03d}/{len(points)}] "
            f"{point} "
            f"→ {status}"
        )


    point_df = pd.DataFrame(
        records
    )


    point_df.to_csv(
        ROAD_POINT_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )


    # --------------------------------------------------------
    # 지점 → 행정동 lookup
    # --------------------------------------------------------

    lookup = (
        point_df
        .set_index(
            "지점명"
        )[
            "행정동"
        ]
        .to_dict()
    )


    segment_records = []


    for _, row in road.iterrows():

        start, end = split_segment(
            row["구간명"]
        )


        start_dong = clean_text(
            lookup.get(
                start,
                ""
            )
        )


        end_dong = clean_text(
            lookup.get(
                end,
                ""
            )
        )


        # ----------------------------------------------------
        # 구간 행정동 판정
        # ----------------------------------------------------

        if (
            start_dong in TARGET_DONGS
            and start_dong == end_dong
        ):

            dong = start_dong

            mapping_status = (
                "자동확정"
            )


        elif (
            start_dong in TARGET_DONGS
            and not end_dong
        ):

            dong = start_dong

            mapping_status = (
                "종료지점검색실패"
            )


        elif (
            end_dong in TARGET_DONGS
            and not start_dong
        ):

            dong = end_dong

            mapping_status = (
                "시작지점검색실패"
            )


        elif (
            start_dong in TARGET_DONGS
            and end_dong in TARGET_DONGS
            and start_dong != end_dong
        ):

            # 서로 다른 동에 걸쳐 있는 도로
            # 임의로 하나를 선택하지 않음
            dong = ""

            mapping_status = (
                "경계구간검토"
            )


        else:

            dong = ""

            mapping_status = (
                "검색실패"
            )


        segment_records.append(
            {
                "도로구간_id":
                    row[
                        "도로구간_id"
                    ],

                "도로명":
                    row[
                        "도로명"
                    ],

                "구간명":
                    row[
                        "구간명"
                    ],

                "시작지점":
                    start,

                "시작행정동":
                    start_dong,

                "종료지점":
                    end,

                "종료행정동":
                    end_dong,

                "행정동":
                    dong,

                "매핑상태":
                    mapping_status,
            }
        )


    result = pd.DataFrame(
        segment_records
    )


    result.to_csv(
        ROAD_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )


    return (
        point_df,
        result,
    )


# ============================================================
# 6. 불법주정차
# ============================================================

def geocode_parking():

    parking = pd.read_csv(
        PARKING_INPUT,
        encoding="utf-8-sig",
    )


    # --------------------------------------------------------
    # 중요:
    # 83,148건을 전부 API 호출하지 않음
    #
    # 단속동 + 단속장소 조합으로
    # 중복 제거 후 검색
    # --------------------------------------------------------

    places = (
        parking[
            [
                "단속동_원본",
                "단속장소",
            ]
        ]
        .dropna(
            subset=[
                "단속장소"
            ]
        )
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )


    print(
        "\n=============================="
    )

    print(
        "불법주정차 TMAP 매핑"
    )

    print(
        "=============================="
    )


    print(
        f"전체 단속건수: "
        f"{len(parking):,}"
    )

    print(
        f"고유 장소: "
        f"{len(places):,}"
    )


    records = []


    for i, row in places.iterrows():

        original_dong = clean_text(
            row[
                "단속동_원본"
            ]
        )

        place = clean_text(
            row[
                "단속장소"
            ]
        )


        # ----------------------------------------------------
        # 검색어 1
        # ----------------------------------------------------

        query = (
            f"천안 {original_dong} "
            f"{place}"
        )


        result = geocode_keyword(
            query
        )


        # ----------------------------------------------------
        # 너무 구체적인 검색어가 실패하면
        # 단속장소만으로 한 번 재검색
        # ----------------------------------------------------

        if not result[
            "행정동유효"
        ]:

            retry_query = (
                f"천안 {place}"
            )


            retry = geocode_keyword(
                retry_query
            )


            if retry[
                "행정동유효"
            ]:

                result = retry


        result[
            "단속동_원본"
        ] = original_dong


        result[
            "단속장소"
        ] = place


        records.append(
            result
        )


        if (
            i == 0
            or (i + 1) % 100 == 0
            or i + 1 == len(places)
        ):

            success = sum(
                record[
                    "행정동유효"
                ]
                for record
                in records
            )


            print(
                f"[{i + 1:,}/{len(places):,}] "
                f"성공 {success:,}"
            )


    place_df = pd.DataFrame(
        records
    )


    place_df.to_csv(
        PARKING_PLACE_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )


    # --------------------------------------------------------
    # 장소 매핑을 원본 83,148건에 다시 병합
    # --------------------------------------------------------

    merged = parking.merge(
        place_df[
            [
                "단속동_원본",
                "단속장소",
                "행정동",
            ]
        ],
        on=[
            "단속동_원본",
            "단속장소",
        ],
        how="left",
        validate="many_to_one",
    )


    # --------------------------------------------------------
    # 단속동별 단일 행정동 매핑을 만들기 위한 참고
    #
    # 주의:
    # 불당동처럼 여러 행정동으로 나뉘는 곳은
    # 단일 매핑으로 강제하지 않음
    # --------------------------------------------------------

    valid = merged[
        merged[
            "행정동"
        ].isin(
            TARGET_DONGS
        )
    ].copy()


    dong_counts = (
        valid
        .groupby(
            [
                "단속동_원본",
                "행정동",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size":
                    "매핑건수"
            }
        )
    )


    # --------------------------------------------------------
    # 한 단속동이 정확히 한 행정동으로만
    # 매핑되는 경우에만 자동확정
    # --------------------------------------------------------

    mapping_records = []


    for original_dong, group in (
        dong_counts.groupby(
            "단속동_원본"
        )
    ):

        unique_dongs = (
            group[
                "행정동"
            ]
            .dropna()
            .unique()
            .tolist()
        )


        if len(
            unique_dongs
        ) == 1:

            dong = (
                unique_dongs[0]
            )

            status = (
                "자동확정"
            )

        else:

            dong = ""

            status = (
                "복수행정동_장소기반필요"
            )


        mapping_records.append(
            {
                "단속동_원본":
                    original_dong,

                "행정동":
                    dong,

                "매핑상태":
                    status,

                "검색된행정동":
                    ", ".join(
                        unique_dongs
                    ),
            }
        )


    mapping_df = pd.DataFrame(
        mapping_records
    )


    mapping_df.to_csv(
        PARKING_MAPPING_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )


    return (
        place_df,
        mapping_df,
    )


# ============================================================
# 결과 요약
# ============================================================

def print_summary(
    smart,
    road_points,
    roads,
    parking_places,
    parking_mapping,
):

    print(
        "\n\n=============================="
    )

    print(
        "TMAP 자동 매핑 결과"
    )

    print(
        "=============================="
    )


    # --------------------------------------------------------
    # 스마트교차로
    # --------------------------------------------------------

    print(
        "\n[스마트교차로]"
    )

    print(
        f"- 전체: "
        f"{len(smart)}개"
    )

    print(
        f"- 성공: "
        f"{smart['행정동유효'].sum()}개"
    )

    print(
        f"- 검토 필요: "
        f"{(~smart['행정동유효']).sum()}개"
    )


    # --------------------------------------------------------
    # 도로 지점
    # --------------------------------------------------------

    print(
        "\n[도로 지점]"
    )

    print(
        f"- 전체: "
        f"{len(road_points)}개"
    )

    print(
        f"- 성공: "
        f"{road_points['행정동유효'].sum()}개"
    )

    print(
        f"- 검토 필요: "
        f"{(~road_points['행정동유효']).sum()}개"
    )


    print(
        "\n[도로구간]"
    )

    print(
        roads[
            "매핑상태"
        ]
        .value_counts()
        .to_string()
    )


    # --------------------------------------------------------
    # 주정차
    # --------------------------------------------------------

    print(
        "\n[불법주정차 장소]"
    )

    print(
        f"- 고유 장소: "
        f"{len(parking_places):,}개"
    )

    print(
        f"- 성공: "
        f"{parking_places['행정동유효'].sum():,}개"
    )

    print(
        f"- 검토 필요: "
        f"{(~parking_places['행정동유효']).sum():,}개"
    )


    print(
        "\n[단속동 매핑 상태]"
    )

    if len(
        parking_mapping
    ):

        print(
            parking_mapping[
                "매핑상태"
            ]
            .value_counts()
            .to_string()
        )

def normalize_place_name(value: str) -> str:
    """
    장소명 비교용 정리
    """

    value = clean_text(value)

    replacements = {
        "APT": "아파트",
        "apt": "아파트",
        "초교": "초등학교",
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    # 교차로 데이터 끝의 3/4/5 제거
    value = re.sub(r"[345]$", "", value)

    # TMAP이 붙이는 표현 제거
    value = value.replace("[교차로]", "")

    # 괄호 안 별칭은 일단 제거
    value = re.sub(r"\([^)]*\)", "", value)

    # 공백 및 특수문자 제거
    value = re.sub(
        r"[^가-힣A-Za-z0-9]",
        "",
        value,
    )

    return value


def name_similarity(
    source_name: str,
    poi_name: str,
) -> float:

    source = normalize_place_name(
        source_name
    )

    target = normalize_place_name(
        poi_name
    )

    if not source or not target:
        return 0.0

    # 한쪽 이름이 다른 쪽에 완전히 포함
    if (
        source in target
        or target in source
    ):
        return 1.0

    return SequenceMatcher(
        None,
        source,
        target,
    ).ratio()

# ============================================================
# main
# ============================================================

def main():

    # --------------------------------------------------------
    # 0. 연결 테스트
    # --------------------------------------------------------

    test_tmap()


    # --------------------------------------------------------
    # 1. 스마트교차로
    # --------------------------------------------------------

    smart = geocode_smart()


    # --------------------------------------------------------
    # 2. 도로
    # --------------------------------------------------------

    road_points, roads = (
        geocode_road()
    )


    # --------------------------------------------------------
    # 3. 불법주정차
    # --------------------------------------------------------

    parking_places, parking_mapping = (
        geocode_parking()
    )


    # --------------------------------------------------------
    # 결과
    # --------------------------------------------------------

    print_summary(
        smart,
        road_points,
        roads,
        parking_places,
        parking_mapping,
    )


    print(
        "\n생성 파일"
    )

    print(
        f"- {SMART_OUTPUT}"
    )

    print(
        f"- {ROAD_POINT_OUTPUT}"
    )

    print(
        f"- {ROAD_OUTPUT}"
    )

    print(
        f"- {PARKING_PLACE_OUTPUT}"
    )

    print(
        f"- {PARKING_MAPPING_OUTPUT}"
    )


if __name__ == "__main__":

    parking = pd.read_csv(
        PARKING_INPUT,
        encoding="utf-8-sig",
    )

    summary = (
        parking
        .groupby(
            "단속동_원본"
        )
        .agg(
            단속건수=("단속장소", "size"),
            고유장소수=("단속장소", "nunique"),
        )
        .sort_values(
            "단속건수",
            ascending=False,
        )
    )

    print(summary.to_string())