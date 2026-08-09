from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from traffic_geocoding import (
    geocode_keyword,
    clean_text,
)


# ============================================================
# 경로
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MAPPING_DIR = ROOT_DIR / "data" / "mapping"

MAPPING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


INPUT = (
    PROCESSED_DIR
    / "교통_불법주정차_기초.csv"
)

CACHE_OUTPUT = (
    MAPPING_DIR
    / "불법주정차_장소_TMAP매핑.csv"
)

OUTPUT = (
    PROCESSED_DIR
    / "교통_불법주정차_행정동매핑.csv"
)


# ============================================================
# 천안시 31개 행정동
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

TARGET_DONGS = (
    DONGNAM
    + SEOBUK
)


# ============================================================
# 법정동 → 행정동
#
# 하나의 행정동으로 바로 연결 가능한 곳만 직접 처리
# 나머지는 TMAP
# ============================================================

DIRECT_MAP = {

    # 읍면
    "목천읍": "목천읍",
    "풍세면": "풍세면",
    "광덕면": "광덕면",
    "북면": "북면",
    "성남면": "성남면",
    "수신면": "수신면",
    "병천면": "병천면",

    "성환읍": "성환읍",
    "성거읍": "성거읍",
    "직산읍": "직산읍",
    "입장면": "입장면",

    # 이름 동일
    "봉명동": "봉명동",
    "신방동": "신방동",
    "백석동": "백석동",

    # 청룡동
    "구성동": "청룡동",
    "청수동": "청룡동",
    "청당동": "청룡동",
    "삼룡동": "청룡동",
    "구룡동": "청룡동",

    # 신안동
    "신부동": "신안동",
    "안서동": "신안동",

    # 일봉동
    "다가동": "일봉동",
    "용곡동": "일봉동",

    # 중앙동
    "대흥동": "중앙동",
    "사직동": "중앙동",
    "영성동": "중앙동",
    "오룡동": "중앙동",

    # 문성동
    "성황동": "문성동",
    "문화동": "문성동",
}


# ============================================================
# 분할 법정동의 허용 행정동
# 잘못된 TMAP 검색 결과 방지
# ============================================================

ALLOWED = {

    "성정동": {
        "성정1동",
        "성정2동",
    },

    "불당동": {
        "불당1동",
        "불당2동",
    },

    "쌍용동": {
        "쌍용1동",
        "쌍용2동",
        "쌍용3동",
    },

    "원성동": {
        "원성1동",
        "원성2동",
    },

    "두정동": {
        "부성1동",
        "부성2동",
    },

    "성성동": {
        "부성1동",
        "부성2동",
    },

    "차암동": {
        "부성1동",
        "부성2동",
    },

    "부대동": {
        "부성1동",
        "부성2동",
    },

    "신당동": {
        "부성1동",
        "부성2동",
    },

    "업성동": {
        "부성1동",
        "부성2동",
    },
}


# ============================================================
# 장소 하나 검색
# ============================================================

def search_place(
    legal_dong,
    place,
):

    legal_dong = clean_text(
        legal_dong
    )

    place = clean_text(
        place
    )

    if not place:

        return {
            "단속동_원본": legal_dong,
            "단속장소": place,
            "행정동": "",
            "검색성공": False,
        }


    # --------------------------------------------------------
    # 1차 검색
    # --------------------------------------------------------

    queries = [
        f"천안 {legal_dong} {place}",
        f"천안 {place}",
    ]


    for query in queries:

        try:

            result = geocode_keyword(
                query
            )

        except Exception:

            continue


        if not result:
            continue


        admin = result.get(
            "행정동",
            "",
        )

        address = str(
            result.get(
                "TMAP주소",
                "",
            )
        )


        if admin not in TARGET_DONGS:
            continue


        # 분할 법정동은
        # 가능한 행정동 범위 검사
        allowed = ALLOWED.get(
            legal_dong
        )

        if (
            allowed is not None
            and admin not in allowed
        ):
            continue


        # TMAP 주소에 원래 법정동이 포함되면 가장 안전
        # 포함되지 않더라도 allowed 범위가 맞으면 사용
        address_match = (
            legal_dong in address
        )


        return {
            "단속동_원본": legal_dong,
            "단속장소": place,
            "행정동": admin,
            "TMAP장소명":
                result.get(
                    "TMAP장소명",
                    "",
                ),
            "TMAP주소": address,
            "주소일치": address_match,
            "검색성공": True,
        }


    return {
        "단속동_원본": legal_dong,
        "단속장소": place,
        "행정동": "",
        "검색성공": False,
    }


# ============================================================
# main
# ============================================================

def main():

    parking = pd.read_csv(
        INPUT,
        encoding="utf-8-sig",
    )


    print(
        f"전체 불법주정차: "
        f"{len(parking):,}건"
    )


    # --------------------------------------------------------
    # 직접 매핑
    # --------------------------------------------------------

    parking[
        "행정동"
    ] = (
        parking[
            "단속동_원본"
        ]
        .map(
            DIRECT_MAP
        )
    )


    direct_count = (
        parking[
            "행정동"
        ]
        .notna()
        .sum()
    )

    print(
        f"직접 매핑: "
        f"{direct_count:,}건"
    )


    # --------------------------------------------------------
    # 직접 매핑되지 않은 장소
    # --------------------------------------------------------

    unresolved = (
        parking[
            parking[
                "행정동"
            ].isna()
        ][
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
        f"TMAP 검색 대상: "
        f"{len(unresolved):,}개 장소"
    )


    # --------------------------------------------------------
    # 기존 캐시
    # --------------------------------------------------------

    if CACHE_OUTPUT.exists():

        cache = pd.read_csv(
            CACHE_OUTPUT,
            encoding="utf-8-sig",
        )

    else:

        cache = pd.DataFrame()


    if not cache.empty:

        completed = set(
            zip(
                cache[
                    "단속동_원본"
                ].astype(str),

                cache[
                    "단속장소"
                ].astype(str),
            )
        )

        unresolved = unresolved[
            ~unresolved.apply(
                lambda row: (
                    str(
                        row[
                            "단속동_원본"
                        ]
                    ),
                    str(
                        row[
                            "단속장소"
                        ]
                    ),
                )
                in completed,
                axis=1,
            )
        ].reset_index(
            drop=True
        )


    print(
        f"실제 신규 API 검색: "
        f"{len(unresolved):,}개"
    )


    # --------------------------------------------------------
    # TMAP 검색
    # --------------------------------------------------------

    new_records = []


    def worker(row):

        return search_place(
            row[
                "단속동_원본"
            ],
            row[
                "단속장소"
            ],
        )


    # API 과부하 방지를 위해 6개 병렬
    with ThreadPoolExecutor(
        max_workers=2
    ) as executor:

        futures = [
            executor.submit(
                worker,
                row,
            )
            for _, row
            in unresolved.iterrows()
        ]


        for i, future in enumerate(
            as_completed(
                futures
            ),
            start=1,
        ):

            try:

                record = (
                    future.result()
                )

            except Exception as e:

                print(
                    "검색 오류:",
                    e,
                )

                continue


            new_records.append(
                record
            )


            if (
                i % 100 == 0
                or i
                == len(futures)
            ):

                print(
                    f"[{i:,}/{len(futures):,}]"
                )


                temp = pd.concat(
                    [
                        cache,
                        pd.DataFrame(
                            new_records
                        ),
                    ],
                    ignore_index=True,
                )


                temp.to_csv(
                    CACHE_OUTPUT,
                    index=False,
                    encoding="utf-8-sig",
                )


    # --------------------------------------------------------
    # 캐시 최종
    # --------------------------------------------------------

    cache = pd.concat(
        [
            cache,
            pd.DataFrame(
                new_records
            ),
        ],
        ignore_index=True,
    )


    cache = (
        cache
        .drop_duplicates(
            subset=[
                "단속동_원본",
                "단속장소",
            ],
            keep="last",
        )
    )


    cache.to_csv(
        CACHE_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )


    # --------------------------------------------------------
    # 원본에 장소 매핑 병합
    # --------------------------------------------------------

    tmap_map = (
        cache[
            [
                "단속동_원본",
                "단속장소",
                "행정동",
            ]
        ]
        .rename(
            columns={
                "행정동":
                    "TMAP행정동"
            }
        )
    )


    parking = parking.merge(
        tmap_map,
        on=[
            "단속동_원본",
            "단속장소",
        ],
        how="left",
    )


    mask = (
        parking[
            "행정동"
        ].isna()
        &
        parking[
            "TMAP행정동"
        ].isin(
            TARGET_DONGS
        )
    )


    parking.loc[
        mask,
        "행정동",
    ] = parking.loc[
        mask,
        "TMAP행정동",
    ]


    parking.drop(
        columns=[
            "TMAP행정동"
        ],
        inplace=True,
    )


    # --------------------------------------------------------
    # 저장
    # --------------------------------------------------------

    parking.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )


    success = (
        parking[
            "행정동"
        ]
        .isin(
            TARGET_DONGS
        )
        .sum()
    )


    print(
        "\n=============================="
    )

    print(
        "불법주정차 매핑 완료"
    )

    print(
        "=============================="
    )

    print(
        f"매핑 성공: "
        f"{success:,}/{len(parking):,}"
    )

    print(
        f"매핑률: "
        f"{success / len(parking) * 100:.2f}%"
    )


    print(
        "\n[행정동별 건수]"
    )

    print(
        parking[
            parking[
                "행정동"
            ].isin(
                TARGET_DONGS
            )
        ]
        .groupby(
            "행정동"
        )
        .size()
        .sort_values(
            ascending=False
        )
        .to_string()
    )


if __name__ == "__main__":
    main()