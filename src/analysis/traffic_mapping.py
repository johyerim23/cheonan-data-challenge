from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]

MAPPING_DIR = ROOT_DIR / "data" / "mapping"

ROAD_MAPPING = MAPPING_DIR / "도로구간_행정동매핑.csv"
SMART_MAPPING = MAPPING_DIR / "스마트교차로_행정동매핑.csv"
PARKING_MAPPING = MAPPING_DIR / "불법주정차_행정동매핑.csv"


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


def detect_dong(text):
    if pd.isna(text):
        return ""

    text = str(text)

    matches = [
        dong
        for dong in TARGET_DONGS
        if dong in text
    ]

    if len(matches) == 1:
        return matches[0]

    return ""


def fill_road():
    df = pd.read_csv(
        ROAD_MAPPING,
        encoding="utf-8-sig"
    )

    for i, row in df.iterrows():

        if pd.notna(row["행정동"]) and str(row["행정동"]).strip():
            continue

        text = (
            str(row["도로명"])
            + " "
            + str(row["구간명"])
        )

        dong = detect_dong(text)

        if dong:
            df.at[i, "행정동"] = dong

    df.to_csv(
        ROAD_MAPPING,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "[도로구간]",
        (df["행정동"].fillna("") != "").sum(),
        "/",
        len(df),
        "자동 매핑"
    )


def fill_smart():
    df = pd.read_csv(
        SMART_MAPPING,
        encoding="utf-8-sig"
    )

    for i, row in df.iterrows():

        if pd.notna(row["행정동"]) and str(row["행정동"]).strip():
            continue

        dong = detect_dong(
            row["교차로명"]
        )

        if dong:
            df.at[i, "행정동"] = dong

    df.to_csv(
        SMART_MAPPING,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "[스마트교차로]",
        (df["행정동"].fillna("") != "").sum(),
        "/",
        len(df),
        "자동 매핑"
    )


def fill_parking():
    df = pd.read_csv(
        PARKING_MAPPING,
        encoding="utf-8-sig"
    )

    for i, row in df.iterrows():

        if pd.notna(row["행정동"]) and str(row["행정동"]).strip():
            continue

        original = str(
            row["단속동_원본"]
        ).strip()

        if original in TARGET_DONGS:
            df.at[i, "행정동"] = original

    df.to_csv(
        PARKING_MAPPING,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "[불법주정차]",
        (df["행정동"].fillna("") != "").sum(),
        "/",
        len(df),
        "자동 매핑"
    )


def show_unmapped():
    print("\n==============================")
    print("미매핑 항목")
    print("==============================")

    road = pd.read_csv(
        ROAD_MAPPING,
        encoding="utf-8-sig"
    )

    smart = pd.read_csv(
        SMART_MAPPING,
        encoding="utf-8-sig"
    )

    parking = pd.read_csv(
        PARKING_MAPPING,
        encoding="utf-8-sig"
    )

    road_missing = road[
        road["행정동"].fillna("").eq("")
    ]

    smart_missing = smart[
        smart["행정동"].fillna("").eq("")
    ]

    parking_missing = parking[
        parking["행정동"].fillna("").eq("")
    ]

    print(
        f"\n도로구간 미매핑: {len(road_missing)}개"
    )

    if len(road_missing):
        print(
            road_missing[
                ["도로명", "구간명"]
            ].to_string(index=False)
        )

    print(
        f"\n스마트교차로 미매핑: {len(smart_missing)}개"
    )

    if len(smart_missing):
        print(
            smart_missing[
                ["교차로명"]
            ].to_string(index=False)
        )

    print(
        f"\n불법주정차 미매핑: {len(parking_missing)}개"
    )

    if len(parking_missing):
        print(
            parking_missing[
                ["단속동_원본"]
            ].to_string(index=False)
        )


def main():
    fill_road()
    fill_smart()
    fill_parking()

    show_unmapped()


if __name__ == "__main__":
    main()