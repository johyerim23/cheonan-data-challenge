import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2


class EmergencyPolicySimulator:

    def __init__(
        self,
        data_dir="data/processed/RL"
    ):
        self.data_dir = data_dir

        # =========================
        # 데이터 로드
        # =========================
        self.region_features = pd.read_csv(
            f"{data_dir}/region_features.csv"
        )

        self.regions = pd.read_csv(
            f"{data_dir}/cheonan_region_db.csv"
        )

        self.ems_stations = pd.read_csv(
            f"{data_dir}/cheonan_ems_station_db.csv"
        )

        self.hospitals_all = pd.read_csv(
            f"{data_dir}/cheonan_hospital_db_all.csv"
        )

        self.emergency_hospitals = pd.read_csv(
            f"{data_dir}/cheonan_emergency_hospital_db.csv"
        )

        self._validate_data()

        # 읍면동 좌표 붙이기
        self.base_state = self.region_features.merge(
            self.regions[
                [
                    "region_id",
                    "latitude",
                    "longitude"
                ]
            ],
            on="region_id",
            how="left"
        )

        # 데이터 기반 보정값 계산
        self._build_calibration()

        # 병원 업그레이드 후보
        self._build_hospital_candidates()

        self.type_cutoffs = {
            "type_dispatch": float(self.base_state["ems_duration_min"].quantile(0.70)),
            "type_transfer": float(self.base_state["hospital_duration_min"].quantile(0.70)),
            "type_alternative": float(self.base_state["alternative_gap_min"].quantile(0.70)),
        }

        self.reset()


    # =========================================================
    # 데이터 검증
    # =========================================================
    def _validate_data(self):

        if len(self.region_features) != 31:
            raise ValueError(
                f"region_features가 31개 지역이 아닙니다: "
                f"{len(self.region_features)}"
            )

        required_region_cols = [
            "region_id",
            "region_name",
            "ems_distance_km",
            "ems_duration_min",
            "hospital_distance_km",
            "hospital_duration_min",
            "occurrence_score_01",
            "vulnerable_score_01",
        ]

        missing = [
            c for c in required_region_cols
            if c not in self.region_features.columns
        ]

        if missing:
            raise ValueError(
                f"region_features 필수 컬럼 누락: {missing}"
            )


    # =========================================================
    # Haversine 직선거리
    # =========================================================
    @staticmethod
    def haversine(
        lat1,
        lon1,
        lat2,
        lon2
    ):
        """
        위경도 두 지점 사이 직선거리(km)
        """

        R = 6371.0

        lat1 = radians(lat1)
        lon1 = radians(lon1)
        lat2 = radians(lat2)
        lon2 = radians(lon2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            sin(dlat / 2) ** 2
            +
            cos(lat1)
            * cos(lat2)
            * sin(dlon / 2) ** 2
        )

        c = 2 * atan2(
            sqrt(a),
            sqrt(1 - a)
        )

        return R * c


    # =========================================================
    # 현재 데이터 기반 거리/시간 보정
    # =========================================================
    def _build_calibration(self):
        """
        현재 관측된 EMS / 병원 이동거리와
        위경도 직선거리를 비교해서

        1) 직선거리 -> 실제 도로거리 보정계수
        2) 실제 도로거리 -> 이동시간 계수

        를 데이터에서 추정한다.
        """

        region_map = self.regions.set_index(
            "region_id"
        )

        ems_map = self.ems_stations.set_index(
            "station_id"
        )

        emergency_map = (
            self.emergency_hospitals
            .set_index("hospital_id")
        )

        # ---------------------------------
        # EMS calibration
        # ---------------------------------
        ems_circuity = []
        ems_min_per_km = []

        for _, row in self.region_features.iterrows():

            station_id = row["station_id"]
            region_id = row["region_id"]

            if station_id not in ems_map.index:
                continue

            if region_id not in region_map.index:
                continue

            station = ems_map.loc[station_id]
            region = region_map.loc[region_id]

            straight_km = self.haversine(
                station["latitude"],
                station["longitude"],
                region["latitude"],
                region["longitude"]
            )

            road_km = row["ems_distance_km"]
            duration = row["ems_duration_min"]

            if straight_km > 0:
                ems_circuity.append(
                    road_km / straight_km
                )

            if road_km > 0:
                ems_min_per_km.append(
                    duration / road_km
                )

        self.ems_circuity = float(
            np.median(ems_circuity)
        )

        self.ems_min_per_km = float(
            np.median(ems_min_per_km)
        )

        # ---------------------------------
        # Hospital calibration
        # ---------------------------------
        hospital_circuity = []
        hospital_min_per_km = []

        for _, row in self.region_features.iterrows():

            hospital_id = row["hospital_id"]
            region_id = row["region_id"]

            if hospital_id not in emergency_map.index:
                continue

            if region_id not in region_map.index:
                continue

            hospital = emergency_map.loc[
                hospital_id
            ]

            region = region_map.loc[
                region_id
            ]

            straight_km = self.haversine(
                region["latitude"],
                region["longitude"],
                hospital["latitude"],
                hospital["longitude"]
            )

            road_km = row["hospital_distance_km"]
            duration = row["hospital_duration_min"]

            if straight_km > 0:
                hospital_circuity.append(
                    road_km / straight_km
                )

            if road_km > 0:
                hospital_min_per_km.append(
                    duration / road_km
                )

        self.hospital_circuity = float(
            np.median(hospital_circuity)
        )

        self.hospital_min_per_km = float(
            np.median(hospital_min_per_km)
        )


    # =========================================================
    # 병원 업그레이드 후보
    # =========================================================
    def _build_hospital_candidates(self):
        """
        천안시 소재 일반 병원 중
        현재 응급의료기관이 아닌 병원을 후보로 사용.
        """

        df = self.hospitals_all.copy()

        # 천안 주소
        df = df[
            df["address"]
            .str.contains("천안", na=False)
        ].copy()

        # 현재 응급기관 제외
        emergency_ids = set(
            self.emergency_hospitals[
                "hospital_id"
            ]
        )

        df = df[
            ~df["institution_id"]
            .isin(emergency_ids)
        ]

        # 일반 병원만
        df = df[
            df["institution_type"] == "병원"
        ].copy()

        df = df.dropna(
            subset=[
                "latitude",
                "longitude"
            ]
        )

        self.hospital_candidates = (
            df.reset_index(drop=True)
        )

        # Cache all candidate-to-region estimates once; training reuses them.
        self.hospital_candidate_times = {}
        for _, hospital in self.hospital_candidates.iterrows():
            self.hospital_candidate_times[hospital["institution_id"]] = np.array(
                [
                    self.estimate_hospital_time(
                        region["latitude"], region["longitude"],
                        hospital["latitude"], hospital["longitude"],
                    )
                    for _, region in self.base_state.iterrows()
                ],
                dtype=float,
            )


    # =========================================================
    # 초기화
    # =========================================================
    def reset(self):

        self.state = self.base_state.copy(
            deep=True
        )

        # 현재 구급차 거점
        self.current_ems = (
            self.ems_stations.copy(
                deep=True
            )
        )

        # 현재 응급병원
        self.current_hospitals = (
            self.emergency_hospitals.copy(
                deep=True
            )
        )

        self.applied_policies = []
        self.upgraded_hospital_ids = []

        return self.state.copy()


    def snapshot(self):
        """Return a deep copy of all mutable policy state."""
        return {
            "state": self.state.copy(deep=True),
            "current_ems": self.current_ems.copy(deep=True),
            "current_hospitals": self.current_hospitals.copy(deep=True),
            "applied_policies": list(self.applied_policies),
            "upgraded_hospital_ids": list(self.upgraded_hospital_ids),
        }


    def restore(self, snapshot):
        """Restore a snapshot created by :meth:`snapshot`."""
        self.state = snapshot["state"].copy(deep=True)
        self.current_ems = snapshot["current_ems"].copy(deep=True)
        self.current_hospitals = snapshot["current_hospitals"].copy(deep=True)
        self.applied_policies = list(snapshot["applied_policies"])
        self.upgraded_hospital_ids = list(snapshot["upgraded_hospital_ids"])


    def _refresh_vulnerability_labels(self):
        """Refresh policy-sensitive vulnerability flags and labels."""
        self.state["type_dispatch"] = (
            self.state["ems_duration_min"] >= self.type_cutoffs["type_dispatch"]
        )
        self.state["type_transfer"] = (
            self.state["hospital_duration_min"] >= self.type_cutoffs["type_transfer"]
        )
        self.state["type_alternative"] = (
            self.state["alternative_gap_min"] >= self.type_cutoffs["type_alternative"]
        )

        type_cols = [
            "type_dispatch", "type_transfer", "type_alternative",
            "type_traffic", "type_demand",
        ]
        self.state["vulnerability_type_count"] = self.state[type_cols].sum(axis=1)
        labels = {
            "type_dispatch": "출동", "type_transfer": "이송",
            "type_alternative": "대체", "type_traffic": "교통",
            "type_demand": "수요",
        }

        def make_label(row):
            selected = [labels[column] for column in type_cols if bool(row[column])]
            return "+".join(selected) if selected else "상대적 비취약"

        self.state["vulnerability_types"] = self.state.apply(make_label, axis=1)
        self.state["is_complex_type"] = self.state["vulnerability_type_count"] >= 2


    # =========================================================
    # 신규 EMS 거점 -> 각 지역 ETA 계산
    # =========================================================
    def estimate_ems_time(
        self,
        source_lat,
        source_lon,
        target_lat,
        target_lon
    ):

        straight_km = self.haversine(
            source_lat,
            source_lon,
            target_lat,
            target_lon
        )

        estimated_road_km = (
            straight_km
            * self.ems_circuity
        )

        estimated_time = (
            estimated_road_km
            * self.ems_min_per_km
        )

        return estimated_time


    # =========================================================
    # 신규 병원 -> 각 지역 ETA 계산
    # =========================================================
    def estimate_hospital_time(
        self,
        region_lat,
        region_lon,
        hospital_lat,
        hospital_lon
    ):

        straight_km = self.haversine(
            region_lat,
            region_lon,
            hospital_lat,
            hospital_lon
        )

        estimated_road_km = (
            straight_km
            * self.hospital_circuity
        )

        estimated_time = (
            estimated_road_km
            * self.hospital_min_per_km
        )

        return estimated_time


    # =========================================================
    # 정책 1-1: 기존 센터 구급차 증차
    # =========================================================
    def add_ambulance_existing_center(
        self,
        station_id
    ):
        """
        현재 단계에서는 '구급차 수 증가'만 반영.

        단순 거리 기반 ETA에서는 같은 센터에
        구급차가 한 대 더 늘어도 거리 자체는 변하지 않으므로
        출동 ETA는 변경하지 않는다.

        이후 응급사건/동시출동 시뮬레이션을 넣으면
        ambulance_count가 중요한 변수가 된다.
        """

        idx = (
            self.current_ems["station_id"]
            == station_id
        )

        if not idx.any():
            raise ValueError(
                f"존재하지 않는 station_id: "
                f"{station_id}"
            )

        self.current_ems.loc[
            idx,
            "ambulance_count"
        ] += 1

        self.applied_policies.append(
            {
                "policy": "ambulance_existing",
                "target": station_id
            }
        )


    # =========================================================
    # 정책 1-2: 신규 구급거점
    # =========================================================
    def add_new_ambulance_base(
        self,
        region_id
    ):
        """
        해당 읍면동 대표점에
        신규 구급차 1대를 배치했다고 가정.

        기존 EMS ETA와 신규 거점 ETA 중
        더 짧은 값을 사용.
        """

        target = self.state[
            self.state["region_id"]
            == region_id
        ]

        if target.empty:
            raise ValueError(
                f"존재하지 않는 region_id: "
                f"{region_id}"
            )

        source_lat = target.iloc[0][
            "latitude"
        ]

        source_lon = target.iloc[0][
            "longitude"
        ]

        for idx, row in self.state.iterrows():

            new_time = self.estimate_ems_time(
                source_lat,
                source_lon,
                row["latitude"],
                row["longitude"]
            )

            self.state.loc[
                idx,
                "ems_duration_min"
            ] = min(
                row["ems_duration_min"],
                new_time
            )

        self._refresh_vulnerability_labels()

        self.applied_policies.append(
            {
                "policy": "ambulance_new_base",
                "target": region_id
            }
        )


    # =========================================================
    # 정책 2: 거점 병원 업그레이드
    # =========================================================
    def upgrade_hospital(
        self,
        institution_id
    ):
        """
        일반 병원을 응급 거점 병원으로
        업그레이드했다고 가정.

        기존 응급실 ETA와 해당 병원 ETA 중
        더 짧은 값을 사용.
        """

        candidate = (
            self.hospital_candidates[
                self.hospital_candidates[
                    "institution_id"
                ]
                == institution_id
            ]
        )

        if candidate.empty:
            raise ValueError(
                "업그레이드 후보가 아닙니다: "
                f"{institution_id}"
            )

        if institution_id in self.upgraded_hospital_ids:
            raise ValueError(f"이미 업그레이드한 병원입니다: {institution_id}")

        self.upgraded_hospital_ids.append(institution_id)

        upgraded = (
            self.hospital_candidates.set_index("institution_id")
            .loc[self.upgraded_hospital_ids]
            .reset_index()
        )

        # The observed baseline first/second choices remain in every candidate set.
        # Only newly upgraded hospitals use calibrated travel-time estimates.
        n_regions = len(self.base_state)
        times = np.column_stack(
            [
                self.base_state["hospital_duration_min"].to_numpy(dtype=float),
                self.base_state["nearest_alternative_time_min"].to_numpy(dtype=float),
            ]
            + [self.hospital_candidate_times[hospital_id]
               for hospital_id in self.upgraded_hospital_ids]
        )
        ids = np.empty((n_regions, times.shape[1]), dtype=object)
        names = np.empty_like(ids)
        types = np.empty_like(ids)
        ids[:, 0] = self.base_state["hospital_id"].to_numpy()
        ids[:, 1] = self.base_state["nearest_alternative_hospital_id"].to_numpy()
        names[:, 0] = self.base_state["hospital_name"].to_numpy()
        names[:, 1] = self.base_state["nearest_alternative_hospital_name"].to_numpy()
        types[:, 0] = self.base_state["hospital_type"].to_numpy()
        types[:, 1] = self.base_state["nearest_alternative_hospital_type"].to_numpy()
        for column, (_, hospital) in enumerate(upgraded.iterrows(), start=2):
            ids[:, column] = hospital["institution_id"]
            names[:, column] = hospital["name"]
            types[:, column] = "업그레이드 응급병원"

        order = np.argsort(times, axis=1, kind="stable")[:, :2]
        first_index = order[:, [0]]
        second_index = order[:, [1]]
        first_time = np.take_along_axis(times, first_index, axis=1).ravel()
        second_time = np.take_along_axis(times, second_index, axis=1).ravel()
        self.state["hospital_id"] = np.take_along_axis(ids, first_index, axis=1).ravel()
        self.state["hospital_name"] = np.take_along_axis(names, first_index, axis=1).ravel()
        self.state["hospital_type"] = np.take_along_axis(types, first_index, axis=1).ravel()
        self.state["hospital_duration_min"] = first_time
        self.state["nearest_alternative_hospital_id"] = np.take_along_axis(
            ids, second_index, axis=1
        ).ravel()
        self.state["nearest_alternative_hospital_name"] = np.take_along_axis(
            names, second_index, axis=1
        ).ravel()
        self.state["nearest_alternative_hospital_type"] = np.take_along_axis(
            types, second_index, axis=1
        ).ravel()
        self.state["nearest_alternative_time_min"] = second_time
        self.state["alternative_gap_min"] = np.maximum(0.0, second_time - first_time)
        baseline_second = self.base_state["nearest_alternative_time_min"].to_numpy(dtype=float)
        ratio = np.divide(
            second_time,
            baseline_second,
            out=np.zeros_like(second_time),
            where=baseline_second > 0,
        ).clip(0.0, 1.0)
        self.state["alternative_score_01"] = (
            self.base_state["alternative_score_01"].to_numpy(dtype=float) * ratio
        )

        self._refresh_vulnerability_labels()

        self.applied_policies.append(
            {
                "policy": "hospital_upgrade",
                "target": institution_id
            }
        )


    # =========================================================
    # 정책 3: 도로망 개선
    # =========================================================
    def improve_road(self, *args, **kwargs):

        raise NotImplementedError(
            "현재 Node/Link 데이터가 없으므로 "
            "도로망 개선 정책은 아직 계산할 수 없습니다."
        )


    # =========================================================
    # 평가 가중치
    # =========================================================
    def _region_weights(self):
        """
        응급발생위험 + 취약인구를 이용해
        지역별 평가 가중치를 생성.

        두 값의 상대 비중은 동일하게 사용.
        """

        weight = (
            self.state["occurrence_score_01"]
            +
            self.state["vulnerable_score_01"]
        )

        # 모두 0일 경우 방어
        if weight.sum() == 0:
            weight = np.ones(
                len(self.state)
            )

        return (
            weight / weight.sum()
        )


    # =========================================================
    # 현재 성능 지표
    # =========================================================
    def calculate_metrics(self):

        weight = self._region_weights()

        ems = self.state[
            "ems_duration_min"
        ]

        hospital = self.state[
            "hospital_duration_min"
        ]

        total = ems + hospital

        return {
            "mean_ems_time":
                float(ems.mean()),

            "mean_hospital_time":
                float(hospital.mean()),

            "mean_total_time":
                float(total.mean()),

            "weighted_ems_time":
                float(
                    np.sum(
                        weight * ems
                    )
                ),

            "weighted_hospital_time":
                float(
                    np.sum(
                        weight * hospital
                    )
                ),

            "weighted_total_time":
                float(
                    np.sum(
                        weight * total
                    )
                ),

            "max_ems_time":
                float(ems.max()),

            "max_hospital_time":
                float(hospital.max()),
        }


    # =========================================================
    # Before / After
    # =========================================================
    def compare_with_baseline(self):

        current = self.state.copy()

        # baseline
        self.state = self.base_state.copy(
            deep=True
        )

        before = self.calculate_metrics()

        # 현재 상태 복원
        self.state = current

        after = self.calculate_metrics()

        result = {}

        for key in before:
            result[key] = {
                "before": before[key],
                "after": after[key],
                "improvement":
                    before[key] - after[key]
            }

        return result


    # =========================================================
    # Reward
    # =========================================================
    def calculate_reward(
        self,
        policy_cost=0.0
    ):

        comparison = (
            self.compare_with_baseline()
        )

        improvement = (
            comparison[
                "weighted_total_time"
            ]["improvement"]
        )

        reward = (
            improvement
            -
            policy_cost
        )

        return float(reward)
