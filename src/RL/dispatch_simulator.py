"""Event-based ambulance dispatch simulator for existing-center increments."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from simulator import EmergencyPolicySimulator


MINUTES_PER_DAY = 24 * 60
MINUTES_PER_YEAR = 365 * MINUTES_PER_DAY


@dataclass(frozen=True)
class DispatchEvent:
    occurred_at_min: float
    region_id: str


class AmbulanceDispatchSimulator:
    """Simulate dispatch queues using observed annual regional demand."""

    def __init__(self, data_dir="data/processed/RL"):
        self.policy_simulator = EmergencyPolicySimulator(data_dir=data_dir)
        self.regions = self.policy_simulator.base_state.set_index("region_id")
        self.stations = self.policy_simulator.ems_stations.set_index("station_id")
        self.emergency_hospitals = (
            self.policy_simulator.emergency_hospitals.set_index("hospital_id")
        )
        self._build_time_matrices()
        self.station_ids = self.stations.index.to_numpy()
        self.region_ids = self.regions.index.to_numpy()
        self.region_index = {
            region_id: index for index, region_id in enumerate(self.region_ids)
        }
        self.dispatch_matrix = self.dispatch_time.loc[
            self.station_ids,
            self.region_ids,
        ].to_numpy(dtype=float)
        self.return_matrix = self.return_time.loc[
            self.station_ids,
            self.region_ids,
        ].to_numpy(dtype=float)
        self.hospital_time = self.regions.loc[
            self.region_ids,
            "hospital_duration_min",
        ].to_numpy(dtype=float)

    def _build_time_matrices(self):
        dispatch_rows = []
        return_rows = []

        for station_id, station in self.stations.iterrows():
            for region_id, region in self.regions.iterrows():
                if station_id == region["station_id"]:
                    dispatch_time = float(region["ems_duration_min"])
                else:
                    dispatch_time = self.policy_simulator.estimate_ems_time(
                        station["latitude"],
                        station["longitude"],
                        region["latitude"],
                        region["longitude"],
                    )

                hospital = self.emergency_hospitals.loc[region["hospital_id"]]
                return_time = self.policy_simulator.estimate_ems_time(
                    hospital["latitude"],
                    hospital["longitude"],
                    station["latitude"],
                    station["longitude"],
                )
                dispatch_rows.append((station_id, region_id, dispatch_time))
                return_rows.append((station_id, region_id, return_time))

        self.dispatch_time = pd.DataFrame(
            dispatch_rows,
            columns=["station_id", "region_id", "dispatch_time_min"],
        ).pivot(
            index="station_id",
            columns="region_id",
            values="dispatch_time_min",
        )
        self.return_time = pd.DataFrame(
            return_rows,
            columns=["station_id", "region_id", "return_time_min"],
        ).pivot(
            index="station_id",
            columns="region_id",
            values="return_time_min",
        )

    def generate_events(self, days=365, seed=42):
        """Generate region-specific Poisson arrivals from observed annual call counts."""
        if days <= 0:
            raise ValueError("days는 1 이상이어야 합니다.")

        rng = np.random.default_rng(seed)
        horizon_min = days * MINUTES_PER_DAY
        expected_counts = (
            self.regions["emergency_calls"].astype(float)
            * horizon_min
            / MINUTES_PER_YEAR
        )
        events = []
        for region_id, expected_count in expected_counts.items():
            event_count = int(rng.poisson(expected_count))
            occurred_at = rng.uniform(0.0, horizon_min, size=event_count)
            events.extend(
                DispatchEvent(float(time), region_id)
                for time in occurred_at
            )
        return sorted(events, key=lambda event: event.occurred_at_min)

    def simulate(self, days=365, seed=42, increments=None):
        """Simulate a common-random-number scenario with optional station increments."""
        increments = increments or {}
        unknown_stations = set(increments) - set(self.stations.index)
        if unknown_stations:
            raise ValueError(f"존재하지 않는 station_id: {sorted(unknown_stations)}")
        if any(count < 0 for count in increments.values()):
            raise ValueError("증차 대수는 음수가 될 수 없습니다.")

        events = self.generate_events(days=days, seed=seed)
        vehicle_station_indices = []
        for station_index, (station_id, station) in enumerate(self.stations.iterrows()):
            count = int(station["ambulance_count"]) + int(increments.get(station_id, 0))
            vehicle_station_indices.extend([station_index] * count)
        vehicle_station_indices = np.asarray(vehicle_station_indices, dtype=int)
        vehicle_available_at = np.zeros(len(vehicle_station_indices), dtype=float)

        occurred_at_values = []
        region_id_values = []
        station_id_values = []
        wait_values = []
        dispatch_values = []
        response_values = []
        hospital_values = []
        return_values = []
        busy_until_values = []
        for event in events:
            region_index = self.region_index[event.region_id]
            dispatch_options = self.dispatch_matrix[
                vehicle_station_indices,
                region_index,
            ]
            arrival_options = (
                np.maximum(event.occurred_at_min, vehicle_available_at)
                + dispatch_options
            )
            vehicle_index = int(arrival_options.argmin())
            station_index = vehicle_station_indices[vehicle_index]
            dispatch_time = dispatch_options[vehicle_index]
            wait_time = max(
                0.0,
                vehicle_available_at[vehicle_index] - event.occurred_at_min,
            )
            arrival_at = event.occurred_at_min + wait_time + dispatch_time
            hospital_time = self.hospital_time[region_index]
            return_time = self.return_matrix[station_index, region_index]
            vehicle_available_at[vehicle_index] = (
                arrival_at + hospital_time + return_time
            )

            occurred_at_values.append(event.occurred_at_min)
            region_id_values.append(event.region_id)
            station_id_values.append(self.station_ids[station_index])
            wait_values.append(wait_time)
            dispatch_values.append(dispatch_time)
            response_values.append(wait_time + dispatch_time)
            hospital_values.append(hospital_time)
            return_values.append(return_time)
            busy_until_values.append(vehicle_available_at[vehicle_index])

        records = pd.DataFrame(
            {
                "occurred_at_min": occurred_at_values,
                "region_id": region_id_values,
                "station_id": station_id_values,
                "wait_time_min": wait_values,
                "dispatch_time_min": dispatch_values,
                "response_time_min": response_values,
                "hospital_time_min": hospital_values,
                "return_time_min": return_values,
                "busy_until_min": busy_until_values,
            }
        )
        if records.empty:
            raise RuntimeError("생성된 신고가 없습니다.")
        metrics = self.calculate_metrics(
            records,
            vehicle_count=len(vehicle_station_indices),
        )
        return records, metrics

    @staticmethod
    def calculate_metrics(records, vehicle_count):
        return {
            "event_count": int(len(records)),
            "vehicle_count": int(vehicle_count),
            "mean_wait_time_min": float(records["wait_time_min"].mean()),
            "p95_wait_time_min": float(records["wait_time_min"].quantile(0.95)),
            "max_wait_time_min": float(records["wait_time_min"].max()),
            "mean_response_time_min": float(records["response_time_min"].mean()),
            "p95_response_time_min": float(
                records["response_time_min"].quantile(0.95)
            ),
            "waited_event_count": int((records["wait_time_min"] > 0).sum()),
            "waited_event_rate": float((records["wait_time_min"] > 0).mean()),
        }
