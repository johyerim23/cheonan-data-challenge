from dispatch_simulator import AmbulanceDispatchSimulator


def run_tests():
    simulator = AmbulanceDispatchSimulator()
    baseline_records, baseline = simulator.simulate(days=30, seed=42)
    repeated_records, repeated = simulator.simulate(days=30, seed=42)
    station_id = simulator.stations.index[0]
    incremented_records, incremented = simulator.simulate(
        days=30,
        seed=42,
        increments={station_id: 1},
    )

    assert baseline == repeated
    assert baseline_records.equals(repeated_records)
    assert len(baseline_records) == len(incremented_records)
    assert baseline["vehicle_count"] + 1 == incremented["vehicle_count"]
    assert (baseline_records["wait_time_min"] >= 0).all()
    assert (baseline_records["response_time_min"] >= baseline_records["dispatch_time_min"]).all()
    assert incremented["mean_wait_time_min"] <= baseline["mean_wait_time_min"]
    assert incremented["mean_response_time_min"] <= baseline["mean_response_time_min"]
    assert set(baseline_records["station_id"]).issubset(set(simulator.stations.index))

    print("Dispatch simulator tests passed")
    print({"baseline": baseline, "incremented": incremented})


if __name__ == "__main__":
    run_tests()
