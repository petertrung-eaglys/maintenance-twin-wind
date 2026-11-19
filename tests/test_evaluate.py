"""Tests for matching alerts to fault events."""

import numpy as np
import pandas as pd
import pytest

from src.health.evaluate import (
    GROUP,
    chance_detection,
    drift_cases,
    drift_table,
    evaluate_scopes,
    match_alerts,
    merge_episodes,
    nearest_matches,
    summarise,
)

LOOKBACK = pd.Timedelta(days=30)


def ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def events(*rows: tuple[str, str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"event": f"E{i}", "turbine_id": t, "component_key": c, "start": ts(s)}
            for i, (t, c, s) in enumerate(rows)
        ]
    )


def alerts(*rows: tuple[str, str, str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "alert": f"A{i}",
                "turbine_id": t,
                "component_key": c,
                "variant": v,
                "start": ts(s),
            }
            for i, (t, c, v, s) in enumerate(rows)
        ]
    )


def test_merge_episodes_joins_close_events_of_one_component() -> None:
    raw = events(
        ("T01", "gearbox", "2022-03-01 00:00"),
        ("T01", "gearbox", "2022-03-05 00:00"),  # 4 days later: same episode
        ("T01", "gearbox", "2022-04-01 00:00"),  # new episode
        ("T01", "pitch", "2022-03-02 00:00"),  # other component
    )

    episodes = merge_episodes(raw, pd.Timedelta(days=7))

    assert episodes[["component_key", "event", "n_events"]].values.tolist() == [
        ["gearbox", "E0", 2],
        ["gearbox", "E2", 1],
        ["pitch", "E3", 1],
    ]


def test_merge_episodes_chain_a_long_repeated_series() -> None:
    starts = [f"2022-03-{d:02d} 00:00" for d in (1, 6, 11, 16)]
    raw = events(*[("T01", "gearbox", s) for s in starts])

    assert len(merge_episodes(raw, pd.Timedelta(days=7))) == 1


def test_match_alerts_within_the_lookback_before_the_episode() -> None:
    episodes = merge_episodes(
        events(("T01", "gearbox", "2022-03-31 00:00")), pd.Timedelta(days=7)
    )
    raised = alerts(
        ("T01", "gearbox", "hybrid", "2022-03-21 00:00"),  # 10 days before: match
        ("T01", "gearbox", "hybrid", "2022-02-01 00:00"),  # too early
        ("T01", "gearbox", "hybrid", "2022-04-02 00:00"),  # after the event
        ("T02", "gearbox", "hybrid", "2022-03-21 00:00"),  # other turbine
        ("T01", "pitch", "hybrid", "2022-03-21 00:00"),  # other component
    )

    matches = match_alerts(raised, episodes, LOOKBACK)

    assert matches[["alert", "event", "lead_hours"]].values.tolist() == [
        ["A0", "E0", 240.0]
    ]


def test_match_alerts_pair_an_alert_with_every_episode_in_reach() -> None:
    episodes = merge_episodes(
        events(
            ("T01", "gearbox", "2022-03-20 00:00"),
            ("T01", "gearbox", "2022-03-31 00:00"),
        ),
        pd.Timedelta(days=7),
    )
    raised = alerts(("T01", "gearbox", "hybrid", "2022-03-15 00:00"))

    assert match_alerts(raised, episodes, LOOKBACK)["event"].tolist() == [
        "E0",
        "E1",
    ]


def test_nearest_matches_link_an_alert_to_its_nearest_episode() -> None:
    episodes = merge_episodes(
        events(
            ("T01", "gearbox", "2022-03-20 00:00"),
            ("T01", "gearbox", "2022-03-31 00:00"),
        ),
        pd.Timedelta(days=7),
    )
    raised = alerts(("T01", "gearbox", "hybrid", "2022-03-15 00:00"))

    matches = nearest_matches(match_alerts(raised, episodes, LOOKBACK))

    assert matches["event"].tolist() == ["E0"]


def test_summarise_lists_models_without_alerts() -> None:
    episodes = merge_episodes(
        events(("T01", "gearbox", "2022-03-31 00:00")), pd.Timedelta(days=7)
    ).assign(farm_code="KLM")
    groups = pd.DataFrame(
        [["KLM", "gearbox", "ThermalModel", "gear_oil_temp", "pinn"]], columns=GROUP
    )
    none = alerts().reindex(columns=[*GROUP, "alert", "turbine_id", "start"])
    turbine_days = pd.Series({("KLM", "T01"): 365.0})

    row = summarise(none, episodes, turbine_days, LOOKBACK, groups).iloc[0]

    assert (row["episodes"], row["detected"], row["alerts"]) == (1, 0, 0)


def test_chance_detection_grows_with_the_alert_rate() -> None:
    low, high = chance_detection(
        np.array([1.0, 10.0]), test_days=365.0, lookback_days=30
    )

    assert (low, high) == pytest.approx((1 - np.exp(-30 / 365), 1 - np.exp(-300 / 365)))


def test_summarise_counts_detections_lead_times_and_false_alarms() -> None:
    episodes = merge_episodes(
        events(
            ("T01", "gearbox", "2022-03-31 00:00"),
            ("T02", "gearbox", "2022-06-30 00:00"),
        ),
        pd.Timedelta(days=7),
    ).assign(farm_code="KLM")
    raised = alerts(
        ("T01", "gearbox", "hybrid", "2022-03-21 00:00"),  # detects T01, 240 h
        ("T01", "gearbox", "hybrid", "2022-09-01 00:00"),  # false alarm
    ).assign(farm_code="KLM", model_class="ThermalModel", target="gear_oil_temp")
    turbine_days = pd.Series({("KLM", "T01"): 365.0, ("KLM", "T02"): 365.0})
    groups = raised[list(GROUP)].drop_duplicates()

    summary = summarise(raised, episodes, turbine_days, LOOKBACK, groups)

    row = summary.iloc[0]
    assert (
        row["episodes"],
        row["detected"],
        row["lead_hours_median"],
        row["alerts"],
        row["false_alarms"],
        row["false_alarms_per_turbine_year"],
    ) == (2, 1, 240.0, 2, 1, pytest.approx(1 / (730 / 365.25)))


def test_drift_table_compares_each_year_with_the_calibration_year() -> None:
    records = {
        "ThermalModel_KLM_T04_gear_oil_temp_baseline": {
            "farm_code": "KLM",
            "turbine_id": "T04",
            "model_class": "ThermalModel",
            "target": "gear_oil_temp",
            "variant": "baseline",
            "yearly_mean_score": {"2021": 0.1, "2022": 0.3, "2023": 4.1},
        }
    }

    drift = drift_table(records, calibration_year=2021)

    assert drift[
        ["turbine_id", "target", "variant", "2022", "2023", "max_drift"]
    ].values.tolist() == [["T04", "gear_oil_temp", "baseline", 0.2, 4.0, 4.0]]


def farm_events(*rows: tuple[str, str, str, str]) -> pd.DataFrame:
    """Events (turbine, component, message, start) of farm KLM."""
    return pd.DataFrame(
        [
            {
                "event": f"E{i}",
                "farm_code": "KLM",
                "turbine_id": tid,
                "component_key": component,
                "message": message,
                "start": ts(start),
            }
            for i, (tid, component, message, start) in enumerate(rows)
        ]
    )


def test_evaluate_scopes_count_only_signal_events_in_the_signal_scope() -> None:
    raw = farm_events(
        ("T01", "gearbox", "Missing gear oil (low rpm)", "2022-03-01 00:00"),
        ("T01", "gearbox", "High temp. gear bearing 1", "2022-06-01 00:00"),
    )
    groups = pd.DataFrame(
        [["KLM", "gearbox", "ThermalModel", "gear_oil_temp", "pinn"]], columns=GROUP
    )
    none = alerts().reindex(columns=[*GROUP, "alert", "turbine_id", "start"])
    days = pd.Series({("KLM", "T01"): 365.0})
    signal = {"gear_oil_temp": ["High temp. gear bearing 1"]}

    summary = evaluate_scopes(
        none, raw, days, LOOKBACK, pd.Timedelta(days=7), groups, signal
    )

    assert summary.set_index("scope")["episodes"].to_dict() == {
        "component": 2,
        "signal": 1,
    }


def test_drift_cases_count_events_during_and_after_the_drift() -> None:
    drifts = pd.DataFrame(
        [
            {
                "model": "ThermalModel_KLM_T01_gear_oil_temp_pinn",
                "farm_code": "KLM",
                "turbine_id": "T01",
                "component": "gearbox",
                "target": "gear_oil_temp",
                "start": ts("2022-01-01 00:00"),
                "end": ts("2022-04-01 00:00"),
            }
        ]
    )
    raw = farm_events(
        ("T01", "gearbox", "Missing gear oil (low rpm)", "2022-02-01 00:00"),
        ("T01", "gearbox", "High temp. gear bearing 1", "2022-04-20 00:00"),
        ("T01", "gearbox", "High temp. gear bearing 1", "2022-08-01 00:00"),  # late
        ("T02", "gearbox", "High temp. gear bearing 1", "2022-02-01 00:00"),
    )
    signal = {"gear_oil_temp": ["High temp. gear bearing 1"]}

    cases = drift_cases(drifts, raw, LOOKBACK, signal)

    assert cases[["component_events", "signal_events"]].values.tolist() == [[2, 1]]
