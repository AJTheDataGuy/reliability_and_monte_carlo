"""Tests for the reliability_funcs_module"""

import pytest
import numpy as np
import pandas as pd

from reliability_funcs import flatten_overlapping_df_timestamps
from reliability_funcs import calculate_time_for_each_event
from reliability_funcs import calculate_time_between_events


@pytest.fixture
def overlap_timeline_data_fixt() -> pd.DataFrame:
    """
    Create a dataframe of overlapping timestamp data
    to use for testing with the different functions
    Uses fictious data

    Creates 5 separate time intervals for Unit 1
    And 3 separate time intervals for Unit 2

    Dates are given in Day/Month/Year format.
    Start and end date lements are meant to go together pair-wise
    So element 0 of start_dates_unit_1 goes with
    element 0 of end_dates_unit_1

    :returns: Pandas DataFrame fixture of overlapping dates
                test data in the format
                |  UNIT  | START_DATE |  END DATE  |
                | UNIT_1 |  1/01/2024 | 20/01/2024 |
    """
    unit_1_array = np.full(10, "UNIT_1")
    unit_2_array = np.full(10, "UNIT_2")

    start_dates_unit_1 = [
        "1/01/2024",
        "5/01/2024",
        "12/01/2024",
        "1/02/2024",
        "3/02/2024",
        "15/02/2024",
        "1/03/2024",
        "10/03/2024",
        "12/03/2024",
        "25/03/2024",
    ]

    end_dates_unit_1 = [
        "10/01/2024",
        "15/01/2024",
        "20/01/2024",
        "10/02/2024",
        "20/02/2024",
        "28/02/2024",
        "5/03/2024",
        "15/03/2024",
        "20/03/2024",
        "30/03/2024",
    ]

    start_dates_unit_2 = [
        "1/01/2024",
        "5/01/2024",
        "7/01/2024",
        "3/01/2024",
        "2/02/2024",
        "1/02/2024",
        "5/05/2024",
        "7/05/2024",
        "11/05/2024",
        "6/05/2024",
    ]

    end_dates_unit_2 = [
        "12/01/2024",
        "10/01/2024",
        "19/01/2024",
        "31/01/2024",
        "9/02/2024",
        "15/02/2024",
        "10/05/2024",
        "12/05/2024",
        "20/05/2024",
        "9/05/2024",
    ]
    # Create column for units
    units_array = np.concatenate([unit_1_array,unit_2_array])
    # Create column for start dates
    start_dates_array = np.concatenate([start_dates_unit_1,start_dates_unit_2])
    end_dates_array = np.concatenate([end_dates_unit_1, end_dates_unit_2])
    test_data_fixt = pd.DataFrame(
        {
            "unit": units_array,
            "start_date": start_dates_array,
            "end_date": end_dates_array,
        }
    )
    yield test_data_fixt


@pytest.fixture
def expected_clean_timeline_data_fixt() -> pd.DataFrame:
    """
    Create a fixture for the expected output of the
    timeline cleaning function given the inputs from the overlapping
    timestamp fixture
    """
    expected_units = [
        "UNIT_1",
        "UNIT_1",
        "UNIT_1",
        "UNIT_1",
        "UNIT_1",
        "UNIT_2",
        "UNIT_2",
        "UNIT_2",
    ]
    expected_starts = [
        "01/01/2024",
        "01/02/2024",
        "01/03/2024",
        "10/03/2024",
        "25/03/2024",
        "01/01/2024",
        "01/02/2024",
        "05/05/2024",
    ]
    expected_ends = [
        "20/01/2024",
        "28/02/2024",
        "05/03/2024",
        "20/03/2024",
        "30/03/2024",
        "31/01/2024",
        "15/02/2024",
        "20/05/2024",
    ]
    expected_df = pd.DataFrame(
        {
            "unit": expected_units,
            "start_date": expected_starts,
            "end_date": expected_ends,
        }
    )
    expected_df["start_date"] = pd.to_datetime(
        expected_df["start_date"], format="%d/%m/%Y"
    )
    expected_df["end_date"] = pd.to_datetime(expected_df["end_date"], format="%d/%m/%Y")
    yield expected_df


@pytest.fixture
def expected_event_time_duration_fixt() -> np.array:
    """
    Create a fixture representing the expected time duration of
    each event in days
    """
    expected_event_time_days = np.array([19, 27, 4, 10, 5, 30, 14, 15])
    yield expected_event_time_days


@pytest.fixture
def expected_time_between_events_fixt():
    """
    Create a fixture representing the expected time between events in days
    Nulls represent switching over to a different unit
    """
    expected_time_between_events = np.array([np.nan, 12, 2, 5, 5, np.nan, 1, 80])
    yield expected_time_between_events


def test_clean_timeline_intervals(overlap_timeline_data_fixt):
    """
    Test that the clean timeline function
    Found the expected 8 intervals between units 1 and 2
    5 intervals for unit 1 and 3 intervals for unit 2
    """
    cleaned_timeline_df = flatten_overlapping_df_timestamps(overlap_timeline_data_fixt)
    assert len(cleaned_timeline_df) == 8


def test_clean_timeline_full(
    overlap_timeline_data_fixt, expected_clean_timeline_data_fixt
):
    """
    Test that the output from the timeline cleaning function
    and the expected outputs match exactly
    """
    actual_output_df = flatten_overlapping_df_timestamps(overlap_timeline_data_fixt)
    pd.testing.assert_frame_equal(actual_output_df, expected_clean_timeline_data_fixt)


def test_event_durations(
    expected_clean_timeline_data_fixt, expected_event_time_duration_fixt
):
    """
    Test that the function to calculate event times is working properly
    And the actual outputs match the expected outputs exactly.
    """
    df_with_event_times = calculate_time_for_each_event(expected_clean_timeline_data_fixt)
    calculated_event_times_days = df_with_event_times["event_duration"].dt.days
    np.testing.assert_array_equal(
        calculated_event_times_days, expected_event_time_duration_fixt
    )


def test_time_between_events(
    expected_clean_timeline_data_fixt, expected_time_between_events_fixt
):
    """
    Test that the function to calculate times between events is working properly
    And the actual outputs match the expected outputs exactly.
    """
    df_with_time_between_events = calculate_time_between_events(expected_clean_timeline_data_fixt)
    calculated_times_between_events = df_with_time_between_events[
        "time_between_events"
    ].dt.days
    np.testing.assert_array_equal(
        calculated_times_between_events, expected_time_between_events_fixt
    )
