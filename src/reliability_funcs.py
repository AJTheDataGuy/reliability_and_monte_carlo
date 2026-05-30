"""Module to calculate reliability engineering metrics
such as time between failures (TBF) and time to repair (TTR)

Can also be used in a more general sense to calculate
event durations (such as with TTR)
and time between events (such as with TBF)

The arguably most important function in this module is used
to flatten a dataframe of overlapping timestamps
into a single timeline / series of events

To Do:
1. Add funcs for mean time to repair and mean time to failure
2. Here or separate module. Combine metrics with probability curves
3. Add some tests
"""

from copy import copy
import pandas as pd


def flatten_overlapping_df_timestamps(
    overlap_df: pd.DataFrame,
    unit_col_name: str = "unit",
    start_col_name: str = "start_date",
    end_col_name: str = "end_date",
    date_format: str = "%d/%m/%Y",
) -> pd.DataFrame:
    """
    Flattens a dataframe of overlapping timestamps
    into a clean timeline of events

    For example, the following data set could represent
    electrical plant unit outages in a system
    (the example is given in D/M/Y format):

    |  UNIT  | START_DATE |  END DATE  |
    | UNIT_1 |  1/01/2024 | 10/01/2024 |
    | UNIT_1 |  5/01/2024 | 15/01/2024 |
    | UNIT_1 |  12/01/2024| 20/01/2024 |

    This should flatten to:

    |  UNIT  | START_DATE |  END DATE  |
    | UNIT_1 |  1/01/2024 | 20/01/2024 |

    :param overlap_df: Pandas dataframe containing overlapping
                        start and end timestamps
    :param unit_col_name: Column name containing string information for the unit
    :param start_col_name: Column name for the start date timestamps
    :param end_col_name: Column name for end date timestamps
    :param date_format: Format to specify how the start and end dates
                        should be formatted. Assumes both the start and
                        end date are in the same format. Formats per:
                        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    :returns: Pandas Dataframe containing a clean timeline of events.
                Start and end date columns are retained.
                Other columns are currently discarded.
    """
    # Make a copy just to be safe with tests
    overlap_df = copy(overlap_df)
    # Step 1: ensure the start and time columns are timestamps
    overlap_df[start_col_name] = pd.to_datetime(
        overlap_df[start_col_name], format=date_format
    )
    overlap_df[end_col_name] = pd.to_datetime(
        overlap_df[end_col_name], format=date_format
    )

    # Step 2: Sort by start times being mindful to partition within unit
    overlap_df_sorted = overlap_df.sort_values([unit_col_name, start_col_name])

    # Step 3: Define some strings which are used for intermediate pandas column names
    lagged_end = "lagged_end"
    new_group_bool = "new_group_bool"
    group = "group"

    # Step 4: Calculate the ending from the previous row
    # Here the group by acts like a SQL partition statement
    # Essemtially like LAG(end_time) OVER (PARTITION BY UNIT) in SQL
    # To do: clean this up a bit
    overlap_df_sorted[lagged_end] = (
        overlap_df_sorted.groupby(unit_col_name)[end_col_name]
        .cummax()
        .groupby(overlap_df_sorted[unit_col_name])
        .shift()
    )

    # Step 5: Find new interval groupings when the start date of the current interval
    # is greater (i.e. does not fall before) the end of the last interval
    overlap_df_sorted[new_group_bool] = (
        overlap_df_sorted[start_col_name] > overlap_df_sorted[lagged_end]
    )

    # Step 6: Create numbered groupings for each group of intervals
    # By running a cumulative sum over the boolean new group column
    # Again the groupby serves a purpose similar to a PARTITION statement in SQL - partition by unit
    overlap_df_sorted[group] = overlap_df_sorted.groupby(unit_col_name)[
        new_group_bool
    ].cumsum()

    # Step 7: Find the min start time and max end time within each group
    # i.e. the true datetime range for the group
    grouped_df = overlap_df_sorted.groupby([unit_col_name, group]).agg(
        {start_col_name: "min", end_col_name: "max"}
    )

    # Step 8: drop the group column as it is no longer needed and then return.
    sorted_df = grouped_df.reset_index().drop(columns=group)
    return sorted_df


def calculate_time_for_each_event(
    sorted_df: pd.DataFrame,
    start_col_name: str = "start_date",
    end_col_name: str = "end_date",
) -> pd.DataFrame:
    """
    Calculates time for each event - start to end.
    For example, can be used to calculate time to repair (TTR)
    As an intermediate to calculating mean time to repair (MTTR)
    Partitions within each unit

    NOTE: timeline of outages must be sorted with the
    flatten_overlapping_df_timestamps function
    Before this function can be used.

    :param sorted_df: DataFrame with a clean timeline of start and end events
                    Recommended to sort using the flatten_overlapping_df_timestamps
                    function
    :param start_col_name: Column name for the start date timestamps
    :param end_col_name: Column name for end date timestamps
    :returns: The original sorted_df DataFrame with an additional column added
            where the additional column calculates the time between the start and
            end timestamps for each row.
    """
    sorted_df["event_duration"] = sorted_df[end_col_name] - sorted_df[start_col_name]
    return sorted_df


def calculate_time_between_events(
    sorted_df: pd.DataFrame,
    unit_col_name: str = "unit",
    start_col_name: str = "start_date",
    end_col_name: str = "end_date",
) -> pd.DataFrame:
    """
    Calculates time between events
    For example can be used to calculate time between failures (TBF)
    As an intermediate to calculating mean time between failures (MTBF)
    (or time between events in general)
    Partitions within each unit

    NOTE: timeline of outages must be sorted with the
    flatten_overlapping_df_timestamps function
    Before this function can be used.

    :param sorted_df: DataFrame with a clean timeline of start and end events
                    Recommended to sort using the flatten_overlapping_df_timestamps
                    function
    :param unit_col_name: Column name containing string information for the unit
    :param start_col_name: Column name for the start date timestamps
    :param end_col_name: Column name for end date timestamps
    :returns: The original sorted_df DataFrame with an additional column added
            where the additional column calculates the time between the start and
            end timestamps for each row.
    """
    lagged_end = "lagged_end"
    sorted_df[lagged_end] = sorted_df.groupby(unit_col_name)[end_col_name].shift(
        periods=1
    )
    sorted_df["time_between_events"] = sorted_df[start_col_name] - sorted_df[lagged_end]
    return sorted_df
