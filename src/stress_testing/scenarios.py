# =====================================================
# Scenarios
#
# Defines the anomaly injection functions and
# the rates (%) for each anomaly type.
#
# injector.py imports everything from here.
# =====================================================

import pandas as pd
import numpy as np

# columns that can be numerically corrupted
PRICE_COLS  = ["Open", "High", "Low", "Close"]
ALL_COLS    = ["Open", "High", "Low", "Close", "Volume"]


# -----------------------------------------------------
# Function: inject_missing_values
# Purpose : Randomly corrupt `rate` % of rows with a
#           missing value in a random column.
# Options : NaN or -999 (common placeholder for missing)
# Label   : "missing_value"
# -----------------------------------------------------
def inject_missing_values(df, rate):
    df = df.copy()

    # calculate how many rows to corrupt
    n = max(1, int(len(df) * rate))

    # pick random row indices
    idx = np.random.choice(df.index, size=n, replace=False)

    for i in idx:
        # randomly pick which column to corrupt
        col = np.random.choice(ALL_COLS)

        # randomly pick how to represent the missing value
        corruption = np.random.choice(["nan", "minus999"])
        df.loc[i, col] = np.nan if corruption == "nan" else -999
        df.loc[i, "data_quality_alert"] = "missing_value"

    return df


# -----------------------------------------------------
# Function: inject_price_spikes
# Purpose : Randomly corrupt `rate` % of rows with an
#           abnormal price in a random price column.
# Options : ×10 (spike up), ×0.01 (crash down),
#           +1000 (add fixed amount), -500 (negative spike)
# Label   : "price_spike"
# -----------------------------------------------------
def inject_price_spikes(df, rate):
    df = df.copy()

    # calculate how many rows to corrupt
    n = max(1, int(len(df) * rate))

    # pick random row indices
    idx = np.random.choice(df.index, size=n, replace=False)

    for i in idx:
        # randomly pick which price column to corrupt
        col = np.random.choice(PRICE_COLS)

        # randomly pick the type of spike
        corruption = np.random.choice(["times10", "times0.01", "plus1000", "minus500"])
        if corruption == "times10":
            df.loc[i, col] = df.loc[i, col] * 10
        elif corruption == "times0.01":
            df.loc[i, col] = df.loc[i, col] * 0.01
        elif corruption == "plus1000":
            df.loc[i, col] = df.loc[i, col] + 1000
        else:
            df.loc[i, col] = df.loc[i, col] - 500

        df.loc[i, "data_quality_alert"] = "price_spike"

    return df


# -----------------------------------------------------
# Function: inject_zero_values
# Purpose : Set clustered blocks (3 rows) in a random
#           column to an abnormal low value.
# Options : 0, -1, 0.0001 (near-zero), 99999 (overflow)
# Label   : "zero_value"
# -----------------------------------------------------
def inject_zero_values(df, rate):
    df = df.copy()

    # calculate how many clusters to inject (each cluster = 3 rows)
    n_clusters = max(1, int(len(df) * rate / 3))

    # pick random start positions for each cluster
    start_positions = np.random.choice(len(df) - 3, size=n_clusters, replace=False)

    for start in start_positions:
        # randomly pick which column to corrupt
        col = np.random.choice(ALL_COLS)

        # randomly pick the abnormal value
        value = np.random.choice([0, -1, 0.0001, 99999])

        # set 3 consecutive rows to the chosen value
        df.loc[start:start + 2, col] = value
        df.loc[start:start + 2, "data_quality_alert"] = "zero_value"

    return df


# -----------------------------------------------------
# Function: inject_duplicates
# Purpose : Randomly duplicate `rate` % of rows.
# Options : exact copy, copy with ±5% price change,
#           copy with date shifted by 1 day
# Label   : "duplicate"
# -----------------------------------------------------
def inject_duplicates(df, rate):
    df = df.copy()

    # calculate how many rows to duplicate
    n = max(1, int(len(df) * rate))

    # pick random rows to duplicate
    duplicated_rows = df.sample(n=n).copy()

    for i in duplicated_rows.index:
        # randomly pick the type of duplicate
        corruption = np.random.choice(["exact", "price_noise", "date_shift"])
        if corruption == "price_noise":
            # slightly change Close by ±5%
            noise = np.random.uniform(0.95, 1.05)
            duplicated_rows.loc[i, "Close"] = duplicated_rows.loc[i, "Close"] * noise
        elif corruption == "date_shift":
            # shift date by 1 day
            duplicated_rows.loc[i, "Date"] = duplicated_rows.loc[i, "Date"] + pd.Timedelta(days=1)

    # label and append duplicates, then sort by date
    duplicated_rows["data_quality_alert"] = "duplicate"
    df = pd.concat([df, duplicated_rows], ignore_index=True)
    df = df.sort_values("Date").reset_index(drop=True)

    return df


# -----------------------------------------------------
# Function: inject_wrong_dates
# Purpose : Randomly shift `rate` % of Date values.
# Options : ±30 days, ±365 days (wrong year),
#           shift to weekend, shift to future
# Label   : "wrong_date"
# -----------------------------------------------------
def inject_wrong_dates(df, rate):
    df = df.copy()

    # calculate how many rows to corrupt
    n = max(1, int(len(df) * rate))

    # pick random row indices
    idx = np.random.choice(df.index, size=n, replace=False)

    for i in idx:
        # randomly pick the type of date corruption
        corruption = np.random.choice(["small_shift", "wrong_year", "weekend", "future"])
        date = pd.to_datetime(df.loc[i, "Date"])

        if corruption == "small_shift":
            # shift ±30 days
            shift = np.random.randint(-30, 30)
            df.loc[i, "Date"] = date + pd.Timedelta(days=int(shift))
        elif corruption == "wrong_year":
            # shift by a full year
            shift = np.random.choice([-365, 365])
            df.loc[i, "Date"] = date + pd.Timedelta(days=int(shift))
        elif corruption == "weekend":
            # shift to the nearest Saturday
            days_to_saturday = (5 - date.weekday()) % 7
            df.loc[i, "Date"] = date + pd.Timedelta(days=int(days_to_saturday))
        else:
            # shift to a random future date (up to 2 years ahead)
            shift = np.random.randint(365, 730)
            df.loc[i, "Date"] = date + pd.Timedelta(days=int(shift))

        df.loc[i, "data_quality_alert"] = "wrong_date"

    return df


# -----------------------------------------------------
# Function: inject_stale_prices
# Purpose : Repeat the same price for 3-5 consecutive
#           rows, simulating a frozen data feed.
# Label   : "stale_price"
# -----------------------------------------------------
def inject_stale_prices(df, rate):
    df = df.copy()

    # calculate how many clusters to inject
    n_clusters = max(1, int(len(df) * rate / 4))

    # pick random start positions
    start_positions = np.random.choice(len(df) - 5, size=n_clusters, replace=False)

    for start in start_positions:
        # randomly pick cluster length between 3 and 5 rows
        length = np.random.randint(3, 6)

        # randomly pick which price column to freeze
        col = np.random.choice(PRICE_COLS)

        # freeze the value at the start row
        frozen_value = df.loc[start, col]
        df.loc[start:start + length - 1, col] = frozen_value
        df.loc[start:start + length - 1, "data_quality_alert"] = "stale_price"

    return df


# -----------------------------------------------------
# Function: inject_ohlc_violations
# Purpose : Create logically impossible OHLC relationships.
#           e.g. High < Low, Open > High, Close < Low
# Label   : "ohlc_violation"
# -----------------------------------------------------
def inject_ohlc_violations(df, rate):
    df = df.copy()

    # calculate how many rows to corrupt
    n = max(1, int(len(df) * rate))

    # pick random row indices
    idx = np.random.choice(df.index, size=n, replace=False)

    for i in idx:
        # randomly pick which OHLC rule to violate
        violation = np.random.choice(["high_lt_low", "open_gt_high", "close_lt_low"])

        if violation == "high_lt_low":
            # swap High and Low so High becomes smaller than Low
            df.loc[i, "High"], df.loc[i, "Low"] = df.loc[i, "Low"], df.loc[i, "High"]
        elif violation == "open_gt_high":
            # set Open much higher than High
            df.loc[i, "Open"] = df.loc[i, "High"] * np.random.uniform(1.1, 1.5)
        else:
            # set Close much lower than Low
            df.loc[i, "Close"] = df.loc[i, "Low"] * np.random.uniform(0.5, 0.9)

        df.loc[i, "data_quality_alert"] = "ohlc_violation"

    return df


# -----------------------------------------------------
# Function: inject_zero_volume
# Purpose : Set Volume to 0 on random trading days
#           while prices remain normal.
# Label   : "zero_volume"
# -----------------------------------------------------
def inject_zero_volume(df, rate):
    df = df.copy()

    # calculate how many rows to corrupt
    n = max(1, int(len(df) * rate))

    # pick random row indices
    idx = np.random.choice(df.index, size=n, replace=False)

    # set volume to 0, prices are untouched
    df.loc[idx, "Volume"] = 0
    df.loc[idx, "data_quality_alert"] = "zero_volume"

    return df


# -----------------------------------------------------
# Function: inject_extreme_gaps
# Purpose : Simulate an extreme overnight price drop
#           (e.g. -80% to -90%) without a stock split.
# Label   : "extreme_gap"
# -----------------------------------------------------
def inject_extreme_gaps(df, rate):
    df = df.copy()

    # calculate how many rows to corrupt
    n = max(1, int(len(df) * rate))

    # pick random row indices (avoid first row)
    idx = np.random.choice(df.index[1:], size=n, replace=False)

    for i in idx:
        # randomly pick drop severity between -80% and -90%
        drop = np.random.uniform(0.10, 0.20)
        df.loc[i, "Close"] = df.loc[i, "Close"] * drop
        df.loc[i, "data_quality_alert"] = "extreme_gap"

    return df


# -----------------------------------------------------
# Function: inject_negative_volume
# Purpose : Set Volume to a negative number,
#           which is physically impossible.
# Label   : "negative_volume"
# -----------------------------------------------------
def inject_negative_volume(df, rate):
    df = df.copy()

    # calculate how many rows to corrupt
    n = max(1, int(len(df) * rate))

    # pick random row indices
    idx = np.random.choice(df.index, size=n, replace=False)

    # flip volume to negative
    df.loc[idx, "Volume"] = -abs(df.loc[idx, "Volume"])
    df.loc[idx, "data_quality_alert"] = "negative_volume"

    return df


# -----------------------------------------------------
# Function: inject_timestamp_conflict
# Purpose : Insert a row with the same date as an
#           existing row but with a different Close price,
#           simulating conflicting data sources.
# Label   : "timestamp_conflict"
# -----------------------------------------------------
def inject_timestamp_conflict(df, rate):
    df = df.copy()

    # calculate how many rows to duplicate with conflict
    n = max(1, int(len(df) * rate))

    # pick random rows to conflict
    conflicting_rows = df.sample(n=n).copy()

    # keep the same date but change Close by ±10–30%
    for i in conflicting_rows.index:
        change = np.random.uniform(1.10, 1.30) * np.random.choice([-1, 1]) + 1
        conflicting_rows.loc[i, "Close"] = conflicting_rows.loc[i, "Close"] * change

    # label and append, sort by date
    conflicting_rows["data_quality_alert"] = "timestamp_conflict"
    df = pd.concat([df, conflicting_rows], ignore_index=True)
    df = df.sort_values("Date").reset_index(drop=True)

    return df


# =====================================================
# Rates Configuration
# Maps each anomaly function to its injection rate.
# Note: class_weight="balanced" should be used during
#       model training to handle class imbalance.
# =====================================================

SCENARIOS = {
    inject_missing_values:    0.02,   # 2%
    inject_price_spikes:      0.01,   # 1%
    inject_zero_values:       0.01,   # 1%
    inject_duplicates:        0.01,   # 1%
    inject_wrong_dates:       0.005,  # 0.5%
    inject_stale_prices:      0.01,   # 1%
    inject_ohlc_violations:   0.005,  # 0.5%
    inject_zero_volume:       0.01,   # 1%
    inject_extreme_gaps:      0.003,  # 0.3%
    inject_negative_volume:   0.005,  # 0.5%
    inject_timestamp_conflict: 0.005, # 0.5%
}
