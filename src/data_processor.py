import pandas as pd
import streamlit as st
import numpy as np

@st.cache_data(show_spinner=False)
def load_and_preprocess_data(file_path_or_buffer, target_col='Close'):
    """
    Loads historical BTC CSV data, handles Kaggle-specific parsing,
    resamples to daily OHLCV if needed, and forward-fills missing dates.
    Returns a clean daily DataFrame and the resolved target column name.
    """
    try:
        df = pd.read_csv(file_path_or_buffer)
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")

    date_cols = ['Timestamp', 'Date', 'Open time', 'datetime', 'time']
    time_col = None
    for col in df.columns:
        if col in date_cols or col.lower() in [c.lower() for c in date_cols]:
            time_col = col
            break

    if time_col is None:
        raise ValueError("Could not auto-detect a Date/Timestamp column.")

    if pd.api.types.is_numeric_dtype(df[time_col]):
        unit = 'ms' if df[time_col].max() > 10**10 else 's'
        df['Date'] = pd.to_datetime(df[time_col], unit=unit)
    else:
        df['Date'] = pd.to_datetime(df[time_col], errors='coerce')

    df = df.dropna(subset=['Date'])

    if target_col not in df.columns:
        matches = [c for c in df.columns if target_col.lower() in c.lower()]
        if matches:
            target_col = matches[0]
        else:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")

    df = df.set_index('Date').sort_index()

    # Resample intraday data to daily OHLCV aggregation (not just last value)
    time_diff = df.index.to_series().diff().median()
    df_numeric = df.select_dtypes(include=['number'])

    if time_diff < pd.Timedelta(days=1):
        agg = {}
        col_lower = {c.lower(): c for c in df_numeric.columns}
        if 'open' in col_lower:   agg[col_lower['open']]   = 'first'
        if 'high' in col_lower:   agg[col_lower['high']]   = 'max'
        if 'low' in col_lower:    agg[col_lower['low']]    = 'min'
        if 'close' in col_lower:  agg[col_lower['close']]  = 'last'
        if 'volume' in col_lower: agg[col_lower['volume']] = 'sum'
        for c in df_numeric.columns:
            if c not in agg:
                agg[c] = 'last'
        df = df_numeric.resample('D').agg(agg)
    else:
        df = df_numeric

    # Fill continuous daily index — forward-fill gaps (weekends / missing days)
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(full_idx).ffill()
    df.index.name = 'Date'

    # Drop any remaining NaN rows at the very start before forward-fill had data
    df = df.dropna(how='all')

    return df, target_col


def add_technical_indicators(df, target_col):
    """
    Adds SMA-50 and SMA-200. Uses min_periods to avoid NaN flooding short series.
    """
    df['SMA_50']  = df[target_col].rolling(window=50,  min_periods=1).mean()
    df['SMA_200'] = df[target_col].rolling(window=200, min_periods=1).mean()
    return df
