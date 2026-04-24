import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def load_and_preprocess_data(file_path_or_buffer, target_col='Close'):
    """
    Loads historical BTC CSV data, handles Kaggle-specific parsing, 
    resamples to daily if needed, and applies ffill for missing dates.
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
        raise ValueError("Could not auto-detect a Date/Timestamp column. Ensure your CSV has a generic time column.")

    if pd.api.types.is_numeric_dtype(df[time_col]):
        if df[time_col].max() > 10**10:
            df['Date'] = pd.to_datetime(df[time_col], unit='ms')
        else:
            df['Date'] = pd.to_datetime(df[time_col], unit='s')
    else:
        df['Date'] = pd.to_datetime(df[time_col], errors='coerce')
        
    df = df.dropna(subset=['Date'])
    
    if target_col not in df.columns:
        matches = [c for c in df.columns if target_col.lower() in c.lower()]
        if matches:
            target_col = matches[0]
        else:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")
            
    df = df.set_index('Date')
    df = df.sort_index()

    time_diff = df.index.to_series().diff().median()
    df_numeric = df.select_dtypes(include=['number'])
    
    if time_diff < pd.Timedelta(days=1):
        df = df_numeric.resample('D').last()
    else:
        df = df_numeric

    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(full_idx)
    df = df.ffill()
    df.index.name = 'Date'
    
    return df, target_col

def add_technical_indicators(df, target_col):
    """
    Adds 50-day and 200-day Simple Moving Averages.
    """
    if len(df) >= 50:
        df['SMA_50'] = df[target_col].rolling(window=50).mean()
    else:
        df['SMA_50'] = None

    if len(df) >= 200:
        df['SMA_200'] = df[target_col].rolling(window=200).mean()
    else:
        df['SMA_200'] = None
        
    return df
