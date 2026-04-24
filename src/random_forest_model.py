import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from src.metrics import calculate_metrics

def run_random_forest(df, target_col, horizon, confidence_level, **kwargs):
    y_raw = df[target_col].dropna().values
    dates = df[target_col].dropna().index
    
    # Differencing for stationarity
    y_diff = np.diff(y_raw)
    
    lags = kwargs.get('rf_lag_days', 14)
    n_estimators = kwargs.get('rf_n_estimators', 100)
    max_depth = kwargs.get('rf_max_depth', 10)
    
    if len(y_diff) <= lags:
        raise ValueError("Not enough data to create lagged features for Random Forest.")
        
    X, _y = [], []
    for i in range(lags, len(y_diff)):
        window = y_diff[i-lags : i]
        
        # Leakage-free rolling stats derived ONLY from the lag window (past data)
        rolling_mean_3 = np.mean(window[-3:]) if len(window) >= 3 else np.mean(window)
        rolling_std_3 = np.std(window[-3:]) if len(window) >= 3 else 0
        rolling_mean_7 = np.mean(window[-7:]) if len(window) >= 7 else np.mean(window)
        
        # Time features
        current_date = dates[i+1] # i+1 because y_diff[0] is dates[1]
        day_of_week = current_date.dayofweek
        day_of_month = current_date.day
        month = current_date.month
        
        features = list(window) + [rolling_mean_3, rolling_std_3, rolling_mean_7, day_of_week, day_of_month, month]
        X.append(features)
        _y.append(y_diff[i])
        
    X, _y = np.array(X), np.array(_y)
    
    split_idx = int(len(_y) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = _y[:split_idx], _y[split_idx:]
    
    # Backtest
    model_bt = RandomForestRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=42
    )
    model_bt.fit(X_train, y_train)
    preds_diff_bt = model_bt.predict(X_test)
    
    # Reconstruct prices for backtest
    actual_prev_prices_bt = y_raw[lags + split_idx : lags + split_idx + len(y_test)]
    preds_price_bt = actual_prev_prices_bt + preds_diff_bt
    actual_prices_bt = y_raw[lags + split_idx + 1 : lags + split_idx + 1 + len(y_test)]
    
    mae, rmse = calculate_metrics(actual_prices_bt, preds_price_bt)
    
    # Full dataset
    model_full = RandomForestRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=42
    )
    model_full.fit(X, _y)
    
    # Iterative forecast (on differences)
    current_lags = list(y_diff[-lags:])
    preds_diff_future = []
    
    last_date = dates[-1]
    
    for step in range(horizon):
        window = current_lags[-lags:]
        
        # Rolling stats for the current iterative step
        rolling_mean_3 = np.mean(window[-3:]) if len(window) >= 3 else np.mean(window)
        rolling_std_3 = np.std(window[-3:]) if len(window) >= 3 else 0
        rolling_mean_7 = np.mean(window[-7:]) if len(window) >= 7 else np.mean(window)
        
        future_date = last_date + pd.Timedelta(days=step+1)
        day_of_week = future_date.dayofweek
        day_of_month = future_date.day
        month = future_date.month
        
        features = list(window) + [rolling_mean_3, rolling_std_3, rolling_mean_7, day_of_week, day_of_month, month]
        next_pred_diff = model_full.predict(np.array([features]))[0]
        preds_diff_future.append(next_pred_diff)
        
        # Update lags for next prediction
        current_lags = current_lags[1:] + [float(next_pred_diff)]
        
    # Reconstruct future prices
    last_actual_price = y_raw[-1]
    preds_price_future = last_actual_price + np.cumsum(preds_diff_future)
    
    # Simulated confidence intervals based on backtest RMSE expanding over time
    margin = rmse * np.sqrt(np.arange(1, horizon + 1)/3.0) 
    if confidence_level == 0.99: margin *= (2.576)
    elif confidence_level == 0.95: margin *= (1.96)
    elif confidence_level == 0.90: margin *= (1.645)
    elif confidence_level == 0.80: margin *= (1.282)
    
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=horizon, freq='D')
    future_forecast = pd.DataFrame({
        'Predicted_Price': preds_price_future,
        'Lower_Bound': preds_price_future - margin,
        'Upper_Bound': preds_price_future + margin
    }, index=future_dates)
    future_forecast.index.name = 'Date'
    
    # In-sample predictions reconstructed
    in_sample_diff = model_full.predict(X)
    actual_prev_prices_in_sample = y_raw[lags : lags + len(in_sample_diff)]
    in_sample_price = actual_prev_prices_in_sample + in_sample_diff
    in_sample_s = pd.Series(in_sample_price, index=dates[lags+1:])
    
    full_curve = pd.concat([
        pd.DataFrame({'yhat': in_sample_s}),
        pd.DataFrame({'yhat': preds_price_future, 'yhat_lower': future_forecast['Lower_Bound'], 'yhat_upper': future_forecast['Upper_Bound']}, index=future_dates)
    ])
    
    return {'forecast_df': future_forecast, 'full_forecast_curve': full_curve, 'mae': mae, 'rmse': rmse}
