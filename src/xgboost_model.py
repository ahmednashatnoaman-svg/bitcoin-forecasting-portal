import pandas as pd
import numpy as np
import xgboost as xgb
from src.metrics import calculate_metrics

def run_xgboost(df, target_col, horizon, confidence_level, **kwargs):
    lags = kwargs.get('lag_days', 14)
    n_estimators = kwargs.get('n_estimators', 100)
    max_depth = kwargs.get('max_depth', 3)
    learning_rate = kwargs.get('learning_rate', 0.1)
    subsample = kwargs.get('subsample', 1.0)
    colsample_bytree = kwargs.get('colsample_bytree', 1.0)
    min_child_weight = kwargs.get('min_child_weight', 1)
    
    # Differencing to make data stationary and allow XGBoost to forecast trends
    y_raw = df[target_col].values
    y_diff = np.diff(y_raw)
    dates_diff = df.index[1:]
    
    X = []
    _y = []
    dates = []
    
    for i in range(lags, len(y_diff)):
        # Base features (lags)
        lags_array = y_diff[i-lags:i]
        feats = list(lags_array)
        
        # Rolling stats of the lags (target-derived, no leakage)
        feats.append(np.mean(lags_array[-3:]))
        feats.append(np.std(lags_array[-3:]) if len(lags_array) >= 3 else 0)
        feats.append(np.mean(lags_array[-7:]) if len(lags_array) >= 7 else np.mean(lags_array))
        feats.append(np.std(lags_array[-7:]) if len(lags_array) >= 7 else np.std(lags_array))
        
        # Seasonal features
        target_date = dates_diff[i]
        feats.extend([target_date.dayofweek, target_date.day, target_date.month])
        X.append(feats)
        _y.append(y_diff[i])
        dates.append(target_date)
        
    X = np.array(X)
    _y = np.array(_y)
    
    if len(X) < 100:
        raise ValueError("Not enough data to train XGBoost with the requested lags.")
        
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = _y[:split_idx], _y[split_idx:]
    
    # Backtest
    model_bt = xgb.XGBRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        learning_rate=learning_rate, 
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
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
    model_full = xgb.XGBRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        learning_rate=learning_rate, 
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        random_state=42
    )
    model_full.fit(X, _y)
    
    # Iterative forecast (on differences)
    current_lags = list(y_diff[-lags:])
    preds_diff_future = []
    
    future_dates_xgb = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=horizon, freq='D')
    
    for i in range(horizon):
        target_date = future_dates_xgb[i]
        
        mean_3 = np.mean(current_lags[-3:])
        std_3 = np.std(current_lags[-3:]) if len(current_lags) >= 3 else 0
        mean_7 = np.mean(current_lags[-7:]) if len(current_lags) >= 7 else np.mean(current_lags)
        std_7 = np.std(current_lags[-7:]) if len(current_lags) >= 7 else np.std(current_lags)
        
        current_feats = current_lags + [mean_3, std_3, mean_7, std_7, target_date.dayofweek, target_date.day, target_date.month]
        
        pred_diff = model_full.predict(np.array([current_feats]))[0]
        preds_diff_future.append(float(pred_diff))
        
        # Update lags for next prediction
        current_lags = current_lags[1:] + [float(pred_diff)]
        
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
    in_sample_s = pd.Series(in_sample_price, index=dates)
    
    full_curve = pd.concat([
        pd.DataFrame({'yhat': in_sample_s}),
        pd.DataFrame({'yhat': preds_price_future, 'yhat_lower': future_forecast['Lower_Bound'], 'yhat_upper': future_forecast['Upper_Bound']}, index=future_dates)
    ])
    
    return {'forecast_df': future_forecast, 'full_forecast_curve': full_curve, 'mae': mae, 'rmse': rmse}
