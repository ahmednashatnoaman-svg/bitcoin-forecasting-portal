import pandas as pd
import numpy as np
from prophet import Prophet
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

def run_prophet(df, target_col, horizon, confidence_level, **kwargs):
    pdf = pd.DataFrame({'ds': df.index, 'y': df[target_col]})
    split_idx = int(len(pdf) * 0.8)
    train_df = pdf.iloc[:split_idx]
    test_df = pdf.iloc[split_idx:]
    
    cps = kwargs.get('changepoint_prior_scale', 0.15)
    sps = kwargs.get('seasonality_prior_scale', 10.0)
    
    model_bt = Prophet(changepoint_prior_scale=cps, seasonality_prior_scale=sps, interval_width=confidence_level)
    model_bt.fit(train_df)
    future_bt = model_bt.make_future_dataframe(periods=len(test_df))
    forecast_bt = model_bt.predict(future_bt)
    preds_bt = forecast_bt['yhat'].iloc[-len(test_df):].values
    
    mae, rmse = calculate_metrics(test_df['y'].values, preds_bt)
    
    model_full = Prophet(changepoint_prior_scale=cps, seasonality_prior_scale=sps, interval_width=confidence_level)
    model_full.fit(pdf)
    future_full = model_full.make_future_dataframe(periods=horizon)
    forecast_full = model_full.predict(future_full)
    
    future_forecast = forecast_full.iloc[-horizon:][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    future_forecast = future_forecast.rename(columns={
        'ds': 'Date', 'yhat': 'Predicted_Price', 'yhat_lower': 'Lower_Bound', 'yhat_upper': 'Upper_Bound'
    }).set_index('Date')
    
    return {
        'forecast_df': future_forecast,
        'full_forecast_curve': forecast_full.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']],
        'mae': mae, 'rmse': rmse
    }

def run_arima(df, target_col, horizon, confidence_level, **kwargs):
    y = df[target_col].dropna()
    split_idx = int(len(y) * 0.8)
    train_y = y.iloc[:split_idx]
    test_y = y.iloc[split_idx:]
    
    alpha = 1.0 - confidence_level
    auto_mode = kwargs.get('auto_arima', True)
    
    if auto_mode:
        model_bt = pm.auto_arima(train_y, start_p=1, start_q=1, max_p=5, max_q=5, m=7, seasonal=True, trace=False, error_action='ignore', suppress_warnings=True, stepwise=True)
        preds_bt = model_bt.predict(n_periods=len(test_y))
        best_order = model_bt.order
        best_seasonal_order = model_bt.seasonal_order
    else:
        p, d, q = kwargs.get('p', 1), kwargs.get('d', 1), kwargs.get('q', 1)
        P, D, Q, m = kwargs.get('seasonal_P', 0), kwargs.get('seasonal_D', 1), kwargs.get('seasonal_Q', 0), kwargs.get('seasonal_m', 30)
        best_order = (p, d, q)
        best_seasonal_order = (P, D, Q, m)
        bt_model = sm.tsa.ARIMA(train_y, order=best_order, seasonal_order=best_seasonal_order).fit()
        preds_bt = bt_model.forecast(steps=len(test_y))

    mae, rmse = calculate_metrics(test_y.values, preds_bt.values)
    
    full_model = sm.tsa.ARIMA(y, order=best_order, seasonal_order=best_seasonal_order)
    full_model_fit = full_model.fit()
    
    forecast_results = full_model_fit.get_forecast(steps=horizon)
    pred_mean = forecast_results.predicted_mean
    pred_ci = forecast_results.conf_int(alpha=alpha)
    
    future_dates = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=horizon, freq='D')
    future_forecast = pd.DataFrame({
        'Predicted_Price': pred_mean.values,
        'Lower_Bound': pred_ci.iloc[:, 0].values,
        'Upper_Bound': pred_ci.iloc[:, 1].values
    }, index=future_dates)
    future_forecast.index.name = 'Date'
    
    in_sample = full_model_fit.predict(start=y.index[0], end=y.index[-1])
    full_curve = pd.concat([
        pd.DataFrame({'yhat': in_sample}),
        pd.DataFrame({'yhat': pred_mean.values, 'yhat_lower': pred_ci.iloc[:, 0].values, 'yhat_upper': pred_ci.iloc[:, 1].values}, index=future_dates)
    ])
    
    return {
        'forecast_df': future_forecast,
        'full_forecast_curve': full_curve,
        'mae': mae, 'rmse': rmse
    }

def run_exponential_smoothing(df, target_col, horizon, confidence_level, **kwargs):
    y = df[target_col].dropna()
    split_idx = int(len(y) * 0.8)
    train_y = y.iloc[:split_idx]
    test_y = y.iloc[split_idx:]
    
    trend = kwargs.get('trend', 'add')
    damped = kwargs.get('damped_trend', False)
    seasonal = kwargs.get('seasonal', 'add')
    seasonal_periods = kwargs.get('seasonal_periods', 7)
    
    bt_model = ExponentialSmoothing(train_y, trend=trend, damped_trend=damped, seasonal=seasonal, seasonal_periods=seasonal_periods, initialization_method="estimated").fit()
    preds_bt = bt_model.forecast(len(test_y))
    mae, rmse = calculate_metrics(test_y.values, preds_bt.values)
    
    full_model = ExponentialSmoothing(y, trend=trend, damped_trend=damped, seasonal=seasonal, seasonal_periods=seasonal_periods, initialization_method="estimated").fit()
    preds_full = full_model.forecast(horizon)
    
    # Simple simulated confidence interval based on RMSE scaling
    z_score = sm.stats.DescrStatsW([0]).zconfint_mean(alpha=1.0-confidence_level)[1] if confidence_level else 1.96
    z_score = abs(sm.stats.proportion_confint(1, 2, alpha=1.0-confidence_level, method='normal')[1] * 2) # rough proxy
    # Better yet, use empirical std dev from residuals
    std_dev = np.std(full_model.resid)
    margin = std_dev * 1.96 * np.sqrt(np.arange(1, horizon + 1)) # simplistic expanding cone
    if confidence_level == 0.99: margin *= (2.576 / 1.96)
    elif confidence_level == 0.90: margin *= (1.645 / 1.96)
    elif confidence_level == 0.80: margin *= (1.282 / 1.96)
    
    future_dates = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=horizon, freq='D')
    future_forecast = pd.DataFrame({
        'Predicted_Price': preds_full.values,
        'Lower_Bound': preds_full.values - margin,
        'Upper_Bound': preds_full.values + margin
    }, index=future_dates)
    future_forecast.index.name = 'Date'
    
    in_sample = full_model.fittedvalues
    full_curve = pd.concat([
        pd.DataFrame({'yhat': in_sample}),
        pd.DataFrame({'yhat': preds_full.values, 'yhat_lower': future_forecast['Lower_Bound'].values, 'yhat_upper': future_forecast['Upper_Bound'].values}, index=future_dates)
    ])
    
    return {'forecast_df': future_forecast, 'full_forecast_curve': full_curve, 'mae': mae, 'rmse': rmse}


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


FORECASTERS = {
    'Prophet': run_prophet,
    'ARIMA': run_arima,
    'Exponential Smoothing': run_exponential_smoothing,
    'XGBoost': run_xgboost,
    'Random Forest (ML Regressor)': run_random_forest
}
