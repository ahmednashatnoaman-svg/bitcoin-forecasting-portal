import pandas as pd
import numpy as np
from prophet import Prophet
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb

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
        model_bt = pm.auto_arima(train_y, start_p=1, start_q=1, max_p=5, max_q=5, m=1, seasonal=False, trace=False, error_action='ignore', suppress_warnings=True, stepwise=True)
        preds_bt = model_bt.predict(n_periods=len(test_y))
        best_order = model_bt.order
    else:
        p, d, q = kwargs.get('p', 1), kwargs.get('d', 1), kwargs.get('q', 1)
        best_order = (p, d, q)
        bt_model = sm.tsa.ARIMA(train_y, order=best_order).fit()
        preds_bt = bt_model.forecast(steps=len(test_y))

    mae, rmse = calculate_metrics(test_y.values, preds_bt.values)
    
    full_model = sm.tsa.ARIMA(y, order=best_order)
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
    
    bt_model = ExponentialSmoothing(train_y, trend=trend, damped_trend=damped, seasonal=None, initialization_method="estimated").fit()
    preds_bt = bt_model.forecast(len(test_y))
    mae, rmse = calculate_metrics(test_y.values, preds_bt.values)
    
    full_model = ExponentialSmoothing(y, trend=trend, damped_trend=damped, seasonal=None, initialization_method="estimated").fit()
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


def create_lagged_features(df, target_col, lags):
    X = []
    _y = []
    dates = []
    # Assumes df is sorted by Date
    vals = df[target_col].values
    idx = df.index
    for i in range(lags, len(vals)):
        X.append(vals[i-lags:i])
        _y.append(vals[i])
        dates.append(idx[i])
    return np.array(X), np.array(_y), dates

def run_xgboost(df, target_col, horizon, confidence_level, **kwargs):
    lags = kwargs.get('lag_days', 14)
    n_estimators = kwargs.get('n_estimators', 100)
    max_depth = kwargs.get('max_depth', 3)
    learning_rate = kwargs.get('learning_rate', 0.1)
    
    X, y, dates = create_lagged_features(df, target_col, lags)
    
    if len(X) < 100:
        raise ValueError("Not enough data to train XGBoost with the requested lags.")
        
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Backtest
    model_bt = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
    model_bt.fit(X_train, y_train)
    preds_bt = model_bt.predict(X_test)
    mae, rmse = calculate_metrics(y_test, preds_bt)
    
    # Full dataset
    model_full = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
    model_full.fit(X, y)
    
    # Iterative forecast
    last_window = X[-1][1:].tolist() + [y[-1]]
    preds_full = []
    current_window = np.array([last_window])
    for _ in range(horizon):
        pred = model_full.predict(current_window)[0]
        preds_full.append(pred)
        new_window = current_window[0][1:].tolist() + [pred]
        current_window = np.array([new_window])
        
    preds_full = np.array(preds_full)
    
    # Simulated confidence intervals based on backtest RMSE expanding over time
    margin = rmse * np.sqrt(np.arange(1, horizon + 1)/3.0) 
    if confidence_level == 0.99: margin *= (2.576)
    elif confidence_level == 0.95: margin *= (1.96)
    elif confidence_level == 0.90: margin *= (1.645)
    elif confidence_level == 0.80: margin *= (1.282)
    
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=horizon, freq='D')
    future_forecast = pd.DataFrame({
        'Predicted_Price': preds_full,
        'Lower_Bound': preds_full - margin,
        'Upper_Bound': preds_full + margin
    }, index=future_dates)
    future_forecast.index.name = 'Date'
    
    # In-sample predictions
    in_sample = model_full.predict(X)
    in_sample_s = pd.Series(in_sample, index=dates)
    
    full_curve = pd.concat([
        pd.DataFrame({'yhat': in_sample_s}),
        pd.DataFrame({'yhat': preds_full, 'yhat_lower': future_forecast['Lower_Bound'], 'yhat_upper': future_forecast['Upper_Bound']}, index=future_dates)
    ])
    
    return {'forecast_df': future_forecast, 'full_forecast_curve': full_curve, 'mae': mae, 'rmse': rmse}


FORECASTERS = {
    'Prophet': run_prophet,
    'ARIMA': run_arima,
    'Exponential Smoothing': run_exponential_smoothing,
    'XGBoost': run_xgboost
}
