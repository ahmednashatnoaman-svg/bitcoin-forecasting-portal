import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from src.metrics import calculate_metrics

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
