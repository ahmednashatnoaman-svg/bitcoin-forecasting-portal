import pandas as pd
import pmdarima as pm
import statsmodels.api as sm
from src.metrics import calculate_metrics

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
