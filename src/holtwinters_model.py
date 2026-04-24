import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from src.metrics import calculate_metrics

def run_exponential_smoothing(df, target_col, horizon, confidence_level, **kwargs):
    """
    Trains Holt-Winters on 80% of data (chronological), evaluates on 20%,
    then retrains on 100% for the final forecast.
    Confidence intervals are derived from residual std of the full model.
    """
    y = df[target_col].dropna()

    if len(y) < 20:
        raise ValueError("Exponential Smoothing requires at least 20 observations.")

    # Chronological 80/20 split
    split_idx = int(len(y) * 0.8)
    train_y = y.iloc[:split_idx]
    test_y  = y.iloc[split_idx:]

    trend            = kwargs.get('trend', 'add')
    damped           = kwargs.get('damped_trend', False)
    seasonal         = kwargs.get('seasonal', 'add')
    seasonal_periods = kwargs.get('seasonal_periods', 7)

    # Guard: need at least 2 full seasonal cycles in training data
    if seasonal is not None and len(train_y) < 2 * seasonal_periods:
        seasonal         = None
        seasonal_periods = None

    def _fit(series):
        return ExponentialSmoothing(
            series,
            trend=trend,
            damped_trend=damped if trend else False,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            initialization_method='estimated',
        ).fit(optimized=True)

    # --- Backtest ---
    model_bt  = _fit(train_y)
    preds_bt  = model_bt.forecast(len(test_y))
    mae, rmse = calculate_metrics(test_y.values, preds_bt.values)

    # --- Full retrain on 100% data ---
    model_full  = _fit(y)
    preds_full  = model_full.forecast(horizon)

    # Confidence interval: residual std expands over horizon (random-walk cone)
    z_map = {0.99: 2.576, 0.95: 1.96, 0.90: 1.645, 0.80: 1.282}
    z = z_map.get(confidence_level, 1.96)
    std_resid = np.std(model_full.resid)
    margin = z * std_resid * np.sqrt(np.arange(1, horizon + 1))

    future_dates = pd.date_range(
        start=y.index[-1] + pd.Timedelta(days=1), periods=horizon, freq='D'
    )
    future_forecast = pd.DataFrame({
        'Predicted_Price': preds_full.values,
        'Lower_Bound':     preds_full.values - margin,
        'Upper_Bound':     preds_full.values + margin,
    }, index=future_dates)
    future_forecast.index.name = 'Date'

    in_sample  = model_full.fittedvalues
    full_curve = pd.concat([
        pd.DataFrame({'yhat': in_sample}),
        pd.DataFrame({
            'yhat':       preds_full.values,
            'yhat_lower': future_forecast['Lower_Bound'].values,
            'yhat_upper': future_forecast['Upper_Bound'].values,
        }, index=future_dates),
    ])

    return {'forecast_df': future_forecast, 'full_forecast_curve': full_curve,
            'mae': mae, 'rmse': rmse}
