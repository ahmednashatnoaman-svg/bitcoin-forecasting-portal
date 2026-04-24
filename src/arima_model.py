import pandas as pd
import numpy as np
import pmdarima as pm
import statsmodels.api as sm
from src.metrics import calculate_metrics

def run_arima(df, target_col, horizon, confidence_level, **kwargs):
    """
    Trains ARIMA on 80% of data (chronological split), evaluates on 20%,
    then retrains on 100% for the final forecast.
    auto_arima=True uses stepwise search; False uses user-specified (p,d,q).
    """
    y = df[target_col].dropna()

    if len(y) < 30:
        raise ValueError("ARIMA requires at least 30 daily observations.")

    # Chronological 80/20 split
    split_idx = int(len(y) * 0.8)
    train_y = y.iloc[:split_idx]
    test_y  = y.iloc[split_idx:]

    alpha      = 1.0 - confidence_level
    auto_mode  = kwargs.get('auto_arima', True)

    # --- Backtest ---
    if auto_mode:
        model_bt = pm.auto_arima(
            train_y, start_p=1, start_q=1, max_p=5, max_q=5,
            m=7, seasonal=True, d=None,           # let auto_arima pick d via ADF test
            information_criterion='aic',
            trace=False, error_action='ignore',
            suppress_warnings=True, stepwise=True,
        )
        preds_bt = model_bt.predict(n_periods=len(test_y))
        best_order          = model_bt.order
        best_seasonal_order = model_bt.seasonal_order
    else:
        p, d, q = kwargs.get('p', 1), kwargs.get('d', 1), kwargs.get('q', 1)
        P, D, Q, m = (kwargs.get('seasonal_P', 1), kwargs.get('seasonal_D', 1),
                      kwargs.get('seasonal_Q', 0), kwargs.get('seasonal_m', 7))
        best_order          = (p, d, q)
        best_seasonal_order = (P, D, Q, m)
        bt_model  = sm.tsa.SARIMAX(train_y, order=best_order,
                                   seasonal_order=best_seasonal_order).fit(disp=False)
        preds_bt  = bt_model.forecast(steps=len(test_y))

    mae, rmse = calculate_metrics(test_y.values, np.asarray(preds_bt))

    # --- Full retrain on 100% data ---
    full_fit = sm.tsa.SARIMAX(y, order=best_order,
                              seasonal_order=best_seasonal_order).fit(disp=False)

    forecast_res = full_fit.get_forecast(steps=horizon)
    pred_mean    = forecast_res.predicted_mean
    pred_ci      = forecast_res.conf_int(alpha=alpha)

    future_dates = pd.date_range(
        start=y.index[-1] + pd.Timedelta(days=1), periods=horizon, freq='D'
    )
    future_forecast = pd.DataFrame({
        'Predicted_Price': pred_mean.values,
        'Lower_Bound':     pred_ci.iloc[:, 0].values,
        'Upper_Bound':     pred_ci.iloc[:, 1].values,
    }, index=future_dates)
    future_forecast.index.name = 'Date'

    in_sample = full_fit.fittedvalues   # avoids predict(start/end) index mismatch
    full_curve = pd.concat([
        pd.DataFrame({'yhat': in_sample}),
        pd.DataFrame({'yhat': pred_mean.values,
                      'yhat_lower': pred_ci.iloc[:, 0].values,
                      'yhat_upper': pred_ci.iloc[:, 1].values}, index=future_dates),
    ])

    return {'forecast_df': future_forecast, 'full_forecast_curve': full_curve,
            'mae': mae, 'rmse': rmse}
