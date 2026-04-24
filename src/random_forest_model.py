import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from src.metrics import calculate_metrics

# Distinct color for this model (used by visualizations)
MODEL_COLOR = '#10B981'

def _build_lag_features(y_diff, dates, lags):
    """
    Builds feature matrix from differenced series.
    All rolling stats computed purely from the lag window — zero look-ahead bias.
    """
    X, targets, date_idx = [], [], []
    for i in range(lags, len(y_diff)):
        window = y_diff[i - lags: i]
        feats  = list(window)
        feats += [
            np.mean(window[-3:]),
            np.std(window[-3:])  if len(window) >= 3 else 0,
            np.mean(window[-7:]) if len(window) >= 7 else np.mean(window),
        ]
        # dates are for the differenced series: diff[i] corresponds to dates[i+1]
        d = dates[i + 1]
        feats += [d.dayofweek, d.day, d.month]
        X.append(feats)
        targets.append(y_diff[i])
        date_idx.append(d)
    return np.array(X), np.array(targets), date_idx


def run_random_forest(df, target_col, horizon, confidence_level, **kwargs):
    """
    Trains Random Forest on differenced price series.
    - 80/20 chronological split for backtest metrics.
    - Full retrain on 100% data for future forecast.
    - Iterative multi-step forecast (no look-ahead).
    - Confidence intervals expand with sqrt(horizon) based on backtest RMSE.
    """
    series    = df[target_col].dropna()
    y_raw     = series.values
    dates     = series.index

    if len(y_raw) < 50:
        raise ValueError("Random Forest requires at least 50 daily observations.")

    # First-difference for stationarity
    y_diff = np.diff(y_raw)

    lags         = kwargs.get('rf_lag_days', 14)
    n_estimators = kwargs.get('rf_n_estimators', 200)
    max_depth    = kwargs.get('rf_max_depth', 10)

    if len(y_diff) <= lags:
        raise ValueError("Not enough data to create lagged features for Random Forest.")

    X, y_target, date_idx = _build_lag_features(y_diff, dates, lags)

    # Chronological 80/20 split
    split_idx = int(len(X) * 0.8)
    X_train, X_test   = X[:split_idx],       X[split_idx:]
    y_train, y_test   = y_target[:split_idx], y_target[split_idx:]

    def _build_model():
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )

    # --- Backtest on test split ---
    model_bt      = _build_model()
    model_bt.fit(X_train, y_train)
    preds_diff_bt = model_bt.predict(X_test)

    # Reconstruct price level for metrics
    prev_bt   = y_raw[lags + split_idx: lags + split_idx + len(y_test)]
    pred_bt   = prev_bt + preds_diff_bt
    actual_bt = y_raw[lags + split_idx + 1: lags + split_idx + 1 + len(y_test)]
    mae, rmse = calculate_metrics(actual_bt, pred_bt)

    # --- Full retrain on 100% data ---
    model_full = _build_model()
    model_full.fit(X, y_target)

    # Iterative multi-step forecast on differences
    current_window    = list(y_diff[-lags:])
    preds_diff_future = []
    future_dates = pd.date_range(
        start=dates[-1] + pd.Timedelta(days=1), periods=horizon, freq='D'
    )

    for i in range(horizon):
        d = future_dates[i]
        w = current_window[-lags:]
        feats = list(w) + [
            np.mean(w[-3:]),
            np.std(w[-3:])  if len(w) >= 3 else 0,
            np.mean(w[-7:]) if len(w) >= 7 else np.mean(w),
            d.dayofweek, d.day, d.month,
        ]
        pred_d = float(model_full.predict(np.array([feats]))[0])
        preds_diff_future.append(pred_d)
        current_window = current_window[1:] + [pred_d]

    preds_price_future = y_raw[-1] + np.cumsum(preds_diff_future)

    z_map  = {0.99: 2.576, 0.95: 1.96, 0.90: 1.645, 0.80: 1.282}
    z      = z_map.get(confidence_level, 1.96)
    margin = z * rmse * np.sqrt(np.arange(1, horizon + 1) / 3.0)

    future_forecast = pd.DataFrame({
        'Predicted_Price': preds_price_future,
        'Lower_Bound':     preds_price_future - margin,
        'Upper_Bound':     preds_price_future + margin,
    }, index=future_dates)
    future_forecast.index.name = 'Date'

    # In-sample curve
    in_sample_diff  = model_full.predict(X)
    prev_in_sample  = y_raw[lags: lags + len(in_sample_diff)]
    in_sample_price = prev_in_sample + in_sample_diff
    in_sample_s     = pd.Series(in_sample_price, index=date_idx)

    full_curve = pd.concat([
        pd.DataFrame({'yhat': in_sample_s}),
        pd.DataFrame({
            'yhat':       preds_price_future,
            'yhat_lower': future_forecast['Lower_Bound'].values,
            'yhat_upper': future_forecast['Upper_Bound'].values,
        }, index=future_dates),
    ])

    return {'forecast_df': future_forecast, 'full_forecast_curve': full_curve,
            'mae': mae, 'rmse': rmse}
