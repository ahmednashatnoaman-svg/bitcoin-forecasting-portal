import pandas as pd
from prophet import Prophet
from src.metrics import calculate_metrics

def run_prophet(df, target_col, horizon, confidence_level, **kwargs):
    pdf = pd.DataFrame({'ds': df.index, 'y': df[target_col]})
    split_idx = int(len(pdf) * 0.8)
    train_df = pdf.iloc[:split_idx]
    test_df = pdf.iloc[split_idx:]
    
    cps = kwargs.get('changepoint_prior_scale', 0.15)
    sps = kwargs.get('seasonality_prior_scale', 10.0)
    
    daily_seasonality = kwargs.get('daily_seasonality', 'auto')
    weekly_seasonality = kwargs.get('weekly_seasonality', 'auto')
    yearly_seasonality = kwargs.get('yearly_seasonality', 'auto')
    
    model_bt = Prophet(
        changepoint_prior_scale=cps, 
        seasonality_prior_scale=sps, 
        interval_width=confidence_level,
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality
    )
    model_bt.fit(train_df)
    future_bt = model_bt.make_future_dataframe(periods=len(test_df))
    forecast_bt = model_bt.predict(future_bt)
    preds_bt = forecast_bt['yhat'].iloc[-len(test_df):].values
    
    mae, rmse = calculate_metrics(test_df['y'].values, preds_bt)
    
    model_full = Prophet(
        changepoint_prior_scale=cps, 
        seasonality_prior_scale=sps, 
        interval_width=confidence_level,
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality
    )
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
        'mae': mae, 'rmse': rmse,
        'model_full': model_full,
        'forecast_full': forecast_full
    }
