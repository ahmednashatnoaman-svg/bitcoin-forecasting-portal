import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from src.ui.styles import ACTIVE_THEME

def render_kpi_cards(results, df, target_col, horizon, confidence, model_mode):
    # Dynamic KPI Cards
    primary_res = list(results.values())[0]
    latest_close = df[target_col].iloc[-1]
    predicted_close = primary_res['forecast_df']['Predicted_Price'].iloc[-1]
    predicted_delta = predicted_close - latest_close
    rmse_margin = primary_res['rmse']

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Close Value", f"${latest_close:,.2f}")
    
    model_metric_label = list(results.keys())[0] if model_mode == 'Compare All Models' else model_mode
    col2.metric(f"Proj. T+{horizon} ({model_metric_label})", f"${predicted_close:,.2f}", f"{predicted_delta:+,.2f} USD")
    col3.metric(f"Confidence Width", f"{(confidence*100):.0f}% Limits")
    col4.metric("Backtest RMSE Float", f"± ${rmse_margin:,.2f}")

def render_main_forecast_chart(df, results, target_col, show_sma):
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(
        x=df.index, y=df[target_col], mode='lines', name='Historical Trajectory',
        line=dict(color=ACTIVE_THEME['history_line'], width=2),
        hovertemplate='Date: %{x}<br>Actual Price: $%{y:,.2f}<extra></extra>'
    ))

    # SMA Overlays
    if show_sma:
        if 'SMA_50' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='50-Day SMA', line=dict(color='#A3A3A3', width=1, dash='dash')))
        if 'SMA_200' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], mode='lines', name='200-Day SMA', line=dict(color='#555555', width=1, dash='dash')))

    m_colors = {
        'Prophet': ACTIVE_THEME['prophet'], 
        'ARIMA': ACTIVE_THEME['arima'], 
        'Exponential Smoothing': ACTIVE_THEME['es'], 
        'XGBoost': ACTIVE_THEME['xgb'], 
        'Random Forest (ML Regressor)': ACTIVE_THEME['rf']
    }
    
    forecast_start_date = df.index[-1]
    
    for m_name, m_res in results.items():
        f_df = m_res['forecast_df']
        c_color = m_colors.get(m_name, ACTIVE_THEME['primary'])
        
        # Transparent Confidence Bounds
        fig.add_trace(go.Scatter(x=f_df.index, y=f_df['Upper_Bound'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(
            x=f_df.index, y=f_df['Lower_Bound'], mode='lines', line=dict(width=0),
            fillcolor=f'rgba{tuple(int(c_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.15,)}', fill='tonexty', name=f'{m_name} Limits', hoverinfo='skip'
        ))
        
        # Core Forecast Trace
        fig.add_trace(go.Scatter(
            x=f_df.index, y=f_df['Predicted_Price'], mode='lines', line=dict(color=c_color, width=3),
            name=f'{m_name} Vector', hovertemplate=f'Date: %{{x}}<br>{m_name} Vector: $%{{y:,.2f}}<br>Bounds active<extra></extra>'
        ))

    # Add Success Highlights Markers
    fig.add_vline(x=forecast_start_date, line_width=2, line_dash="dash", line_color=ACTIVE_THEME['primary'])
    
    max_price = df[target_col].max()
    fig.add_annotation(
        x=forecast_start_date, y=max_price, text="Forecast Start", showarrow=True,
        arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor=ACTIVE_THEME['primary'],
        ax=-50, ay=-30, font=dict(size=14, color=ACTIVE_THEME['text']),
        bgcolor=ACTIVE_THEME['card_bg'], bordercolor=ACTIVE_THEME['primary'], borderpad=4
    )

    fig.update_layout(
        template=ACTIVE_THEME['plotly_template'], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Timeline", yaxis_title="Price Denomination (USD)",
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

def render_data_explorer(df, target_col, results):
    st.subheader("Data Explorer Mapping")
    colA, colB = st.columns([0.7, 0.3])
    with colA:
        st.dataframe(df, use_container_width=True)
    with colB:
        st.write("Summary Statistics")
        st.dataframe(df[target_col].describe(), use_container_width=True)
        if results:
            primary_res = list(results.values())[0]
            csv_data = primary_res['forecast_df'].to_csv().encode('utf-8')
            st.download_button("Export Predictive File Matrix (CSV)", data=csv_data, file_name='btc_outcomes.csv', mime='text/csv')

def render_diagnostics(results):
    st.subheader("Model Diagnostic Benchmarks (80/20 Test Split)")
    perf_data = [{"Algorithmic Core": k, "Mean Absolute Error (USD)": f"${v['mae']:,.2f}", "Root Mean Squared Error (USD)": f"${v['rmse']:,.2f}"} for k, v in results.items()]
    st.table(pd.DataFrame(perf_data))
    st.info("The RMSE provides a rigid constraint representing the standard deviation of unexplained variance in historical bounds testing.")
    
    if 'Prophet' in results:
        st.subheader("Prophet Components & Seasonality Insights")
        try:
            fig_comp = results['Prophet']['model_full'].plot_components(results['Prophet']['forecast_full'])
            st.pyplot(fig_comp)
        except Exception as e:
            st.warning(f"Could not render Prophet components: {e}")
