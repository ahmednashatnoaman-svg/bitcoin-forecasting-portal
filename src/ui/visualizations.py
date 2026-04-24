import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from src.ui.styles import ACTIVE_THEME

# Distinct color per model — consistent across all charts
MODEL_COLORS = {
    'Prophet':                      '#6366F1',   # Indigo
    'ARIMA':                        '#F59E0B',   # Amber
    'Exponential Smoothing':        '#EC4899',   # Pink
    'XGBoost':                      '#10B981',   # Emerald
    'Random Forest (ML Regressor)': '#EF4444',   # Red
}


def render_kpi_cards(results, df, target_col, horizon, confidence, model_mode):
    """Renders the four top-level KPI metric cards."""
    primary_res     = list(results.values())[0]
    latest_price    = df[target_col].iloc[-1]
    predicted_price = primary_res['forecast_df']['Predicted_Price'].iloc[-1]
    delta           = predicted_price - latest_price
    rmse            = primary_res['rmse']

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Price",              f"${latest_price:,.2f}")
    model_label = list(results.keys())[0] if model_mode == 'Compare All Models' else model_mode
    col2.metric(f"T+{horizon} Forecast ({model_label})", f"${predicted_price:,.2f}", f"{delta:+,.2f} USD")
    col3.metric("Confidence Level",          f"{int(confidence * 100)}%")
    col4.metric("Backtest RMSE",             f"± ${rmse:,.2f}")


def render_main_forecast_chart(df, results, target_col, show_sma_50=False, show_sma_200=False):
    """Renders the main Plotly forecast chart with per-model color coding."""
    fig = go.Figure()

    # 1. Historical price trace
    fig.add_trace(go.Scatter(
        x=df.index, y=df[target_col],
        mode='lines', name='Historical Price',
        line=dict(color=ACTIVE_THEME['history_line'], width=2),
        hovertemplate='%{x}<br>Price: $%{y:,.2f}<extra></extra>',
    ))

    # 2. Optional SMA overlays — each with its own distinct color
    if show_sma_50 and 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_50'], mode='lines',
            name='SMA 50', line=dict(color='#FBBF24', width=1.5, dash='dash'),
        ))
    if show_sma_200 and 'SMA_200' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_200'], mode='lines',
            name='SMA 200', line=dict(color='#818CF8', width=1.5, dash='dot'),
        ))

    # 3. Per-model forecast traces with distinct colors
    forecast_start = df.index[-1]

    for m_name, m_res in results.items():
        f_df  = m_res['forecast_df']
        color = MODEL_COLORS.get(m_name, ACTIVE_THEME['primary'])
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

        # Confidence band (filled area)
        fig.add_trace(go.Scatter(
            x=f_df.index, y=f_df['Upper_Bound'],
            mode='lines', line=dict(width=0),
            showlegend=False, hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            x=f_df.index, y=f_df['Lower_Bound'],
            mode='lines', line=dict(width=0),
            fillcolor=f'rgba({r},{g},{b},0.12)', fill='tonexty',
            name=f'{m_name} CI', hoverinfo='skip',
        ))

        # Main forecast line
        fig.add_trace(go.Scatter(
            x=f_df.index, y=f_df['Predicted_Price'],
            mode='lines', name=m_name,
            line=dict(color=color, width=3),
            hovertemplate=f'%{{x}}<br>{m_name}: $%{{y:,.2f}}<extra></extra>',
        ))

    # 4. Forecast start marker
    fig.add_vline(x=forecast_start, line_width=2, line_dash='dash',
                  line_color=ACTIVE_THEME['primary'])
    fig.add_annotation(
        x=forecast_start, y=df[target_col].max(),
        text='Forecast Start', showarrow=True,
        arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor=ACTIVE_THEME['primary'], ax=-50, ay=-30,
        font=dict(size=13, color=ACTIVE_THEME['text']),
        bgcolor=ACTIVE_THEME['card_bg'],
        bordercolor=ACTIVE_THEME['primary'], borderpad=4,
    )

    fig.update_layout(
        template=ACTIVE_THEME['plotly_template'],
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title='Date', yaxis_title='Price (USD)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_data_explorer(df, target_col, results):
    """Renders the raw data table and summary statistics."""
    st.subheader("Data Explorer")
    colA, colB = st.columns([0.7, 0.3])
    with colA:
        st.dataframe(df, use_container_width=True)
    with colB:
        st.write("Summary Statistics")
        st.dataframe(df[target_col].describe(), use_container_width=True)
        if results:
            csv_data = list(results.values())[0]['forecast_df'].to_csv().encode('utf-8')
            st.download_button("Export Forecast CSV", data=csv_data,
                               file_name='btc_forecast.csv', mime='text/csv')


def render_diagnostics(results):
    """Renders model performance table and Prophet component plots."""
    st.subheader("Model Diagnostics — 80/20 Chronological Backtest")
    perf = [{
        "Model":  k,
        "MAE (USD)":  f"${v['mae']:,.2f}",
        "RMSE (USD)": f"${v['rmse']:,.2f}",
        "Color":  MODEL_COLORS.get(k, '#888'),
    } for k, v in results.items()]
    st.table(pd.DataFrame(perf).drop(columns='Color'))

    st.info("Metrics computed on the held-out 20% test set. "
            "Lower MAE/RMSE = better historical fit.")

    if 'Prophet' in results:
        st.subheader("Prophet Seasonality Components")
        try:
            fig_comp = results['Prophet']['model_full'].plot_components(
                results['Prophet']['forecast_full']
            )
            st.pyplot(fig_comp)
        except Exception as e:
            st.warning(f"Could not render Prophet components: {e}")
