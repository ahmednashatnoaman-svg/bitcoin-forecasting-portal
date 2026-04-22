import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

import data_processor as dp
import forecaster as fc

# 1. Page Configuration & Theme Dictionary
st.set_page_config(layout="wide", page_title="BTC Forecast Pro", page_icon="🏦", initial_sidebar_state="expanded")

THEMES = {
    'Dark Neon': {
        'primary': '#00FFA3', 'bg': '#0E1117', 'card_bg': '#1E212B', 'text': '#FFFFFF', 
        'plotly_template': 'plotly_dark', 'history_line': '#8E9EB6',
        'prophet': '#00FFA3', 'arima': '#00B8FF', 'es': '#FF007A', 'xgb': '#FFB800'
    },
    'Midnight Corporate': {
        'primary': '#3B82F6', 'bg': '#0F172A', 'card_bg': '#1E293B', 'text': '#F8FAFC', 
        'plotly_template': 'plotly_dark', 'history_line': '#94A3B8',
        'prophet': '#3B82F6', 'arima': '#10B981', 'es': '#8B5CF6', 'xgb': '#F59E0B'
    },
    'Light Minimal': {
        'primary': '#2563EB', 'bg': '#F1F5F9', 'card_bg': '#FFFFFF', 'text': '#0F172A', 
        'plotly_template': 'plotly_white', 'history_line': '#64748B',
        'prophet': '#2563EB', 'arima': '#059669', 'es': '#7C3AED', 'xgb': '#D97706'
    }
}

# 2. State Management Initialization
if 'config' not in st.session_state:
    st.session_state.config = {
        'model': 'Prophet', 'horizon': 30, 'confidence': 0.95, 'target_col': 'Close',
        'show_sma': False, 'model_kwargs': {}, 'theme': 'Midnight Corporate'
    }
if 'is_generated' not in st.session_state:
    st.session_state.is_generated = False

def update_config(key, value):
    st.session_state.config[key] = value

# Top UI Theme Selection
theme_choice = st.session_state.config['theme']
active_theme = THEMES[theme_choice]

# Inject Dynamic CSS
st.markdown(f"""
    <style>
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        .block-container {{
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 95%;
        }}
        
        /* Metric Cards Styling */
        div[data-testid="metric-container"] {{
            background-color: {active_theme['card_bg']};
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(150, 150, 150, 0.1);
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            transition: transform 0.2s ease-in-out;
        }}
        div[data-testid="metric-container"]:hover {{
            transform: translateY(-2px);
        }}
        
        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 20px;
        }}
        .stTabs [data-baseweb="tab"] {{
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0px 0px;
            gap: 10px;
            padding-top: 10px;
            padding-bottom: 10px;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {active_theme['card_bg']} !important;
            border-bottom: 3px solid {active_theme['primary']};
        }}
        
        /* Generate Button */
        .stButton>button {{
            height: 50px;
            font-size: 18px;
            font-weight: 600;
        }}
    </style>
""", unsafe_allow_html=True)

# 3. Application Layout 
colA, colB = st.columns([0.8, 0.2])
with colA:
    st.title("🏦 BTC Forecast Pro")
    st.markdown("A premium business intelligence dashboard for cryptocurrency time-series analysis.")
with colB:
    st.selectbox("🎨 UI Theme Pattern", list(THEMES.keys()), index=list(THEMES.keys()).index(theme_choice), key='theme_sel', on_change=lambda: update_config('theme', st.session_state.theme_sel))

st.divider()

# Sidebar Panel
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/46/Bitcoin.svg", width=60)
    st.markdown("## Control Panel")
    
    with st.expander("📂 Dataset Configuration", expanded=True):
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=['csv'])
        
    with st.expander("⚙️ Model Parameters", expanded=True):
        model_choice = st.selectbox("Forecasting Engine", ['Prophet', 'ARIMA', 'Exponential Smoothing', 'XGBoost', 'Compare All Models'], index=0)
        st.session_state.config['model'] = model_choice
        
        horizon = st.slider("Forecast Horizon (Days)", min_value=7, max_value=365, value=st.session_state.config['horizon'])
        update_config('horizon', horizon)
        
        confidence = st.selectbox("Confidence Bounds", [0.80, 0.90, 0.95, 0.99], index=2)
        update_config('confidence', confidence)

    with st.expander("🔬 Expert Configurations", expanded=False):
        kwargs = {}
        if model_choice in ['Prophet', 'Compare All Models']:
            st.markdown("#### Prophet Settings")
            kwargs['changepoint_prior_scale'] = st.slider("Changepoint Prior Scale", 0.01, 0.50, 0.15)
            kwargs['seasonality_prior_scale'] = st.slider("Seasonality Prior Scale", 1.0, 20.0, 10.0)
        
        if model_choice in ['ARIMA', 'Compare All Models']:
            st.markdown("#### ARIMA Settings")
            kwargs['auto_arima'] = st.checkbox("Use Auto-ARIMA Defaults", True)
            if not kwargs['auto_arima']:
                kwargs['p'] = st.number_input("AR order (p)", 0, 5, 1)
                kwargs['d'] = st.number_input("Differencing (d)", 0, 2, 1)
                kwargs['q'] = st.number_input("MA order (q)", 0, 5, 1)

        if model_choice in ['XGBoost', 'Compare All Models']:
            st.markdown("#### XGBoost Settings")
            kwargs['lag_days'] = st.slider("Lag Features (Days)", 7, 60, 14, key='x1')
            kwargs['n_estimators'] = st.slider("Trees (n_estimators)", 50, 500, 100, key='x2')
            kwargs['max_depth'] = st.slider("Max Depth", 2, 10, 3, key='x3')
            kwargs['learning_rate'] = st.selectbox("Learning Rate", [0.01, 0.05, 0.1, 0.2], index=2)

        if model_choice in ['Exponential Smoothing', 'Compare All Models']:
            st.markdown("#### Holt-Winters Settings")
            kwargs['trend'] = st.selectbox("Trend Type", ['add', 'mul'], index=0)
            kwargs['damped_trend'] = st.checkbox("Dampen Trend", False)

        st.session_state.config['model_kwargs'] = kwargs

    with st.expander("✨ Advanced Visuals"):
        show_sma = st.checkbox("Overlay 50 & 200 Day SMA", value=st.session_state.config['show_sma'])
        update_config('show_sma', show_sma)
        
    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button("Generate Forecast", type="primary", use_container_width=True)
    if generate_btn:
        st.session_state.is_generated = True

if uploaded_file is None:
    st.info("👋 Upload a Kaggle BTC dataset (CSV) from the sidebar to begin analysis. The app handles 1M granularity scaling instantly.")
    st.stop()

# 4. Processing Pipeline
try:
    raw_df, fallback_target = dp.load_and_preprocess_data(uploaded_file, target_col='Close')
except Exception as e:
    st.error(f"Error processing file: {e}")
    st.stop()

numerical_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
if not numerical_cols:
     st.error("No numerical columns found in dataset.")
     st.stop()
     
# Populate Target Column Dynamically
with st.sidebar:
    with st.expander("📂 Dataset Configuration", expanded=True):
        # Determine the default index dynamically to prevent index out-of-bounds error
        target_default_index = numerical_cols.index(fallback_target) if fallback_target in numerical_cols else 0
        target_selection = st.selectbox("Target Feature", numerical_cols, index=target_default_index)
        update_config('target_col', target_selection)

target_col = st.session_state.config['target_col']
df = dp.add_technical_indicators(raw_df.copy(), target_col)

if not st.session_state.is_generated:
    st.info("👈 Please define your parameters in the sidebar and click 'Generate Forecast' to execute the models.")
    st.stop()

# 5. Dashboard View Logic
tab1, tab2, tab3 = st.tabs(["📈 Intelligence Dashboard", "📊 Raw Data Explorer", "🧬 Algorithmic Diagnostics"])

with tab1:
    horizon = st.session_state.config['horizon']
    confidence = st.session_state.config['confidence']
    model_mode = st.session_state.config['model']
    model_kwargs = st.session_state.config.get('model_kwargs', {})

    with st.spinner('Synchronizing Data & Executing Predictive Algorithms...'):
        results = {}
        models_to_run = [model_mode] if model_mode != 'Compare All Models' else ['Prophet', 'ARIMA', 'Exponential Smoothing', 'XGBoost']
        
        for m in models_to_run:
            try:
                results[m] = fc.FORECASTERS[m](df, target_col, horizon, confidence, **model_kwargs)
            except Exception as e:
                st.warning(f"Failed to fit {m}: {e}")

    if not results:
        st.error("Severe Error: No models could be successfully fitted.")
        st.stop()

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

    st.markdown("<br>", unsafe_allow_html=True)

    # Master Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df[target_col], mode='lines', name='Historical Trajectory',
        line=dict(color=active_theme['history_line'], width=2),
        hovertemplate='Date: %{x}<br>Actual Price: $%{y:,.2f}<extra></extra>'
    ))

    if st.session_state.config['show_sma']:
        if 'SMA_50' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='50-Day SMA', line=dict(color='#A3A3A3', width=1, dash='dash')))
        if 'SMA_200' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], mode='lines', name='200-Day SMA', line=dict(color='#555555', width=1, dash='dash')))

    m_colors = {'Prophet': active_theme['prophet'], 'ARIMA': active_theme['arima'], 'Exponential Smoothing': active_theme['es'], 'XGBoost': active_theme['xgb']}
    
    forecast_start_date = df.index[-1]
    
    for m_name, m_res in results.items():
        f_df = m_res['forecast_df']
        c_color = m_colors[m_name]
        
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

    # Add Success Highlights Markers (Forecast Start Point)
    fig.add_vline(x=forecast_start_date, line_width=2, line_dash="dash", line_color=active_theme['primary'])
    # Optional annotation marker
    max_price = df[target_col].max()
    fig.add_annotation(
        x=forecast_start_date,
        y=max_price,
        text="Forecast Start",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor=active_theme['primary'],
        ax=-50,
        ay=-30,
        font=dict(size=14, color=active_theme['text']),
        bgcolor=active_theme['card_bg'],
        bordercolor=active_theme['primary'],
        borderpad=4
    )

    fig.update_layout(
        template=active_theme['plotly_template'], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Timeline", yaxis_title="Price Denomination (USD)",
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Data Explorer Mapping")
    colA, colB = st.columns([0.7, 0.3])
    with colA:
        st.dataframe(df, use_container_width=True)
    with colB:
        st.write("Summary Statistics")
        st.dataframe(df[target_col].describe(), use_container_width=True)
        csv_data = primary_res['forecast_df'].to_csv().encode('utf-8')
        st.download_button("Export Predictive File Matrix (CSV)", data=csv_data, file_name='btc_outcomes.csv', mime='text/csv')

with tab3:
    st.subheader("Model Diagnostic Benchmarks (80/20 Test Split)")
    perf_data = [{"Algorithmic Core": k, "Mean Absolute Error (USD)": f"${v['mae']:,.2f}", "Root Mean Squared Error (USD)": f"${v['rmse']:,.2f}"} for k, v in results.items()]
    st.table(pd.DataFrame(perf_data))
    st.info("The RMSE provides a rigid constraint representing the standard deviation of unexplained variance in historical bounds testing. Algorithms with lower RMSEs correctly matched actual pricing movements closer in the final backtesting window.")
