import streamlit as st
import numpy as np
import pandas as pd

import src.data_processor as dp
from src.prophet_model import run_prophet
from src.arima_model import run_arima
from src.holtwinters_model import run_exponential_smoothing
from src.xgboost_model import run_xgboost
from src.random_forest_model import run_random_forest

from src.ui.styles import inject_custom_css
from src.ui.sidebar import render_sidebar, render_target_selection
from src.ui.visualizations import render_kpi_cards, render_main_forecast_chart, render_data_explorer, render_diagnostics

# 1. Page Configuration
st.set_page_config(layout="wide", page_title="BTC Forecast Pro", page_icon="🏦", initial_sidebar_state="expanded")

FORECASTERS = {
    'Prophet': run_prophet,
    'ARIMA': run_arima,
    'Exponential Smoothing': run_exponential_smoothing,
    'XGBoost': run_xgboost,
    'Random Forest (ML Regressor)': run_random_forest
}

# 2. State Management Initialization
if 'config' not in st.session_state:
    st.session_state.config = {
        'model': 'Prophet', 'horizon': 30, 'confidence': 0.95, 'target_col': 'Close',
        'show_sma': False, 'model_kwargs': {}
    }
if 'is_generated' not in st.session_state:
    st.session_state.is_generated = False

def update_config(key, value):
    st.session_state.config[key] = value

inject_custom_css()

# 3. Header
col1, col2 = st.columns([0.80, 0.20])
with col1:
    st.title("🏦 BTC Forecast Pro")
    st.markdown("A premium business intelligence dashboard for cryptocurrency time-series analysis.")
with col2:
    st.write("###") 
    st.image("src/ui/assets/iti.png", use_container_width=True)

st.divider()

# 4. Sidebar & Inputs
sidebar_results = render_sidebar()
uploaded_file = sidebar_results['uploaded_file']
model_choice  = sidebar_results['model_choice']
update_config('horizon', sidebar_results['horizon'])
update_config('confidence', sidebar_results['confidence'])
update_config('model_kwargs', sidebar_results['kwargs'])
update_config('show_sma_50',  sidebar_results['show_sma_50'])
update_config('show_sma_200', sidebar_results['show_sma_200'])

if sidebar_results['generate_btn']:
    st.session_state.is_generated = True

# Handle Kaggle dataset download
if sidebar_results['kaggle_download'] and sidebar_results['kaggle_slug']:
    slug = sidebar_results['kaggle_slug'].strip()
    try:
        import kagglehub, glob
        with st.spinner(f"⬇️ Downloading **{slug}** from Kaggle..."):
            path = kagglehub.dataset_download(slug)
            csvs = glob.glob(f"{path}/**/*.csv", recursive=True)
            if not csvs:
                st.error("No CSV files found in the downloaded Kaggle dataset.")
                st.stop()
            st.session_state['kaggle_csv_path'] = csvs[0]
            st.session_state['kaggle_slug_name'] = slug
            st.success(f"✅ Loaded `{csvs[0].split('/')[-1]}` from Kaggle.")
    except Exception as e:
        st.error(f"Kaggle download failed: {e}")
        st.stop()

# Resolve data source priority: upload > kaggle download
data_source = uploaded_file
if data_source is None and 'kaggle_csv_path' in st.session_state:
    data_source = st.session_state['kaggle_csv_path']

if data_source is None:
    st.info(
        "👋 Upload a BTC dataset or paste a **Kaggle slug** and click Download.\n\n"
        "**Example Kaggle Slugs:**\n"
        "- `novandraanugrah/bitcoin-historical-datasets-2018-2024`\n"
        "- `imranbukhari/comprehensive-btcusd-1m-data`"
    )
    st.stop()

# 5. Data Processing
try:
    raw_df, fallback_target = dp.load_and_preprocess_data(data_source, target_col='Close')
except Exception as e:
    st.error(f"Error processing file: {e}")
    st.stop()

numerical_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
target_selection = render_target_selection(numerical_cols, fallback_target)
update_config('target_col', target_selection)

target_col = st.session_state.config['target_col']
df = dp.add_technical_indicators(raw_df.copy(), target_col)

if not st.session_state.is_generated:
    st.info("👈 Please define your parameters in the sidebar and click 'Generate Forecast' to execute the models.")
    st.stop()

# 6. Forecasting Logic
horizon = st.session_state.config['horizon']
confidence = st.session_state.config['confidence']
model_kwargs = st.session_state.config.get('model_kwargs', {})

with st.spinner('Synchronizing Data & Executing Predictive Algorithms...'):
    results = {}
    models_to_run = [model_choice] if model_choice != 'Compare All Models' else ['Prophet', 'ARIMA', 'Exponential Smoothing', 'XGBoost', 'Random Forest (ML Regressor)']
    
    for m in models_to_run:
        try:
            results[m] = FORECASTERS[m](df, target_col, horizon, confidence, **model_kwargs)
        except Exception as e:
            st.warning(f"Failed to fit {m}: {e}")

if not results:
    st.error("Severe Error: No models could be successfully fitted.")
    st.stop()

# 7. Main Dashboard Views
tab1, tab2, tab3 = st.tabs(["📈 Intelligence Dashboard", "📊 Raw Data Explorer", "🧬 Algorithmic Diagnostics"])

with tab1:
    render_kpi_cards(results, df, target_col, horizon, confidence, model_choice)
    st.markdown("<br>", unsafe_allow_html=True)
    render_main_forecast_chart(df, results, target_col,
                               st.session_state.config.get('show_sma_50', False),
                               st.session_state.config.get('show_sma_200', False))

with tab2:
    render_data_explorer(df, target_col, results)

with tab3:
    render_diagnostics(results)
