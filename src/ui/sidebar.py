import streamlit as st

def render_sidebar():
    """
    Renders the unified control panel sidebar and returns user configurations.
    """
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/4/46/Bitcoin.svg", width=60)
        st.markdown("## Control Panel")
        
        with st.expander("📂 Dataset Configuration", expanded=True):
            uploaded_file = st.file_uploader("Upload CSV Dataset", type=['csv'])
            
        with st.expander("⚙️ Model Parameters", expanded=True):
            model_choice = st.selectbox(
                "Forecasting Engine", 
                ['Prophet', 'ARIMA', 'Exponential Smoothing', 'XGBoost', 'Random Forest (ML Regressor)', 'Compare All Models'], 
                index=0
            )
            
            horizon = st.slider("Forecast Horizon (Days)", min_value=7, max_value=365, value=st.session_state.config.get('horizon', 30))
            confidence = st.selectbox("Confidence Bounds", [0.80, 0.90, 0.95, 0.99], index=2)

        kwargs = {}
        with st.expander("🔬 Expert Configurations", expanded=False):
            if model_choice in ['Prophet', 'Compare All Models']:
                st.markdown("#### Prophet Settings")
                kwargs['changepoint_prior_scale'] = st.slider("Changepoint Prior Scale", 0.01, 0.50, 0.15)
                kwargs['seasonality_prior_scale'] = st.slider("Seasonality Prior Scale", 1.0, 20.0, 10.0)
                kwargs['daily_seasonality'] = st.checkbox("Daily Seasonality (Prophet)", value=False)
                kwargs['weekly_seasonality'] = st.checkbox("Weekly Seasonality (Prophet)", value=True)
                kwargs['yearly_seasonality'] = st.checkbox("Yearly Seasonality (Prophet)", value=True)
            
            if model_choice in ['ARIMA', 'Compare All Models']:
                st.markdown("#### ARIMA Settings")
                kwargs['auto_arima'] = st.checkbox("Use Auto-ARIMA Defaults", True)
                if not kwargs['auto_arima']:
                    kwargs['p'] = st.number_input("AR order (p)", 0, 5, 1)
                    kwargs['d'] = st.number_input("Differencing (d)", 0, 2, 1)
                    kwargs['q'] = st.number_input("MA order (q)", 0, 5, 1)
                    kwargs['seasonal_P'] = st.number_input("Seasonal AR (P)", 0, 5, 0)
                    kwargs['seasonal_D'] = st.number_input("Seasonal Differencing (D)", 0, 2, 1)
                    kwargs['seasonal_Q'] = st.number_input("Seasonal MA (Q)", 0, 5, 0)
                    kwargs['seasonal_m'] = st.number_input("Seasonal Period (m)", 0, 365, 30)

            if model_choice in ['XGBoost', 'Compare All Models']:
                st.markdown("#### XGBoost Settings")
                kwargs['lag_days'] = st.slider("Lag Features (Days)", 7, 60, 14, key='x1')
                kwargs['n_estimators'] = st.slider("Trees (n_estimators)", 50, 500, 100, key='x2')
                kwargs['max_depth'] = st.slider("Max Depth", 2, 10, 3, key='x3')
                kwargs['learning_rate'] = st.selectbox("Learning Rate", [0.01, 0.05, 0.1, 0.2], index=2)
                kwargs['subsample'] = st.slider("Subsample Ratio", 0.1, 1.0, 0.8, key='x4')
                kwargs['colsample_bytree'] = st.slider("Feature Sample Ratio", 0.1, 1.0, 0.8, key='x5')
                kwargs['min_child_weight'] = st.slider("Min Child Weight", 1, 15, 1, key='x6')

            if model_choice in ['Random Forest (ML Regressor)', 'Compare All Models']:
                st.markdown("#### Random Forest Settings")
                kwargs['rf_lag_days'] = st.slider("Lag Features (Days)", 7, 60, 14, key='rf1')
                kwargs['rf_n_estimators'] = st.slider("Trees (n_estimators)", 50, 500, 100, key='rf2')
                kwargs['rf_max_depth'] = st.slider("Max Depth", 2, 20, 10, key='rf3')

            if model_choice in ['Exponential Smoothing', 'Compare All Models']:
                st.markdown("#### Holt-Winters Settings")
                kwargs['trend'] = st.selectbox("Trend Type", ['add', 'mul'], index=0)
                kwargs['damped_trend'] = st.checkbox("Dampen Trend", False)

        with st.expander("✨ Advanced Visuals"):
            show_sma = st.checkbox("Overlay 50 & 200 Day SMA", value=st.session_state.config.get('show_sma', False))
            
        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button("Generate Forecast", type="primary", use_container_width=True)
        
    return {
        'uploaded_file': uploaded_file,
        'model_choice': model_choice,
        'horizon': horizon,
        'confidence': confidence,
        'kwargs': kwargs,
        'show_sma': show_sma,
        'generate_btn': generate_btn
    }

def render_target_selection(numerical_cols, fallback_target):
    with st.sidebar:
        with st.expander("📂 Dataset Configuration", expanded=True):
            target_default_index = numerical_cols.index(fallback_target) if fallback_target in numerical_cols else 0
            target_selection = st.selectbox("Target Feature", numerical_cols, index=target_default_index)
            return target_selection
