import streamlit as st

ACTIVE_THEME = {
    'primary': '#2563EB', 'bg': '#F1F5F9', 'card_bg': '#FFFFFF', 'text': '#0F172A', 
    'plotly_template': 'plotly_white', 'history_line': '#64748B',
    'prophet': '#2563EB', 'arima': '#059669', 'es': '#7C3AED', 'xgb': '#D97706', 'rf': '#EC4899'
}

def inject_custom_css():
    st.markdown(f"""
        <style>
            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            .block-container {{
                padding-top: 1rem;
                padding-bottom: 2rem;
                max-width: 95%;
            }}
            
            /* Metric Cards Styling */
            div[data-testid="metric-container"] {{
                background-color: {ACTIVE_THEME['card_bg']};
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
                background-color: {ACTIVE_THEME['card_bg']} !important;
                border-bottom: 3px solid {ACTIVE_THEME['primary']};
            }}
            
            /* Generate Button */
            .stButton>button {{
                height: 50px;
                font-size: 18px;
                font-weight: 600;
            }}
        </style>
    """, unsafe_allow_html=True)
