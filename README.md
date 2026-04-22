# Bitcoin Price Forecasting Portal

A premium Streamlit application designed for financial time-series analysis to forecast Bitcoin (BTC) price trends. This project features a robust data pipeline, professional UI aesthetics (dark mode, glassmorphism), and rigorous time-series algorithms explicitly configured for high volatility logic.

## Dataset
This app is tested using standard Kaggle datasets, specifically:
**[Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)**

Note: The app is capable of auto-detecting timestamps and downsampling large, minute-by-minute interval datasets to daily prices so that browser memory remains stable.

## How the Models Handle Crypto-Volatility
Cryptocurrencies, and Bitcoin in particular, exhibit high volatility with frequent regime changes, price shocks, and dramatic trend shifts.

1. **Facebook Prophet**: 
Prophet excels in volatile environments when tuned correctly. We have adjusted the `changepoint_prior_scale` hyperparameter from its default (0.05) to `0.15`. This explicitly tells the algorithm to increase its flexibility in detecting trend shifts—making it highly suited for sudden bull runs or crashes typical in cryptographic assets. It natively models heavy seasonality.

2. **Auto-ARIMA (pmdarima)**:
Instead of assuming fixed autoregressive orders `(p,d,q)`, we employ an automated ARIMA search algorithm. The model finds the optimal integration (`d`) to stationarise the volatile series and tests multiple combination boundaries. This adaptive modeling guarantees that sudden stochastic shifts are captured in the parameter tuning, avoiding static baseline errors.

## Installation & Setup

1. **Clone and Setup Virtual Environment:**
```bash
git clone <your-repo>
cd <repo-folder>
python3 -m venv venv
source venv/bin/activate
```

2. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the Application:**
```bash
streamlit run app.py
```

Upload your Kaggle CSV to the sidebar to begin forecasting.

## Required Tech Stack
-   **Streamlit**: Frontend presentation and state mapping.
-   **Plotly**: Interactive visualization rendering.
-   **Prophet / Statsmodels (pmdarima)**: Heavy lifting core engines.
-   **Pandas / Scikit-learn**: Data cleaning, metric calculation, and subset management.
