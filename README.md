<div align="center">
  <h1>🏦 Bitcoin Price Forecasting Portal</h1>
  <p><i>Advanced Time-Series Analysis & Machine Learning Dashboard for Cryptocurrencies</i></p>

  ![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
  ![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0-FF4B4B)
  ![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-orange)
  ![Prophet](https://img.shields.io/badge/Prophet-Ready-blueviolet)
  ![License](https://img.shields.io/badge/License-MIT-green)
</div>

---

A premium Streamlit application designed for sophisticated financial time-series analysis to forecast Bitcoin (BTC) price trends. This project features a robust automated data pipeline, professional UI aesthetics, and rigorous machine learning algorithms configured specifically to handle high-volatility financial data.
Data source used for development: [Comprehensive BTC/USD 1M Data](https://www.kaggle.com/datasets/imranbukhari/comprehensive-btcusd-1m-data)

## 🌟 Key Features

- **Dynamic Data Ingestion**: Automatically detects Kaggle-style CSV formatting, standardizing timestamps and downsampling large minute-by-minute interval datasets to robust daily frequencies.
- **Zero-Leakage Feature Engineering**: Implements rigorous moving averages and rolling standard deviations derived *strictly* from lagged target data, ensuring models never look into the future during multi-step forecasting.
- **Expert Configuration Panel**: Total granular control over algorithmic hyperparameters, from ARIMA seasonal orders to XGBoost tree configurations.
- **"Compare All Models" Mode**: Execute all forecasting engines simultaneously and visualize overlapping confidence bounds and consensus trajectories on a single interactive Plotly canvas.

## 🧠 Forecasting Engines

The portal utilizes multiple parallel algorithms tailored to handle cryptocurrency volatility:

1. **XGBoost (Extreme Gradient Boosting)**
   - Utilizes advanced target-derived rolling statistics.
   - Exposes extensive hyperparameters (`subsample`, `colsample_bytree`, `min_child_weight`) for regularization against market noise.
2. **Random Forest (ML Regressor)**
   - A robust bagging algorithm that naturally resists overfitting and handles non-linear financial patterns.
3. **Facebook Prophet**
   - Tuned `changepoint_prior_scale` for maximum flexibility in detecting rapid trend shifts and sudden crypto market cap surges.
4. **Auto-ARIMA (PMDarima) & Seasonal ARIMA**
   - Automatically optimizes autoregressive integrations `(p,d,q)` or allows for deep manual configuration of Seasonal components `(P,D,Q,m)`.
5. **Exponential Smoothing (Holt-Winters)**
   - Captures additive/multiplicative trends with damped forecasting modes to stabilize long-horizon predictions.

## 📂 Project Structure

```text
.
├── app.py                      # Main Streamlit dashboard application
├── requirements.txt            # Dependency definitions
├── README.md                   # Project documentation
├── src/                        # Core backend package
│   ├── data_processor.py       # Data cleaning and technical indicator logic
│   ├── prophet_model.py        # Prophet architecture
│   ├── arima_model.py          # ARIMA engine
│   ├── holtwinters_model.py    # Holt-Winters smoothing
│   ├── xgboost_model.py        # XGBoost ML engine
│   ├── random_forest_model.py  # Random Forest ML engine
│   ├── metrics.py              # Performance evaluation utilities
│   └── __init__.py             # Package initializer
├── notebooks/                  # Experimental Jupyter Notebooks
├── docs/                       # Reference documents (ignored in git)
└── data/                       # Local dataset directory
```

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/bitcoin-forecasting-portal.git
   cd bitcoin-forecasting-portal
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Dashboard:**
   ```bash
   streamlit run app.py
   ```

## 📊 Usage Guide

1. Launch the app and upload a historical Bitcoin price CSV (e.g., from Kaggle).
2. Use the **Dataset Configuration** sidebar to select your target feature (typically `Close` or `Weighted_Price`).
3. Set your **Forecast Horizon** (e.g., 30 Days) and **Confidence Bounds**.
4. Choose a single engine or **Compare All Models**.
5. Unfurl the **Expert Configurations** panel to fine-tune specific algorithmic behaviors.
6. Click **Generate Forecast** to run the pipeline and render the interactive intelligence dashboard.

---
*Built with ❤️ for Time-Series Analysis & Financial Data Science.*
