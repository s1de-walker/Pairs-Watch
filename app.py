import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import altair as alt
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

st.title("Pairs Watch")
st.write("Pairs Watch app is a quantitative finance tool that helps users analyze potential pairs trading opportunities.")
st.divider()

# User selections
ticker_options = ["SPY", "QQQ", "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "AMD", "META", "NFLX", "NKE", "ADDYY"]

col1, col2 = st.columns(2)
with col1:
    ticker1 = st.selectbox("Select first stock", ticker_options, index=0)

available_tickers = [t for t in ticker_options if t != ticker1]
with col2:
    ticker2 = st.selectbox("Select second stock", available_tickers, index=0)

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Select Start Date", datetime.today() - timedelta(days=180))
with col2:
    end_date = st.date_input("Select End Date", datetime.today())

# Validation checks
error_flag = False

if end_date < start_date:
    st.error("🚨 End Date cannot be earlier than Start Date. Please select a valid range.")
    error_flag = True

if start_date > datetime.today().date() or end_date > datetime.today().date():
    st.error("🚨 Dates cannot be in the future. Please select a valid range.")
    error_flag = True

if not error_flag:
    data = yf.download([ticker1, ticker2], start=start_date, end=end_date)["Close"]
    returns = data.pct_change().dropna()
    cm_returns = (returns + 1).cumprod() - 1
    cm_returns.columns = data.columns

    cm_returns_melted = cm_returns.reset_index().melt(id_vars="Date", var_name="Stock", value_name="Cumulative Return")
    color_map = {
        cm_returns.columns[0]: "#FF4500",
        cm_returns.columns[1]: "#0074D9",
    }
    fig = px.line(
        cm_returns_melted,
        x="Date",
        y="Cumulative Return",
        color="Stock",
        title="Cumulative Returns",
        color_discrete_map=color_map
    )
    st.plotly_chart(fig)
    st.divider()

    scatter_plot = alt.Chart(returns.reset_index()).mark_circle(size=60).encode(
        x=alt.X(f"{ticker1}:Q", title=f"{ticker1} Daily Return", axis=alt.Axis(format=".2%")),
        y=alt.Y(f"{ticker2}:Q", title=f"{ticker2} Daily Return", axis=alt.Axis(format=".2%")),
        tooltip=["Date:T", f"{ticker1}:Q", f"{ticker2}:Q"]
    ).properties(title="Daily Returns Scatter Plot")
    st.altair_chart(scatter_plot, use_container_width=True)

    X = sm.add_constant(returns[ticker1])
    Y = returns[ticker2]
    model = sm.OLS(Y, X).fit()
    r_squared = model.rsquared
    beta = model.params[ticker1]
    spread = returns[ticker2] - beta * returns[ticker1]
    adf_pvalue = adfuller(spread)[1]

    col1, col2, col3 = st.columns(3)
    col1.metric(label="R-Squared", value=f"{r_squared:.3f}")
    col2.metric(label="OLS Beta", value=f"{beta:.3f}")
    col3.metric(label="ADF P-Value", value=f"{adf_pvalue:.3f}")

    if adf_pvalue < 0.05:
        col3.write(f"✅ The spread is **stationary** (p-value: {adf_pvalue:.3f})")
    else:
        col3.write(f"❌ The spread is **non-stationary** (p-value: {adf_pvalue:.3f})")

    st.divider()
    st.subheader("Cointegration residuals")
    df_coint = model.resid
    df_coint_plot = pd.DataFrame({"Time": returns.index, "Residuals": df_coint})
    coint_chart = alt.Chart(df_coint_plot).mark_line().encode(
        x=alt.X("Time:T", title="Time"),
        y=alt.Y("Residuals:Q", title="Cointegration Residuals")
    ).properties(title="Cointegration Residuals Over Time")
    st.altair_chart(coint_chart, use_container_width=True)

    percentile_options = [1, 3, 5]
    selected_percentile = st.selectbox("Select Percentile", percentile_options, index=1)
    lower_percentile = np.quantile(df_coint, selected_percentile / 100)
    upper_percentile = np.quantile(df_coint, 1 - selected_percentile / 100)
    df_coint_plot["Lower Percentile"] = lower_percentile
    df_coint_plot["Upper Percentile"] = upper_percentile
    df_coint_plot.set_index("Time", inplace=True)
    st.line_chart(df_coint_plot)
