# Libraries
#--------------------------------------------------------------------------------------------------------------------------------------------
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import altair as alt
import plotly.express as px

# Import the statsmodels module for regression and the adfuller function
# Import statsmodels.formula.api
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
#--------------------------------------------------------------------------------------------------------------------------------------------

# PART 1: TAKING INPUTS AND INITIALIZATION
#=========================================
# Streamlit UI title
st.title("Pairs Watch")

st.write(":grey[Pairs Watch app is a quantitative finance tool that helps users analyze potential pairs trading opportunities.]")
st.write("")


#User selections

# Stock selection list
ticker_options = ["SPY", "QQQ", "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "AMD", "META", "NFLX", "NKE", "ADDYY", "BMW.DE", "MBG.DE","VOW.DE", "PAH3.DE","AML.L","RR.L", "QUAL", "MTUM", "USMV", "VLUE"]

# User inputs
col1, col2 = st.columns(2)
with col1:
    ticker1 = st.selectbox("Select first stock", ticker_options, index=0)

available_tickers = [t for t in ticker_options if t != ticker1]
with col2:
    ticker2 = st.selectbox("Select second stock", available_tickers, index=0)

# Date range selection
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Select Start Date", datetime.today() - timedelta(days=365))
with col2:
    end_date = st.date_input("Select End Date",  datetime.today())


#--------------------------------------------------------------------------------------------------------------------------------------------
# **Validation Checks**
error_flag = False  # Flag to control execution

if end_date < start_date:
    st.error("🚨 End Date cannot be earlier than Start Date. Please select a valid range.")
    error_flag = True

if start_date > datetime.today().date() or end_date > datetime.today().date():
    st.error("🚨 Dates cannot be in the future. Please select a valid range.")
    error_flag = True
#--------------------------------------------------------------------------------------------------------------------------------------------

if not error_flag:
    # Fetch stock data
    #-------------------------------------------------------------------------
    data = yf.download([ticker1, ticker2], start=start_date, end=end_date + timedelta(days=1))["Close"]
    
    returns = data.pct_change().dropna()
    cm_returns = (returns + 1).cumprod() - 1
    
    # Ensure correct column names
    cm_returns.columns = data.columns  # This preserves the order returned by yfinance
    
    # Plot cumulative returns
    st.subheader("Market Summary")
    #-------------------------------------------------------------------------
    # Fetch the last traded price (close) for each stock
    last_close_ticker1 = data[ticker1].iloc[-1]
    last_close_ticker2 = data[ticker2].iloc[-1]
    
    # Calculate the percentage change for each stock
    pct_change_ticker1 = returns[ticker1].iloc[-1] * 100
    pct_change_ticker2 = returns[ticker2].iloc[-1] * 100
    
    
    # Display metrics in two columns without labels
    col1, col2 = st.columns(2)
    
    col1.metric(f"{ticker1}", f"${last_close_ticker1:.2f}", f"{pct_change_ticker1:.2f}%")
    col2.metric(f"{ticker2}", f"${last_close_ticker2:.2f}", f"{pct_change_ticker2:.2f}%")
        
    # Reshape data for Plotly
    cm_returns_melted = cm_returns.reset_index().melt(id_vars="Date", var_name="Stock", value_name="Cumulative Return")
    
    # Define custom colors
    color_map = {
        cm_returns.columns[0]: "#fb580d",  # Fiery Orange
        cm_returns.columns[1]: "#5cc8e2",  # Electric Blue
    }
    
    # Create Plotly figure
    fig = px.line(
        cm_returns_melted,
        x="Date",
        y="Cumulative Return",
        color="Stock",
        title="Cumulative Returns",
        color_discrete_map=color_map
    )
    
    # Show chart in Streamlit
    st.plotly_chart(fig)
    #--------------------------------------------------------------------------------------------------------------------------------------------
    
    # PART 2: CHECKS
    #===============
    st.divider()
    st.subheader("Cointegration")
    #-------------------------------------------------------------------------
    
    # Create Altair Scatter Plot
    #-------------------------------------------------------------------------
    scatter_plot = alt.Chart(returns.reset_index()).mark_circle(size=60).encode(
        x=alt.X(f"{ticker1}:Q", title=f"{ticker1} Daily Return", axis=alt.Axis(format=".2%")),
        y=alt.Y(f"{ticker2}:Q", title=f"{ticker2} Daily Return", axis=alt.Axis(format=".2%")),
        tooltip=["Date:T", f"{ticker1}:Q", f"{ticker2}:Q"]
    ).properties(
        title="Daily Returns Scatter Plot"
    )
    
    # Show the plot in Streamlit
    st.altair_chart(scatter_plot, use_container_width=True)
    
    # Display summary
    #-------------------------------------------------------------------------
    
    # OLS Regression
    #-------------------------------------------------------------------------
    # Prepare independent (X) and dependent (Y) variables
    X = returns[ticker1]  # Predictor (Independent variable)
    Y = returns[ticker2]  # Response (Dependent variable)
    
    # Add constant term for intercept
    X = sm.add_constant(X)
    
    # Run OLS regression
    model = sm.OLS(Y, X).fit()
    
    # Display regression summary in Streamlit
    #-------------------------------------------------------------------------
    st.markdown("OLS Regression Results")
    #st.text(model.summary())
    
    # Extract key regression metrics
    r_squared = model.rsquared
    beta = model.params[ticker1]
    
    # Compute ADF test on residual (spread)
    spread = returns[ticker2] - beta * returns[ticker1]
    adf_pvalue = adfuller(spread)[1]
    
    # Display results
    # Create three columns
    col1, col2, col3 = st.columns(3)
    
    # Display R-squared in the first column
    col1.metric(label="R-Squared", value=f"{r_squared:.3f}")
    # Use Streamlit's built-in color formatting
    col1.markdown(f"*:grey[Relationship Strength: {ticker1} explains {r_squared*100:.0f}% of the variation in {ticker2}]*")
    
    # Display OLS Beta in the second column
    col2.metric(label="OLS Beta", value=f"{beta:.3f}")
    col2.write(f"*:grey[Effect of {ticker1} on {ticker2}: A 1-unit increase in {ticker1} is associated with a {beta:.2f} increase in {ticker2}.\n]*")
    
    # Display ADF Test P-Value in the third column
    col3.metric(label="ADF P-Value", value=f"{adf_pvalue:.3f}")
    
    if adf_pvalue < 0.05:
        col3.write(f"*:grey[✅ The spread is **stationary** (p-value: {adf_pvalue:.3f})] :grey[. It means the SPREAD has a constant mean and variance over time, suggesting a stable relationship that is likely to revert to its average.]*")
    else:
        col3.write(f"*❌ The spread is **non-stationary** (p-value: {adf_pvalue:.3f})*")
    #--------------------------------------------------------------------------------------------------------------------------------------------
    
    # PART 3: COINTEGRATION RESIDUALS
    #================================
    st.write(" ")
    st.write(" ")
    st.markdown("### Cointegration residuals")
    #-------------------------------------------------------------------------
    
    # Compute the cointegration residuals
    #-------------------------------------------------------------------------
    df_coint = model.resid
    
    # Convert residuals to DataFrame for plotting
    df_coint_plot = pd.DataFrame({"Time": returns.index, "Residuals": df_coint})
    
    # Create Altair line chart
    coint_chart = alt.Chart(df_coint_plot).mark_line().encode(
        x=alt.X("Time:T", title="Time"),
        y=alt.Y("Residuals:Q", title="Cointegration Residuals")
    ).properties(
        title="Cointegration Residuals Over Time"
    )
    
    
    # Final plot
    #-------------------------------------------------------------------------

    
    # User input for percentile selection
    percentile_options = [0.3, 0.62, 1, 3, 5]

    # Create three columns
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_percentile = st.selectbox("Select Percentile", percentile_options, index=2)
    
    # Compute actual percentile values
    lower_percentile = np.quantile(df_coint, selected_percentile / 100)  # Convert to fraction
    upper_percentile = np.quantile(df_coint, 1 - selected_percentile / 100)  # Convert to fraction
    
    # Add percentile values to DataFrame for plotting
    df_coint_plot["Lower Percentile"] = lower_percentile
    df_coint_plot["Upper Percentile"] = upper_percentile
    
    # Set "Time" as the index for Streamlit's line chart
    df_coint_plot.set_index("Time", inplace=True)
    
    # Display in Streamlit
    st.line_chart(df_coint_plot)
    #--------------------------------------------------------------------------------------------------------------------------------------------
    #-Part 2
    #--------------------------------------------------------------------------------------------------------------------------------------------
    #st.write("⚠️ Work in progress beyond this ⚠️")
    st.divider()
    #-------------------------------------------------------------------------
    st.subheader("Relative Risk")
    #st.markdown("Rolling Volatility Ratio")


    # Calculate the difference in days between the start and end dates
    date_diff = (end_date - start_date).days


    # Create three columns
    col1, col2, col3 = st.columns(3)
    with col1:
        # Numeric input for rolling window period selection (in days)
        rolling_window = st.number_input("Enter Rolling Window Period (in days)", min_value=1, max_value=365, value=30)
    

    # Check if the rolling window period is greater than the date difference
    if rolling_window > date_diff:
        st.error("Rolling window period cannot be greater than the difference between the start and end dates.")
    else:
        # Calculate rolling volatility for each stock
        rolling_volatility_ticker1 = returns[ticker1].rolling(window=rolling_window).std().dropna()
        rolling_volatility_ticker2 = returns[ticker2].rolling(window=rolling_window).std().dropna()
    
        # Calculate rolling volatility ratio (ticker1 / ticker2)
        rolling_volatility_ratio = rolling_volatility_ticker1 / rolling_volatility_ticker2
    
        # Drop NaN values to start the chart from where the data is available
        rolling_volatility_ratio = rolling_volatility_ratio.dropna()
    
        # Create a DataFrame for plotting
        rolling_volatility_df = pd.DataFrame({
            'Date': rolling_volatility_ratio.index,
            'Rolling Volatility Ratio': rolling_volatility_ratio.values
        })
    
        # Create Plotly figure for rolling volatility ratio
        fig_volatility_ratio = px.line(
            rolling_volatility_df,
            x='Date',
            y='Rolling Volatility Ratio',
            title=f"Rolling Volatility Ratio ({ticker1} / {ticker2})",
            labels={'Rolling Volatility Ratio': 'Volatility Ratio'}
        )
    
        # Show chart in Streamlit
        st.plotly_chart(fig_volatility_ratio)

        st.write("***:grey[A sudden spike (in either direction) suggests market stress or some event affecting one more than the other.]***")
        
