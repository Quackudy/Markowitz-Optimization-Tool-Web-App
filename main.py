import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



plt.rcParams['figure.facecolor'] = '#0e1117'  
plt.rcParams['axes.facecolor'] = '#0e1117'
plt.rcParams['axes.edgecolor'] = 'white'  
plt.rcParams['axes.labelcolor'] = 'white' 
plt.rcParams['xtick.color'] = 'white'  
plt.rcParams['ytick.color'] = 'white' 
plt.rcParams['legend.facecolor'] = '#1e2530'  
plt.rcParams['legend.edgecolor'] = 'white'  
plt.rcParams['legend.labelcolor'] = 'white' 


# Web Title
st.title('Portfolio Optimization Tool')

##################################################
# Clear cache to reload data after adjust input
def clear_cache():
    fetch_data.clear()

# Configure stock data
def configure() :
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        period = st.selectbox(
            "Select period:",
            ["1y", "2y", "5y", "10y"],
            key="period",
            index=2,
            on_change=clear_cache
        )

    with col2:
        interval = st.selectbox(
            "Select interval:",
            ["1d", "5d", "1mo", "3mo"],
            key="interval",
            index=2,
            on_change=clear_cache
        )

    with col3:
        log_mode = st.checkbox("Log scale", key="log", help="Use logarithmic scale")
        
        # Add some space to checkbox
        st.markdown(
        """
        <style>
        .st-key-log {
            margin-top: 35px;  /* Adjust spacing here */
        }
        </style>
        """,
        unsafe_allow_html=True
        )

    return period, interval, log_mode


# Fetch data using yfinance
@st.cache_data
def fetch_data(ticker):
    ticker = ticker.strip()
    try:
        if ticker:
            # Download data from yfinance
            data = yf.download(ticker, period=period, interval=interval)
            if data.empty:  # Check if the Dataframe is empty
                st.warning(f"No data found for ticker: {ticker}")
                return None
            return data['Close']  # Return the 'Close' prices
    except Exception as e:
        st.error(f"Error fetching data for ticker: {ticker}. Details: {str(e)}")
        return None
##################################################
    
#Input tickers
tickers_input = st.text_input("Enter comma-separated stock tickers (Start with only 3-4 stocks):", "AAPL,JNJ,GLD,BND")
tickers = tickers_input.split(',')


period, interval, log = configure()

data = {} # Normal Price (For Visualization)
data_log = {} # Log Scaled Price (For Visualization)
data_percent_change = {} # Price Change (For Optimization)

# Fetch price data for each ticker
for ticker in tickers:
    prices = fetch_data(ticker.strip())
    data_percent_change[ticker] = prices
    if prices is not None:
        prices_log = np.log10(prices)
        data[ticker] = ((prices - prices.iloc[0]) / prices.iloc[0]) * 100  # Percent change
        data_log[ticker] = ((prices_log - prices_log.iloc[0]) / prices_log.iloc[0]) * 100  # Percent change (log)

try:
    # Concat into single DataFrame and calculate percent change
    data_percent_change = pd.concat(data_percent_change, axis=1).pct_change().dropna()

    #Log scale?
    if not log:
        df = pd.concat(data.values(), axis=1)
        df.columns = data.keys()  # Set column names to ticker names
        st.line_chart(df, x_label="Date", y_label="Percent Change (%)")
    else:
        df_log = pd.concat(data_log.values(), axis=1)
        df_log.columns = data_log.keys()  # Set column names to ticker names
        st.line_chart(df_log, x_label="Date", y_label="Percent Change (Log Scale, %)")

except Exception as e:
    st.error(f"Error plotting data. Details: {str(e)}")

##################################################

st.markdown("## Visualize Correlation Matrix")

col = st.columns([2,6])
with col[0]:
    corr_method = st.selectbox("Select correlation method:", ["pearson", "kendall", "spearman"], key="corr_method")
with col[1]:
    slide = st.slider("A correlation more than selected threshold will be filtered out:", 0.0, 1.0, 0.7, 0.05, key="filter")

col = st.columns([1,10,1])
with col[1] :
    corr = data_percent_change.corr(method=corr_method)

    # Create masks
    triMask = np.triu(np.ones_like(corr, dtype=bool))  # Upper triangle mask
    filterMask = corr > slide  # Threshold filter mask
    mixMask = triMask | filterMask  # Combine masks

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap='crest', mask=mixMask, annot=True, fmt=".2f", ax=ax)

    st.pyplot(fig)

#################################################

st.markdown("## Markowitz Portfolio Optimization")

@st.cache_data
def optimize(sample):

    month = interval[:-2]  # Find how many months per interval
    day = interval[:-1]  # Find how many days per interval

    # Calculate the annualized return and covariance matrix
    if interval.endswith('d'):  # Daily data
        avgRet = data_percent_change.mean() * 252 / int(day)  # 252 is the number of trading days
        cov_matrix = data_percent_change.cov() * 252 / int(day) 
    elif interval.endswith('mo'):  # Monthly data
        avgRet = data_percent_change.mean() * 12  / int(month)  
        cov_matrix = data_percent_change.cov() * 12  / int(month) 

    n = data_percent_change.shape[1]  # Number of stocks
    #sample = 500*n**3  # Number of sample for Monte Carlo simulation

    results = np.zeros((sample, 3))  # Store results

    # Track max value
    maxSharpe = 0 # Maximum Sharpe ratio 
    maxVol = 0 # Maximum Volatility
    maxRet = 0 # Maximum Return
    maxWeights = np.zeros(n)

    # Monte Carlo simulation for random portfolios
    for i in range(sample):
        weights = np.random.rand(n) 
        weights /= np.sum(weights)  

        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  
        ret = avgRet @ weights 
        sharpe = (ret - 0.03) / vol  # Given risk-free rate of 3%


        if sharpe > maxSharpe:
            maxSharpe = sharpe
            maxWeights = weights
            maxVol = vol
            maxRet = ret

        results[i, 0] = ret
        results[i, 1] = vol
        results[i, 2] = sharpe


    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(results[:, 1], results[:, 0], c=results[:, 2], cmap='viridis') #Plot max sharpe ratio as a red cross
    sc = ax.scatter(maxVol, maxRet, marker='x',c='red', label='Max Sharpe Ratio')
 
    ax.set_title('Visualization of efficient frontier')
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    fig.colorbar(sc, ax=ax, label='Sharpe Ratio')
    ax.legend()

    return fig, maxSharpe, maxWeights

# Button to trigger optimization

col = st.columns([2,11])

with col[0] :
    optimize_button = st.button('Optimize', on_click=optimize.clear())
with col[1] :
    sample_num = st.select_slider("Choose number of samples for Monte Carlo simulation:", options=[1000, 3000, 5000, 10000, 30000, 50000, 100000, 300000], value=10000, key="sample_num")

if optimize_button :
    fig, max_sharpe_ratio, optimize_weight = optimize(sample_num)

    st.pyplot(fig)


    optimize_weight = pd.DataFrame(optimize_weight, index=tickers, columns=['Weight'])
    
    st.markdown(
        "<div style='text-align: center;'>The weights shown below is the optimal weights for each stock in the portfolio</div>", 
        unsafe_allow_html=True
    )

    st.markdown(
        "<h3 style='text-align: center;'>Sharpe Ratio: {:.2f}</h3>".format(max_sharpe_ratio), 
        unsafe_allow_html=True
    )

    col = st.columns([3,3,3])
    with col[1]:
        st.table(optimize_weight)



