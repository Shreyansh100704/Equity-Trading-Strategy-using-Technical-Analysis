import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def MACD(df):
    """ 
    Computes the Moving Average Convergence Divergence (MACD) of a stock
    """
    
    df.insert(column="EMA12",value=df.Close.ewm(span=12).mean(), loc=0)
    df.insert(column="EMA26",value=df.Close.ewm(span=26).mean(), loc=1)
    df.insert(column="MACD",value=df.EMA12-df.EMA26, loc=2)
    df.insert(column="Signal",value=df.MACD.ewm(span=9).mean(), loc=3)
    
    
    
def get_RSI(df):
    """
    Computes the Relative Strength Index (RSI) of a stock
    """
    
    df["MA14"] = df["Adj Close"].rolling(window=14).mean()
    df["Daily Returns"] = df["Adj Close"].pct_change()
    df["Upmove"] = df["Daily Returns"].apply(lambda x: x if x>0 else 0)
    df["Downmove"] = df["Daily Returns"].apply(lambda x: abs(x) if x<0 else 0)
    df["Avg Up"] = df["Upmove"].ewm(span=25).mean()
    df["Avg Down"] = df["Downmove"].ewm(span=25).mean()
    
    df["RS"] = df["Avg Up"]/df["Avg Down"]
    df["RSI"] = df["RS"].apply(lambda x: 100-(100/(x+1)))
    df["RSI_MA"] = df['RSI'].rolling(window=14).mean()
    return df.dropna()


def volume_momentum_strategy(df):
    """ 
    Create curve of Moving Average of Volumes with respect to Volumes for past 10 days 
    """
    
    df['Volume_MA'] = df['Volume'].rolling(window=10).mean()


def annualized_returns (returns, period_per_year = 252):
    n_days = returns.shape[0]
    returns_per_year = ((returns+1).prod()**(period_per_year/n_days))-1
    return returns_per_year

def annualized_volatility(returns):
    sigma = returns.std()
    vol =  sigma*(252**0.5)
    #return round(vol,4)
    return vol






def sharpe_ratio(r, riskfree_rate=0.07327, periods_per_year=252):
    """
        Compute annualized sharpe ratio
        In India 10 Years Government Bond has a 7.327% yield
    """
    rf_per_period = (1+riskfree_rate)**(10/periods_per_year)-1
    annual_excess_returns = annualized_returns(r)
    excess_returns = annual_excess_returns-rf_per_period
    #annual_excess_returns = annualized_returns(excess_returns)
    annual_vol = annualized_volatility(r)
    return round((annual_excess_returns/annual_vol), 2)

def drawdown(returns:pd.Series,df, portfolio_value=10000):
    wealth_index = portfolio_value*(returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.Series(drawdowns)



def strategy(df):
    MACD(df)
    get_RSI(df)
    volume_momentum_strategy(df)
    
    #To get Signals
    df["MACD_Buy_Signal"] = ((df['MACD'] > df['Signal']) & (df['MACD'].shift(1) <= df['Signal'].shift(1)))
    df['RSI_Buy_Signal'] = (df['RSI'] > 60) & (df['RSI'].shift(1) <= 60)
    df["MACD_Sell_Signal"] = (df['MACD'] < df['Signal']) & (df['MACD'].shift(1) >= df['Signal'].shift(1))
    df['RSI_MA_Buy_Signal'] = (df['RSI_MA'] > df['RSI']) & (df['RSI_MA'].shift(1) <= df['RSI'])
    df['RSI_MA_Sell_Signal'] = (df['RSI_MA'] < df['RSI']) & (df['RSI_MA'].shift(1) >= df['RSI'])
    df['Vol_Buy_Signal'] = (df['Volume_MA'] > df['Volume']) & (df['Volume_MA'].shift(1) <= df['Volume'])
    df['Vol_Sell_Signal'] = (df['Volume_MA'] < df['Volume']) & (df['Volume_MA'].shift(1) >= df['Volume'])
    
    df.insert(loc=22, column="Position", value=0)
    
    df["Buy_Signal"] = ((df['MACD'] > df['Signal']) & df['RSI_Buy_Signal'])  |  (df['RSI_MA_Buy_Signal']) | (df['MACD_Buy_Signal'] & df['Vol_Buy_Signal'])
    df["Sell_Signal"] = (df['RSI_MA_Sell_Signal'] | df["MACD_Sell_Signal"]) | (df["MACD_Sell_Signal"] & df['Vol_Sell_Signal'])
    
    for i in range(len(df)):
        if (df["Buy_Signal"][i]==True):
            df["Position"][i]=1
        elif (df["Sell_Signal"][i]==True):
            df["Position"][i]=-1
        else:
            df["Position"][i] = df["Position"].shift(1)[i]
            
    df.insert(column="Opened",value=None, loc=7)
    df.insert(column="Closed",value=None, loc=8)

    
    df['Opened'] = np.where(df['Position'] > df['Position'].shift(1), df['Adj Close'], None)
    df['Closed'] = np.where(df['Position'] < df['Position'].shift(1), df['Adj Close'], None)
    
    print("The strategy is implemented and it is ready for backtesting.")
    
    


def benchmark_rets(benchmark):
    ben_daily_returns = benchmark["Adj Close"].pct_change()
    benchmark_returns = (((1+ben_daily_returns).cumprod())-1)
    return ((benchmark_returns[-1])*100)
    
def backtest(name, start):
    """
    Parameters: name: Give ticker symbol of the stock in string format
                start: Provide the date from which backtesting starts in datetime format
    """

    df = yf.download(name, start=start)
    strategy(df)
    benchmark = yf.download('^NSEI', start=start)
    benchmark_return = benchmark_rets(benchmark)
    today_index = (df.shape[0]) - 1
    closed = df["Closed"].dropna()
    opened = df["Opened"].dropna()
    opened_date = opened.index.tolist()
    opened_price = opened.tolist()
    closed_price = closed.tolist()
    current_price = df["Adj Close"][today_index]
    
    start = df.index[0]
    returns=pd.Series([])
    returns_l = []
    
    #To prevent our portfolio from short selling in first couple of trades
    for i in range(4):
        if closed.index[i]<opened.index[i]:
            closed = closed.drop(closed.index[i])
        else:
            break
    
    
    #To compute returns of every position whether it was closed or open
    for i in range(len(closed)):
        if len(opened) == len(closed): 
            returns[i] = (1+(closed[i] - opened[i])/opened[i])
        if len(opened) == len(closed)+1:
            returns[i] = (1+(closed[i] - opened[i])/opened[i])
            returns_l = returns.tolist()
            returns_l.append(1+(current_price - opened[i])/opened[i])
    
    
    
    #To get the efficiency of strategy as win percentage
    no_of_successful_trades = 0
    for i in range(len(returns)):
        if returns[i]>1:
            no_of_successful_trades += 1
    win = str(round(((no_of_successful_trades/len(closed))*100), 2)) + "%"
    
    
    initial_portfolio_value = 10000
    no_of_stocks = initial_portfolio_value//opened[0]
    initial_buy = opened[0]*no_of_stocks
    initial_sold = closed[0]*no_of_stocks
    rem_amount_initial = initial_portfolio_value - initial_buy

    stocks = pd.Series([(initial_portfolio_value//opened[0]),])
    rem_amount = pd.Series([rem_amount_initial,])
    portfolio_value = pd.Series([(initial_sold + rem_amount[0]),])
    
    
    #To get the number of stocks we can buy based on our portfolio value and also update portfolio value after each trade
    for i in range(1,len(opened)):
            stocks[i] = portfolio_value[i-1]//opened[i]
            if i<len(closed):
                rem_amount[i] = (portfolio_value[i-1]) - (stocks[i]*opened[i])
                portfolio_value[i] = (stocks[i]*closed[i]) + rem_amount[i]
            if i==len(closed):
                rem_amount[i] = (portfolio_value[i-1]) - (stocks[i]*opened[i])
                portfolio_value[i] = (stocks[i]*current_price) + rem_amount[i]

            bought = opened*no_of_stocks
            sold = closed*no_of_stocks
    
    #To get the overall profit in given timeframe
    if len(opened) == len(closed):
        total_return = (returns.prod() -1)
        total_profit = portfolio_value[len(opened)-1] - initial_portfolio_value
    if len(opened) == len(closed)+1:
        returns_series = pd.Series(returns_l)
        total_return = (returns_series.prod() -1)
        
        total_profit = portfolio_value[len(closed)] - initial_portfolio_value
    ret = (total_profit/initial_portfolio_value)*100
    
    profit = [] 
    invested = []
    
    #To maintain the record of individual profits from each trade
    for i in range(len(opened)):
        if i<len(closed):
            profit.append((closed_price[i] - opened_price[i])*stocks[i])
            invested.append(opened_price[i]*stocks[i])
        if i==len(closed):
                profit.append((current_price)-(opened_price[-1])*stocks[i])
                invested.append(opened_price[i]*stocks[i])
                
                
    #To express the dates on which stocks were bought and sold in a more convenient way
    if len(opened) == len(closed):
        buy_dates = pd.to_datetime(opened.index).strftime("%d-%m-%Y")
        sell_dates = pd.to_datetime(closed.index).strftime("%d-%m-%Y")
        
    if len(opened) == len(closed)+1:
        buy_dates = pd.to_datetime(opened_date).strftime("%d-%m-%Y")
        sell_dates = pd.to_datetime(closed.index).strftime("%d-%m-%Y")
        closed_date = closed.index.tolist()
        
        closed_date.append(np.nan)
        closed_date = pd.to_datetime(closed_date[:]).strftime("%d-%m-%Y")
        
        
    
    position = pd.Series([])
    
    #To maintain the record of position of each trade whether it is open aur squared off
    for i in range(len(closed)):
        if opened.index[i] < closed.index[i]:
            position[i] = "Closed"
        if len(closed)<len(opened):
            position[len(closed)] = "Open"
            
            

    
    highest_profit = str(round(((returns.max()-1)*100),2)) + "%"
    highest_loss = str(-round(((returns.min()-1)*100), 2)) + "%"

    
    
    daily_returns = df["Adj Close"].pct_change()
    ann_return = annualized_returns(daily_returns)
    sharpe__ratio = sharpe_ratio(daily_returns)
    drawdowns = drawdown(returns, portfolio_value)
    max_drawdown = drawdowns.min()
    
    
    #To create summary
    pf_summary = [len(opened), len(closed), len(closed)+len(opened), str(round(ret, 2)) + "%", highest_profit, highest_loss, str(round((-max_drawdown*100), 2)) + "%", str(round((ann_return*100),2)) + "%", sharpe__ratio, win, str(round(benchmark_return,2)) + "%" ]
    
    data = ["No. of Buy", "No. of Sell", "Number of trades", "Portfolio Return", "Largest profit making trade", "Largest loss making trade", "Maximum Drawdown", "Annualized return", "Sharpe ratio", "Win", "Benchmark return" ]
    details = pd.DataFrame(data, columns=["Summary"])
    i_summary = pd.DataFrame(pf_summary, columns=["Summary"])
    summary = pd.concat([details, i_summary], axis=1)
    summary.to_csv(name[:-3] + "_Summary.csv")
            
            
    #To create portfolio with records of each trade in given timeframe when all positions were closed
    if len(buy_dates) == len(sell_dates):
        portfolio = pd.DataFrame({
                            "Buy Date": buy_dates,
                            "Stocks": stocks.astype(int),
                            "Sell Date": sell_dates,
                            "Invested Amount (Rs.)": opened.values*stocks,
                            "Profit & Loss (Rs.)": profit,
                            "Returns (%)": (returns - 1)*100,
                            "Opening Balance (Rs.)": portfolio_value.shift(1),
                            "Closing Balance (Rs.)": portfolio_value,
                            "Position": position,
                            

        })
        
        
    #To create portfolio with records of each trade in given timeframe when some positions are open
    if len(opened) == len(closed)+1:
        portfolio = pd.DataFrame({
                            "Buy Date": buy_dates,
                            "Stocks": stocks.astype(int),
                            "Sell Date": closed_date,
                            "Invested Amount (Rs.)": opened_price*stocks,
                            "Profit & Loss (Rs.)": profit,
                            "Returns (%)": (returns_series - 1)*100,
                            "Opening Balance (Rs.)": portfolio_value.shift(1),
                            "Closing Balance (Rs.)": portfolio_value,
                            "Position": position,
        })
        
        
    portfolio["Opening Balance (Rs.)"][0] = initial_portfolio_value
    portfolio.to_csv(name[:-3] + "_Portfolio.csv")
    
    #To plot the graph of Price vs Date and marking buy and sell points on that
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Adj Close'], label='Close Price', alpha=0.7)
    plt.scatter(df.index, df['Opened'], label='Buy Signal', marker='^', color='green', alpha=1)
    plt.scatter(df.index, df['Closed'], label='Sell Signal', marker='v', color='red', alpha=1)
    plt.title('Buy and Sell Signals')
    plt.legend()
    plt.show()
    

name = input("Enter the ticker symbol of stock: ")
start = input("Enter the date from which you want to start backtesting: ")
backtest(name, start)
    