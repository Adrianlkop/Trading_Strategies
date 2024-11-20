import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class StatisticalArbitrageStrategy:
    def __init__(self, pairs_list, lookback_period=252, zscore_threshold=2.0, 
                 holding_period=20, stop_loss=0.05, leverage=1.5):
        """
        Initialize the Statistical Arbitrage Strategy
        
        Parameters:
        pairs_list (list): List of pairs of tickers to trade
        lookback_period (int): Period for calculating statistics
        zscore_threshold (float): Z-score threshold for trading signals
        holding_period (int): Maximum holding period for positions
        stop_loss (float): Stop loss percentage
        leverage (float): Leverage ratio
        """
        self.pairs = pairs_list
        self.lookback = lookback_period
        self.zscore_thresh = zscore_threshold
        self.holding_period = holding_period
        self.stop_loss = stop_loss
        self.leverage = leverage
        self.positions = {}
        self.portfolio_value = 1000000  # Initial portfolio value
        
    def fetch_data(self, start_date, end_date):
        """Fetch and prepare price data for all pairs"""
        self.data = {}
        for pair in self.pairs:
            try:
                stock1 = yf.download(pair[0], start=start_date, end=end_date)['Adj Close']
                stock2 = yf.download(pair[1], start=start_date, end=end_date)['Adj Close']
                
                # Ensure both stocks have data
                common_dates = stock1.index.intersection(stock2.index)
                self.data[pair] = pd.DataFrame({
                    'stock1': stock1[common_dates],
                    'stock2': stock2[common_dates]
                })
            except Exception as e:
                print(f"Error fetching data for pair {pair}: {str(e)}")
                
    def calculate_cointegration(self, pair_data):
        """Calculate cointegration statistics for a pair"""
        score, pvalue, _ = coint(pair_data['stock1'], pair_data['stock2'])
        return score, pvalue
    
    def calculate_hedge_ratio(self, pair_data):
        """Calculate optimal hedge ratio using OLS regression"""
        X = pair_data['stock1'].values.reshape(-1, 1)
        y = pair_data['stock2'].values
        beta = np.linalg.lstsq(X, y, rcond=None)[0][0]
        return beta
    
    def calculate_spread(self, pair_data, hedge_ratio):
        """Calculate the spread between two stocks"""
        spread = pair_data['stock1'] - hedge_ratio * pair_data['stock2']
        return spread
    
    def calculate_zscore(self, spread):
        """Calculate z-score of the spread"""
        mean = spread.rolling(window=self.lookback).mean()
        std = spread.rolling(window=self.lookback).std()
        zscore = (spread - mean) / std
        return zscore
    
    def generate_signals(self, pair_data):
        """Generate trading signals based on z-score"""
        # Calculate hedge ratio and spread
        hedge_ratio = self.calculate_hedge_ratio(pair_data)
        spread = self.calculate_spread(pair_data, hedge_ratio)
        zscore = self.calculate_zscore(spread)
        
        # Generate signals
        signals = pd.DataFrame(index=pair_data.index)
        signals['zscore'] = zscore
        signals['long_entry'] = (zscore < -self.zscore_thresh) & (zscore.shift(1) >= -self.zscore_thresh)
        signals['short_entry'] = (zscore > self.zscore_thresh) & (zscore.shift(1) <= self.zscore_thresh)
        signals['exit'] = abs(zscore) < 0.5
        
        return signals, hedge_ratio
    
    def calculate_position_size(self, pair_data, hedge_ratio):
        """Calculate position sizes based on volatility and correlation"""
        # Calculate volatility
        vol1 = pair_data['stock1'].pct_change().std() * np.sqrt(252)
        vol2 = pair_data['stock2'].pct_change().std() * np.sqrt(252)
        
        # Calculate correlation
        corr = pair_data['stock1'].corr(pair_data['stock2'])
        
        # Adjust position size based on volatility and correlation
        position_size1 = self.portfolio_value * self.leverage * (1 / vol1) / len(self.pairs)
        position_size2 = position_size1 * hedge_ratio * (vol1 / vol2)
        
        return position_size1, position_size2
    
    def calculate_returns(self, pair_data, signals, hedge_ratio, pair):
        """Calculate strategy returns for a pair"""
        position_size1, position_size2 = self.calculate_position_size(pair_data, hedge_ratio)
        
        # Initialize position trackers
        position = 0
        entry_price1 = 0
        entry_price2 = 0
        holding_days = 0
        
        returns = pd.DataFrame(index=pair_data.index)
        returns['returns'] = 0.0
        
        for i in range(1, len(pair_data)):
            if position == 0:  # No position
                if signals['long_entry'].iloc[i]:
                    position = 1
                    entry_price1 = pair_data['stock1'].iloc[i]
                    entry_price2 = pair_data['stock2'].iloc[i]
                    holding_days = 0
                elif signals['short_entry'].iloc[i]:
                    position = -1
                    entry_price1 = pair_data['stock1'].iloc[i]
                    entry_price2 = pair_data['stock2'].iloc[i]
                    holding_days = 0
                    
            else:  # In position
                # Calculate returns
                if position == 1:
                    stock1_return = (pair_data['stock1'].iloc[i] - entry_price1) / entry_price1
                    stock2_return = -(pair_data['stock2'].iloc[i] - entry_price2) / entry_price2
                else:
                    stock1_return = -(pair_data['stock1'].iloc[i] - entry_price1) / entry_price1
                    stock2_return = (pair_data['stock2'].iloc[i] - entry_price2) / entry_price2
                
                total_return = stock1_return + stock2_return * hedge_ratio
                returns['returns'].iloc[i] = total_return
                
                # Check exit conditions
                stop_loss_triggered = abs(total_return) > self.stop_loss
                holding_period_reached = holding_days >= self.holding_period
                zscore_exit = signals['exit'].iloc[i]
                
                if stop_loss_triggered or holding_period_reached or zscore_exit:
                    position = 0
                    returns['returns'].iloc[i] = total_return
                else:
                    holding_days += 1
                    
        return returns
    
    def run_backtest(self, start_date, end_date):
        """Run the complete backtest"""
        self.fetch_data(start_date, end_date)
        all_returns = pd.DataFrame()
        
        for pair in self.pairs:
            pair_data = self.data[pair]
            
            # Check for cointegration
            score, pvalue = self.calculate_cointegration(pair_data)
            if pvalue > 0.05:
                print(f"Pair {pair} is not cointegrated (p-value: {pvalue:.4f})")
                continue
                
            signals, hedge_ratio = self.generate_signals(pair_data)
            pair_returns = self.calculate_returns(pair_data, signals, hedge_ratio, pair)
            
            if all_returns.empty:
                all_returns = pair_returns
            else:
                all_returns = all_returns.add(pair_returns, fill_value=0)
        
        # Calculate portfolio metrics
        self.portfolio_metrics(all_returns)
        self.plot_results(all_returns)
        
    def portfolio_metrics(self, returns):
        """Calculate and display portfolio performance metrics"""
        # Basic metrics
        total_return = (1 + returns['returns']).cumprod().iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        daily_std = returns['returns'].std()
        annual_std = daily_std * np.sqrt(252)
        sharpe_ratio = annual_return / annual_std
        
        # Drawdown analysis
        cum_returns = (1 + returns['returns']).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # Print metrics
        print("\nPortfolio Performance Metrics:")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annual Return: {annual_return:.2%}")
        print(f"Annual Volatility: {annual_std:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        
        # Calculate additional risk metrics
        sorted_returns = returns['returns'].sort_values()
        var_95 = sorted_returns.quantile(0.05)
        cvar_95 = sorted_returns[sorted_returns <= var_95].mean()
        
        print(f"Value at Risk (95%): {var_95:.2%}")
        print(f"Conditional VaR (95%): {cvar_95:.2%}")
        
    def plot_results(self, returns):
        """Plot strategy results"""
        plt.style.use('seaborn')
        fig, axes = plt.subplots(3, 1, figsize=(15, 15))
        
        # Cumulative returns
        cum_returns = (1 + returns['returns']).cumprod()
        cum_returns.plot(ax=axes[0], title='Cumulative Returns', color='blue')
        axes[0].set_ylabel('Cumulative Return')
        
        # Drawdown
        drawdown = cum_returns / cum_returns.expanding().max() - 1
        drawdown.plot(ax=axes[1], title='Drawdown', color='red')
        axes[1].set_ylabel('Drawdown')
        
        # Return distribution
        sns.histplot(returns['returns'], kde=True, ax=axes[2])
        axes[2].set_title('Return Distribution')
        axes[2].set_xlabel('Daily Returns')
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Define pairs of related stocks from different sectors
    pairs = [
        ("XOM", "CVX"),  
        ("JPM", "BAC"),  
        ("MSFT", "AAPL"),  
        ("PG", "KO"),  
        ("HD", "LOW") 
    ]
    
    # Initialize and run strategy
    strategy = StatisticalArbitrageStrategy(
        pairs_list=pairs,
        lookback_period=252,  # One year of trading days
        zscore_threshold=2.0,
        holding_period=20,
        stop_loss=0.05,
        leverage=1.5
    )
    
    # Run backtest for the last 5 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    strategy.run_backtest(start_date, end_date)
