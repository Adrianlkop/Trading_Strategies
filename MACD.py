import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class MACDStrategy:
    def __init__(self, ticker, start_date, end_date=None, interval="1d", 
                 initial_investment=100000, stop_loss=0.05, take_profit=0.1):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.interval = interval
        self.initial_investment = initial_investment
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.df = None
        
    def fetch_data(self):
        """Fetch and prepare data with error handling"""
        try:
            self.df = yf.download(self.ticker, start=self.start_date, 
                                end=self.end_date, interval=self.interval)
            if self.df.empty:
                raise ValueError("No data fetched for the specified ticker and date range")
            return True
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return False

    def add_technical_indicators(self):
        """Add various technical indicators to the dataset"""
        # MACD
        macd = ta.macd(self.df['Close'])
        self.df = pd.concat([self.df, macd], axis=1)
        
        # RSI for confirmation
        self.df['RSI'] = ta.rsi(self.df['Close'], length=14)
        
        # Volatility indicator (Average True Range)
        self.df['ATR'] = ta.atr(self.df['High'], self.df['Low'], self.df['Close'], length=14)
        
        # Volume analysis
        self.df['Volume_MA'] = self.df['Volume'].rolling(window=20).mean()
        self.df['Volume_Ratio'] = self.df['Volume'] / self.df['Volume_MA']

    def generate_signals(self):
        """Generate trading signals with additional filters"""
        # Basic MACD signals
        self.df['Buy_Signal'] = np.where(
            (self.df['MACD_12_26_9'] > self.df['MACDs_12_26_9']) & 
            (self.df['MACD_12_26_9'].shift(1) <= self.df['MACDs_12_26_9'].shift(1)) &
            (self.df['RSI'] > 30) &  # Oversold condition
            (self.df['Volume_Ratio'] > 1.2),  # Above average volume
            1, 0
        )
        
        self.df['Sell_Signal'] = np.where(
            (self.df['MACD_12_26_9'] < self.df['MACDs_12_26_9']) & 
            (self.df['MACD_12_26_9'].shift(1) >= self.df['MACDs_12_26_9'].shift(1)) &
            (self.df['RSI'] < 70),  # Overbought condition
            -1, 0
        )
        
        self.df['Signal'] = self.df['Buy_Signal'] + self.df['Sell_Signal']
        self.df['Position'] = self.df['Signal'].replace(to_replace=0, method='ffill')

    def calculate_returns(self):
        """Calculate strategy returns with stop loss and take profit"""
        self.df['Entry_Price'] = np.nan
        self.df.loc[self.df['Buy_Signal'] == 1, 'Entry_Price'] = self.df['Close']
        self.df['Entry_Price'] = self.df['Entry_Price'].ffill()

        # Initialize stop loss and take profit levels
        self.df['Stop_Loss'] = np.where(self.df['Position'] == 1, 
                                      self.df['Entry_Price'] * (1 - self.stop_loss), np.nan)
        self.df['Take_Profit'] = np.where(self.df['Position'] == 1,
                                        self.df['Entry_Price'] * (1 + self.take_profit), np.nan)

        # Calculate daily returns
        self.df['Daily_Return'] = self.df['Close'].pct_change()
        
        # Adjust position based on stop loss and take profit
        for i in range(1, len(self.df)):
            if self.df['Position'].iloc[i-1] == 1:
                if self.df['Low'].iloc[i] <= self.df['Stop_Loss'].iloc[i-1]:
                    self.df.loc[self.df.index[i], 'Position'] = 0
                    self.df.loc[self.df.index[i], 'Signal'] = -1
                elif self.df['High'].iloc[i] >= self.df['Take_Profit'].iloc[i-1]:
                    self.df.loc[self.df.index[i], 'Position'] = 0
                    self.df.loc[self.df.index[i], 'Signal'] = -1

        self.df['Strategy_Return'] = self.df['Position'].shift(1) * self.df['Daily_Return']
        self.df['Cumulative_Return'] = (1 + self.df['Strategy_Return']).cumprod() - 1
        self.df['Net_Return'] = self.initial_investment * (1 + self.df['Cumulative_Return'])

    def calculate_metrics(self):
        """Calculate and return performance metrics"""
        # Basic metrics
        total_return = (self.df['Net_Return'].iloc[-1] - self.initial_investment) / self.initial_investment
        buy_triggers = self.df['Buy_Signal'].sum()
        sell_triggers = abs(self.df['Sell_Signal'].sum())
        
        # Drawdown analysis
        self.df['Peak'] = self.df['Net_Return'].cummax()
        self.df['Drawdown'] = (self.df['Net_Return'] - self.df['Peak']) / self.df['Peak']
        max_drawdown = self.df['Drawdown'].min()
        
        # Trade analysis
        trades = self.df[self.df['Signal'] != 0].copy()
        trades['Trade_Return'] = trades['Net_Return'].shift(-1) - trades['Net_Return']
        
        winning_trades = trades[trades['Trade_Return'] > 0]
        losing_trades = trades[trades['Trade_Return'] < 0]
        
        win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
        average_win = winning_trades['Trade_Return'].mean() if not winning_trades.empty else 0
        average_loss = losing_trades['Trade_Return'].mean() if not losing_trades.empty else 0
        
        # Risk metrics
        sharpe_ratio = np.sqrt(252) * (self.df['Strategy_Return'].mean() / self.df['Strategy_Return'].std())
        
        return {
            'Total Return': f"{total_return:.2%}",
            'Number of Trades': len(trades),
            'Win Rate': f"{win_rate:.2%}",
            'Average Win': f"${average_win:.2f}",
            'Average Loss': f"${average_loss:.2f}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Buy Triggers': buy_triggers,
            'Sell Triggers': sell_triggers
        }

    def plot_results(self):
        """Plot strategy results with improved visualization"""
        plt.style.use('seaborn')
        fig, axs = plt.subplots(5, 1, figsize=(15, 20), gridspec_kw={'height_ratios': [3, 2, 2, 2, 3]})
        
        # Price and Signals
        axs[0].plot(self.df.index, self.df['Close'], label='Close Price', color='blue', alpha=0.7)
        axs[0].scatter(self.df.index[self.df['Buy_Signal'] == 1], 
                      self.df['Close'][self.df['Buy_Signal'] == 1], 
                      marker='^', color='green', label='Buy Signal', s=100)
        axs[0].scatter(self.df.index[self.df['Sell_Signal'] == -1], 
                      self.df['Close'][self.df['Sell_Signal'] == -1], 
                      marker='v', color='red', label='Sell Signal', s=100)
        axs[0].set_title('Price Action and Signals')
        axs[0].legend()
        
        # MACD
        axs[1].plot(self.df.index, self.df['MACD_12_26_9'], label='MACD', color='blue')
        axs[1].plot(self.df.index, self.df['MACDs_12_26_9'], label='Signal Line', color='red')
        axs[1].bar(self.df.index, self.df['MACDh_12_26_9'], label='Histogram', color='grey', alpha=0.3)
        axs[1].set_title('MACD Indicator')
        axs[1].legend()
        
        # RSI
        axs[2].plot(self.df.index, self.df['RSI'], label='RSI', color='purple')
        axs[2].axhline(y=70, color='r', linestyle='--', alpha=0.5)
        axs[2].axhline(y=30, color='g', linestyle='--', alpha=0.5)
        axs[2].set_title('RSI Indicator')
        axs[2].legend()
        
        # Volume
        axs[3].bar(self.df.index, self.df['Volume'], label='Volume', color='blue', alpha=0.3)
        axs[3].plot(self.df.index, self.df['Volume_MA'], label='Volume MA', color='red')
        axs[3].set_title('Volume Analysis')
        axs[3].legend()
        
        # Strategy Returns
        axs[4].plot(self.df.index, self.df['Net_Return'], label='Strategy Return', color='green')
        axs[4].plot(self.df.index, self.df['Peak'], label='Peak Value', color='blue', linestyle='--', alpha=0.5)
        axs[4].fill_between(self.df.index, self.df['Net_Return'], self.df['Peak'], 
                           color='red', alpha=0.3, label='Drawdown')
        axs[4].set_title('Strategy Performance')
        axs[4].legend()
        
        plt.tight_layout()
        plt.show()

    def run_strategy(self):
        """Run the complete strategy"""
        if not self.fetch_data():
            return None
            
        self.add_technical_indicators()
        self.generate_signals()
        self.calculate_returns()
        metrics = self.calculate_metrics()
        
        return metrics

# Example usage
if __name__ == "__main__":
    strategy = MACDStrategy(
        ticker="AAPL",
        start_date="2023-01-01",
        end_date="2024-01-01",
        initial_investment=100000,
        stop_loss=0.05,
        take_profit=0.10
    )
    
    metrics = strategy.run_strategy()
    if metrics:
        print("\nStrategy Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
            
        strategy.plot_results()
