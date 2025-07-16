import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
import warnings
warnings.filterwarnings('ignore')

class PairsTradingBacktest:
    def __init__(self, data_file='data/all_prices_aligned.csv'):
        self.df = None
        self.pair = None
        self.spread = None
        self.z_score = None
        self.signals = None
        self.returns = None
        self.load_data(data_file)
    
    def load_data(self, data_file):
        """Load and clean price data"""
        self.df = pd.read_csv(data_file)
        # Clean data similar to cointegration analysis
        self.df = self.df[self.df['Date'] != 'Date']
        self.df = self.df[pd.to_datetime(self.df['Date'], errors='coerce').notna()]
        for col in self.df.columns[1:]:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        self.df = self.df.dropna(subset=self.df.columns[1:].tolist(), how='any')
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Ensure deterministic column ordering
        date_col = self.df[['Date']]
        price_cols = self.df.drop('Date', axis=1)
        price_cols = price_cols.reindex(sorted(price_cols.columns), axis=1)
        self.df = pd.concat([date_col, price_cols], axis=1)
        
        print(f"Loaded data: {self.df.shape[0]} days, {self.df.shape[1]-1} assets")
    
    def find_best_pair(self):
        """Find the most cointegrated pair using Engle-Granger test"""
        from itertools import combinations
        
        # Log transform prices for better stationarity
        price_data = self.df.copy()
        for col in price_data.columns[1:]:
            price_data[col] = np.log(price_data[col])
        
        best_pvalue = 1.0
        best_pair = None
        best_spread = None
        
        tickers = price_data.columns[1:]
        print(f"Testing {len(list(combinations(tickers, 2)))} pairs for cointegration...")
        
        for ticker1, ticker2 in combinations(tickers, 2):
            s1 = price_data[ticker1].astype(float)
            s2 = price_data[ticker2].astype(float)
            
            try:
                score, pvalue, _ = coint(s1, s2)
                if pvalue < best_pvalue:
                    best_pvalue = pvalue
                    best_pair = (ticker1, ticker2)
                    # Calculate spread using linear regression
                    beta = np.polyfit(s2, s1, 1)[0]
                    best_spread = s1 - beta * s2
            except:
                continue
        
        self.pair = best_pair
        self.spread = best_spread
        print(f"Best pair: {best_pair[0]} & {best_pair[1]} (p-value: {best_pvalue:.6f})")
        return best_pair, best_pvalue
    
    def calculate_zscore(self, window=30):
        """Calculate rolling z-score of the spread"""
        if self.spread is None:
            raise ValueError("Must find best pair first")
        
        # Calculate rolling mean and std
        rolling_mean = self.spread.rolling(window=window).mean()
        rolling_std = self.spread.rolling(window=window).std()
        
        # Calculate z-score
        self.z_score = (self.spread - rolling_mean) / rolling_std
        print(f"Calculated z-score with {window}-day rolling window")
    
    def generate_signals(self, entry_threshold=1.0, exit_threshold=0.5, max_holding_days=10):
        """Generate trading signals based on z-score with realistic constraints"""
        if self.z_score is None:
            raise ValueError("Must calculate z-score first")
        
        signals = pd.DataFrame(index=self.z_score.index)
        signals['z_score'] = self.z_score
        signals['position'] = 0  # 0=neutral, 1=long spread, -1=short spread
        signals['entry'] = False
        signals['exit'] = False
        signals['holding_days'] = 0
        
        current_position = 0
        holding_days = 0
        
        for i in range(1, len(signals)):
            z = signals['z_score'].iloc[i]
            prev_position = current_position
            holding_days += 1 if current_position != 0 else 0
            
            # Entry signals (only when neutral)
            if current_position == 0:
                if z < -entry_threshold:  # Long spread (buy stock1, sell stock2)
                    current_position = 1
                    signals['entry'].iloc[i] = True
                    holding_days = 1
                elif z > entry_threshold:  # Short spread (sell stock1, buy stock2)
                    current_position = -1
                    signals['entry'].iloc[i] = True
                    holding_days = 1
            
            # Exit signals
            elif current_position == 1:  # Currently long spread
                if z > -exit_threshold or holding_days >= max_holding_days:  # Exit long position
                    current_position = 0
                    signals['exit'].iloc[i] = True
                    holding_days = 0
            
            elif current_position == -1:  # Currently short spread
                if z < exit_threshold or holding_days >= max_holding_days:  # Exit short position
                    current_position = 0
                    signals['exit'].iloc[i] = True
                    holding_days = 0
            
            signals['position'].iloc[i] = current_position
            signals['holding_days'].iloc[i] = holding_days
        
        # Shift signals forward by 1 bar to avoid lookahead bias
        signals['position_shifted'] = signals['position'].shift(1).fillna(0)
        signals['entry_shifted'] = signals['entry'].shift(1).fillna(False)
        signals['exit_shifted'] = signals['exit'].shift(1).fillna(False)
        
        self.signals = signals
        print(f"Generated signals: {signals['entry'].sum()} entries, {signals['exit'].sum()} exits")
        print(f"Max holding period constraint: {max_holding_days} days")
        return signals
    
    def simulate_pnl(self, transaction_cost=0.001):
        """Simulate PnL with transaction costs and detailed trade tracking"""
        if self.signals is None:
            raise ValueError("Must generate signals first")
        
        # Get price data for the pair
        ticker1, ticker2 = self.pair
        price1 = self.df[ticker1].values
        price2 = self.df[ticker2].values
        
        # Calculate daily returns
        ret1 = np.diff(np.log(price1))
        ret2 = np.diff(np.log(price2))
        
        # Align with signals (remove first row since we have n-1 returns)
        signals_aligned = self.signals.iloc[1:].copy()
        signals_aligned['ret1'] = ret1
        signals_aligned['ret2'] = ret2
        
        # Use shifted position (no lookahead bias)
        prev_position = signals_aligned['position_shifted'].values
        
        # Calculate strategy returns before transaction costs
        strategy_returns_gross = prev_position * (ret1 - ret2)  # Long-short return
        
        # Calculate transaction costs
        position_changes = np.diff(np.concatenate([[0], prev_position]))
        trade_occurs = np.abs(position_changes) > 0
        
        # Transaction cost = 0.1% round-trip cost when entering/exiting positions
        transaction_costs = trade_occurs * transaction_cost
        strategy_returns_net = strategy_returns_gross - transaction_costs
        
        signals_aligned['strategy_return_gross'] = strategy_returns_gross
        signals_aligned['strategy_return_net'] = strategy_returns_net
        signals_aligned['transaction_costs'] = transaction_costs
        signals_aligned['cumulative_return_gross'] = (1 + signals_aligned['strategy_return_gross']).cumprod() - 1
        signals_aligned['cumulative_return_net'] = (1 + signals_aligned['strategy_return_net']).cumprod() - 1
        
        self.returns = signals_aligned
        
        # Track individual trades
        self.trade_stats = self.calculate_trade_statistics()
        
        # Calculate performance metrics
        total_return_gross = signals_aligned['cumulative_return_gross'].iloc[-1]
        total_return_net = signals_aligned['cumulative_return_net'].iloc[-1]
        annual_return_gross = (1 + total_return_gross) ** (252 / len(signals_aligned)) - 1
        annual_return_net = (1 + total_return_net) ** (252 / len(signals_aligned)) - 1
        volatility = signals_aligned['strategy_return_net'].std() * np.sqrt(252)
        sharpe_ratio = annual_return_net / volatility if volatility > 0 else 0
        max_drawdown = self.calculate_max_drawdown(signals_aligned['cumulative_return_net'])
        
        total_transaction_costs = signals_aligned['transaction_costs'].sum()
        
        print(f"\nPerformance: Strategy Performance:")
        print(f"Number of Trades: {self.trade_stats['num_trades']}")
        print(f"Average Holding Period: {self.trade_stats['avg_holding_period']:.1f} days")
        print(f"Win Rate: {self.trade_stats['win_rate']:.1%}")
        print(f"Total Return (Gross): {total_return_gross:.2%}")
        print(f"Total Return (Net): {total_return_net:.2%}")
        print(f"Annualized Return (Gross): {annual_return_gross:.2%}")
        print(f"Annualized Return (Net): {annual_return_net:.2%}")
        print(f"Total Transaction Costs: {total_transaction_costs:.4f} ({total_transaction_costs*100:.2f}%)")
        print(f"Volatility: {volatility:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        
        return signals_aligned
    
    def calculate_trade_statistics(self):
        """Calculate detailed trade statistics"""
        if self.returns is None:
            return {}
        
        trades = []
        current_trade = None
        
        for i, row in self.returns.iterrows():
            if row['entry_shifted'] and current_trade is None:
                # Start new trade
                current_trade = {
                    'entry_date': i,
                    'entry_position': row['position_shifted'],
                    'entry_return': 0,
                    'holding_period': 0
                }
            
            if current_trade is not None:
                current_trade['holding_period'] += 1
                current_trade['entry_return'] += row['strategy_return_net']
                
                if row['exit_shifted']:
                    # End trade
                    current_trade['exit_date'] = i
                    current_trade['total_return'] = current_trade['entry_return']
                    trades.append(current_trade)
                    current_trade = None
        
        # Handle case where last trade doesn't close
        if current_trade is not None:
            current_trade['exit_date'] = self.returns.index[-1]
            current_trade['total_return'] = current_trade['entry_return']
            trades.append(current_trade)
        
        if not trades:
            return {
                'num_trades': 0,
                'avg_holding_period': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'trades': []
            }
        
        num_trades = len(trades)
        holding_periods = [trade['holding_period'] for trade in trades]
        returns = [trade['total_return'] for trade in trades]
        
        winners = [r for r in returns if r > 0]
        losers = [r for r in returns if r <= 0]
        
        stats = {
            'num_trades': num_trades,
            'avg_holding_period': np.mean(holding_periods),
            'win_rate': len(winners) / num_trades if num_trades > 0 else 0,
            'avg_win': np.mean(winners) if winners else 0,
            'avg_loss': np.mean(losers) if losers else 0,
            'trades': trades
        }
        
        return stats
    
    def calculate_max_drawdown(self, cumulative_returns):
        """Calculate maximum drawdown"""
        cumulative_wealth = 1 + cumulative_returns
        running_max = cumulative_wealth.expanding().max()
        drawdown = (cumulative_wealth - running_max) / running_max
        return drawdown.min()
    
    def plot_results(self):
        """Plot z-score with signals and equity curve"""
        if self.returns is None:
            raise ValueError("Must simulate PnL first")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Z-score with entry/exit points
        dates = self.df['Date'].iloc[1:]  # Align with returns data
        z_scores_aligned = self.z_score.iloc[1:]  # Align with dates
        
        ax1.plot(dates, z_scores_aligned, label='Z-Score', linewidth=1.5, color='blue')
        ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Entry Threshold (+1)')
        ax1.axhline(y=-1, color='red', linestyle='--', alpha=0.7, label='Entry Threshold (-1)')
        ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Exit Threshold (+0.5)')
        ax1.axhline(y=-0.5, color='orange', linestyle='--', alpha=0.7, label='Exit Threshold (-0.5)')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Mark entry and exit points using returns data indices
        entry_mask = self.returns['entry_shifted']
        exit_mask = self.returns['exit_shifted']
        
        if entry_mask.any():
            entry_dates_plot = dates[entry_mask]
            entry_zscores_plot = z_scores_aligned[entry_mask]
            ax1.scatter(entry_dates_plot, entry_zscores_plot, color='green', s=100, marker='^', 
                       label=f'Entry Points ({entry_mask.sum()})', zorder=5)
        
        if exit_mask.any():
            exit_dates_plot = dates[exit_mask]
            exit_zscores_plot = z_scores_aligned[exit_mask]
            ax1.scatter(exit_dates_plot, exit_zscores_plot, color='red', s=100, marker='v', 
                       label=f'Exit Points ({exit_mask.sum()})', zorder=5)
        
        ax1.set_title(f'Spread Z-Score and Trading Signals: {self.pair[0]} & {self.pair[1]}', fontsize=14)
        ax1.set_ylabel('Z-Score', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Equity curve
        ax2.plot(dates, self.returns['cumulative_return_net'] * 100, label='Strategy Return (Net)', 
                linewidth=2, color='green')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('Cumulative Strategy Returns (Net)', fontsize=14)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def run_full_backtest(self, zscore_window=30, entry_threshold=1.0, exit_threshold=0.5, 
                         max_holding_days=10, transaction_cost=0.001):
        """Run complete backtest pipeline with enhanced features"""
        print("Starting Enhanced Pairs Trading Backtest...")
        print("=" * 60)
        
        # Step 1: Find best pair
        self.find_best_pair()
        
        # Step 2: Calculate z-score
        self.calculate_zscore(window=zscore_window)
        
        # Step 3: Generate signals with constraints
        self.generate_signals(entry_threshold=entry_threshold, exit_threshold=exit_threshold,
                            max_holding_days=max_holding_days)
        
        # Step 4: Simulate PnL with transaction costs
        self.simulate_pnl(transaction_cost=transaction_cost)
        
        # Step 5: Plot results
        self.plot_results()
        
        return self.returns

class MultiPairBacktest(PairsTradingBacktest):
    def __init__(self, data_file='data/all_prices_aligned.csv'):
        super().__init__(data_file)
        self.top_pairs = []
        self.pair_results = {}
        self.portfolio_returns = None
        self.portfolio_stats = {}
    
    def find_top_pairs(self, top_n=3):
        """Find the top N most cointegrated pairs"""
        from itertools import combinations
        
        # Log transform prices for better stationarity
        price_data = self.df.copy()
        for col in price_data.columns[1:]:
            price_data[col] = np.log(price_data[col])
        
        pair_results = []
        tickers = price_data.columns[1:]
        print(f"Testing {len(list(combinations(tickers, 2)))} pairs for cointegration...")
        
        for ticker1, ticker2 in combinations(tickers, 2):
            s1 = price_data[ticker1].astype(float)
            s2 = price_data[ticker2].astype(float)
            
            try:
                score, pvalue, _ = coint(s1, s2)
                # Calculate spread using linear regression
                beta = np.polyfit(s2, s1, 1)[0]
                spread = s1 - beta * s2
                
                pair_results.append({
                    'pair': (ticker1, ticker2),
                    'pvalue': pvalue,
                    'spread': spread,
                    'beta': beta
                })
            except:
                continue
        
        # Sort by p-value and take top N
        pair_results.sort(key=lambda x: x['pvalue'])
        self.top_pairs = pair_results[:top_n]
        
        print(f"\nTop {top_n} cointegrated pairs:")
        for i, result in enumerate(self.top_pairs, 1):
            pair = result['pair']
            print(f"{i}. {pair[0]} & {pair[1]} (p-value: {result['pvalue']:.6f})")
        
        return self.top_pairs
    
    def run_single_pair_strategy(self, pair_data, pair_name, zscore_window=20, 
                                entry_threshold=1.0, exit_threshold=0.0, 
                                max_holding_days=10, transaction_cost=0.001):
        """Run strategy for a single pair"""
        
        # Calculate z-score
        spread = pair_data['spread']
        rolling_mean = spread.rolling(window=zscore_window).mean()
        rolling_std = spread.rolling(window=zscore_window).std()
        z_score = (spread - rolling_mean) / rolling_std
        
        # Generate signals
        signals = pd.DataFrame(index=z_score.index)
        signals['z_score'] = z_score
        signals['position'] = 0
        signals['entry'] = False
        signals['exit'] = False
        signals['holding_days'] = 0
        
        current_position = 0
        holding_days = 0
        
        for i in range(1, len(signals)):
            z = signals['z_score'].iloc[i]
            holding_days += 1 if current_position != 0 else 0
            
            # Entry signals
            if current_position == 0:
                if z < -entry_threshold:  # Long spread
                    current_position = 1
                    signals['entry'].iloc[i] = True
                    holding_days = 1
                elif z > entry_threshold:  # Short spread
                    current_position = -1
                    signals['entry'].iloc[i] = True
                    holding_days = 1
            
            # Exit signals
            elif current_position != 0:
                if (current_position == 1 and z > exit_threshold) or \
                   (current_position == -1 and z < -exit_threshold) or \
                   holding_days >= max_holding_days:
                    current_position = 0
                    signals['exit'].iloc[i] = True
                    holding_days = 0
            
            signals['position'].iloc[i] = current_position
            signals['holding_days'].iloc[i] = holding_days
        
        # Shift signals to avoid lookahead bias
        signals['position_shifted'] = signals['position'].shift(1).fillna(0)
        signals['entry_shifted'] = signals['entry'].shift(1).fillna(False)
        signals['exit_shifted'] = signals['exit'].shift(1).fillna(False)
        
        # Calculate returns
        ticker1, ticker2 = pair_data['pair']
        price1 = self.df[ticker1].values
        price2 = self.df[ticker2].values
        
        ret1 = np.diff(np.log(price1))
        ret2 = np.diff(np.log(price2))
        
        signals_aligned = signals.iloc[1:].copy()
        signals_aligned['ret1'] = ret1
        signals_aligned['ret2'] = ret2
        
        prev_position = signals_aligned['position_shifted'].values
        strategy_returns_gross = prev_position * (ret1 - ret2)
        
        # Transaction costs
        position_changes = np.diff(np.concatenate([[0], prev_position]))
        trade_occurs = np.abs(position_changes) > 0
        transaction_costs = trade_occurs * transaction_cost
        strategy_returns_net = strategy_returns_gross - transaction_costs
        
        signals_aligned['strategy_return_net'] = strategy_returns_net
        signals_aligned['cumulative_return'] = (1 + signals_aligned['strategy_return_net']).cumprod() - 1
        
        # Calculate trade statistics
        trades = []
        current_trade = None
        
        for i, row in signals_aligned.iterrows():
            if row['entry_shifted'] and current_trade is None:
                current_trade = {'entry_date': i, 'entry_return': 0, 'holding_period': 0}
            
            if current_trade is not None:
                current_trade['holding_period'] += 1
                current_trade['entry_return'] += row['strategy_return_net']
                
                if row['exit_shifted']:
                    current_trade['total_return'] = current_trade['entry_return']
                    trades.append(current_trade)
                    current_trade = None
        
        # Handle unclosed trade
        if current_trade is not None:
            current_trade['total_return'] = current_trade['entry_return']
            trades.append(current_trade)
        
        num_trades = len(trades)
        avg_holding = np.mean([t['holding_period'] for t in trades]) if trades else 0
        win_rate = len([t for t in trades if t['total_return'] > 0]) / num_trades if num_trades > 0 else 0
        
        return {
            'pair_name': pair_name,
            'signals': signals_aligned,
            'num_trades': num_trades,
            'avg_holding_period': avg_holding,
            'win_rate': win_rate,
            'total_return': signals_aligned['cumulative_return'].iloc[-1],
            'z_score': z_score
        }
    
    def run_multi_pair_backtest(self, top_n=3, zscore_window=20, entry_threshold=1.0, 
                               exit_threshold=0.0, max_holding_days=10, transaction_cost=0.001):
        """Run backtest on multiple pairs and combine into portfolio"""
        print("Starting Multi-Pair Portfolio Backtest...")
        print("=" * 60)
        
        # Find top pairs
        self.find_top_pairs(top_n=top_n)
        
        # Run strategy for each pair
        all_returns = []
        
        for i, pair_data in enumerate(self.top_pairs):
            pair_name = f"{pair_data['pair'][0]}-{pair_data['pair'][1]}"
            print(f"\nRunning strategy for pair {i+1}: {pair_name}")
            
            result = self.run_single_pair_strategy(
                pair_data, pair_name, zscore_window, entry_threshold, 
                exit_threshold, max_holding_days, transaction_cost
            )
            
            self.pair_results[pair_name] = result
            all_returns.append(result['signals']['strategy_return_net'])
            
            print(f"  Trades: {result['num_trades']}")
            print(f"  Win Rate: {result['win_rate']:.1%}")
            print(f"  Total Return: {result['total_return']:.2%}")
        
        # Combine into portfolio (equal weight)
        portfolio_returns = pd.concat(all_returns, axis=1).mean(axis=1)
        portfolio_cumulative = (1 + portfolio_returns).cumprod() - 1
        
        self.portfolio_returns = pd.DataFrame({
            'daily_return': portfolio_returns,
            'cumulative_return': portfolio_cumulative
        })
        
        # Calculate portfolio statistics
        total_return = portfolio_cumulative.iloc[-1]
        annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        max_drawdown = self.calculate_max_drawdown(portfolio_cumulative)
        
        total_trades = sum([result['num_trades'] for result in self.pair_results.values()])
        
        self.portfolio_stats = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades
        }
        
        print(f"\nPerformance: Portfolio Performance Summary:")
        print(f"Total Pairs: {len(self.pair_results)}")
        print(f"Total Trades: {total_trades}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {annual_return:.2%}")
        print(f"Volatility: {volatility:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        
        return self.portfolio_returns
    
    def plot_multi_pair_results(self):
        """Plot individual pair results and combined portfolio"""
        if not self.pair_results or self.portfolio_returns is None:
            raise ValueError("Must run multi-pair backtest first")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        dates = self.df['Date'].iloc[1:]  # Align with returns data
        
        # Plot 1: Individual pair equity curves
        ax1 = axes[0, 0]
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for i, (pair_name, result) in enumerate(self.pair_results.items()):
            cumulative_returns = result['signals']['cumulative_return'] * 100
            ax1.plot(dates, cumulative_returns, label=pair_name, 
                    color=colors[i % len(colors)], linewidth=1.5)
        
        ax1.set_title('Individual Pair Strategy Returns', fontsize=14)
        ax1.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Portfolio equity curve
        ax2 = axes[0, 1]
        portfolio_curve = self.portfolio_returns['cumulative_return'] * 100
        ax2.plot(dates, portfolio_curve, label='Portfolio', 
                color='black', linewidth=2.5)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax2.set_title('Combined Portfolio Equity Curve', fontsize=14)
        ax2.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Z-scores for each pair
        ax3 = axes[1, 0]
        for i, (pair_name, result) in enumerate(self.pair_results.items()):
            z_scores = result['z_score'].iloc[1:]  # Align with dates
            ax3.plot(dates, z_scores, label=pair_name, 
                    color=colors[i % len(colors)], alpha=0.7)
        
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Entry Threshold')
        ax3.axhline(y=-1, color='red', linestyle='--', alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, label='Exit Threshold')
        ax3.set_title('Z-Scores for All Pairs', fontsize=14)
        ax3.set_ylabel('Z-Score', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Performance comparison bar chart
        ax4 = axes[1, 1]
        pair_names = list(self.pair_results.keys())
        pair_returns = [result['total_return'] * 100 for result in self.pair_results.values()]
        portfolio_return = self.portfolio_stats['total_return'] * 100
        
        bars = ax4.bar(pair_names + ['Portfolio'], pair_returns + [portfolio_return],
                      color=colors[:len(pair_names)] + ['black'])
        ax4.set_title('Total Returns Comparison', fontsize=14)
        ax4.set_ylabel('Total Return (%)', fontsize=12)
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, pair_returns + [portfolio_return]):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed pair statistics
        print(f"\nDetails: Detailed Pair Statistics:")
        print("=" * 80)
        print(f"{'Pair':<15} {'Trades':<8} {'Win Rate':<10} {'Avg Hold':<10} {'Total Return':<12}")
        print("-" * 80)
        
        for pair_name, result in self.pair_results.items():
            print(f"{pair_name:<15} {result['num_trades']:<8} "
                  f"{result['win_rate']:<10.1%} {result['avg_holding_period']:<10.1f} "
                  f"{result['total_return']:<12.2%}")
        
        print("-" * 80)
        print(f"{'PORTFOLIO':<15} {self.portfolio_stats['total_trades']:<8} "
              f"{'N/A':<10} {'N/A':<10} {self.portfolio_stats['total_return']:<12.2%}")

def main():
    # Run single pair backtest (original)
    print("Running Single Pair Backtest:")
    single_backtest = PairsTradingBacktest()
    single_results = single_backtest.run_full_backtest(
        zscore_window=20,
        entry_threshold=1.0,
        exit_threshold=0.5,
        max_holding_days=10,
        transaction_cost=0.001
    )
    
    print("\n" + "="*80 + "\n")
    
    # Run multi-pair portfolio backtest (new)
    print("Running Multi-Pair Portfolio Backtest:")
    multi_backtest = MultiPairBacktest()
    portfolio_results = multi_backtest.run_multi_pair_backtest(
        top_n=3,
        zscore_window=20,
        entry_threshold=1.0,
        exit_threshold=0.0,  # Exit at z-score = 0
        max_holding_days=10,
        transaction_cost=0.001
    )
    
    multi_backtest.plot_multi_pair_results()
    
    return single_backtest, multi_backtest

if __name__ == "__main__":
    backtest, results = main() 