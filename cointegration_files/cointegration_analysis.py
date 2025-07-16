import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def engle_granger_test(series1, series2):
    score, pvalue, _ = coint(series1, series2)
    spread = series1 - np.polyfit(series2, series1, 1)[0] * series2
    return pvalue, spread


def johansen_test(df_pair):
    # Johansen expects a 2D array
    result = coint_johansen(df_pair, det_order=0, k_ar_diff=1)
    # Take the smallest p-value (most significant cointegration)
    trace_stat = result.lr1[0]
    crit_95 = result.cvt[0, 1]
    pvalue = np.exp(-trace_stat / crit_95)  # Not a true p-value, but a score for ranking
    spread = df_pair @ result.evec[:, 0]
    return pvalue, spread


def compute_cointegration(df):
    pairs = list(itertools.combinations(df.columns[1:], 2))  # skip 'Date' column
    results = []
    for a, b in pairs:
        s1 = df[a].astype(float)
        s2 = df[b].astype(float)
        # Engle-Granger
        eg_pval, eg_spread = engle_granger_test(s1, s2)
        # Johansen
        j_pval, j_spread = johansen_test(df[[a, b]].astype(float))
        results.append({
            'pair': (a, b),
            'engle_granger_pval': eg_pval,
            'engle_granger_spread': eg_spread,
            'johansen_score': j_pval,
            'johansen_spread': j_spread
        })
    return results


def print_and_plot_top_pairs(results, method='engle_granger_pval', top_n=5):
    sorted_results = sorted(results, key=lambda x: x[method])[:top_n]
    for i, res in enumerate(sorted_results, 1):
        a, b = res['pair']
        print(f"{i}. Pair: {a} & {b}")
        print(f"   Engle-Granger p-value: {res['engle_granger_pval']:.4g}")
        print(f"   Johansen score: {res['johansen_score']:.4g}")
        # Plot Engle-Granger spread
        plt.figure(figsize=(10, 3))
        plt.plot(res['engle_granger_spread'], label='Engle-Granger Spread')
        plt.title(f"Engle-Granger Spread: {a} & {b}")
        plt.legend()
        plt.show()
        # Plot Johansen spread
        plt.figure(figsize=(10, 3))
        plt.plot(res['johansen_spread'], label='Johansen Spread', color='orange')
        plt.title(f"Johansen Spread: {a} & {b}")
        plt.legend()
        plt.show()


def rolling_cointegration(df, window=252):
    pairs = list(itertools.combinations(df.columns[1:], 2))
    results = []
    for a, b in pairs:
        s1 = df[a].astype(float)
        s2 = df[b].astype(float)
        min_pval = 1.0
        min_spread = None
        min_start = None
        valid_windows = 0
        for start in range(0, len(df) - window + 1):
            end = start + window
            win1 = s1.iloc[start:end]
            win2 = s2.iloc[start:end]
            if win1.isnull().any() or win2.isnull().any():
                continue
            pval, spread = engle_granger_test(win1, win2)
            valid_windows += 1
            if pval < min_pval:
                min_pval = pval
                min_spread = spread
                min_start = start
        if valid_windows == 0:
            min_pval = np.nan
        results.append({
            'pair': (a, b),
            'min_rolling_pval': min_pval,
            'spread': min_spread,
            'window_start': min_start,
            'valid_windows': valid_windows
        })
    return results

def print_significant_pairs(results, threshold=0.05):
    # Filter pairs with significant p-values
    significant = [r for r in results if not np.isnan(r['engle_granger_pval']) and r['engle_granger_pval'] < threshold]
    
    if not significant:
        print(f"No pairs found with p-value < {threshold}")
        return
    
    print(f"\nKey: Significant Cointegrated Pairs (p-value < {threshold}):")
    print("=" * 70)
    print(f"{'Rank':<4} {'Pair':<15} {'P-Value':<12} {'Johansen Score':<15}")
    print("-" * 70)
    
    sorted_pairs = sorted(significant, key=lambda x: x['engle_granger_pval'])
    for i, res in enumerate(sorted_pairs, 1):
        a, b = res['pair']
        pair_str = f"{a} & {b}"
        print(f"{i:<4} {pair_str:<15} {res['engle_granger_pval']:<12.6f} {res['johansen_score']:<15.4f}")
    
    print("=" * 70)
    print(f"Total significant pairs: {len(sorted_pairs)}")

def print_rolling_summary(results, threshold=0.05):
    # Filter pairs with significant rolling p-values
    significant = [r for r in results if not np.isnan(r['min_rolling_pval']) and r['min_rolling_pval'] < threshold]
    
    if not significant:
        print(f"No pairs found with rolling p-value < {threshold}")
        return
    
    print(f"\nRolling: Rolling Window Significant Pairs (p-value < {threshold}):")
    print("=" * 80)
    print(f"{'Rank':<4} {'Pair':<15} {'Min P-Value':<12} {'Valid Windows':<15}")
    print("-" * 80)
    
    sorted_pairs = sorted(significant, key=lambda x: x['min_rolling_pval'])
    for i, res in enumerate(sorted_pairs, 1):
        a, b = res['pair']
        pair_str = f"{a} & {b}"
        print(f"{i:<4} {pair_str:<15} {res['min_rolling_pval']:<12.6f} {res['valid_windows']:<15}")
    
    print("=" * 80)
    print(f"Total significant rolling pairs: {len(sorted_pairs)}")

def print_and_plot_top_rolling_pairs(results, df, window=252, top_n=5):
    # Filter out pairs with nan p-values
    filtered = [r for r in results if not np.isnan(r['min_rolling_pval'])]
    if not filtered:
        print("No valid windows found for any pair. Try lowering the window size or check your data for missing values.")
        return
    sorted_results = sorted(filtered, key=lambda x: x['min_rolling_pval'])[:top_n]
    for i, res in enumerate(sorted_results, 1):
        a, b = res['pair']
        print(f"{i}. Pair: {a} & {b}")
        print(f"   Min rolling Engle-Granger p-value: {res['min_rolling_pval']:.4g}")
        if res['window_start'] is not None:
            start = res['window_start']
            end = start + window
            dates = pd.to_datetime(df['Date'].iloc[start:end])
            plt.figure(figsize=(12, 4))
            plt.plot(dates, res['spread'], label='Rolling Window Spread', linewidth=2)
            plt.title(f"Rolling Window Spread: {a} & {b} (p-value: {res['min_rolling_pval']:.4g})")
            plt.xlabel('Date')
            plt.ylabel('Spread')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            print("   No valid window found.")


def main():
    df = pd.read_csv('data/all_prices_aligned.csv')
    print(f"Original data shape: {df.shape}")
    
    # Remove duplicate header rows (where Date column contains 'Date' string)
    df = df[df['Date'] != 'Date']
    print(f"After removing duplicate headers: {df.shape}")
    
    # Drop rows where 'Date' is not a valid date
    df = df[pd.to_datetime(df['Date'], errors='coerce').notna()]
    print(f"After removing invalid dates: {df.shape}")
    
    # Convert all price columns to float, coerce errors to NaN
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with any NaN values in price columns
    df = df.dropna(subset=df.columns[1:].tolist(), how='any')
    print(f"After removing rows with NaN values: {df.shape}")
    
    if df.empty:
        print("ERROR: No valid data remaining after cleaning!")
        return
    
    # Log-transform the price columns
    for col in df.columns[1:]:
        df[col] = np.log(df[col])
    
    print(f"Final data shape for analysis: {df.shape}")
    print(f"Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    
    print("\nRolling window cointegration analysis (15-day window):")
    rolling_results = rolling_cointegration(df, window=15)
    print_rolling_summary(rolling_results, threshold=0.05)
    print_and_plot_top_rolling_pairs(rolling_results, df, window=15, top_n=5)
    
    # Still run full-period analysis for reference
    print("\nFull-period cointegration analysis:")
    results = compute_cointegration(df)
    print_significant_pairs(results, threshold=0.05)
    
    best = min(results, key=lambda x: x['engle_granger_pval'])
    a, b = best['pair']
    print(f"\nBest: Best cointegrated pair: {a} & {b}")
    print(f"   Engle-Granger p-value: {best['engle_granger_pval']:.6f}")
    print(f"   Johansen score: {best['johansen_score']:.4f}")
    plt.figure(figsize=(12, 4))
    plt.plot(best['engle_granger_spread'], label='Engle-Granger Spread', linewidth=2)
    plt.title(f"Best Pair Engle-Granger Spread: {a} & {b} (p-value: {best['engle_granger_pval']:.4g})")
    plt.xlabel('Time')
    plt.ylabel('Spread')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(12, 4))
    plt.plot(best['johansen_spread'], label='Johansen Spread', color='orange', linewidth=2)
    plt.title(f"Best Pair Johansen Spread: {a} & {b}")
    plt.xlabel('Time')
    plt.ylabel('Spread')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main() 