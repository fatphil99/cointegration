#!/usr/bin/env python3
"""
Test script to verify that pairs trading strategy produces identical results
"""

import numpy as np
import pandas as pd
from enhanced_pairs_strategy import EnhancedPairsStrategy
import time

def test_reproducibility(n_runs=3):
    """Test that strategy produces identical results across multiple runs"""
    print(f"Testing reproducibility across {n_runs} runs...")
    print("=" * 60)
    
    results = []
    
    for run in range(n_runs):
        print(f"\nRun {run + 1}:")
        print("-" * 30)
        
        # Create new strategy instance for each run
        strategy = EnhancedPairsStrategy()
        
        # Run the analysis
        start_time = time.time()
        metrics, pairs, mc_results = strategy.run_enhanced_analysis()
        end_time = time.time()
        
        if metrics is not None:
            run_result = {
                'run': run + 1,
                'total_return': metrics.get('total_return', 0),
                'annual_return': metrics.get('annual_return', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'num_trades': metrics.get('num_trades', 0),
                'execution_time': end_time - start_time
            }
            
            if mc_results:
                run_result.update({
                    'mc_mean_return': mc_results.get('mean_return', 0),
                    'mc_var_95': mc_results.get('var_95', 0),
                    'mc_prob_profit': mc_results.get('prob_profit', 0)
                })
            
            results.append(run_result)
            
            print(f"  Total Return: {run_result['total_return']:.6f}")
            print(f"  Sharpe Ratio: {run_result['sharpe_ratio']:.6f}")
            print(f"  Num Trades: {run_result['num_trades']}")
            if mc_results:
                print(f"  MC Mean Return: {run_result['mc_mean_return']:.6f}")
                print(f"  MC VaR 95%: {run_result['mc_var_95']:.6f}")
            print(f"  Execution Time: {run_result['execution_time']:.2f}s")
        else:
            print(f"  ERROR: Run {run + 1} failed")
    
    # Check if all results are identical
    if len(results) < 2:
        print("\nERROR: Insufficient successful runs for comparison")
        return False
    
    print(f"\n\nReproducibility Analysis:")
    print("=" * 60)
    
    # Compare key metrics
    metrics_to_check = ['total_return', 'annual_return', 'sharpe_ratio', 'max_drawdown', 'num_trades']
    if mc_results:
        metrics_to_check.extend(['mc_mean_return', 'mc_var_95', 'mc_prob_profit'])
    
    all_identical = True
    
    for metric in metrics_to_check:
        values = [result[metric] for result in results if metric in result]
        if len(values) > 1:
            are_identical = all(abs(v - values[0]) < 1e-10 for v in values)
            variance = np.var(values) if len(values) > 1 else 0
            
            print(f"{metric:20} | Identical: {'YES' if are_identical else 'NO':3} | "
                  f"Values: {values} | Variance: {variance:.2e}")
            
            if not are_identical:
                all_identical = False
    
    print("\n" + "=" * 60)
    if all_identical:
        print("SUCCESS: All runs produced IDENTICAL results!")
        print("Your strategy is now fully reproducible.")
    else:
        print("WARNING: Results differ between runs.")
        print("There may be remaining sources of randomness.")
    
    print(f"\nExecution times: {[r['execution_time'] for r in results]}")
    print(f"Average execution time: {np.mean([r['execution_time'] for r in results]):.2f}s")
    
    return all_identical

if __name__ == "__main__":
    test_reproducibility(n_runs=3) 