import os
import json
import pandas as pd
from glob import glob

def compare_models():
    report_files = glob('reports/models/*.json')
    if not report_files:
        print("No model reports found in reports/models/. Run src/model_tester.py first.")
        return

    data = []
    for f in report_files:
        with open(f, 'r') as jf:
            report = json.load(jf)
            data.append({
                'Model': report['model'],
                'Period': report['period'],
                'Return %': round(report['total_return_pct'], 2),
                'Benchmark %': round(report.get('benchmark_return_pct', 0), 2),
                'Win Rate %': round(report['win_rate_pct'], 1),
                'Trades': report['total_trades']
            })

    df = pd.DataFrame(data)
    
    # Pivot to see BEAR and LIVE side-by-side per model
    pivot_df = df.pivot(index='Model', columns='Period', values=['Return %', 'Win Rate %', 'Trades'])
    
    print("\n" + "="*80)
    print("      🏆 NEURAL SENTINEL MODEL COMPARISON REPORT")
    print("="*80)
    
    if pivot_df.empty:
        print("Not enough data to compare. Ensure models have both BEAR and LIVE reports.")
    else:
        print(pivot_df.to_string())
    
    print("\n" + "="*80)
    print("Note: BEAR (2023-2024) is a static benchmark.")
    print("      LIVE (2025-Present) tracks performance on most recent data.")
    print("="*80)

if __name__ == "__main__":
    compare_models()
