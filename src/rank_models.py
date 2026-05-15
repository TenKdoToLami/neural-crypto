import pandas as pd
import os
from glob import glob
import numpy as np

def calculate_fitness(row):
    """
    Calculates a fitness score for a specific model strategy.
    Target: 100-200 trades per year across all coins.
    """
    wr = row['win_rate']
    profit = row['avg_profit']
    trades_per_year = row['trades_per_year']
    
    # 1. Base Score (Win Rate * Profit)
    # We use a non-linear boost for profit to prioritize "Whale Hunters"
    base_score = wr * (profit * 100) 
    
    # 2. Trade Frequency Multiplier (The "Goldilocks" curve)
    if 100 <= trades_per_year <= 250:
        freq_mult = 1.0
    elif trades_per_year < 50:
        freq_mult = 0.3  # Too rare to trust
    elif trades_per_year < 100:
        freq_mult = 0.7  # A bit too quiet
    elif trades_per_year > 500:
        freq_mult = 0.5  # Over-trading (Fee trap)
    else:
        freq_mult = 0.8  # Acceptable but not ideal
        
    return base_score * freq_mult

def main():
    print("🏆 Ranking Neural Trading Models (The 'OOS-First' Method)")
    print("-" * 60)
    
    report_dirs = glob("reports/evaluation/*/")
    all_rankings = []
    
    for d in report_dirs:
        m_name = os.path.basename(os.path.normpath(d))
        stats_path = os.path.join(d, 'asset_performance.csv')
        
        if not os.path.exists(stats_path):
            continue
            
        df = pd.read_csv(stats_path)
        
        # Split into OOS (2020-2023) and IS (2024-2026)
        df_oos = df[df['year'] <= 2023]
        df_is = df[df['year'] >= 2024]
        
        if df_oos.empty or df_is.empty:
            continue
            
        # Group by strategy settings (Entry/Exit)
        settings_groups = df.groupby(['entry_conf', 'exit_conf'])
        
        for (ent, ext), group in settings_groups:
            # OOS Metrics
            oos_group = group[group['year'] <= 2023]
            is_group = group[group['year'] >= 2024]
            
            if oos_group.empty: continue
            
            # Global Metrics for this setting
            total_years = group['year'].nunique()
            total_trades = group['trades'].sum()
            trades_per_year = total_trades / total_years
            
            avg_wr_oos = oos_group['win_rate'].mean()
            avg_profit_oos = oos_group['avg_profit'].mean()
            
            avg_wr_is = is_group['win_rate'].mean() if not is_group.empty else 0
            avg_profit_is = is_group['avg_profit'].mean() if not is_group.empty else 0
            
            # Calculate Scores
            score_oos = calculate_fitness({'win_rate': avg_wr_oos, 'avg_profit': avg_profit_oos, 'trades_per_year': trades_per_year})
            score_is = calculate_fitness({'win_rate': avg_wr_is, 'avg_profit': avg_profit_is, 'trades_per_year': trades_per_year})
            
            # Final Weighted Score: OOS counts for 70% of the total rank
            final_score = (score_oos * 0.7) + (score_is * 0.3)
            
            all_rankings.append({
                'Model': m_name,
                'Entry': ent,
                'Exit': ext,
                'Final_Score': final_score,
                'OOS_WinRate': f"{avg_wr_oos:.1%}",
                'OOS_WinRate_Val': avg_wr_oos,
                'OOS_Profit': f"{avg_profit_oos:.2%}",
                'OOS_Profit_Val': avg_profit_oos,
                'Trades/Year': round(trades_per_year, 1),
                'Stability': "✅" if avg_wr_oos > 0.5 and avg_wr_is > 0.5 else "⚠️"
            })

    # Convert to DataFrame
    rank_df = pd.DataFrame(all_rankings)
    if rank_df.empty:
        print("No valid models found to rank. Run evaluations first.")
        return

    # --- STRICT PROFESSIONAL FILTER ---
    # User Requirements: WinRate >= 60%, Profit >= 1%, Trades 50-300
    rank_df['Passed_Strict'] = (
        (rank_df['OOS_WinRate_Val'] >= 0.60) & 
        (rank_df['OOS_Profit_Val'] >= 0.01) & 
        (rank_df['Trades/Year'] >= 50) & 
        (rank_df['Trades/Year'] <= 300)
    )
    
    # 1. Identify "Winners" (Models that have at least one passing setting)
    winning_models = rank_df[rank_df['Passed_Strict']]['Model'].unique()
    
    # 2. Identify "Trash" (Models that failed ALL settings)
    all_models = rank_df['Model'].unique()
    trash_models = [m for m in all_models if m not in winning_models]
    
    # 3. Perform the Purge
    if trash_models:
        print(f"\n🗑️  PURGING {len(trash_models)} UNDERPERFORMING MODELS...")
        for m in trash_models:
            # Delete Report Folder
            report_path = os.path.join('reports/evaluation', m)
            if os.path.exists(report_path):
                import shutil
                shutil.rmtree(report_path)
            
            # Delete Model File
            model_files = glob(f"models/{m}.pth")
            for f in model_files:
                os.remove(f)
            print(f"  [x] Deleted: {m}")
    
    # 4. Display the survivors
    rank_df = rank_df[rank_df['Passed_Strict']]
    
    if rank_df.empty:
        print("\n⚠️  No models passed the strict filter (60% WR / 1% Profit / 50-300 Trades).")
        return
        
    top_10 = rank_df.sort_values('Final_Score', ascending=False).head(15)
    
    print("\n🏆 TOP PERFORMING SURVIVORS:")
    print(top_10[['Model', 'Entry', 'Exit', 'OOS_WinRate', 'OOS_Profit', 'Trades/Year', 'Final_Score']].to_string(index=False))
    
    winner = top_10.iloc[0]
    print("\n👑 THE GRAND WINNER:")
    print(f"Model: {winner['Model']}")
    print(f"Recommended Settings: Entry {winner['Entry']} / Exit {winner['Exit']}")
    print(f"Reason: Met all strict criteria with {winner['OOS_WinRate']} Win Rate and {winner['OOS_Profit']} avg profit.")

if __name__ == "__main__":
    main()
