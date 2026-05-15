import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from glob import glob
import re

st.set_page_config(page_title="Neural-Crypto Analytics", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧬 Neural-Crypto: Intelligence Dashboard")
st.markdown("---")

def parse_neurons(name):
    match = re.search(r'(\d+)N', name)
    return int(match.group(1)) if match else 256

@st.cache_data
def load_data():
    report_dirs = glob("reports/evaluation/*/")
    all_stats = []
    for d in report_dirs:
        m_name = os.path.basename(os.path.normpath(d))
        stats_path = os.path.join(d, 'asset_performance.csv')
        if os.path.exists(stats_path):
            df = pd.read_csv(stats_path)
            df['model'] = m_name
            df['neurons'] = parse_neurons(m_name)
            # Detect Version
            df['version'] = 'V2 (VPT+Attention)' if 'V2_' in m_name else 'V1 (Legacy)'
            all_stats.append(df)
    return pd.concat(all_stats) if all_stats else None

df = load_data()

if df is None:
    st.error("No data found. Run your evaluation script first!")
else:
    # --- TOP LEVEL METRICS ---
    st.subheader("🚀 High-Level Performance Comparison")
    m_v1 = df[df['version'] == 'V1 (Legacy)']
    m_v2 = df[df['version'] == 'V2 (VPT+Attention)']
    
    col1, col2, col3 = st.columns(3)
    if not m_v1.empty:
        best_v1 = m_v1['win_rate'].max()
        col1.metric("Best V1 Win Rate", f"{best_v1:.1%}", delta=None)
    if not m_v2.empty:
        best_v2 = m_v2['win_rate'].max()
        improvement = (best_v2 - best_v1) if not m_v1.empty else 0
        col2.metric("Best V2 Win Rate", f"{best_v2:.1%}", delta=f"{improvement:.1%} vs V1", delta_color="normal")
    
    total_models = df['model'].nunique()
    col3.metric("Models Evaluated", total_models)
    
    st.markdown("---")
    # Sidebar Controls
    st.sidebar.header("🕹️ Control Panel")
    neuron_types = sorted(df['neurons'].unique())
    selected_neurons = st.sidebar.multiselect("Brain Size (Neurons)", neuron_types, default=neuron_types)
    
    years = sorted(df['year'].unique())
    selected_years = st.sidebar.multiselect("Years to Analyze", years, default=years)
    
    # NEW: Trade Frequency Filter
    st.sidebar.subheader("📊 Trade Frequency")
    min_t, max_t = st.sidebar.slider("Trades per Year Range", 0, 2000, (50, 1000))
    
    top_n = st.sidebar.slider("Show Top 'N' Models", 3, 20, 10)
    
    # Apply Basic Filters
    df_filtered = df[df['neurons'].isin(selected_neurons) & df['year'].isin(selected_years)]
    
    # Aggregate Stats (Include Version in Groupby)
    leaderboard = df_filtered.groupby(['model', 'neurons', 'version']).agg({
        'win_rate': 'mean', 'avg_profit': 'mean', 'trades': 'sum'
    }).reset_index()
    
    # Calculate Trades Per Year
    num_years = df_filtered['year'].nunique() if not df_filtered.empty else 1
    leaderboard['trades_per_year'] = (leaderboard['trades'] / num_years).round(1)
    
    # APPLY FREQUENCY FILTER
    leaderboard = leaderboard[(leaderboard['trades_per_year'] >= min_t) & (leaderboard['trades_per_year'] <= max_t)]
    
    if leaderboard.empty:
        st.warning("No models found matching these filters! Try widening the 'Trades per Year' range.")
        st.stop()

    top_models = leaderboard.sort_values('win_rate', ascending=False).head(top_n)['model'].tolist()
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 Leaderboard", "📉 Time & Stability", "🧠 Brain Size", "🗃️ Master Explorer", "🔍 Model Deep-Dive"])

    with tab1:
        st.header(f"Top {top_n} Models ({min_t}-{max_t} trades/yr)")
        display_lb = leaderboard.sort_values('win_rate', ascending=False).head(top_n).copy()
        display_lb['Win Rate %'] = (display_lb['win_rate'] * 100).round(1)
        display_lb['Avg Profit %'] = (display_lb['avg_profit'] * 100).round(2)
        display_lb['Total Return %'] = (display_lb['avg_profit'] * display_lb['trades'] * 100).round(1)
        
        st.dataframe(display_lb[['model', 'neurons', 'Win Rate %', 'Avg Profit %', 'Total Return %', 'trades_per_year']], use_container_width=True)
        
        st.subheader("Profitability Matrix (V1 vs V2)")
        # We color by Version now to make it more "visible"
        fig = px.scatter(leaderboard, 
                         x='win_rate', 
                         y='avg_profit', 
                         color='version', 
                         symbol='version',
                         size='trades',
                         hover_name='model', 
                         color_discrete_map={'V2 (VPT+Attention)': '#00FFCC', 'V1 (Legacy)': '#FF3366'},
                         template="plotly_dark")
        
        # Add a benchmark line for 60% win rate
        fig.add_vline(x=0.6, line_dash="dash", line_color="gray", annotation_text="Target Accuracy")
        
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Reliability Over Time")
        yearly_df = df_filtered[df_filtered['model'].isin(top_models)].groupby(['year', 'model'])['win_rate'].mean().reset_index()
        fig_line = px.line(yearly_df, x='year', y='win_rate', color='model', markers=True)
        st.plotly_chart(fig_line, use_container_width=True)

    with tab3:
        st.header("Smaller Brains vs Larger Brains")
        neuron_agg = df_filtered.groupby(['neurons', 'year'])['win_rate'].mean().reset_index()
        fig_neurons = px.bar(neuron_agg, x='year', y='win_rate', color='neurons', barmode='group')
        st.plotly_chart(fig_neurons, use_container_width=True)

    with tab4:
        st.header("🗃️ Master Strategy Explorer")
        strategy_table = df_filtered.groupby(['model', 'neurons', 'entry_conf', 'exit_conf']).agg({
            'win_rate': 'mean', 'avg_profit': 'mean', 'trades': 'sum'
        }).reset_index()
        
        # Apply Frequency Filter to Master Table
        strategy_table['trades_per_year'] = (strategy_table['trades'] / num_years).round(1)
        strategy_table = strategy_table[(strategy_table['trades_per_year'] >= min_t) & (strategy_table['trades_per_year'] <= max_t)]
        
        master_table = strategy_table.copy()
        master_table['Win Rate %'] = (master_table['win_rate'] * 100).round(2)
        master_table['Avg Profit %'] = (master_table['avg_profit'] * 100).round(3)
        master_table['Total Return %'] = (master_table['avg_profit'] * master_table['trades'] * 100).round(1)
        
        # FEATURE: Filter for Best Setting per Model
        if st.checkbox("Only show best setting for each model", value=False):
            # Sort by win rate then pick top 1 for each model
            master_table = master_table.sort_values(['model', 'Win Rate %'], ascending=False).groupby('model').head(1)
        
        st.dataframe(master_table.sort_values(['Win Rate %', 'Avg Profit %'], ascending=False), use_container_width=True, height=600)

    with tab5:
        st.header("🔍 Individual Model Deep-Dive")
        target_model = st.selectbox("Select Model to Inspect", options=sorted(df['model'].unique()))
        
        m_df = df[df['model'] == target_model]
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Performance by Asset")
            asset_m = m_df.groupby('asset')['win_rate'].mean().reset_index()
            fig_asset = px.bar(asset_m, x='asset', y='win_rate', color='win_rate', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig_asset, use_container_width=True)
            
        with c2:
            st.subheader("Performance by Year")
            year_m = m_df.groupby('year')['win_rate'].mean().reset_index()
            fig_year_m = px.line(year_m, x='year', y='win_rate', markers=True)
            st.plotly_chart(fig_year_m, use_container_width=True)
            
        st.subheader("Setting Efficiency (Heatmap)")
        # Show which Entry/Exit combos worked best for THIS model
        pivot_m = m_df.groupby(['entry_conf', 'exit_conf'])['win_rate'].mean().unstack()
        fig_heat = px.imshow(pivot_m, text_auto=True, color_continuous_scale='RdYlGn',
                             labels=dict(x="Exit Threshold", y="Entry Threshold", color="Win Rate"))
        st.plotly_chart(fig_heat, use_container_width=True)
        
        # Load Raw Trade Log if available
        trade_log_path = os.path.join('reports/evaluation', target_model, 'cleaned_trade_log.csv')
        if os.path.exists(trade_log_path):
            st.subheader("Recent Trade Sample (Cleaned Log)")
            trades = pd.read_csv(trade_log_path)
            st.dataframe(trades.head(50), use_container_width=True)
