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
            all_stats.append(df)
    return pd.concat(all_stats) if all_stats else None

df = load_data()

if df is None:
    st.error("No data found. Run your evaluation script first!")
else:
    st.sidebar.header("🕹️ Control Panel")
    neuron_types = sorted(df['neurons'].unique())
    selected_neurons = st.sidebar.multiselect("Brain Size (Neurons)", neuron_types, default=neuron_types)
    years = sorted(df['year'].unique())
    selected_years = st.sidebar.multiselect("Years to Analyze", years, default=years)
    top_n = st.sidebar.slider("Show Top 'N' Models", 3, 20, 10)
    
    df_filtered = df[df['neurons'].isin(selected_neurons) & df['year'].isin(selected_years)]
    
    leaderboard = df_filtered.groupby(['model', 'neurons']).agg({
        'win_rate': 'mean', 'avg_profit': 'mean', 'trades': 'sum'
    }).reset_index()
    
    top_models = leaderboard.sort_values('win_rate', ascending=False).head(top_n)['model'].tolist()
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 Leaderboard", "📉 Time & Stability", "🧠 Brain Size", "🗃️ Master Explorer", "🔍 Model Deep-Dive"])

    with tab1:
        st.header(f"Top {top_n} Models by Win Rate")
        display_lb = leaderboard.sort_values('win_rate', ascending=False).head(top_n).copy()
        display_lb['Win Rate %'] = (display_lb['win_rate'] * 100).round(1)
        display_lb['Avg Profit %'] = (display_lb['avg_profit'] * 100).round(2)
        st.dataframe(display_lb[['model', 'neurons', 'Win Rate %', 'Avg Profit %', 'trades']], use_container_width=True)
        
        st.subheader("Profitability Matrix")
        fig = px.scatter(leaderboard, x='win_rate', y='avg_profit', color='neurons', size='trades',
                         hover_name='model', color_continuous_scale='Viridis')
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
        
        master_table = strategy_table.copy()
        master_table['Win Rate %'] = (master_table['win_rate'] * 100).round(2)
        master_table['Profit %'] = (master_table['avg_profit'] * 100).round(3)
        
        # FEATURE: Filter for Best Setting per Model
        if st.checkbox("Only show best setting for each model", value=False):
            # Sort by win rate then pick top 1 for each model
            master_table = master_table.sort_values(['model', 'Win Rate %'], ascending=False).groupby('model').head(1)
        
        st.dataframe(master_table.sort_values(['Win Rate %', 'Profit %'], ascending=False), use_container_width=True, height=600)

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
