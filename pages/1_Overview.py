"""
================================================================================
PAGE 1: OVERVIEW
================================================================================
File: deployment/pages/1_Overview.py

Executive summary dashboard with key metrics and insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.loaders import load_all_data
from utils.plotting import plot_risk_distribution, plot_temporal_trend
from config import Config

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")

st.title("📊 Executive Overview")
st.markdown("High-level summary of fraud detection results")

# Load data
try:
    data = load_all_data()
    contracts = data['contracts']
    agents = data['agents']
    
    # ========================================================================
    # MERGE NETWORK DATA
    # ========================================================================
    
    # Try to load and merge agent communities data
    try:
        agent_communities_path = Path(Config.DATA_DIR) / 'agent_communities.csv'
        if agent_communities_path.exists():
            agent_communities = pd.read_csv(agent_communities_path)
            
            # Only merge the columns that exist
            merge_cols = ['agent_id', 'neighbor_avg_risk', 'community_id']
            available_merge_cols = [col for col in merge_cols if col in agent_communities.columns]
            
            # Merge network features into agents
            agents = agents.merge(
                agent_communities[available_merge_cols],
                on='agent_id',
                how='left',
                suffixes=('', '_network')
            )
            
            network_coverage = agents['neighbor_avg_risk'].notna().sum()
            st.sidebar.success(f"✅ Network data loaded ({network_coverage:,} agents)")
        else:
            st.sidebar.warning("⚠️ Network analysis not available")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Could not load network data: {str(e)[:50]}")
    
    # ========================================================================
    # KEY METRICS
    # ========================================================================
    
    st.markdown("## 🎯 Key Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_contracts = len(contracts)
        st.metric("Total Contracts", f"{total_contracts:,}")
    
    with col2:
        high_risk_contracts = (contracts['risk_prediction'] == 1).sum()
        risk_rate = high_risk_contracts / total_contracts * 100
        st.metric(
            "High-Risk Contracts",
            f"{high_risk_contracts:,}",
            f"{risk_rate:.1f}%"
        )
    
    with col3:
        high_risk_agents = (agents['risk_score'] > 0.7).sum()
        st.metric("High-Risk Agents", f"{high_risk_agents:,}")
    
    with col4:
        ml_caught = contracts.get('ml_caught_not_rules', pd.Series([False]*len(contracts))).sum()
        st.metric(
            "ML-Detected (Not Rules)",
            f"{ml_caught:,}",
            help="Contracts ML flagged that simple rules missed"
        )
    
    with col5:
        if 'community_summary' in data:
            high_risk_communities = (data['community_summary']['avg_risk_score'] > 0.6).sum()
            st.metric("High-Risk Communities", f"{high_risk_communities:,}")
        else:
            st.metric("High-Risk Communities", "N/A")
    
    # ========================================================================
    # RISK DISTRIBUTION
    # ========================================================================
    
    st.markdown("---")
    st.markdown("## 📈 Risk Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = plot_risk_distribution(contracts, 'combined_risk_score', "Contract Risk Score Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = plot_risk_distribution(agents, 'risk_score', "Agent Risk Score Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # TEMPORAL TRENDS
    # ========================================================================
    
    if 'awardYear' in contracts.columns:
        st.markdown("---")
        st.markdown("## 📅 Temporal Trends")
        
        yearly = contracts.groupby('awardYear').agg({
            'risk_prediction': ['sum', 'count']
        }).reset_index()
        yearly.columns = ['Year', 'High_Risk', 'Total']
        yearly['Risk_Rate'] = yearly['High_Risk'] / yearly['Total'] * 100
        
        # Custom muted color scale for bars
        custom_colorscale = [
            [0.0, '#5a9e6f'],   # Muted green
            [0.5, '#c9b56c'],   # Muted yellow
            [1.0, '#b05555']    # Muted red
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=yearly['Year'],
            y=yearly['High_Risk'],
            name='High-Risk Contracts',
            marker=dict(
                color=yearly['Risk_Rate'],  # Color by risk rate
                colorscale=custom_colorscale,
                showscale=True,
                colorbar=dict(
                    title="Risk<br>Rate %",
                    thickness=15,
                    len=0.5
                )
            )
        ))
        fig.add_trace(go.Scatter(
            x=yearly['Year'],
            y=yearly['Risk_Rate'],
            name='Risk Rate (%)',
            yaxis='y2',
            mode='lines+markers',
            line=dict(color='#a88b5c', width=3),  # Muted orange
            marker=dict(size=8)
        ))
        fig.update_layout(
            title="Yearly Risk Trends",
            xaxis_title="Year",
            yaxis_title="Number of High-Risk Contracts",
            yaxis2=dict(
                title="Risk Rate (%)",
                overlaying='y',
                side='right'
            ),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # TOP INSIGHTS
    # ========================================================================
    
    st.markdown("---")
    st.markdown("## 💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔍 ML-Specific Findings")
        st.markdown(f"""
        - **{ml_caught:,} contracts** flagged by ML but not by simple price/duration rules
        - These represent **hidden risks** in "normal-looking" contracts
        - **{(contracts['risk_prediction']==1).sum() - ml_caught:,} contracts** flagged by both ML and rules
        """)
    
    with col2:
        st.markdown("### 🕸️ Network Effects")
        
        # Check if network data is available
        if 'neighbor_avg_risk' in agents.columns:
            # Count agents with network data
            agents_with_network = agents['neighbor_avg_risk'].notna().sum()
            
            if agents_with_network > 0:
                # Filter to only agents with network data for analysis
                agents_networked = agents[agents['neighbor_avg_risk'].notna()]
                high_neighbor_risk = (agents_networked['neighbor_avg_risk'] > 0.6).sum()
                
                st.markdown(f"""
                - **{agents_with_network:,} agents** analyzed in network ({agents_with_network/len(agents)*100:.1f}%)
                - **{high_neighbor_risk:,} agents** have high-risk neighbors (>0.6)
                - Risk propagation through buyer-supplier networks detected
                - Community-level risk clustering identified
                """)
                
                # Show network coverage
                st.caption(f"ℹ️ Network analysis covers {agents_with_network:,} out of {len(agents):,} total agents")
            else:
                st.info("📊 Network data loaded but no valid neighbor risk scores found")
        else:
            st.info("""
            📊 **Network analysis not available**
            
            To enable network effects analysis:
            - Ensure `agent_communities.csv` exists in data folder
            - The system will automatically merge network features
            
            Network analysis reveals:
            - Risk propagation patterns
            - Community clustering
            - Supplier-buyer relationships
            """)
    
    # ========================================================================
    # TOP RISKY ENTITIES
    # ========================================================================
    
    st.markdown("---")
    st.markdown("## ⚠️ Top Risk Entities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Top 10 Riskiest Contracts")
        top_contracts = contracts.nlargest(10, 'combined_risk_score')[
            ['lotId', 'buyerId', 'supplierId', 'combined_risk_score']
        ].copy()
        # Format risk score
        top_contracts['combined_risk_score'] = top_contracts['combined_risk_score'].round(4)
        st.dataframe(top_contracts, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### Top 10 Riskiest Agents")
        display_cols = ['agent_id', 'risk_score', 'risk_category']
        
        # Add community_id if available
        if 'community_id' in agents.columns:
            display_cols.append('community_id')
        
        top_agents = agents.nlargest(10, 'risk_score')[display_cols].copy()
        # Format risk score
        top_agents['risk_score'] = top_agents['risk_score'].round(4)
        st.dataframe(top_agents, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"❌ Error loading overview data: {e}")
    st.exception(e)  # Show full traceback for debugging