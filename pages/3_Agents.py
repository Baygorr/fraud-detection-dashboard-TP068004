"""
================================================================================
PAGE 3: AGENTS - FIXED VERSION
================================================================================
File: deployment/pages/3_Agents.py

Agent-level risk profiling with network effects (BINARY CLASSIFICATION)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.loaders import load_all_data
from utils.plotting import plot_risk_distribution
from config import Config

st.set_page_config(page_title="Agents", page_icon="👥", layout="wide")

st.title("👥 Agent Risk Profiling")

# Binary classification threshold
BINARY_THRESHOLD = 0.5

try:
    data = load_all_data()
    agents = data['agents']
    
    # ========================================================================
    # MERGE NETWORK DATA IF AVAILABLE
    # ========================================================================
    
    try:
        agent_communities_path = Path(Config.DATA_DIR) / 'agent_communities.csv'
        if agent_communities_path.exists():
            agent_communities = pd.read_csv(agent_communities_path)
            
            # Merge columns
            merge_cols = ['agent_id', 'neighbor_avg_risk', 'community_id', 'degree']
            available_merge_cols = [col for col in merge_cols if col in agent_communities.columns]
            
            agents = agents.merge(
                agent_communities[available_merge_cols],
                on='agent_id',
                how='left',
                suffixes=('', '_network')
            )
    except Exception as e:
        pass
    
    # ========================================================================
    # ENSURE BINARY CLASSIFICATION
    # ========================================================================
    
    # Standardize risk categories to binary
    if 'risk_category' in agents.columns:
        # Map any variations to standard binary categories
        agents['risk_category'] = agents['risk_category'].map({
            'Low': 'Low Risk',
            'Low Risk': 'Low Risk',
            'High': 'High Risk',
            'High Risk': 'High Risk',
            'Medium': 'Low Risk',  # If any legacy data exists
            'Medium Risk': 'Low Risk'
        })
    else:
        # Create risk_category from risk_score if missing
        agents['risk_category'] = agents['risk_score'].apply(
            lambda x: 'High Risk' if x >= BINARY_THRESHOLD else 'Low Risk'
        )
    
    # Ensure risk_prediction column exists
    if 'risk_prediction' not in agents.columns:
        agents['risk_prediction'] = (agents['risk_score'] >= BINARY_THRESHOLD).astype(int)
    
    # ========================================================================
    # FILTERS
    # ========================================================================
    
    st.sidebar.markdown("## 🔍 Filters")
    
    risk_threshold = st.sidebar.slider(
        "Minimum Risk Score",
        0.0, 1.0, 0.0, 0.05,
        help="Filter agents by minimum risk probability"
    )
    
    # Binary risk category filter
    risk_category = st.sidebar.multiselect(
        "Risk Category",
        ['Low Risk', 'High Risk'],
        default=['Low Risk', 'High Risk'],
        help="Select risk categories to display"
    )
    
    # Agent type filter (if available)
    if 'agent_type' in agents.columns:
        agent_types = agents['agent_type'].dropna().unique().tolist()
        selected_types = st.sidebar.multiselect(
            "Agent Type",
            agent_types,
            default=agent_types,
            help="Filter by buyer or supplier"
        )
        
        filtered = agents[
            (agents['risk_score'] >= risk_threshold) &
            (agents['risk_category'].isin(risk_category)) &
            (agents['agent_type'].isin(selected_types))
        ]
    else:
        filtered = agents[
            (agents['risk_score'] >= risk_threshold) &
            (agents['risk_category'].isin(risk_category))
        ]
    
    # ========================================================================
    # SUMMARY METRICS
    # ========================================================================
    
    st.markdown("## 📊 Agent Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Agents", 
            f"{len(filtered):,}",
            help="Total number of agents matching filters"
        )
    
    with col2:
        high_risk = (filtered['risk_score'] >= BINARY_THRESHOLD).sum()
        high_risk_pct = (high_risk / len(filtered) * 100) if len(filtered) > 0 else 0
        st.metric(
            "High-Risk Agents", 
            f"{high_risk:,}",
            delta=f"{high_risk_pct:.1f}%",
            delta_color="inverse",
            help=f"Agents with risk score ≥ {BINARY_THRESHOLD}"
        )
    
    with col3:
        avg_risk = filtered['risk_score'].mean()
        st.metric(
            "Avg Risk Score", 
            f"{avg_risk:.3f}",
            help="Mean risk probability across filtered agents"
        )
    
    with col4:
        if 'neighbor_avg_risk' in filtered.columns and filtered['neighbor_avg_risk'].notna().sum() > 0:
            high_neighbor = (filtered['neighbor_avg_risk'] >= BINARY_THRESHOLD).sum()
            st.metric(
                "High-Risk Neighbors", 
                f"{high_neighbor:,}",
                help="Agents with high-risk network connections"
            )
        else:
            st.metric(
                "Network Data", 
                "N/A",
                help="Enable network analysis to see neighbor metrics"
            )
    
    # ========================================================================
    # RISK DISTRIBUTION
    # ========================================================================
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Risk Score Distribution")
        fig = plot_risk_distribution(filtered, 'risk_score', "Agent Risk Distribution")
        
        # Add threshold line
        fig.add_vline(
            x=BINARY_THRESHOLD, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Threshold ({BINARY_THRESHOLD})",
            annotation_position="top"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Binary Classification Summary")
        
        # Calculate metrics
        low_risk_count = (filtered['risk_score'] < BINARY_THRESHOLD).sum()
        high_risk_count = (filtered['risk_score'] >= BINARY_THRESHOLD).sum()
        total = len(filtered)
        
        if total > 0:
            low_pct = (low_risk_count / total * 100)
            high_pct = (high_risk_count / total * 100)
        else:
            low_pct = high_pct = 0
        
        # Create bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Low Risk', 'High Risk'],
            y=[low_risk_count, high_risk_count],
            text=[f"{low_risk_count:,}<br>({low_pct:.1f}%)", 
                  f"{high_risk_count:,}<br>({high_pct:.1f}%)"],
            textposition='auto',
            marker_color=['#5a9e6f', '#b05555'],
            hovertemplate='<b>%{x}</b><br>Count: %{y:,}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Risk Classification",
            xaxis_title="Category",
            yaxis_title="Number of Agents",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # NETWORK EFFECTS
    # ========================================================================
    
    if 'neighbor_avg_risk' in filtered.columns and filtered['neighbor_avg_risk'].notna().sum() > 0:
        st.markdown("---")
        st.markdown("## 🕸️ Network Risk Propagation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Agent Risk vs Neighbor Risk")
            
            # Filter to agents with network data
            filtered_with_network = filtered[filtered['neighbor_avg_risk'].notna()].copy()
            
            # Sample for performance
            sample_size = min(2000, len(filtered_with_network))
            sample_data = filtered_with_network.sample(n=sample_size, random_state=42)
            
            fig = px.scatter(
                sample_data,
                x='risk_score',
                y='neighbor_avg_risk',
                color='risk_category',
                title="Risk Propagation Analysis",
                color_discrete_map={
                    'Low Risk': '#5a9e6f',
                    'High Risk': '#b05555'
                },
                opacity=0.6,
                hover_data=['agent_id', 'degree'] if 'degree' in sample_data.columns else ['agent_id']
            )
            
            # Add diagonal reference line
            fig.add_trace(go.Scatter(
                x=[0, 1], 
                y=[0, 1],
                mode='lines',
                line=dict(dash='dash', color='gray', width=1),
                name='y=x',
                showlegend=True
            ))
            
            # Add threshold lines
            fig.add_hline(
                y=BINARY_THRESHOLD,
                line_dash="dot",
                line_color="red",
                opacity=0.5,
                annotation_text="High Risk Threshold",
                annotation_position="right"
            )
            fig.add_vline(
                x=BINARY_THRESHOLD,
                line_dash="dot",
                line_color="red",
                opacity=0.5
            )
            
            fig.update_layout(
                xaxis_title="Agent Risk Score",
                yaxis_title="Neighbor Avg Risk Score",
                height=450
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🔍 Risk Propagation Insights")
            
            # Calculate quadrants
            both_high = (
                (filtered_with_network['risk_score'] >= BINARY_THRESHOLD) &
                (filtered_with_network['neighbor_avg_risk'] >= BINARY_THRESHOLD)
            ).sum()
            
            agent_high_neighbor_low = (
                (filtered_with_network['risk_score'] >= BINARY_THRESHOLD) &
                (filtered_with_network['neighbor_avg_risk'] < BINARY_THRESHOLD)
            ).sum()
            
            agent_low_neighbor_high = (
                (filtered_with_network['risk_score'] < BINARY_THRESHOLD) &
                (filtered_with_network['neighbor_avg_risk'] >= BINARY_THRESHOLD)
            ).sum()
            
            both_low = (
                (filtered_with_network['risk_score'] < BINARY_THRESHOLD) &
                (filtered_with_network['neighbor_avg_risk'] < BINARY_THRESHOLD)
            ).sum()
            
            total_with_network = len(filtered_with_network)
            
            st.markdown(f"""
            **Network Risk Patterns** (n={total_with_network:,}):
            
            - 🔴 **Both High Risk**: {both_high:,} agents ({both_high/total_with_network*100:.1f}%)
              - Agent and neighbors both high risk
              - Highest fraud propagation concern
            
            - 🟡 **Agent High, Neighbors Low**: {agent_high_neighbor_low:,} agents ({agent_high_neighbor_low/total_with_network*100:.1f}%)
              - High-risk agent in low-risk network
              - Potential isolated fraud cases
            
            - 🟡 **Agent Low, Neighbors High**: {agent_low_neighbor_high:,} agents ({agent_low_neighbor_high/total_with_network*100:.1f}%)
              - Low-risk agent in high-risk network
              - At risk of exposure to fraud
            
            - 🟢 **Both Low Risk**: {both_low:,} agents ({both_low/total_with_network*100:.1f}%)
              - Agent and neighbors both low risk
              - Minimal fraud concern
            """)
            
            # Correlation analysis
            if len(filtered_with_network) > 0:
                correlation = filtered_with_network['risk_score'].corr(
                    filtered_with_network['neighbor_avg_risk']
                )
                st.markdown(f"""
                **Risk Correlation**: {correlation:.3f}
                - {'Positive' if correlation > 0 else 'Negative'} correlation indicates 
                  {"fraud tends to cluster in networks" if correlation > 0 else "fraud is distributed"}
                """)
    
    # ========================================================================
    # NETWORK CENTRALITY ANALYSIS
    # ========================================================================
    
    if 'degree' in filtered.columns and filtered['degree'].notna().sum() > 0:
        st.markdown("---")
        st.markdown("## 🌐 Network Centrality Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Network Degree vs Risk")
            
            filtered_with_network = filtered[filtered['degree'].notna()].copy()
            sample_size = min(2000, len(filtered_with_network))
            sample_data = filtered_with_network.sample(n=sample_size, random_state=42)
            
            fig = px.scatter(
                sample_data,
                x='degree',
                y='risk_score',
                color='risk_category',
                title="Network Centrality vs Risk Score",
                color_discrete_map={
                    'Low Risk': '#5a9e6f',
                    'High Risk': '#b05555'
                },
                opacity=0.6,
                hover_data=['agent_id']
            )
            
            # Add threshold line
            fig.add_hline(
                y=BINARY_THRESHOLD,
                line_dash="dash",
                line_color="red",
                annotation_text=f"High Risk Threshold",
                annotation_position="right"
            )
            
            fig.update_layout(
                xaxis_title="Network Degree (# of connections)",
                yaxis_title="Risk Score",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Key Insights")
            
            # High-risk highly connected agents
            high_risk_connected = filtered_with_network[
                (filtered_with_network['risk_score'] >= BINARY_THRESHOLD) &
                (filtered_with_network['degree'] > filtered_with_network['degree'].quantile(0.75))
            ]
            
            # Low-risk highly connected agents
            low_risk_connected = filtered_with_network[
                (filtered_with_network['risk_score'] < BINARY_THRESHOLD) &
                (filtered_with_network['degree'] > filtered_with_network['degree'].quantile(0.75))
            ]
            
            total_connections_high = high_risk_connected['degree'].sum() if len(high_risk_connected) > 0 else 0
            total_connections_low = low_risk_connected['degree'].sum() if len(low_risk_connected) > 0 else 0
            
            st.markdown(f"""
            **Highly Connected Agents** (top 25% degree):
            
            🔴 **High Risk + High Degree**:
            - **{len(high_risk_connected):,}** agents
            - **{total_connections_high:.0f}** total connections
            - High impact: fraud can propagate widely
            
            🟢 **Low Risk + High Degree**:
            - **{len(low_risk_connected):,}** agents
            - **{total_connections_low:.0f}** total connections
            - Lower concern: trusted central actors
            
            **Impact Assessment**:
            - Network centrality amplifies fraud impact
            - High-risk central agents require priority investigation
            - Monitor for collusion patterns in connected groups
            """)
    
    # ========================================================================
    # REPEATED WINNERS ANALYSIS
    # ========================================================================
    
    if 'degree' in filtered.columns and filtered['degree'].notna().sum() > 0:
        st.markdown("---")
        st.markdown("## 🏆 Repeated Winners Analysis")
        
        filtered_with_network = filtered[filtered['degree'].notna()].copy()
        
        if len(filtered_with_network) > 0:
            # Top 10% by degree = repeated winners
            threshold_degree = filtered_with_network['degree'].quantile(0.9)
            repeated_winners = filtered_with_network[filtered_with_network['degree'] >= threshold_degree]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Repeated Winners", f"{len(repeated_winners):,}")
                
                high_risk_winners = (repeated_winners['risk_score'] >= BINARY_THRESHOLD).sum()
                st.metric("High-Risk Winners", f"{high_risk_winners:,}")
                
                if len(repeated_winners) > 0:
                    pct_high_risk = (high_risk_winners / len(repeated_winners) * 100)
                    st.metric("High-Risk %", f"{pct_high_risk:.1f}%")
            
            with col2:
                # Distribution chart
                winner_risk_dist = repeated_winners['risk_category'].value_counts()
                
                fig = px.pie(
                    values=winner_risk_dist.values,
                    names=winner_risk_dist.index,
                    title="Repeated Winners Risk Distribution",
                    color=winner_risk_dist.index,
                    color_discrete_map={
                        'Low Risk': '#5a9e6f',
                        'High Risk': '#b05555'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                st.markdown("### 📊 Findings")
                st.markdown(f"""
                **Detection Method**:
                - Identified top 10% by contract count
                - ML assesses risk through:
                  - Contract pattern anomalies
                  - Network position analysis
                  - Value distribution irregularities
                
                **Risk Assessment**:
                - {high_risk_winners:,} / {len(repeated_winners):,} flagged as high risk
                - {'⚠️ High concentration' if pct_high_risk > 30 else '✓ Acceptable distribution'}
                """)
    
        # ========================================================================
        # RISK CATEGORY BREAKDOWN
        # ========================================================================
        
        st.markdown("---")
        st.markdown("## 📊 Detailed Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Risk Category Distribution")
            
            category_counts = filtered['risk_category'].value_counts()
            
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Agent Risk Categories",
                color=category_counts.index,
                color_discrete_map={
                    'Low Risk': '#5a9e6f',
                    'High Risk': '#b05555'
                },
                hole=0.4
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Risk by Agent Type")
            
            if 'agent_type' in filtered.columns:
                # *** FIXED: Keep column name consistent ***
                # Aggregate by type (don't rename yet)
                type_risk = filtered.groupby('agent_type').agg({
                    'risk_score': 'mean',
                    'agent_id': 'count'
                }).reset_index()
                type_risk.columns = ['agent_type', 'Avg Risk Score', 'Count']
                
                # Add high-risk count (using the same column name)
                high_risk_by_type = filtered[filtered['risk_score'] >= BINARY_THRESHOLD].groupby('agent_type').size().reset_index(name='High Risk Count')
                type_risk = type_risk.merge(high_risk_by_type, on='agent_type', how='left')
                
                type_risk['High Risk Count'] = type_risk['High Risk Count'].fillna(0).astype(int)
                type_risk['High Risk %'] = (type_risk['High Risk Count'] / type_risk['Count'] * 100).round(1)
                
                # NOW rename for display
                type_risk = type_risk.rename(columns={'agent_type': 'Agent Type'})
                
                # Reorder columns
                type_risk = type_risk[['Agent Type', 'Count', 'High Risk Count', 'High Risk %', 'Avg Risk Score']]
                type_risk['Avg Risk Score'] = type_risk['Avg Risk Score'].round(4)
                
                st.dataframe(type_risk, use_container_width=True, hide_index=True)
            else:
                st.info("Agent type information not available")
        
        # ========================================================================
        # TOP RISKY AGENTS TABLE
        # ========================================================================
        
        st.markdown("---")
        st.markdown("## ⚠️ Top Risk Agents")
        
        # Select columns to display
        display_cols = ['agent_id', 'risk_score', 'risk_prediction', 'risk_category']
        
        if 'agent_type' in filtered.columns:
            display_cols.append('agent_type')
        if 'degree' in filtered.columns:
            display_cols.append('degree')
        if 'neighbor_avg_risk' in filtered.columns:
            display_cols.append('neighbor_avg_risk')
        if 'community_id' in filtered.columns:
            display_cols.append('community_id')
        if 'contract_count' in filtered.columns:
            display_cols.append('contract_count')
        
        available_cols = [c for c in display_cols if c in filtered.columns]
        
        top_agents = filtered.nlargest(100, 'risk_score')[available_cols].copy()
        
        # Format numeric columns
        for col in ['risk_score', 'neighbor_avg_risk']:
            if col in top_agents.columns:
                top_agents[col] = top_agents[col].round(4)
        
        # Rename for clarity
        column_rename = {
            'risk_prediction': 'Prediction (0/1)',
            'risk_score': 'Risk Probability',
            'neighbor_avg_risk': 'Neighbor Avg Risk',
            'contract_count': '# Contracts'
        }
        top_agents = top_agents.rename(columns=column_rename)
        
        st.dataframe(
            top_agents, 
            use_container_width=True, 
            height=500, 
            hide_index=True
        )

    st.markdown("---")
    st.markdown("## 🔍 Agent Detail Viewer")

    st.info("💡 **Tip**: Select an agent from the table above to view detailed analysis")

    # Agent selector
    col1, col2 = st.columns([2, 1])

    with col1:
        # Get list of high-risk agents for quick access
        high_risk_agent_ids = filtered[filtered['risk_score'] >= BINARY_THRESHOLD]['agent_id'].tolist()
        
        selected_agent_id = st.selectbox(
            "Select Agent to Analyze",
            options=sorted(filtered['agent_id'].unique()),
            format_func=lambda x: f"Agent {x} - Risk: {filtered[filtered['agent_id']==x]['risk_score'].values[0]:.3f}",
            help="Choose an agent to see detailed profile"
        )

    with col2:
        # Quick stats for selected agent
        agent_info = filtered[filtered['agent_id'] == selected_agent_id].iloc[0]
        st.metric("Agent Risk Score", f"{agent_info['risk_score']:.3f}")
        st.metric("Risk Category", agent_info['risk_category'])

    if selected_agent_id:
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Agent Profile", 
            "📄 Contract History", 
            "🕸️ Network & Communities",
            "🔍 Risk Explanation"
        ])
        
        # ====================================================================
        # TAB 1: AGENT PROFILE
        # ====================================================================
        
        with tab1:
            st.markdown(f"### Agent Profile: {selected_agent_id}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Basic Information")
                st.markdown(f"""
                - **Agent ID**: {selected_agent_id}
                - **Risk Score**: {agent_info['risk_score']:.4f}
                - **Risk Prediction**: {'🔴 High Risk' if agent_info['risk_prediction'] == 1 else '🟢 Low Risk'}
                - **Risk Category**: {agent_info['risk_category']}
                """)
                
                if 'agent_type' in agent_info.index:
                    st.markdown(f"- **Agent Type**: {agent_info['agent_type']}")
                
                if 'contract_count' in agent_info.index:
                    st.markdown(f"- **Total Contracts**: {agent_info['contract_count']:,}")
            
            with col2:
                st.markdown("#### Network Statistics")
                if 'degree' in agent_info.index:
                    st.markdown(f"- **Network Degree**: {agent_info['degree']}")
                    
                    # Compare to average
                    avg_degree = filtered['degree'].mean()
                    degree_pct = (agent_info['degree'] / avg_degree - 1) * 100
                    st.markdown(f"  - {degree_pct:+.1f}% vs average")
                
                if 'neighbor_avg_risk' in agent_info.index:
                    st.markdown(f"- **Neighbor Avg Risk**: {agent_info['neighbor_avg_risk']:.4f}")
                    
                    neighbor_status = "🔴 High" if agent_info['neighbor_avg_risk'] >= BINARY_THRESHOLD else "🟢 Low"
                    st.markdown(f"  - Risk Network: {neighbor_status}")
                
                if 'pagerank' in agent_info.index:
                    st.markdown(f"- **PageRank**: {agent_info['pagerank']:.6f}")
            
            with col3:
                st.markdown("#### Community Information")
                if 'community_id' in agent_info.index:
                    comm_id = agent_info['community_id']
                    st.markdown(f"- **Community ID**: {comm_id}")
                    
                    # Get community info
                    try:
                        comm_summary = pd.read_csv(Path(Config.DATA_DIR) / 'community_risk_summary.csv')
                        comm_info = comm_summary[comm_summary['community_id'] == comm_id].iloc[0]
                        
                        st.markdown(f"- **Community Size**: {comm_info['size']:,} agents")
                        st.markdown(f"- **Community Avg Risk**: {comm_info['avg_risk_score']:.4f}")
                        st.markdown(f"- **High Risk %**: {comm_info['high_risk_pct']:.1f}%")
                        
                        comm_status = "🔴 High Risk" if comm_info['avg_risk_score'] >= BINARY_THRESHOLD else "🟢 Low Risk"
                        st.markdown(f"- **Community Status**: {comm_status}")
                    except:
                        st.markdown("- Community details unavailable")
                else:
                    st.markdown("- No community assignment")
            
            # Risk comparison visualization
            st.markdown("#### Risk Comparison")
            
            # Get percentile
            percentile = (filtered['risk_score'] < agent_info['risk_score']).sum() / len(filtered) * 100
            
            fig = go.Figure()
            
            # Add histogram of all agents
            fig.add_trace(go.Histogram(
                x=filtered['risk_score'],
                nbinsx=50,
                name='All Agents',
                marker_color='lightblue',
                opacity=0.7
            ))
            
            # Add vertical line for selected agent
            fig.add_vline(
                x=agent_info['risk_score'],
                line_dash="dash",
                line_color="red",
                line_width=3,
                annotation_text=f"This Agent (Top {100-percentile:.1f}%)",
                annotation_position="top"
            )
            
            # Add threshold line
            fig.add_vline(
                x=BINARY_THRESHOLD,
                line_dash="dot",
                line_color="orange",
                annotation_text="Threshold",
                annotation_position="bottom"
            )
            
            fig.update_layout(
                title="Agent Risk Score in Context",
                xaxis_title="Risk Score",
                yaxis_title="Number of Agents",
                showlegend=False,
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # ====================================================================
        # TAB 2: CONTRACT HISTORY
        # ====================================================================
        
        with tab2:
            st.markdown(f"### Contract History for Agent {selected_agent_id}")
            
            # Load contract data
            try:
                contracts_df = pd.read_csv(Path(Config.DATA_DIR) / 'contract_risk_predictions.csv')
                
                # Get contracts where agent is buyer or supplier
                agent_contracts = contracts_df[
                    (contracts_df['buyerId'] == selected_agent_id) | 
                    (contracts_df['supplierId'] == selected_agent_id)
                ].copy()
                
                # Add role column
                agent_contracts['agent_role'] = agent_contracts.apply(
                    lambda row: 'Buyer' if row['buyerId'] == selected_agent_id else 'Supplier',
                    axis=1
                )
                
                # Add counterparty column
                agent_contracts['counterparty_id'] = agent_contracts.apply(
                    lambda row: row['supplierId'] if row['buyerId'] == selected_agent_id else row['buyerId'],
                    axis=1
                )
                
                if len(agent_contracts) > 0:
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Contracts", f"{len(agent_contracts):,}")
                    
                    with col2:
                        as_buyer = (agent_contracts['agent_role'] == 'Buyer').sum()
                        st.metric("As Buyer", f"{as_buyer:,}")
                    
                    with col3:
                        as_supplier = (agent_contracts['agent_role'] == 'Supplier').sum()
                        st.metric("As Supplier", f"{as_supplier:,}")
                    
                    with col4:
                        high_risk_contracts = (agent_contracts['risk_score'] >= BINARY_THRESHOLD).sum()
                        high_risk_pct = (high_risk_contracts / len(agent_contracts) * 100)
                        st.metric("High Risk Contracts", f"{high_risk_contracts:,}", f"{high_risk_pct:.1f}%")
                    
                    # Temporal analysis
                    if 'awardYear' in agent_contracts.columns:
                        st.markdown("#### Contract Activity Over Time")
                        
                        yearly_stats = agent_contracts.groupby('awardYear').agg({
                            'lotId': 'count',
                            'risk_score': 'mean'
                        }).reset_index()
                        yearly_stats.columns = ['Year', 'Contract Count', 'Avg Risk Score']
                        
                        fig = go.Figure()
                        
                        # Bar chart for count
                        fig.add_trace(go.Bar(
                            x=yearly_stats['Year'],
                            y=yearly_stats['Contract Count'],
                            name='Contract Count',
                            yaxis='y',
                            marker_color='lightblue'
                        ))
                        
                        # Line chart for risk
                        fig.add_trace(go.Scatter(
                            x=yearly_stats['Year'],
                            y=yearly_stats['Avg Risk Score'],
                            name='Avg Risk Score',
                            yaxis='y2',
                            line=dict(color='red', width=3),
                            mode='lines+markers'
                        ))
                        
                        fig.update_layout(
                            title="Yearly Contract Activity & Risk",
                            xaxis_title="Year",
                            yaxis=dict(title="Contract Count"),
                            yaxis2=dict(title="Avg Risk Score", overlaying='y', side='right'),
                            hovermode='x unified',
                            height=350
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk distribution by role
                    st.markdown("#### Risk Distribution by Role")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        buyer_contracts = agent_contracts[agent_contracts['agent_role'] == 'Buyer']
                        if len(buyer_contracts) > 0:
                            fig = go.Figure()
                            fig.add_trace(go.Histogram(
                                x=buyer_contracts['risk_score'],
                                nbinsx=30,
                                marker_color='blue',
                                opacity=0.7
                            ))
                            fig.add_vline(x=BINARY_THRESHOLD, line_dash="dash", line_color="red")
                            fig.update_layout(
                                title=f"As Buyer (n={len(buyer_contracts)})",
                                xaxis_title="Risk Score",
                                yaxis_title="Count",
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No contracts as buyer")
                    
                    with col2:
                        supplier_contracts = agent_contracts[agent_contracts['agent_role'] == 'Supplier']
                        if len(supplier_contracts) > 0:
                            fig = go.Figure()
                            fig.add_trace(go.Histogram(
                                x=supplier_contracts['risk_score'],
                                nbinsx=30,
                                marker_color='green',
                                opacity=0.7
                            ))
                            fig.add_vline(x=BINARY_THRESHOLD, line_dash="dash", line_color="red")
                            fig.update_layout(
                                title=f"As Supplier (n={len(supplier_contracts)})",
                                xaxis_title="Risk Score",
                                yaxis_title="Count",
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No contracts as supplier")
                    
                    # Detailed contract table
                    st.markdown("#### All Contracts")
                    
                    # Prepare display columns
                    display_cols = ['lotId', 'awardYear', 'agent_role', 'counterparty_id', 
                                'risk_score', 'predicted_risk_label']
                    
                    if 'awardPrice_log' in agent_contracts.columns:
                        display_cols.append('awardPrice_log')
                    if 'cpv' in agent_contracts.columns:
                        display_cols.append('cpv')
                    if 'typeOfContract' in agent_contracts.columns:
                        display_cols.append('typeOfContract')
                    
                    available_cols = [c for c in display_cols if c in agent_contracts.columns]
                    contract_table = agent_contracts[available_cols].copy()
                    
                    # Format and rename
                    contract_table['risk_score'] = contract_table['risk_score'].round(4)
                    contract_table = contract_table.rename(columns={
                        'lotId': 'Contract ID',
                        'awardYear': 'Year',
                        'agent_role': 'Role',
                        'counterparty_id': 'Counterparty',
                        'risk_score': 'Risk Score',
                        'predicted_risk_label': 'Risk Label',
                        'awardPrice_log': 'Log Price',
                        'cpv': 'CPV Code',
                        'typeOfContract': 'Type'
                    })
                    
                    # Sort by risk score
                    contract_table = contract_table.sort_values('Risk Score', ascending=False)
                    
                    st.dataframe(contract_table, use_container_width=True, height=400, hide_index=True)
                    
                    # Download option
                    csv = agent_contracts.to_csv(index=False)
                    st.download_button(
                        "📥 Download Agent Contracts",
                        csv,
                        f"agent_{selected_agent_id}_contracts.csv",
                        "text/csv"
                    )
                else:
                    st.warning(f"No contracts found for Agent {selected_agent_id}")
            
            except Exception as e:
                st.error(f"Error loading contract data: {e}")
        
        # ====================================================================
        # TAB 3: NETWORK & COMMUNITIES
        # ====================================================================
        
        with tab3:
            st.markdown(f"### Network Analysis for Agent {selected_agent_id}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Community Membership")
                
                if 'community_id' in agent_info.index:
                    comm_id = agent_info['community_id']
                    
                    try:
                        # Load community data
                        agent_communities_df = pd.read_csv(Path(Config.DATA_DIR) / 'agent_communities.csv')
                        comm_summary = pd.read_csv(Path(Config.DATA_DIR) / 'community_risk_summary.csv')
                        
                        # Get community members
                        community_members = agent_communities_df[
                            agent_communities_df['community_id'] == comm_id
                        ].copy()
                        
                        comm_info = comm_summary[comm_summary['community_id'] == comm_id].iloc[0]
                        
                        st.markdown(f"""
                        **Community {comm_id} Details:**
                        - **Total Members**: {comm_info['size']:,}
                        - **Avg Risk Score**: {comm_info['avg_risk_score']:.4f}
                        - **High Risk Count**: {comm_info['high_risk_count']:,}
                        - **High Risk %**: {comm_info['high_risk_pct']:.1f}%
                        - **Avg Degree**: {comm_info['avg_degree']:.2f}
                        - **Status**: {'🔴 High Risk Community' if comm_info['avg_risk_score'] >= BINARY_THRESHOLD else '🟢 Low Risk Community'}
                        """)
                        
                        # Risk distribution in community
                        fig = px.histogram(
                            community_members,
                            x='risk_score',
                            nbins=30,
                            title=f"Risk Distribution in Community {comm_id}",
                            labels={'risk_score': 'Risk Score', 'count': 'Number of Agents'}
                        )
                        
                        # Highlight selected agent
                        fig.add_vline(
                            x=agent_info['risk_score'],
                            line_dash="dash",
                            line_color="red",
                            annotation_text="This Agent"
                        )
                        
                        fig.add_vline(
                            x=BINARY_THRESHOLD,
                            line_dash="dot",
                            line_color="orange",
                            annotation_text="Threshold"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Top members in community
                        st.markdown("##### Top 10 Risky Members in Community")
                        top_members = community_members.nlargest(10, 'risk_score')[
                            ['agent_id', 'risk_score', 'risk_category', 'degree']
                        ].copy()
                        
                        # Highlight if selected agent is in top 10
                        top_members['is_selected'] = top_members['agent_id'] == selected_agent_id
                        top_members['risk_score'] = top_members['risk_score'].round(4)
                        
                        st.dataframe(
                            top_members.style.apply(
                                lambda row: ['background-color: #ffe0e0' if row['is_selected'] else '' for _ in row],
                                axis=1
                            ),
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    except Exception as e:
                        st.error(f"Error loading community data: {e}")
                else:
                    st.info("Agent not assigned to any community")
            
            with col2:
                st.markdown("#### Network Connections")
                
                try:
                    contracts_df = pd.read_csv(Path(Config.DATA_DIR) / 'contract_risk_predictions.csv')
                    
                    # Get all connected agents
                    connections = contracts_df[
                        (contracts_df['buyerId'] == selected_agent_id) | 
                        (contracts_df['supplierId'] == selected_agent_id)
                    ].copy()
                    
                    # Extract counterparties
                    counterparties = []
                    for _, row in connections.iterrows():
                        if row['buyerId'] == selected_agent_id:
                            counterparties.append(row['supplierId'])
                        else:
                            counterparties.append(row['buyerId'])
                    
                    unique_counterparties = len(set(counterparties))
                    
                    st.metric("Unique Connections", f"{unique_counterparties:,}")
                    st.metric("Total Interactions", f"{len(connections):,}")
                    
                    # Most frequent counterparties
                    if counterparties:
                        from collections import Counter
                        counter_counts = Counter(counterparties)
                        top_counterparties = counter_counts.most_common(10)
                        
                        st.markdown("##### Top 10 Frequent Counterparties")
                        
                        cp_df = pd.DataFrame(top_counterparties, columns=['Agent ID', 'Contract Count'])
                        
                        # Get risk scores for counterparties
                        cp_risks = []
                        for cp_id in cp_df['Agent ID']:
                            cp_risk = agents[agents['agent_id'] == cp_id]['risk_score'].values
                            cp_risks.append(cp_risk[0] if len(cp_risk) > 0 else None)
                        
                        cp_df['Risk Score'] = cp_risks
                        cp_df['Risk Score'] = cp_df['Risk Score'].round(4)
                        
                        st.dataframe(cp_df, use_container_width=True, hide_index=True)
                        
                        # Visualization
                        fig = px.bar(
                            cp_df,
                            x='Contract Count',
                            y='Agent ID',
                            orientation='h',
                            color='Risk Score',
                            color_continuous_scale='Reds',
                            title="Frequent Counterparties by Risk"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error loading network connections: {e}")
        
        # ====================================================================
        # TAB 4: RISK EXPLANATION
        # ====================================================================
        
        with tab4:
            st.markdown(f"### Risk Explanation for Agent {selected_agent_id}")
            
            st.markdown(f"""
            **Risk Assessment Summary:**
            - **Risk Score**: {agent_info['risk_score']:.4f}
            - **Classification**: {'🔴 High Risk (≥0.5)' if agent_info['risk_score'] >= BINARY_THRESHOLD else '🟢 Low Risk (<0.5)'}
            - **Percentile**: Top {100 - percentile:.1f}% riskiest
            """)
            
            st.markdown("#### Key Risk Factors")
            
            # Analyze different risk components
            risk_factors = []
            
            # 1. Individual risk score
            if agent_info['risk_score'] >= BINARY_THRESHOLD:
                risk_factors.append({
                    'Factor': 'High Agent Risk Score',
                    'Value': f"{agent_info['risk_score']:.4f}",
                    'Impact': '🔴 High',
                    'Explanation': 'Agent has elevated individual risk based on contract patterns'
                })
            
            # 2. Network influence
            if 'neighbor_avg_risk' in agent_info.index:
                if agent_info['neighbor_avg_risk'] >= BINARY_THRESHOLD:
                    risk_factors.append({
                        'Factor': 'High-Risk Network',
                        'Value': f"{agent_info['neighbor_avg_risk']:.4f}",
                        'Impact': '🔴 High',
                        'Explanation': 'Agent is connected to other high-risk entities'
                    })
                else:
                    risk_factors.append({
                        'Factor': 'Low-Risk Network',
                        'Value': f"{agent_info['neighbor_avg_risk']:.4f}",
                        'Impact': '🟢 Low',
                        'Explanation': 'Agent\'s connections have lower risk profiles'
                    })
            
            # 3. Network centrality
            if 'degree' in agent_info.index:
                avg_degree = filtered['degree'].mean()
                if agent_info['degree'] > avg_degree * 1.5:
                    risk_factors.append({
                        'Factor': 'High Network Centrality',
                        'Value': f"{agent_info['degree']} connections",
                        'Impact': '🟡 Medium',
                        'Explanation': 'Agent is highly connected - risk can propagate widely if fraudulent'
                    })
            
            # 4. Community risk
            if 'community_id' in agent_info.index:
                try:
                    comm_summary = pd.read_csv(Path(Config.DATA_DIR) / 'community_risk_summary.csv')
                    comm_info = comm_summary[comm_summary['community_id'] == agent_info['community_id']].iloc[0]
                    
                    if comm_info['avg_risk_score'] >= BINARY_THRESHOLD:
                        risk_factors.append({
                            'Factor': 'High-Risk Community',
                            'Value': f"{comm_info['avg_risk_score']:.4f}",
                            'Impact': '🔴 High',
                            'Explanation': f"Part of Community {agent_info['community_id']} with elevated risk"
                        })
                except:
                    pass
            
            # Display risk factors table
            if risk_factors:
                factors_df = pd.DataFrame(risk_factors)
                st.dataframe(factors_df, use_container_width=True, hide_index=True)
            
            st.markdown("#### Model Features (if available)")
            
            st.info("""
            **How Risk is Calculated:**
            
            The agent risk score is derived from:
            1. **Contract-level predictions**: ML model analyzes each contract for fraud indicators
            2. **Aggregation**: Agent risk = average of all their contract risk scores
            3. **Network effects**: Considers connections to other high-risk entities
            4. **Community analysis**: Evaluates risk clustering in network communities
            
            **Key Features Used:**
            - Contract value anomalies
            - Procurement transparency indicators
            - Competition levels
            - Regulatory compliance
            - Network centrality metrics
            - Temporal patterns
            """)
            
            # Recommendation based on risk level
            st.markdown("#### Recommended Actions")
            
            if agent_info['risk_score'] >= 0.7:
                st.error(f"""
                **🔴 Critical Risk Level (Score: {agent_info['risk_score']:.3f})**
                
                Immediate Actions:
                - Priority investigation required
                - Review all high-risk contracts
                - Audit procurement procedures
                - Check for collusion patterns with connected entities
                - Consider suspension pending investigation
                """)
            elif agent_info['risk_score'] >= BINARY_THRESHOLD:
                st.warning(f"""
                **🟡 Elevated Risk Level (Score: {agent_info['risk_score']:.3f})**
                
                Recommended Actions:
                - Enhanced monitoring
                - Sample contract audits
                - Review unusual patterns
                - Monitor network connections
                """)
            else:
                st.success(f"""
                **🟢 Low Risk Level (Score: {agent_info['risk_score']:.3f})**
                
                Status:
                - Standard monitoring sufficient
                - No immediate concerns
                - Continue routine oversight
                """)   
        
        # ========================================================================
        # DOWNLOAD OPTIONS
        # ========================================================================
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download filtered data
            csv = filtered.to_csv(index=False)
            st.download_button(
                "📥 Download Filtered Agent Data",
                csv,
                "agent_risk_profiles.csv",
                "text/csv",
                help="Download all filtered agents with risk scores"
            )
        
        with col2:
            # Download high-risk agents only
            high_risk_agents = filtered[filtered['risk_score'] >= BINARY_THRESHOLD]
            csv_high_risk = high_risk_agents.to_csv(index=False)
            st.download_button(
                "📥 Download High-Risk Agents Only",
                csv_high_risk,
                "high_risk_agents.csv",
                "text/csv",
                help=f"Download only agents with risk score ≥ {BINARY_THRESHOLD}"
            )
except Exception as e:
    st.error(f"Error loading agent data: {e}")
    st.stop()