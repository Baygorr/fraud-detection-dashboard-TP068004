"""
================================================================================
PAGE 4: NETWORK ANALYSIS (BINARY CLASSIFICATION - FIXED VISUALIZATION)
================================================================================
File: deployment/pages/4_Network.py

Community detection and risk propagation with interactive network graphs
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.loaders import load_all_data
from config import Config

st.set_page_config(page_title="Network", page_icon="🕸️", layout="wide")

st.title("🕸️ Network Analysis & Community Detection")

# Binary threshold
BINARY_THRESHOLD = 0.5

try:
    data = load_all_data()
    agents = data['agents']
    
    if 'community_summary' not in data:
        st.warning("⚠️ Community data not available. Run network analysis in setup_dashboard.py first.")
        st.stop()
    
    communities = data['communities'] if 'communities' in data else agents
    community_summary = data['community_summary']
    
    # ========================================================================
    # LOAD CONTRACT DATA FOR VISUALIZATION
    # ========================================================================
    
    # Try to load contract predictions
    contracts_path = Path(Config.DATA_DIR) / 'contract_risk_predictions.csv'
    
    if not contracts_path.exists():
        st.error("❌ Contract data not found. Please run setup_dashboard.py first.")
        st.stop()
    
    # Load contracts (limit to prevent memory issues)
    contracts = pd.read_csv(contracts_path)
    st.sidebar.success(f"✅ Loaded {len(contracts):,} contracts")
    
    # ========================================================================
    # COMMUNITY OVERVIEW
    # ========================================================================
    
    st.markdown("## 📊 Community Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Communities", f"{len(community_summary):,}")
    
    with col2:
        high_risk_comm = (community_summary['avg_risk_score'] >= BINARY_THRESHOLD).sum()
        st.metric("High-Risk Communities", f"{high_risk_comm:,}")
    
    with col3:
        avg_size = community_summary['size'].mean()
        st.metric("Avg Community Size", f"{avg_size:.1f}")
    
    with col4:
        largest = community_summary['size'].max()
        st.metric("Largest Community", f"{largest:,}")
    
    # ========================================================================
    # IMPROVED RISK DISTRIBUTION (TOP N COMMUNITIES)
    # ========================================================================
    
    st.markdown("---")
    st.markdown("## 📈 Risk Distribution Across Communities")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        top_n = st.slider(
            "Number of Communities to Display",
            min_value=50,
            max_value=min(500, len(community_summary)),
            value=min(200, len(community_summary)),
            step=50,
            help="Reduce this to make the chart less crowded"
        )
    
    with col2:
        sort_by = st.selectbox(
            "Sort By",
            ["Risk Score", "Community Size", "High Risk Count"],
            help="Choose how to sort communities"
        )
    
    # Sort communities
    if sort_by == "Risk Score":
        display_communities = community_summary.nlargest(top_n, 'avg_risk_score')
    elif sort_by == "Community Size":
        display_communities = community_summary.nlargest(top_n, 'size')
    else:
        display_communities = community_summary.nlargest(top_n, 'high_risk_count')
    
    # Create improved bar chart
    fig = go.Figure()
    
    colors = ['#b05555' if x >= BINARY_THRESHOLD else '#5a9e6f' 
              for x in display_communities['avg_risk_score']]
    
    fig.add_trace(go.Bar(
        x=display_communities['community_id'],
        y=display_communities['avg_risk_score'],
        marker_color=colors,
        hovertemplate=(
            '<b>Community %{x}</b><br>' +
            'Avg Risk Score: %{y:.3f}<br>' +
            'Size: %{customdata[0]}<br>' +
            'High Risk Count: %{customdata[1]}<br>' +
            'High Risk %: %{customdata[2]:.1f}%<br>' +
            '<extra></extra>'
        ),
        customdata=display_communities[['size', 'high_risk_count', 'high_risk_pct']].values
    ))
    
    fig.add_hline(
        y=BINARY_THRESHOLD,
        line_dash="dash",
        line_color="red",
        annotation_text=f"High Risk Threshold ({BINARY_THRESHOLD})",
        annotation_position="right"
    )
    
    fig.update_layout(
        title=f"Top {top_n} Communities by {sort_by}",
        xaxis_title="Community ID",
        yaxis_title="Average Risk Score",
        showlegend=False,
        height=500,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # INTERACTIVE NETWORK GRAPH (ENHANCED - ALL COMMUNITIES + YEAR FILTER)
    # ========================================================================
    
    st.markdown("---")
    st.markdown("## 🕸️ Interactive Network Visualization")
    
    # Sort all communities by risk score
    all_communities_sorted = community_summary.sort_values('avg_risk_score', ascending=False)
    
    # Create filter options
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Option to view by community or all communities
        view_mode = st.radio(
            "View Mode",
            ["Specific Community", "All Communities by Year"],
            horizontal=True,
            help="Choose to view a specific community or all communities filtered by year"
        )
    
    with col2:
        # Year filter (applies to both modes)
        available_years = sorted(contracts['awardYear'].unique())
        selected_year = st.selectbox(
            "Filter by Year",
            ["All Years"] + available_years,
            help="Filter contracts by award year"
        )
    
    with col3:
        max_nodes = st.slider(
            "Max Nodes",
            min_value=50,
            max_value=500,
            value=100,
            step=50,
            help="Limit nodes for performance"
        )
    
    # Community selection (only shown in specific community mode)
    if view_mode == "Specific Community":
        st.markdown("### Select Community")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Show all communities, not just top 20
            selected_community = st.selectbox(
                "Select Community to Visualize",
                all_communities_sorted['community_id'].values,
                format_func=lambda x: f"Community {x} (Risk: {all_communities_sorted[all_communities_sorted['community_id']==x]['avg_risk_score'].values[0]:.3f}, Size: {all_communities_sorted[all_communities_sorted['community_id']==x]['size'].values[0]})",
                help="Choose any community to see its network structure"
            )
        
        with col2:
            # Show selected community info
            comm_info = all_communities_sorted[all_communities_sorted['community_id'] == selected_community].iloc[0]
            st.metric("Community Size", f"{comm_info['size']:,}")
            st.metric("Avg Risk", f"{comm_info['avg_risk_score']:.3f}")
        
        # Get agents in selected community
        community_agents = communities[
            communities['community_id'] == selected_community
        ].copy()
        
        # Limit to top N by risk score
        community_agents = community_agents.nlargest(max_nodes, 'risk_score')
        agent_ids_set = set(community_agents['agent_id'].values)
        
        st.markdown(f"**Showing:** {len(community_agents)} agents from Community {selected_community}")
        
    else:  # All Communities by Year
        st.markdown("### Viewing All Communities")
        
        if selected_year == "All Years":
            # Get top N agents across all communities
            community_agents = communities.nlargest(max_nodes, 'risk_score')
            agent_ids_set = set(community_agents['agent_id'].values)
            st.info(f"📊 Showing top {max_nodes} highest-risk agents across all communities and years")
        else:
            st.info(f"📊 Showing top {max_nodes} highest-risk agents across all communities for year {selected_year}")
            # We'll filter by year through contracts later
            community_agents = communities.nlargest(max_nodes * 2, 'risk_score')  # Get more to ensure enough after filtering
            agent_ids_set = set(community_agents['agent_id'].values)
    
    # ========================================================================
    # FILTER CONTRACTS BY YEAR AND AGENTS
    # ========================================================================
    
    # Filter contracts
    edges = contracts[
        contracts['buyerId'].isin(agent_ids_set) | 
        contracts['supplierId'].isin(agent_ids_set)
    ].copy()
    
    # Apply year filter if selected
    if selected_year != "All Years":
        edges = edges[edges['awardYear'] == selected_year]
        st.markdown(f"**Year Filter:** {selected_year} - Found {len(edges):,} contracts")
        
        # Update agent set based on filtered contracts
        agent_ids_in_year = set(edges['buyerId'].unique()) | set(edges['supplierId'].unique())
        agent_ids_set = agent_ids_set.intersection(agent_ids_in_year)
        
        # Update community_agents to only include agents with contracts in this year
        community_agents = community_agents[community_agents['agent_id'].isin(agent_ids_set)]
        
        # Limit to max_nodes after filtering
        if len(community_agents) > max_nodes:
            community_agents = community_agents.nlargest(max_nodes, 'risk_score')
            agent_ids_set = set(community_agents['agent_id'].values)
            
            # Re-filter edges with updated agent set
            edges = edges[
                edges['buyerId'].isin(agent_ids_set) | 
                edges['supplierId'].isin(agent_ids_set)
            ]
    
    st.markdown(f"**Network Data:** {len(community_agents)} agents, {len(edges):,} contracts")
    
    # ========================================================================
    # BUILD AND VISUALIZE NETWORK
    # ========================================================================
    
    if len(edges) > 0 and len(community_agents) > 0:
        # Limit edges for performance
        max_edges = 1000
        if len(edges) > max_edges:
            edges = edges.nlargest(max_edges, 'risk_score')
            st.info(f"📊 Displaying top {max_edges} highest-risk contracts out of {len(edges):,} for performance")
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes with attributes from community_agents
        for _, agent in community_agents.iterrows():
            G.add_node(
                agent['agent_id'],
                risk_score=agent['risk_score'],
                risk_category=agent.get('risk_category', 'Unknown'),
                degree=agent.get('degree', 0),
                community_id=agent.get('community_id', -1)
            )
        
        # Add edges from contracts
        for _, edge in edges.iterrows():
            buyer = edge['buyerId']
            supplier = edge['supplierId']
            
            # Add nodes if they don't exist
            if buyer not in G.nodes():
                buyer_data = communities[communities['agent_id'] == buyer]
                if len(buyer_data) > 0:
                    buyer_risk = buyer_data['risk_score'].iloc[0]
                    buyer_comm = buyer_data['community_id'].iloc[0]
                    buyer_cat = buyer_data.get('risk_category', pd.Series(['Unknown'])).iloc[0]
                else:
                    buyer_risk = 0.5
                    buyer_comm = -1
                    buyer_cat = 'Unknown'
                G.add_node(buyer, risk_score=buyer_risk, risk_category=buyer_cat, 
                          degree=0, community_id=buyer_comm)
            
            if supplier not in G.nodes():
                supplier_data = communities[communities['agent_id'] == supplier]
                if len(supplier_data) > 0:
                    supplier_risk = supplier_data['risk_score'].iloc[0]
                    supplier_comm = supplier_data['community_id'].iloc[0]
                    supplier_cat = supplier_data.get('risk_category', pd.Series(['Unknown'])).iloc[0]
                else:
                    supplier_risk = 0.5
                    supplier_comm = -1
                    supplier_cat = 'Unknown'
                G.add_node(supplier, risk_score=supplier_risk, risk_category=supplier_cat,
                          degree=0, community_id=supplier_comm)
            
            # Add edge with contract information
            if not G.has_edge(buyer, supplier):
                G.add_edge(
                    buyer,
                    supplier,
                    risk_probability=edge.get('risk_score', 0.5),
                    contract_id=edge.get('lotId', 'N/A'),
                    year=edge.get('awardYear', 'N/A')
                )
        
        st.markdown(f"**Network Graph:** {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
            # Calculate layout
            with st.spinner("Calculating network layout..."):
                try:
                    if G.number_of_nodes() < 100:
                        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
                    else:
                        pos = nx.spring_layout(G, k=3, iterations=30, seed=42)
                except:
                    pos = nx.random_layout(G, seed=42)
            
            # Create edge traces
            edge_trace = []
            
            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                
                risk_prob = edge[2].get('risk_probability', 0.5)
                edge_color = '#b05555' if risk_prob >= BINARY_THRESHOLD else '#5a9e6f'
                edge_width = 1 + (risk_prob * 2)
                
                edge_trace.append(go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=edge_width, color=edge_color),
                    hoverinfo='skip',
                    showlegend=False,
                    opacity=0.5
                ))
            
            # Create node trace
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            node_size = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                risk_score = G.nodes[node]['risk_score']
                risk_cat = G.nodes[node]['risk_category']
                degree = G.degree(node)
                comm_id = G.nodes[node].get('community_id', 'N/A')
                
                node_text.append(
                    f"Agent: {node}<br>" +
                    f"Community: {comm_id}<br>" +
                    f"Risk Score: {risk_score:.3f}<br>" +
                    f"Category: {risk_cat}<br>" +
                    f"Connections: {degree}"
                )
                
                node_color.append(risk_score)
                node_size.append(8 + (degree * 1.5))
            
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    showscale=True,
                    colorscale='RdYlGn_r',
                    size=node_size,
                    color=node_color,
                    colorbar=dict(
                        title="Risk Score",
                        thickness=15,
                        len=0.7,
                        x=1.02
                    ),
                    line=dict(width=1.5, color='white')
                ),
                showlegend=False
            )
            
            # Create figure
            fig = go.Figure(data=edge_trace + [node_trace])
            
            # Set title based on view mode
            if view_mode == "Specific Community":
                year_str = f" ({selected_year})" if selected_year != "All Years" else ""
                title = f"Community {selected_community} Network{year_str}"
            else:
                year_str = f" - Year {selected_year}" if selected_year != "All Years" else ""
                title = f"All Communities Network{year_str}"
            
            fig.update_layout(
                title=title,
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=20, r=20, t=60),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=700,
                plot_bgcolor='rgba(240,240,240,0.9)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Network statistics
            st.markdown("### 📊 Network Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Nodes", f"{len(G.nodes()):,}")
                st.metric("Edges", f"{len(G.edges()):,}")
            
            with col2:
                density = nx.density(G)
                st.metric("Network Density", f"{density:.4f}")
                
                avg_degree = sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0
                st.metric("Avg Degree", f"{avg_degree:.2f}")
            
            with col3:
                try:
                    clustering = nx.average_clustering(G)
                    st.metric("Avg Clustering", f"{clustering:.4f}")
                except:
                    st.metric("Avg Clustering", "N/A")
                
                high_risk_nodes = sum(1 for n in G.nodes() 
                                    if G.nodes[n]['risk_score'] >= BINARY_THRESHOLD)
                st.metric("High Risk Nodes", f"{high_risk_nodes:,}")
            
            with col4:
                high_risk_pct = (high_risk_nodes / len(G.nodes()) * 100) if len(G.nodes()) > 0 else 0
                st.metric("High Risk %", f"{high_risk_pct:.1f}%")
                
                num_communities = len(set(G.nodes[n].get('community_id', -1) for n in G.nodes()))
                st.metric("Communities", f"{num_communities:,}")
            
            # Show detailed table
            st.markdown("### 👥 Network Members")
            
            display_cols = ['agent_id', 'risk_score', 'risk_category']
            if 'community_id' in community_agents.columns:
                display_cols.append('community_id')
            if 'degree' in community_agents.columns:
                display_cols.append('degree')
            if 'neighbor_avg_risk' in community_agents.columns:
                display_cols.append('neighbor_avg_risk')
            
            available_cols = [c for c in display_cols if c in community_agents.columns]
            agent_table = community_agents[available_cols].copy()
            
            for col in ['risk_score', 'neighbor_avg_risk']:
                if col in agent_table.columns:
                    agent_table[col] = agent_table[col].round(4)
            
            # Sort by risk score
            agent_table = agent_table.sort_values('risk_score', ascending=False)
            
            st.dataframe(agent_table, use_container_width=True, height=400)
        
        else:
            st.info("ℹ️ No network connections found for visualization.")
    else:
        if selected_year != "All Years":
            st.warning(f"⚠️ No contract data found for the selected criteria in year {selected_year}. Try selecting a different year or 'All Years'.")
        else:
            st.info("ℹ️ No contract data found for the selected agents.")
    
    # ========================================================================
    # COMMUNITY SIZE VS RISK (SCATTER PLOT)
    # ========================================================================
    
    st.markdown("---")
    st.markdown("## 📊 Community Size vs Risk Analysis")
    
    fig = px.scatter(
        community_summary,
        x='size',
        y='avg_risk_score',
        size='high_risk_count',
        color='high_risk_pct',
        hover_data=['community_id', 'avg_degree'],
        title="Community Size vs Average Risk Score",
        labels={
            'size': 'Community Size (# agents)',
            'avg_risk_score': 'Average Risk Score',
            'high_risk_pct': 'High Risk %',
            'high_risk_count': 'High Risk Count'
        },
        color_continuous_scale='Reds'
    )
    
    fig.add_hline(
        y=BINARY_THRESHOLD,
        line_dash="dash",
        line_color="red",
        annotation_text=f"High Risk Threshold",
        annotation_position="right"
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # COMMUNITY DETAILS TABLE
    # ========================================================================
    
    st.markdown("---")
    st.markdown("## 🔍 Community Risk Details")

    high_risk_communities = community_summary[
        community_summary['avg_risk_score'] >= BINARY_THRESHOLD
    ].sort_values('avg_risk_score', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Top 20 Highest Risk Communities")
        
        display_cols = [
            'community_id', 'size', 'avg_risk_score', 
            'high_risk_count', 'high_risk_pct', 'avg_neighbor_risk'
        ]
        available_cols = [c for c in display_cols if c in high_risk_communities.columns]
        
        top_communities = high_risk_communities[available_cols].head(20).copy()
        
        for col in ['avg_risk_score', 'avg_neighbor_risk']:
            if col in top_communities.columns:
                top_communities[col] = top_communities[col].round(4)
        
        if 'high_risk_pct' in top_communities.columns:
            top_communities['high_risk_pct'] = top_communities['high_risk_pct'].round(1)
        
        st.dataframe(top_communities, use_container_width=True, height=400)
    
    with col2:
        st.markdown("### 📊 Risk Propagation Insights")
        
        very_high_risk = (community_summary['avg_risk_score'] >= 0.7).sum()
        high_risk = (
            (community_summary['avg_risk_score'] >= BINARY_THRESHOLD) &
            (community_summary['avg_risk_score'] < 0.7)
        ).sum()
        low_risk = (community_summary['avg_risk_score'] < BINARY_THRESHOLD).sum()
        
        total_comms = len(community_summary)
        
        st.markdown(f"""
        **Risk Distribution:**
        - 🔴 **Very High Risk** (≥0.7): {very_high_risk:,} ({very_high_risk/total_comms*100:.1f}%)
        - 🟡 **High Risk** (≥0.5, <0.7): {high_risk:,} ({high_risk/total_comms*100:.1f}%)
        - 🟢 **Low Risk** (<0.5): {low_risk:,} ({low_risk/total_comms*100:.1f}%)
        
        **Network Patterns:**
        - Risk propagates through buyer-supplier relationships
        - Tightly connected communities show correlated risk
        - High-degree nodes amplify fraud impact
        - Collusion patterns detected in dense clusters
        
        **Key Findings:**
        - {very_high_risk + high_risk:,} communities flagged as high risk
        - Largest community: {community_summary['size'].max():,} agents
        - Average community size: {community_summary['size'].mean():.1f} agents
        """)
    
    # ========================================================================
    # DOWNLOAD OPTIONS
    # ========================================================================
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = community_summary.to_csv(index=False)
        st.download_button(
            "📥 Download Community Summary",
            csv,
            "community_summary.csv",
            "text/csv",
            help="Download all community statistics"
        )
    
    with col2:
        csv = communities.to_csv(index=False)
        st.download_button(
            "📥 Download Agent-Community Mapping",
            csv,
            "agent_communities.csv",
            "text/csv",
            help="Download agent-level community assignments"
        )

except Exception as e:
    st.error(f"❌ Error loading network data: {e}")
    st.exception(e)