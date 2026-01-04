"""
================================================================================
FRAUD DETECTION DASHBOARD - MAIN APP (FIXED)
================================================================================
File: deployment/app.py

Streamlit multi-page dashboard for procurement fraud detection
Shows ML predictions, network analysis, and contract risk patterns
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import pickle

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.loaders import load_all_data
from config import Config

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Procurement Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD CSS
# ============================================================================

def load_css():
    
    # Additional CSS for feature table
    st.markdown("""
    <style>
    .feature-table {
        font-size: 0.9rem;
    }
    .stDataFrame {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    logo_path = Path("assets/SDG 16.png")
    if logo_path.exists():
        try:
            st.image(str(logo_path))
        except:
            st.info("🎯 SDG 16: Peace, Justice and Strong Institutions")
    
    st.title("🔍 Fraud Detection")
    st.markdown("---")
    
    st.markdown("### Navigation")
    st.markdown("""
    - **Overview**: Executive summary
    - **Contracts**: Risk analysis by contract
    - **Agents**: Buyer/supplier risk profiles
    - **Network**: Community detection & risk propagation
    - **Explainability**: Model insights
    """)
    
    st.markdown("---")
    st.markdown("### Dataset Info")
    try:
        data = load_all_data()
        st.metric("Total Contracts", f"{len(data['contracts']):,}")
        st.metric("Total Agents", f"{len(data['agents']):,}")
        st.metric("High-Risk Contracts", 
                 f"{(data['contracts']['risk_prediction'] == 1).sum():,}")
    except Exception as e:
        st.error(f"Error loading data: {e}")

# ============================================================================
# MAIN PAGE
# ============================================================================

st.title("🔍 Procurement Fraud Detection System")
st.markdown("### Welcome to the Fraud Detection Dashboard")

st.markdown("""
**By:**  
**AUSTIN BAY QI HERN**  
TP068004  
APD3F2505CS(DA) – Computer Science (Data Analytics Specialism)

---

This dashboard provides a comprehensive analysis of procurement fraud risks using:

- **Predictive Modelling**: Random Forest, Neural Network, Decision Tree  
- **Graph Analytics**: GAT Conv and network structure analysis  
- **Community Detection**: Identifying high-risk clusters and risk propagation patterns  
- **Contract and Agent Profiling**: Statistical and ML-based insights  

""")

# ============================================================================
# QUICK STATS
# ============================================================================

st.markdown("---")
st.markdown("## 📊 Quick Statistics")

try:
    data = load_all_data()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Contracts",
            f"{len(data['contracts']):,}",
            help="Total number of procurement contracts analyzed"
        )
    
    with col2:
        high_risk = (data['contracts']['risk_prediction'] == 1).sum()
        risk_pct = high_risk / len(data['contracts']) * 100
        st.metric(
            "High-Risk Contracts",
            f"{high_risk:,}",
            f"{risk_pct:.1f}%"
        )
    
    with col3:
        high_risk_agents = (data['agents']['risk_score'] > 0.7).sum()
        st.metric(
            "High-Risk Agents",
            f"{high_risk_agents:,}",
            help="Agents with risk score > 0.7"
        )
    
    with col4:
        if data['communities'] is not None and 'community_id' in data['communities'].columns:
            n_communities = data['communities']['community_id'].nunique()
            st.metric(
                "Risk Communities",
                f"{n_communities:,}",
                help="Detected communities in network"
            )
        else:
            st.metric("Risk Communities", "N/A")
    
    # Risk Distribution Chart
    st.markdown("### Risk Score Distribution")
    
    # Create histogram with gradient colors
    hist_data = data['contracts']['risk_score']
    counts, bins = np.histogram(hist_data, bins=50)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Create color scale based on risk score (green -> yellow -> red)
    colors = []
    for bc in bin_centers:
        if bc < 0.3:
            # Low risk: green to light green
            colors.append(f'rgb({int(100 + bc*200)}, {int(200 - bc*100)}, {int(100)})')
        elif bc < 0.7:
            # Medium risk: yellow to orange
            colors.append(f'rgb({int(255)}, {int(200 - (bc-0.3)*250)}, {int(50)})')
        else:
            # High risk: orange to red
            colors.append(f'rgb({int(255)}, {int(50 - (bc-0.7)*150)}, {int(50)})')
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=counts,
        marker=dict(
            color=bin_centers,
            colorscale='RdYlGn_r',  # Reversed Red-Yellow-Green
            showscale=True,
            colorbar=dict(
                title="Risk Score",
                tickvals=[0, 0.3, 0.5, 0.7, 1.0],
                ticktext=['Low', 'Medium-Low', 'Threshold', 'Medium-High', 'High']
            )
        ),
        hovertemplate='Risk Score: %{x:.2f}<br>Contracts: %{y:,}<extra></extra>'
    ))
    
    fig.add_vline(x=0.5, line_dash="dash", line_color="white", line_width=2,
                  annotation_text="Threshold (0.5)", 
                  annotation_position="top right",
                  annotation=dict(font_size=12, font_color="white"))
    
    fig.update_layout(
        title="Contract Risk Score Distribution",
        xaxis_title="Risk Score",
        yaxis_title="Number of Contracts",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Temporal Trend
    if 'awardYear' in data['contracts'].columns:
        st.markdown("### Risk Trends Over Time")
        yearly_risk = data['contracts'].groupby('awardYear').agg({
            'risk_prediction': ['sum', 'count']
        }).reset_index()
        yearly_risk.columns = ['Year', 'High_Risk', 'Total']
        yearly_risk['Risk_Rate'] = yearly_risk['High_Risk'] / yearly_risk['Total'] * 100
        
        fig = go.Figure()
        
        # Add gradient fill under the line
        fig.add_trace(go.Scatter(
            x=yearly_risk['Year'],
            y=yearly_risk['Risk_Rate'],
            mode='lines',
            name='Risk Rate (%)',
            line=dict(width=0),
            fillcolor='rgba(255, 100, 100, 0)',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Main line with gradient color
        fig.add_trace(go.Scatter(
            x=yearly_risk['Year'],
            y=yearly_risk['Risk_Rate'],
            mode='lines+markers',
            name='Risk Rate (%)',
            line=dict(
                color='rgba(255, 107, 107, 0.8)',
                width=3
            ),
            marker=dict(
                size=10,
                color=yearly_risk['Risk_Rate'],
                colorscale='Viridis',  # Green to purple gradient
                showscale=False,
                line=dict(width=2, color='white')
            ),
            fill='tonexty',
            fillcolor='rgba(99, 110, 250, 0.2)',
            hovertemplate='<b>Year %{x}</b><br>Risk Rate: %{y:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title="Yearly Risk Rate Trend",
            xaxis_title="Year",
            yaxis_title="Risk Rate (%)",
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    # ============================================================================
    # FEATURE IMPORTANCE TABLE
    # ============================================================================
    
    st.markdown("---")
    st.markdown("## 🎯 Model Features & Importance")
    
    # Load feature importance from model metadata - UPDATED FOR LIGHTGBM
    model_metadata_path = Path("models/lgbm_model_metadata.pkl")
    
    # Find the most recent LightGBM feature importance file
    feature_importance_files = list(Path("models").glob("lgbm_feature_importance_*.csv"))
    if feature_importance_files:
        # Sort by modification time and get the most recent
        feature_importance_csv = max(feature_importance_files, key=lambda p: p.stat().st_mtime)
    else:
        feature_importance_csv = Path("models/lgbm_feature_importance.csv")  # fallback
    
    feature_descriptions = {
        'low_transparency_risk': 'Low transparency in procurement process (binary indicator)',
        'competition_x_transparency': 'Interaction effect between competition level and transparency',
        'composite_risk_score': 'Weighted composite score combining multiple risk indicators',
        'low_competition_risk': 'Insufficient competition detected (fewer bidders than expected)',
        'duration_anomaly': 'Contract duration deviates significantly from normal patterns',
        'weighted_compliance_risk': 'Weighted score of regulatory compliance violations',
        'regulatory_compliance_risk': 'Non-compliance with EU procurement regulations',
        'awardYear': 'Year when contract was awarded (temporal feature)',
        'contractDuration': 'Duration of contract in days',
        'cpv': 'Common Procurement Vocabulary code (contract category)',
        'src_degree': 'Number of contracts/connections for buyer (network centrality)',
        'dst_degree': 'Number of contracts/connections for supplier (network centrality)',
        'degree_diff': 'Difference in network degree between buyer and supplier',
        'core_difference': 'Difference in k-core values (network density measure)',
        'pagerank_diff': 'Difference in PageRank scores (influence disparity)',
        'dst_pagerank': 'PageRank score of supplier (influence in network)',
        'src_pagerank': 'PageRank score of buyer (influence in network)',
        'typeOfContract': 'Contract type (services, supplies, works, mixed)',
        'onBehalf': 'Whether contract is on behalf of another entity',
        'fraAgreement': 'Framework agreement indicator (pre-negotiated terms)',
        'outOfDirectives': 'Contract falls outside EU procurement directives',
        'gpa': 'Government Procurement Agreement (international trade agreement)',
        'edge_betweenness': 'Betweenness centrality of buyer-supplier relationship',
        'both_high_influence': 'Both buyer and supplier have high network influence',
        'correctionsNb': 'Number of corrections/amendments made to contract',
        'cancelled': 'Contract cancellation status',
        'awardPrice_log': 'Log-transformed contract award price (normalized)'
    }
    
    feature_categories = {
        'Risk Indicators': ['low_transparency_risk', 'composite_risk_score', 'low_competition_risk', 
                           'duration_anomaly', 'weighted_compliance_risk', 'regulatory_compliance_risk'],
        'Network Features': ['src_degree', 'dst_degree', 'degree_diff', 'core_difference', 
                            'pagerank_diff', 'dst_pagerank', 'src_pagerank', 'edge_betweenness', 
                            'both_high_influence'],
        'Contract Attributes': ['awardYear', 'contractDuration', 'cpv', 'typeOfContract', 
                               'awardPrice_log', 'correctionsNb', 'cancelled'],
        'Regulatory Features': ['onBehalf', 'fraAgreement', 'outOfDirectives', 'gpa'],
        'Interaction Features': ['competition_x_transparency'],
        'Graph Embeddings': []  # Will be populated dynamically for emb_* features
    }
    
    # Try to load feature importance
    feature_data = []
    
    if feature_importance_csv.exists():
        # Load from CSV - NOTE: columns are 'Feature' and 'Importance' (capitalized)
        importance_df = pd.read_csv(feature_importance_csv)
        
        # Get selected features from metadata
        selected_features = importance_df['Feature'].tolist()
        
        # Create feature table
        for idx, row in importance_df.iterrows():
            feature = row['Feature']
            
            # Determine category
            category = 'Other'
            for cat, features in feature_categories.items():
                if feature in features:
                    category = cat
                    break
            
            # Check if it's an embedding feature
            if feature.startswith('emb_'):
                category = 'Graph Embeddings'
            
            feature_data.append({
                'Rank': idx + 1,
                'Feature Name': feature,
                'Category': category,
                'Importance Score': row['Importance'],
                'Description': feature_descriptions.get(feature, 
                    'Graph embedding dimension' if feature.startswith('emb_') else 'No description available')
            })
    
    else:
        # Fallback: Use metadata only
        if model_metadata_path.exists():
            with open(model_metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                selected_features = metadata.get('selected_features', [])
                feature_importance_dict = metadata.get('feature_importance', {})
                
                # Sort by importance if available
                if feature_importance_dict:
                    sorted_features = sorted(feature_importance_dict.items(), 
                                           key=lambda x: x[1], reverse=True)
                else:
                    sorted_features = [(f, None) for f in selected_features]
                
                for idx, (feature, importance) in enumerate(sorted_features, 1):
                    category = 'Other'
                    for cat, features in feature_categories.items():
                        if feature in features:
                            category = cat
                            break
                    
                    # Check if it's an embedding feature
                    if feature.startswith('emb_'):
                        category = 'Graph Embeddings'
                    
                    feature_data.append({
                        'Rank': idx,
                        'Feature Name': feature,
                        'Category': category,
                        'Importance Score': importance,
                        'Description': feature_descriptions.get(feature, 
                            'Graph embedding dimension' if feature.startswith('emb_') else 'No description available')
                    })
    
    if feature_data:
        features_df = pd.DataFrame(feature_data)
        
        # Add importance level
        if features_df['Importance Score'].notna().any():
            max_importance = features_df['Importance Score'].max()
            features_df['Importance Level'] = features_df['Importance Score'].apply(
                lambda x: '🔴 High' if x >= max_importance * 0.7 
                else '🟡 Medium' if x >= max_importance * 0.3 
                else '🟢 Low' if pd.notna(x) else 'N/A'
            )
        else:
            features_df['Importance Level'] = 'N/A'
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Features Used", len(features_df))
        
        with col2:
            if features_df['Importance Score'].notna().any():
                top_feature = features_df.iloc[0]['Feature Name']
                st.metric("Top Feature", top_feature)
        
        with col3:
            network_features = features_df[features_df['Category'] == 'Network Features']
            st.metric("Network Features", len(network_features))
        
        with col4:
            # Updated: Count embedding features
            embedding_features = features_df[features_df['Category'] == 'Graph Embeddings']
            if len(embedding_features) > 0:
                st.metric("Graph Embeddings", len(embedding_features))
            else:
                risk_features = features_df[features_df['Category'] == 'Risk Indicators']
                st.metric("Risk Indicators", len(risk_features))
        
        # Display feature table
        st.markdown("### 📋 Selected Features for Model Training")
        
        # Prepare display dataframe
        display_df = features_df.copy()
        
        # Format importance score
        if display_df['Importance Score'].notna().any():
            display_df['Importance Score'] = display_df['Importance Score'].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
            )
        
        # Reorder columns for better display
        column_order = ['Feature Name', 'Category', 'Importance Level', 
                       'Importance Score', 'Description']
        display_df = display_df.set_index('Rank', drop=False)

        display_df = display_df[column_order]
        
        # Style the dataframe
        def highlight_importance(val):
            if '🔴 High' in str(val):
                return 'background-color: #ffe0e0; font-weight: bold'
            elif '🟡 Medium' in str(val):
                return 'background-color: #fff4cc; font-weight: bold'
            elif '🟢 Low' in str(val):
                return 'background-color: #e0ffe0'
            return ''
        
        styled_df = display_df.style.set_properties(
            **{'background-color': '', 'color': ''},  # empty string means default table style
            subset=['Importance Level']
        ).set_properties(**{
            'text-align': 'left',
            'font-size': '13px',
            'border': '1px solid #e0e0e0'
        })
        
        st.dataframe(styled_df, use_container_width=True, height=600)
        
        # Feature category breakdown
        st.markdown("### 📊 Feature Distribution by Category")
        
        category_counts = features_df['Category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        
        fig = px.bar(
            category_counts,
            x='Category',
            y='Count',
            color='Category',
            title="Number of Features by Category",
            labels={'Count': 'Number of Features'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top 10 features importance visualization
        if features_df['Importance Score'].notna().any():
            st.markdown("### 🎯 Top 10 Most Important Features")
            
            top10 = features_df.nsmallest(10, 'Rank').copy()
            top10['Importance Score'] = top10['Importance Score'].astype(float)
            
            fig = px.bar(
                top10,
                x='Importance Score',
                y='Feature Name',
                orientation='h',
                color='Importance Score',
                color_continuous_scale='Reds',
                title="Feature Importance Scores (Top 10)",
                labels={'Importance Score': 'Importance', 'Feature Name': 'Feature'}
            )
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("⚠️ Feature importance data not available. Model metadata file may be missing.")
        st.info("Expected file: `models/lgbm_model_metadata.pkl` or LightGBM feature importance CSV")

except Exception as e:
    st.error(f"Error loading dashboard data: {e}")
    st.info("Please ensure all data files are present in the data/ folder")
    import traceback
    with st.expander("Show full error"):
        st.code(traceback.format_exc())

