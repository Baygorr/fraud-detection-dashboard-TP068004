"""
================================================================================
PAGE 2: CONTRACTS
================================================================================
File: deployment/pages/2_Contracts.py

Detailed contract-level risk analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.loaders import load_all_data
from config import Config

st.set_page_config(page_title="Contracts", page_icon="📄", layout="wide")

st.title("📄 Contract Risk Analysis")

try:
    data = load_all_data()
    contracts = data['contracts']
    
    # ========================================================================
    # FILTERS
    # ========================================================================
    
    st.sidebar.markdown("## Filters")
    
    risk_filter = st.sidebar.selectbox(
        "Risk Category",
        ['All', 'Low', 'Medium', 'High']
    )
    
    if 'awardYear' in contracts.columns:
        years = sorted(contracts['awardYear'].unique())
        year_filter = st.sidebar.multiselect(
            "Award Year",
            years,
            default=years
        )
    else:
        year_filter = None
    
    # Apply filters
    filtered = contracts.copy()
    
    if risk_filter != 'All':
        filtered = filtered[filtered['risk_category'] == risk_filter]
    
    if year_filter:
        filtered = filtered[filtered['awardYear'].isin(year_filter)]
    
    # ========================================================================
    # SUMMARY STATS
    # ========================================================================
    
    st.markdown("## 📊 Filtered Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Contracts", f"{len(filtered):,}")
    
    with col2:
        high_risk = (filtered['risk_prediction'] == 1).sum()
        st.metric("High-Risk", f"{high_risk:,}")
    
    with col3:
        if 'ml_caught_not_rules' in filtered.columns:
            ml_only = filtered['ml_caught_not_rules'].sum()
            st.metric("ML-Detected Only", f"{ml_only:,}")
    
    with col4:
        avg_risk = filtered['combined_risk_score'].mean()
        st.metric("Avg Risk Score", f"{avg_risk:.3f}")
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Risk Category Distribution")
        risk_counts = filtered['risk_category'].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Risk Categories",
            color_discrete_map={
                'Low': Config.LOW_RISK_COLOR,
                'Medium': Config.MEDIUM_RISK_COLOR,
                'High': Config.HIGH_RISK_COLOR
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Buyer vs Supplier Risk")
        fig = px.scatter(
            filtered.sample(min(1000, len(filtered))),
            x='buyer_risk_score',
            y='supplier_risk_score',
            color='risk_category',
            title="Buyer vs Supplier Risk Relationship",
            color_discrete_map={
                'Low': Config.LOW_RISK_COLOR,
                'Medium': Config.MEDIUM_RISK_COLOR,
                'High': Config.HIGH_RISK_COLOR
            },
            opacity=0.6
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # ML-SPECIFIC FINDINGS
    # ========================================================================
    
    if 'ml_caught_not_rules' in filtered.columns:
        st.markdown("---")
        st.markdown("## 🤖 ML-Specific Detections")
        
        ml_caught = filtered[filtered['ml_caught_not_rules'] == True]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Total ML-Only Detections", f"{len(ml_caught):,}")
            st.metric("Percentage of High-Risk", 
                     f"{len(ml_caught) / (filtered['risk_prediction']==1).sum() * 100:.1f}%")
        
        with col2:
            st.markdown("### What This Means")
            st.markdown("""
            These are contracts that:
            - **ML flagged as high-risk** (combined_risk_score ≥ 0.5)
            - **Not flagged by simple rules** (normal price, duration)
            - Represent **hidden fraud patterns** ML learned from network structure,
              agent behavior, and complex feature interactions
            """)
    
    # ========================================================================
    # DATA TABLE
    # ========================================================================
    
    st.markdown("---")
    st.markdown("## 📋 Contract Details")
    
    display_cols = [
        'lotId', 'buyerId', 'supplierId',
        'combined_risk_score', 'risk_category',
        'buyer_risk_score', 'supplier_risk_score'
    ]
    
    if 'awardYear' in filtered.columns:
        display_cols.insert(3, 'awardYear')
    
    if 'awardPrice_log' in filtered.columns:
        display_cols.insert(4, 'awardPrice_log')
    
    available_cols = [c for c in display_cols if c in filtered.columns]
    
    st.dataframe(
        filtered[available_cols].sort_values('combined_risk_score', ascending=False),
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = filtered.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data",
        data=csv,
        file_name="filtered_contracts.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error(f"Error loading contract data: {e}")