# ============================================================================
# UTILS: PLOTTING
# ============================================================================
"""
File: deployment/utils/plotting.py
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from config import Config

def plot_risk_distribution(df, score_col='risk_score', title="Risk Score Distribution"):
    """Plot risk score histogram with gradient color scale (green → yellow → red)"""
    
    # Create a copy with the score for coloring
    df_plot = df.copy()
    
    # Create histogram data manually for gradient coloring
    hist_data = np.histogram(df_plot[score_col].dropna(), bins=50)
    bin_edges = hist_data[1]
    bin_counts = hist_data[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Custom muted color scale: green → yellow → orange → red
    custom_colorscale = [
        [0.0, '#5a9e6f'],   # Muted green (low risk)
        [0.3, '#7fb069'],   # Lighter green
        [0.5, '#c9b56c'],   # Muted yellow
        [0.7, '#d4a574'],   # Muted orange
        [0.85, '#c97b7b'],  # Muted coral
        [1.0, '#b05555']    # Muted red (high risk)
    ]
    
    fig = go.Figure()
    
    # Add bars with gradient color based on risk score
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=bin_counts,
        marker=dict(
            color=bin_centers,  # Color based on the risk score
            colorscale=custom_colorscale,  # Custom muted gradient
            showscale=True,
            colorbar=dict(
                title="Risk<br>Score",
                thickness=15,
                len=0.7
            ),
            cmin=0,  # Min value for color scale
            cmax=1   # Max value for color scale
        ),
        width=(bin_edges[1] - bin_edges[0]) * 0.9,  # Bar width
        hovertemplate='Risk Score: %{x:.3f}<br>Count: %{y}<extra></extra>'
    ))
    
    # Add threshold line
    fig.add_vline(
        x=Config.RISK_THRESHOLD,
        line_dash="dash",
        line_color="#8b4545",  # Muted dark red
        line_width=2,
        annotation_text=f"Threshold ({Config.RISK_THRESHOLD})",
        annotation_position="top"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title='Risk Score',
        yaxis_title='Count',
        showlegend=False,
        hovermode='closest'
    )
    
    return fig

def plot_feature_importance(importance_df):
    """Plot feature importance with muted gradient colors"""
    # Custom muted blue-purple gradient
    custom_colorscale = [
        [0.0, '#9eb3c2'],   # Light muted blue
        [0.5, '#6b85a3'],   # Medium blue
        [1.0, '#4a6580']    # Dark muted blue
    ]
    
    fig = px.bar(
        importance_df.head(20),
        x='importance',
        y='feature',
        orientation='h',
        title="Top 20 Most Important Features",
        labels={'importance': 'Importance', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale=custom_colorscale
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig

def plot_confusion_matrix(cm, labels=['Low Risk', 'High Risk']):
    """Plot confusion matrix with muted colors"""
    # Custom muted gradient for confusion matrix
    custom_colorscale = [
        [0.0, '#e8f4ea'],   # Very light green
        [0.5, '#c9b56c'],   # Muted yellow
        [1.0, '#b05555']    # Muted red
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        colorscale=custom_colorscale,
        reversescale=False
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual"
    )
    return fig

def plot_roc_curve(fpr, tpr, auc_score):
    """Plot ROC curve with muted colors"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC (AUC={auc_score:.3f})',
        line=dict(color='#6b85a3', width=3)  # Muted blue
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='#999999', width=1, dash='dash')  # Muted gray
    ))
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        showlegend=True
    )
    return fig

def plot_temporal_trend(df, date_col, risk_col, title="Risk Trend Over Time"):
    """Plot risk trends over time with muted gradient"""
    temporal = df.groupby(date_col)[risk_col].mean().reset_index()
    
    # Custom muted color scale
    custom_colorscale = [
        [0.0, '#5a9e6f'],   # Muted green
        [0.5, '#c9b56c'],   # Muted yellow
        [1.0, '#b05555']    # Muted red
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=temporal[date_col],
        y=temporal[risk_col],
        mode='lines+markers',
        line=dict(
            color='#a88b5c',  # Muted orange/brown for trend line
            width=3
        ),
        marker=dict(
            size=8,
            color=temporal[risk_col],  # Gradient based on risk score
            colorscale=custom_colorscale,
            showscale=True,
            colorbar=dict(
                title="Risk<br>Score",
                thickness=15,
                len=0.5
            ),
            cmin=temporal[risk_col].min(),
            cmax=temporal[risk_col].max()
        )
    ))
    fig.update_layout(
        title=title,
        xaxis_title=date_col,
        yaxis_title="Average Risk Score",
        hovermode='x unified'
    )
    return fig

def plot_community_risk(community_summary):
    """Plot community risk summary with muted gradient colors"""
    # Custom muted color scale
    custom_colorscale = [
        [0.0, '#5a9e6f'],   # Muted green
        [0.5, '#c9b56c'],   # Muted yellow
        [1.0, '#b05555']    # Muted red
    ]
    
    fig = go.Figure()
    
    # Bar chart with muted gradient
    fig.add_trace(go.Bar(
        x=community_summary['community_id'],
        y=community_summary['avg_risk_score'],
        name='Avg Risk Score',
        marker=dict(
            color=community_summary['avg_risk_score'],
            colorscale=custom_colorscale,
            showscale=True,
            colorbar=dict(
                title="Avg Risk",
                x=1.15,
                thickness=15,
                len=0.5
            )
        )
    ))
    
    # Line chart with muted blue
    fig.add_trace(go.Scatter(
        x=community_summary['community_id'],
        y=community_summary['high_risk_pct'],
        name='High Risk %',
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='#6b85a3', width=3),  # Muted blue for contrast
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Risk Distribution Across Communities",
        xaxis_title="Community ID",
        yaxis_title="Average Risk Score",
        yaxis2=dict(
            title="High Risk Percentage (%)",
            overlaying='y',
            side='right'
        ),
        hovermode='x unified'
    )
    return fig