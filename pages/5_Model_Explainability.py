"""
================================================================================
PAGE 5: MODEL EXPLAINABILITY (ENHANCED)
================================================================================
File: deployment/pages/5_Model_Explainability.py

Comprehensive model performance analysis and feature importance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.loaders import load_model, load_all_data
from config import Config

st.set_page_config(page_title="Model Explainability", page_icon="📊", layout="wide")

st.title("📊 Model Explainability & Performance")

try:
    # Load model and data
    model_data = load_model()
    data = load_all_data()
    
    model = model_data['model']
    metadata = model_data['metadata']
    
    if metadata is None:
        st.error("❌ Model metadata not found. Please ensure rf_model_metadata.pkl exists.")
        st.stop()
    
    # ========================================================================
    # MODEL OVERVIEW
    # ========================================================================
    
    st.markdown("## 🤖 Model Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Type", "Random Forest")
        st.metric("Classification", "Binary (High/Low Risk)")
    
    with col2:
        f1_score = metadata['test_metrics']['f1_score']
        st.metric("F1 Score", f"{f1_score:.4f}")
        
        precision = metadata['test_metrics']['precision']
        st.metric("Precision", f"{precision:.4f}")
    
    with col3:
        recall = metadata['test_metrics']['recall']
        st.metric("Recall", f"{recall:.4f}")
        
        roc_auc = metadata['test_metrics']['roc_auc']
        st.metric("ROC-AUC", f"{roc_auc:.4f}")
    
    with col4:
        accuracy = metadata['test_metrics'].get('accuracy', 0)
        st.metric("Accuracy", f"{accuracy:.4f}")
        
        n_features = len(metadata['selected_features'])
        st.metric("Features Used", f"{n_features}")
    
    # ========================================================================
    # BEST HYPERPARAMETERS
    # ========================================================================
    
    st.markdown("---")
    st.markdown("## ⚙️ Model Hyperparameters")
    
    if 'best_params' in metadata:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Optimal Parameters")
            
            params_df = pd.DataFrame([
                {"Parameter": k, "Value": str(v)} 
                for k, v in metadata['best_params'].items()
            ])
            
            st.dataframe(params_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### 📊 Key Settings")
            
            best_params = metadata['best_params']
            
            st.markdown(f"""
            **Tree Configuration:**
            - Number of Trees: `{best_params.get('n_estimators', 'N/A')}`
            - Max Depth: `{best_params.get('max_depth', 'N/A')}`
            - Min Samples Split: `{best_params.get('min_samples_split', 'N/A')}`
            - Min Samples Leaf: `{best_params.get('min_samples_leaf', 'N/A')}`
            
            **Feature Selection:**
            - Max Features: `{best_params.get('max_features', 'N/A')}`
            
            **Training:**
            - Bootstrap: `{best_params.get('bootstrap', 'N/A')}`
            - Random State: `{best_params.get('random_state', 42)}`
            """)
    else:
        st.info("Hyperparameter information not available in metadata.")
    
    # ========================================================================
    # PERFORMANCE METRICS (DETAILED)
    # ========================================================================
    
    st.markdown("---")
    st.markdown("## 📈 Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Classification Metrics")
        
        test_metrics = metadata['test_metrics']
        
        metrics_data = {
            'Metric': ['F1 Score', 'Precision', 'Recall', 'Accuracy', 'ROC-AUC'],
            'Score': [
                test_metrics['f1_score'],
                test_metrics['precision'],
                test_metrics['recall'],
                test_metrics.get('accuracy', 0),
                test_metrics['roc_auc']
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df['Score'] = metrics_df['Score'].round(4)
        
        # Create bar chart
        fig = px.bar(
            metrics_df,
            x='Metric',
            y='Score',
            title="Model Performance Metrics",
            color='Score',
            color_continuous_scale='RdYlGn',
            range_color=[0, 1]
        )
        
        fig.update_layout(
            yaxis_range=[0, 1],
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Macro/Micro Metrics")
        
        # Check if we have macro/micro metrics - FIX: use space not underscore
        if 'classification_report' in metadata and 'macro avg' in metadata.get('classification_report', {}):
            report = metadata['classification_report']
            
            macro_micro_data = {
                'Metric': ['Precision', 'Recall', 'F1-Score'],
                'Macro Avg': [
                    report['macro avg']['precision'],
                    report['macro avg']['recall'],
                    report['macro avg']['f1-score']
                ],
                'Weighted Avg': [
                    report['weighted avg']['precision'],
                    report['weighted avg']['recall'],
                    report['weighted avg']['f1-score']
                ]
            }
            
            mm_df = pd.DataFrame(macro_micro_data)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=mm_df['Metric'],
                y=mm_df['Macro Avg'],
                name='Macro Avg',
                marker_color='#5a9e6f'
            ))
            
            fig.add_trace(go.Bar(
                x=mm_df['Metric'],
                y=mm_df['Weighted Avg'],
                name='Weighted Avg',
                marker_color='#4a7c9e'
            ))
            
            fig.update_layout(
                title="Macro vs Weighted Averages",
                yaxis_title="Score",
                yaxis_range=[0, 1],
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Macro/Micro metrics not available in metadata.")
    
    # ========================================================================
    # CONFUSION MATRIX
    # ========================================================================
    
    st.markdown("---")
    st.markdown("## 🔲 Confusion Matrix")
    
    if 'confusion_matrix' in metadata:
        cm = np.array(metadata['confusion_matrix'])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Low Risk', 'Predicted High Risk'],
                y=['Actual Low Risk', 'Actual High Risk'],
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 20},
                colorscale='Blues',
                showscale=True
            ))
            
            fig.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted Label",
                yaxis_title="Actual Label",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Confusion Matrix Breakdown")
            
            tn, fp, fn, tp = cm.ravel()
            total = cm.sum()
            
            st.markdown(f"""
            **True Negatives (TN):** {tn:,}
            - Correctly predicted as Low Risk
            - {tn/total*100:.1f}% of total
            
            **True Positives (TP):** {tp:,}
            - Correctly predicted as High Risk
            - {tp/total*100:.1f}% of total
            
            **False Positives (FP):** {fp:,}
            - Incorrectly flagged as High Risk
            - {fp/total*100:.1f}% of total (Type I Error)
            
            **False Negatives (FN):** {fn:,}
            - Missed High Risk cases
            - {fn/total*100:.1f}% of total (Type II Error)
            
            ---
            
            **Total Predictions:** {total:,}
            **Correct:** {tn + tp:,} ({(tn + tp)/total*100:.1f}%)
            **Errors:** {fp + fn:,} ({(fp + fn)/total*100:.1f}%)
            """)
    else:
        st.info("Confusion matrix not available. Add it to metadata during model training.")
    
    # ========================================================================
    # FEATURE IMPORTANCE (WITH ACTUAL NAMES)
    # ========================================================================
    
    st.markdown("---")
    st.markdown("## 🎯 Feature Importance Analysis")
    
    # Get feature importances from model
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = metadata['selected_features']
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📊 Top 20 Most Important Features")
            
            # Plot top 20
            top_n = min(20, len(importance_df))
            top_features = importance_df.head(top_n)
            
            fig = px.bar(
                top_features,
                x='Importance',
                y='Feature',
                orientation='h',
                title=f"Top {top_n} Features by Importance",
                color='Importance',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=600,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🔝 Top 5 Features")
            
            top_5 = importance_df.head(5)
            
            for idx, row in top_5.iterrows():
                st.metric(
                    row['Feature'],
                    f"{row['Importance']:.4f}"
                )
            
            st.markdown("---")
            
            st.markdown("### 💡 Key Insights")
            
            # Categorize features
            domain_features = [f for f in feature_names if any(x in f.lower() for x in ['risk', 'compliance', 'transparency', 'competition'])]
            network_features = [f for f in feature_names if any(x in f.lower() for x in ['degree', 'pagerank', 'betweenness', 'influence'])]
            contract_features = [f for f in feature_names if any(x in f.lower() for x in ['price', 'duration', 'cpv', 'award'])]
            
            st.markdown(f"""
            **Feature Categories:**
            - 🏢 Domain features: {len(domain_features)}
            - 🕸️ Network features: {len(network_features)}
            - 📄 Contract features: {len(contract_features)}
            
            **Top Feature:**
            - `{top_5.iloc[0]['Feature']}`
            - Importance: {top_5.iloc[0]['Importance']:.4f}
            - {top_5.iloc[0]['Importance']/importance_df['Importance'].sum()*100:.1f}% of total importance
            
            **Cumulative Importance:**
            - Top 5: {top_5['Importance'].sum()/importance_df['Importance'].sum()*100:.1f}%
            - Top 10: {importance_df.head(10)['Importance'].sum()/importance_df['Importance'].sum()*100:.1f}%
            """)
        
        # ====================================================================
        # FEATURE IMPORTANCE TABLE
        # ====================================================================
        
        st.markdown("---")
        st.markdown("### 📋 Complete Feature Importance Table")
        
        # Add cumulative importance
        importance_df['Cumulative Importance'] = importance_df['Importance'].cumsum()
        importance_df['Cumulative %'] = (importance_df['Cumulative Importance'] / importance_df['Importance'].sum() * 100).round(2)
        importance_df['Importance %'] = (importance_df['Importance'] / importance_df['Importance'].sum() * 100).round(2)
        
        # Format for display
        display_df = importance_df.copy()
        display_df['Importance'] = display_df['Importance'].round(4)
        display_df['Cumulative Importance'] = display_df['Cumulative Importance'].round(4)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400,
            hide_index=True
        )
        
        # Download button
        csv = importance_df.to_csv(index=False)
        st.download_button(
            "📥 Download Feature Importance",
            csv,
            "feature_importance.csv",
            "text/csv"
        )
    
    else:
        st.warning("Feature importance not available for this model type.")
    
    # ========================================================================
    # TRAINING INFORMATION
    # ========================================================================
    
    st.markdown("---")
    st.markdown("## 📚 Training Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Dataset Split")
        
        if 'data_split' in metadata:
            split_info = metadata['data_split']
            
            st.markdown(f"""
            **Training Set:**
            - Size: {split_info.get('train_size', 'N/A'):,} samples
            
            **Validation Set:**
            - Size: {split_info.get('val_size', 'N/A'):,} samples
            
            **Test Set:**
            - Size: {split_info.get('test_size', 'N/A'):,} samples
            
            **Total:**
            - {split_info.get('total_size', 'N/A'):,} contracts
            """)
        else:
            st.info("Dataset split information not available.")
    
    with col2:
        st.markdown("### ⚖️ Class Balance")
        
        if 'class_distribution' in metadata:
            class_dist = metadata['class_distribution']
            
            fig = go.Figure(data=[go.Pie(
                labels=['Low Risk', 'High Risk'],
                values=[
                    class_dist.get('low_risk', 0),
                    class_dist.get('high_risk', 0)
                ],
                marker_colors=['#5a9e6f', '#b05555'],
                hole=0.4
            )])
            
            fig.update_layout(
                title="Training Data Distribution",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Class distribution not available.")
    
    with col3:
        st.markdown("### ⏱️ Training Details")
        
        st.markdown(f"""
        
        **Feature Selection:**
        - Method: {metadata.get('feature_selection_method', 'N/A')}
        - Original features: {metadata.get('n_original_features', 'N/A')}
        - Selected features: {len(metadata['selected_features'])}
        
        **Threshold:**
        - Classification: 0.5 (binary)
        """)
    
    # ========================================================================
    # MODEL COMPARISON (if available)
    # ========================================================================
    
    if 'cv_scores' in metadata:
        st.markdown("---")
        st.markdown("## 🔄 Cross-Validation Results")
        
        cv_scores = metadata['cv_scores']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 CV Score Distribution")
            
            cv_df = pd.DataFrame({
                'Fold': [f"Fold {i+1}" for i in range(len(cv_scores))],
                'Score': cv_scores
            })
            
            fig = px.bar(
                cv_df,
                x='Fold',
                y='Score',
                title="Cross-Validation Scores",
                color='Score',
                color_continuous_scale='Viridis'
            )
            
            fig.add_hline(
                y=np.mean(cv_scores),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {np.mean(cv_scores):.4f}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 CV Statistics")
            
            st.metric("Mean CV Score", f"{np.mean(cv_scores):.4f}")
            st.metric("Std Dev", f"{np.std(cv_scores):.4f}")
            st.metric("Min Score", f"{np.min(cv_scores):.4f}")
            st.metric("Max Score", f"{np.max(cv_scores):.4f}")

except FileNotFoundError as e:
    st.error(f"❌ Model files not found: {e}")
    st.info("Please ensure the following files exist:\n"
            "- rf_best_model.joblib (or .pkl)\n"
            "- rf_model_metadata.pkl\n"
            "- feature_selector.pkl")

except Exception as e:
    st.error(f"❌ Error loading model explainability: {e}")
    st.exception(e)