"""
================================================================================
UTILITIES: DATA LOADERS (FIXED - WITH load_model)
================================================================================
File: deployment/utils/loaders.py

Handles loading and standardizing data from CSV files
Automatically maps column names to what dashboard expects
"""

import pandas as pd
import numpy as np
import json
import joblib
import pickle
import streamlit as st
from pathlib import Path

from config import Config

def standardize_column_names(df, file_type='contracts'):
    """
    Standardize column names to what dashboard expects
    
    Args:
        df: DataFrame to standardize
        file_type: 'contracts' or 'agents'
    
    Returns:
        df with standardized column names
    """
    # Column mappings: what_dashboard_expects -> [possible_names_in_data]
    COLUMN_ALIASES = {
        'risk_score': ['combined_risk_score', 'risk_probability', 'risk_score'],
        'risk_prediction': ['predicted_risk_label', 'risk_prediction'],
        'agent_id': ['buyerId', 'supplierId', 'agent_id']
    }
    
    for standard_name, aliases in COLUMN_ALIASES.items():
        # Check if any alias exists in the dataframe
        for alias in aliases:
            if alias in df.columns and standard_name not in df.columns:
                # Use the first available alias
                df[standard_name] = df[alias]
                break
    
    return df


def load_contracts():
    """Load contract risk patterns with standardized columns"""
    file_path = Path(Config.DATA_DIR) / "contract_risk_patterns.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Contract data not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Standardize column names
    df = standardize_column_names(df, file_type='contracts')
    
    # Ensure required columns exist
    if 'risk_score' not in df.columns:
        if 'combined_risk_score' in df.columns:
            df['risk_score'] = df['combined_risk_score']
        elif 'buyer_risk_score' in df.columns and 'supplier_risk_score' in df.columns:
            # Recreate combined risk score
            df['risk_score'] = 0.3 * df['buyer_risk_score'] + 0.7 * df['supplier_risk_score']
        else:
            raise ValueError("Cannot find risk_score column or alternatives")
    
    if 'risk_prediction' not in df.columns:
        if 'predicted_risk_label' in df.columns:
            df['risk_prediction'] = df['predicted_risk_label']
        else:
            # Infer from risk_score
            df['risk_prediction'] = (df['risk_score'] >= 0.5).astype(int)
    
    return df


def load_agents():
    """Load agent risk scores with standardized columns"""
    file_path = Path(Config.DATA_DIR) / "agent_risk_scores.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Agent data not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Standardize column names
    df = standardize_column_names(df, file_type='agents')
    
    # Ensure required columns exist
    if 'risk_score' not in df.columns:
        raise ValueError("Agent data must have 'risk_score' column")
    
    if 'risk_prediction' not in df.columns:
        df['risk_prediction'] = (df['risk_score'] >= 0.5).astype(int)
    
    if 'risk_category' not in df.columns:
        df['risk_category'] = pd.cut(
            df['risk_score'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low', 'Medium', 'High']
        )
    
    return df


def load_embeddings():
    """Load agent embeddings"""
    file_path = Path(Config.DATA_DIR) / "agent_embeddings.csv"
    
    if not file_path.exists():
        # Try .npy format
        npy_path = Path(Config.DATA_DIR) / "agent_embeddings.npy"
        if npy_path.exists():
            embeddings = np.load(npy_path)
            df = pd.DataFrame(embeddings, columns=['embedding_1', 'embedding_2'])
            return df
        raise FileNotFoundError(f"Embeddings not found")
    
    return pd.read_csv(file_path)


def load_communities():
    """Load agent communities"""
    file_path = Path(Config.DATA_DIR) / "agent_communities.csv"
    
    if not file_path.exists():
        return None
    
    df = pd.read_csv(file_path)
    df = standardize_column_names(df, file_type='agents')
    
    return df


def load_community_summary():
    """Load community risk summary"""
    file_path = Path(Config.DATA_DIR) / "community_risk_summary.csv"
    
    if not file_path.exists():
        return None
    
    return pd.read_csv(file_path)


def load_feature_config():
    """Load feature configuration (model info, selected features, etc.)"""
    file_path = Path(Config.DATA_DIR) / "feature_config.json"
    
    if not file_path.exists():
        return None
    
    with open(file_path, 'r') as f:
        return json.load(f)


def load_model():
    """
    Load the trained model and its components
    
    Returns:
        dict with keys: 'model', 'metadata', 'feature_selector', 'label_encoders'
    """
    # Initialize the dictionary FIRST
    model_data = {}
    
    # Try to find model in models directory
    model_dir = Path(Config.MODELS_DIR) if hasattr(Config, 'MODELS_DIR') else Path(Config.DATA_DIR).parent / 'models'
    
    # Load model
    try:
        model_path = model_dir / "rf_best_model.joblib"
        if model_path.exists():
            model_data['model'] = joblib.load(model_path)
        else:
            # Try .pkl format
            model_path = model_dir / "rf_best_model.pkl"
            with open(model_path, 'rb') as f:
                model_data['model'] = pickle.load(f)
    except Exception as e:
        raise FileNotFoundError(f"Could not load model from {model_dir}: {e}")
    
    # Load metadata
    try:
        metadata_path = model_dir / "rf_model_metadata.pkl"
        with open(metadata_path, 'rb') as f:
            model_data['metadata'] = pickle.load(f)
    except FileNotFoundError:
        model_data['metadata'] = None
    
    # Load feature selector
    try:
        selector_path = model_dir / "feature_selector.pkl"
        with open(selector_path, 'rb') as f:
            model_data['feature_selector'] = pickle.load(f)
    except FileNotFoundError:
        model_data['feature_selector'] = None
    
    # Load label encoders
    try:
        encoders_path = model_dir / "label_encoders.pkl"
        with open(encoders_path, 'rb') as f:
            model_data['label_encoders'] = pickle.load(f)
    except FileNotFoundError:
        model_data['label_encoders'] = {}
    
    return model_data


def load_all_data():
    """
    Load all dashboard data with graceful handling for missing files.
    
    Returns:
        dict with keys: 'contracts', 'agents', 'embeddings', 
                       'communities', 'community_summary', 'config'
    """
    data = {
        'contracts': None,
        'agents': None,
        'embeddings': None,
        'communities': None,
        'patterns': None
    }
    
    # Check for required files
    required_files = {
        'contracts': Config.DATA_DIR / "contract_risk_predictions.csv",
        'agents': Config.DATA_DIR / "agent_risk_scores.csv",
    }
    
    optional_files = {
        'embeddings': Config.DATA_DIR / "agent_embeddings.csv",
        'communities': Config.DATA_DIR / "agent_communities.csv",
        'patterns': Config.DATA_DIR / "contract_risk_patterns.csv",
    }
    
    missing_required = [k for k, v in required_files.items() if not v.exists()]
    
    if missing_required:
        st.error(f"""
        ⚠️ **Missing Required Data Files**
        
        The following files are missing: {missing_required}
        
        **For Local Development:**
        ```
        python setup_dashboard.py
        ```
        
        **For Cloud Deployment:**
        Please upload the data files to `data/` folder or use Git LFS.
        """)
        st.stop()
    
    # Load required files
    for key, filepath in required_files.items():
        data[key] = pd.read_csv(filepath)
    
    # Load optional files (won't fail if missing)
    for key, filepath in optional_files.items():
        if filepath.exists():
            try:
                data[key] = pd.read_csv(filepath)
            except Exception as e:
                st.warning(f"Could not load {filepath.name}: {e}")
                data[key] = pd.DataFrame()
        else:
            data[key] = pd.DataFrame()
    
    return data


def get_data_summary():
    """Get summary statistics about loaded data"""
    data = load_all_data()
    
    summary = {
        'total_contracts': len(data['contracts']),
        'total_agents': len(data['agents']),
        'high_risk_contracts': (data['contracts']['risk_prediction'] == 1).sum(),
        'high_risk_agents': (data['agents']['risk_category'] == 'High').sum(),
        'avg_contract_risk': data['contracts']['risk_score'].mean(),
        'avg_agent_risk': data['agents']['risk_score'].mean(),
    }
    
    if data['communities'] is not None:
        summary['n_communities'] = data['communities']['community_id'].nunique()
    
    if data['config'] is not None:
        summary['model_f1'] = data['config']['model_info']['test_f1_score']
        summary['model_auc'] = data['config']['model_info']['test_roc_auc']
        summary['n_features'] = data['config']['features']['n_selected']
    
    return summary


# ============================================================================
# COLUMN NAME REFERENCE
# ============================================================================

EXPECTED_COLUMNS = {
    'contracts': {
        'required': ['risk_score', 'risk_prediction', 'buyerId', 'supplierId'],
        'optional': ['awardYear', 'awardPrice_log', 'cpv', 'risk_category', 
                    'buyer_risk_score', 'supplier_risk_score', 'combined_risk_score']
    },
    'agents': {
        'required': ['agent_id', 'risk_score', 'risk_prediction', 'risk_category'],
        'optional': ['agent_type', 'contract_count', 'max_risk', 'risk_std']
    },
    'communities': {
        'required': ['agent_id', 'community_id'],
        'optional': ['risk_score', 'degree', 'betweenness', 'pagerank']
    }
}


def validate_data_schema(df, data_type='contracts'):
    """
    Validate that dataframe has required columns
    
    Args:
        df: DataFrame to validate
        data_type: 'contracts', 'agents', or 'communities'
    
    Returns:
        tuple: (is_valid, missing_columns)
    """
    required_cols = EXPECTED_COLUMNS[data_type]['required']
    missing = [col for col in required_cols if col not in df.columns]
    
    is_valid = len(missing) == 0
    
    return is_valid, missing


if __name__ == "__main__":
    # Test loading
    print("Testing data loaders...")
    
    try:
        data = load_all_data()
        summary = get_data_summary()
        
        print("\n✅ Data loaded successfully!")
        print(f"\n📊 Summary:")
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        print("\n📋 Available columns:")
        print(f"   Contracts: {list(data['contracts'].columns)}")
        print(f"   Agents: {list(data['agents'].columns)}")
        
        # Test model loading
        print("\n🤖 Testing model loading...")
        model_data = load_model()
        print(f"   ✓ Model loaded: {type(model_data['model'])}")
        if model_data['metadata']:
            print(f"   ✓ Metadata: F1={model_data['metadata']['test_metrics']['f1_score']:.4f}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")