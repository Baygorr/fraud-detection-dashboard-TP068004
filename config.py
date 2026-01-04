# ============================================================================
# CONFIG FILE
# ============================================================================
"""
File: deployment/config.py

Configuration for Fraud Detection Dashboard
Supports both local and Streamlit Cloud deployment
"""

import os
from pathlib import Path

class Config:
    """Configuration class for dashboard settings."""
    
    # Detect if running on Streamlit Cloud
    IS_CLOUD = os.environ.get('STREAMLIT_SHARING_MODE') is not None or \
               os.environ.get('STREAMLIT_SERVER_HEADLESS') == 'true'
    
    # Base directory - works for both local and cloud
    if IS_CLOUD:
        BASE_DIR = Path(__file__).parent
    else:
        BASE_DIR = Path(__file__).parent
    
    # Data and model directories
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    ASSETS_DIR = BASE_DIR / "assets"
    
    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    ASSETS_DIR.mkdir(exist_ok=True)
    
    # Model files
    MODEL_FILE = MODEL_DIR / "lgbm_best_model.joblib"
    MODEL_METADATA = MODEL_DIR / "lgbm_model_metadata.pkl"
    FEATURE_SELECTOR = MODEL_DIR / "feature_selector.pkl"
    LABEL_ENCODERS = MODEL_DIR / "label_encoders.pkl"
    
    # Data files
    CONTRACT_PREDICTIONS = DATA_DIR / "contract_risk_predictions.csv"
    AGENT_RISK_SCORES = DATA_DIR / "agent_risk_scores.csv"
    AGENT_EMBEDDINGS = DATA_DIR / "agent_embeddings.csv"
    CONTRACT_PATTERNS = DATA_DIR / "contract_risk_patterns.csv"
    AGENT_COMMUNITIES = DATA_DIR / "agent_communities.csv"
    COMMUNITY_SUMMARY = DATA_DIR / "community_risk_summary.csv"
    
    # Dashboard settings
    PAGE_TITLE = "Procurement Fraud Detection"
    PAGE_ICON = "🔍"
    LAYOUT = "wide"
    
    # Risk thresholds
    HIGH_RISK_THRESHOLD = 0.7
    MEDIUM_RISK_THRESHOLD = 0.4
    
    @classmethod
    def get_data_path(cls, filename: str) -> Path:
        """Get full path for a data file."""
        return cls.DATA_DIR / filename
    
    @classmethod
    def get_model_path(cls, filename: str) -> Path:
        """Get full path for a model file."""
        return cls.MODEL_DIR / filename
    
    @classmethod
    def file_exists(cls, filepath: Path) -> bool:
        """Check if a file exists."""
        return filepath.exists()
    
    @classmethod
    def print_status(cls):
        """Print configuration status for debugging."""
        print(f"Running on Cloud: {cls.IS_CLOUD}")
        print(f"Base Directory: {cls.BASE_DIR}")
        print(f"Data Directory: {cls.DATA_DIR}")
        print(f"Model Directory: {cls.MODEL_DIR}")
        print(f"\nData Files:")
        for attr in ['CONTRACT_PREDICTIONS', 'AGENT_RISK_SCORES', 'AGENT_EMBEDDINGS']:
            path = getattr(cls, attr)
            status = "✅" if path.exists() else "❌"
            print(f"  {status} {attr}: {path.name}")
        print(f"\nModel Files:")
        for attr in ['MODEL_FILE', 'MODEL_METADATA', 'FEATURE_SELECTOR']:
            path = getattr(cls, attr)
            status = "✅" if path.exists() else "❌"
            print(f"  {status} {attr}: {path.name}")