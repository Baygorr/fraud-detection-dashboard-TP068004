"""
Data loader with external storage support for Streamlit Cloud
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import requests
import os

# Option 1: Google Drive (public links)
DATA_URLS = {

    # Add more files...
}

@st.cache_data(ttl=3600)
def load_from_url(url: str) -> pd.DataFrame:
    """Load CSV from URL with caching."""
    return pd.read_csv(url)

@st.cache_data
def load_data_smart(filename: str, local_path: Path = None):
    """
    Load data from local path or external URL.
    Prioritizes local files for faster development.
    """
    # Try local first
    if local_path and local_path.exists():
        return pd.read_csv(local_path)
    
    # Fall back to URL
    if filename in DATA_URLS:
        return load_from_url(DATA_URLS[filename])
    
    raise FileNotFoundError(f"Cannot find {filename} locally or remotely")
