"""
================================================================================
OPTIMIZED SETUP SCRIPT - LIGHTGBM MODEL (IRT-ALIGNED)
================================================================================
Faster version with progress tracking and IRT-based binary classification
Uses trained LightGBM model with graph embeddings for fraud detection

For Streamlit Cloud: Set SKIP_REGENERATION=True to use pre-generated data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from tqdm import tqdm
import time
import os

# ============================================================================
# CLOUD DEPLOYMENT CHECK
# ============================================================================

# Set to True for Streamlit Cloud deployment (uses pre-generated data)
# Set to False for local regeneration from preprocessing outputs
SKIP_REGENERATION = os.environ.get('STREAMLIT_CLOUD', 'false').lower() == 'true' or \
                    os.environ.get('SKIP_REGENERATION', 'false').lower() == 'true'

# Check if running on Streamlit Cloud
IS_CLOUD = os.environ.get('STREAMLIT_SHARING_MODE') is not None

if IS_CLOUD:
    SKIP_REGENERATION = True
    print("☁️  Running on Streamlit Cloud - using pre-generated data")

# Import local services
from services.contract_analysis import ContractAnalyzer
from services.network_analysis import NetworkAnalyzer
from config import Config

# ============================================================================
# CHECK IF DATA ALREADY EXISTS
# ============================================================================

required_files = [
    Config.DATA_DIR / "contract_risk_predictions.csv",
    Config.DATA_DIR / "agent_risk_scores.csv",
    Config.DATA_DIR / "agent_embeddings.csv",
]

all_files_exist = all(f.exists() for f in required_files)

if SKIP_REGENERATION and all_files_exist:
    print("="*80)
    print("✅ PRE-GENERATED DATA FOUND - SKIPPING REGENERATION")
    print("="*80)
    print("\nExisting files:")
    for f in required_files:
        if f.exists():
            print(f"  ✓ {f.name}")
    print("\n🚀 Ready to launch dashboard!")
    print("   Run: streamlit run app.py")
    print("="*80)
    exit(0)

if SKIP_REGENERATION and not all_files_exist:
    print("="*80)
    print("⚠️  SKIP_REGENERATION=True but data files are missing!")
    print("="*80)
    print("\nMissing files:")
    for f in required_files:
        if not f.exists():
            print(f"  ✗ {f.name}")
    print("\nPlease either:")
    print("  1. Upload the missing data files to deployment/data/")
    print("  2. Set SKIP_REGENERATION=False and run locally with preprocessing data")
    print("="*80)
    exit(1)

# ============================================================================
# CONTINUE WITH DATA GENERATION (LOCAL ONLY)
# ============================================================================

print("="*80)
print("🚀 FRAUD DETECTION DASHBOARD - LIGHTGBM MODEL DATA GENERATION")
print("="*80)

start_time = time.time()

# ============================================================================
# STEP 0: LOAD TRAINED LIGHTGBM MODEL AND COMPONENTS
# ============================================================================

print("\n🤖 Step 0: Loading trained LightGBM model...")

MODEL_DIR = Path(r"C:\Users\ASUS\Documents\FYP\deployment\models")

# Load model
try:
    model = joblib.load(MODEL_DIR / "lgbm_best_model.joblib")
    print("✅ Loaded: lgbm_best_model.joblib")
except FileNotFoundError:
    with open(MODEL_DIR / "lgbm_best_model.pkl", 'rb') as f:
        model = pickle.load(f)
    print("✅ Loaded: lgbm_best_model.pkl")

# Load feature selector
with open(MODEL_DIR / "feature_selector.pkl", 'rb') as f:
    feature_selector = pickle.load(f)

# Load metadata
with open(MODEL_DIR / "lgbm_model_metadata.pkl", 'rb') as f:
    metadata = pickle.load(f)

# Load label encoders
try:
    with open(MODEL_DIR / "label_encoders.pkl", 'rb') as f:
        label_encoders = pickle.load(f)
except FileNotFoundError:
    label_encoders = {}

# Load IRT metadata (for consistent labeling)
GRAPH_DATASET = Path(r"C:\Users\ASUS\Documents\FYP\Preprocessing_Output_Folder\Graph_Dataset_(4)")
try:
    with open(GRAPH_DATASET / "edge_labeling_metadata.pkl", 'rb') as f:
        irt_metadata = pickle.load(f)
    IRT_THRESHOLD = irt_metadata['threshold_value']  # -0.6561
    print(f"✅ Loaded IRT metadata: threshold = {IRT_THRESHOLD:.4f}")
except FileNotFoundError:
    # Fallback to 0.5 if metadata not found
    IRT_THRESHOLD = 0.5
    print(f"⚠️  IRT metadata not found, using default threshold = {IRT_THRESHOLD}")

selected_features = metadata['selected_features']
all_features_from_model = metadata.get('all_features', selected_features)  # Features used during training

# NOTE: We need to use the EXACT features the model was trained on
print(f"📋 Model: {len(selected_features)} selected features, F1={metadata['test_metrics']['f1_score']:.4f}")
print(f"   All features used during training: {len(all_features_from_model)}")

# ============================================================================
# STEP 1: LOAD DATA (WITH PROGRESS)
# ============================================================================

print("\n📂 Step 1: Loading datasets...")

# Load with low_memory=False to avoid warnings
print("   Loading train edges...")
train_edges = pd.read_csv(GRAPH_DATASET / "train_edges_with_embeddings.csv", low_memory=False)
print("   Loading validation edges...")
val_edges = pd.read_csv(GRAPH_DATASET / "val_edges_with_embeddings.csv", low_memory=False)
print("   Loading test edges...")
test_edges = pd.read_csv(GRAPH_DATASET / "test_edges_with_embeddings.csv", low_memory=False)
# Combine - DO THIS ONLY ONCE
all_edges = pd.concat([train_edges, val_edges, test_edges], ignore_index=True)
del train_edges, val_edges, test_edges  # Free memory

print(f"✅ Loaded {len(all_edges):,} total edges")

# ============================================================================
# STEP 2: PREPARE FEATURES (OPTIMIZED)
# ============================================================================

print("\n🔧 Step 2: Preparing features for inference...")

# Use the EXACT features the model was trained on (from metadata)
# This ensures feature_selector.transform() works correctly
features_to_use = all_features_from_model

# Check which features are available in the dataset
available_features = [f for f in features_to_use if f in all_edges.columns]
missing_features = [f for f in features_to_use if f not in all_edges.columns]

print(f"   Features from model training: {len(features_to_use)}")
print(f"   Features available in data:   {len(available_features)}")

if missing_features:
    print(f"⚠️  Missing {len(missing_features)} features from training:")
    # Show first 5 missing features
    for f in missing_features[:5]:
        print(f"      - {f}")
    if len(missing_features) > 5:
        print(f"      ... and {len(missing_features) - 5} more")
    
    # If too many features missing, try to find matching embeddings
    if len(missing_features) > 10:
        print("\n⚠️  Many features missing. Checking for alternative embedding columns...")
        embedding_cols_in_data = [col for col in all_edges.columns if col.startswith('emb_')]
        print(f"   Found {len(embedding_cols_in_data)} embedding columns in data")
        
        # If the model expects emb_average_* but data has emb_l1_* (or vice versa)
        # We need to retrain or use compatible data
        print("\n❌ ERROR: Feature mismatch between model and data!")
        print("   The model was trained on different embedding features than what's in the data.")
        print("   Options:")
        print("   1. Use the same embedding data that was used for training")
        print("   2. Retrain the model with the current embedding features")
        print("   3. Run inference without feature selection (see below)")
        
        # FALLBACK: Skip feature selection and use model directly on available features
        print("\n🔄 Attempting fallback: Using model without feature selector...")
        
        # Get the features the model actually expects (from feature_importances_)
        try:
            model_feature_names = model.feature_name_
            print(f"   Model expects {len(model_feature_names)} features")
            
            # Check overlap
            overlap = set(model_feature_names) & set(all_edges.columns)
            print(f"   Overlap with data: {len(overlap)} features")
            
            if len(overlap) == len(model_feature_names):
                print("   ✅ All model features found in data! Proceeding without feature selector.")
                available_features = list(model_feature_names)
                USE_FEATURE_SELECTOR = False
            else:
                missing_model = set(model_feature_names) - set(all_edges.columns)
                print(f"   ❌ Missing {len(missing_model)} features that model expects")
                raise ValueError(f"Cannot proceed: model expects features not in data")
        except AttributeError:
            print("   Could not get feature names from model")
            raise ValueError("Feature mismatch and cannot determine model features")
else:
    USE_FEATURE_SELECTOR = True

def prepare_features_optimized(df, feature_list, label_encoders):
    """Optimized feature preparation"""
    X = df[feature_list].copy()
    
    # Vectorized encoding for categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in label_encoders:
            le = label_encoders[col]
            # Faster: create mapping dict
            mapping = {label: idx for idx, label in enumerate(le.classes_)}
            X[col] = X[col].astype(str).map(mapping).fillna(-1).astype(int)
        else:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Vectorized fillna for numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    
    return X

print(f"   Using {len(available_features)} features for inference")

# Prepare features ONCE
X_all = prepare_features_optimized(all_edges, available_features, label_encoders)

# Apply feature selection only if features match
if USE_FEATURE_SELECTOR:
    X_all_selected = feature_selector.transform(X_all)
    print(f"✅ Prepared and selected features: {X_all_selected.shape}")
else:
    X_all_selected = X_all.values
    print(f"✅ Prepared features (no selector): {X_all_selected.shape}")

# ============================================================================
# STEP 3: RUN INFERENCE (SINGLE PASS)
# ============================================================================

print("\n🤖 Step 3: Running model inference...")

# Single prediction pass
print("   Predicting risk labels...")
risk_predictions = model.predict(X_all_selected)
print("   Calculating risk probabilities...")
risk_probabilities = model.predict_proba(X_all_selected)[:, 1]

# Add to dataframe
all_edges['predicted_risk_label'] = risk_predictions
all_edges['risk_probability'] = risk_probabilities
all_edges['risk_score'] = risk_probabilities

# Binary classification based on model predictions (0.5 threshold)
high_risk_count = (risk_predictions == 1).sum()
print(f"✅ Generated predictions for {len(all_edges):,} contracts")
print(f"   • High Risk (1): {high_risk_count:,} ({high_risk_count/len(risk_predictions)*100:.1f}%)")
print(f"   • Low Risk (0):  {len(all_edges) - high_risk_count:,} ({(1 - high_risk_count/len(risk_predictions))*100:.1f}%)")
print(f"   • Mean Risk Score: {risk_probabilities.mean():.4f}")

# Save contract predictions
output_cols = ['buyerId', 'supplierId', 'awardYear', 'predicted_risk_label', 'risk_probability', 'risk_score']
for col in ['awardPrice_log', 'lotId', 'tedCanId', 'cpv', 'typeOfContract']:
    if col in all_edges.columns:
        output_cols.append(col)

contract_predictions = all_edges[output_cols].copy()
output_file = Path(Config.DATA_DIR) / "contract_risk_predictions.csv"
contract_predictions.to_csv(output_file, index=False)
print(f"✅ Saved: {output_file}")

# ============================================================================
# STEP 4: AGENT-LEVEL AGGREGATION (BINARY CLASSIFICATION)
# ============================================================================

print("\n👤 Step 4: Aggregating to agent-level risk scores (BINARY)...")

# Combine buyer and supplier in one operation
buyer_agg = all_edges.groupby('buyerId').agg({
    'risk_probability': ['mean', 'max', 'std', 'count'],
    'predicted_risk_label': 'mean'
}).reset_index()
buyer_agg.columns = ['agent_id', 'risk_score', 'max_risk', 'risk_std', 'contract_count', 'risk_prediction']
buyer_agg['agent_type'] = 'buyer'

supplier_agg = all_edges.groupby('supplierId').agg({
    'risk_probability': ['mean', 'max', 'std', 'count'],
    'predicted_risk_label': 'mean'
}).reset_index()
supplier_agg.columns = ['agent_id', 'risk_score', 'max_risk', 'risk_std', 'contract_count', 'risk_prediction']
supplier_agg['agent_type'] = 'supplier'

agent_risk_scores = pd.concat([buyer_agg, supplier_agg], ignore_index=True)

# *** BINARY CLASSIFICATION (consistent with model training) ***
# Agent is high-risk if average risk probability >= 0.5
agent_risk_scores['risk_prediction'] = (agent_risk_scores['risk_score'] >= 0.5).astype(int)
agent_risk_scores['risk_category'] = agent_risk_scores['risk_prediction'].map({
    0: 'Low Risk',
    1: 'High Risk'
})

output_file = Path(Config.DATA_DIR) / "agent_risk_scores.csv"
agent_risk_scores.to_csv(output_file, index=False)

low_risk_count = (agent_risk_scores['risk_prediction'] == 0).sum()
high_risk_count = (agent_risk_scores['risk_prediction'] == 1).sum()

print(f"✅ Generated risk scores for {len(agent_risk_scores):,} agents")
print(f"   • Low Risk (0):  {low_risk_count:,} ({low_risk_count/len(agent_risk_scores)*100:.1f}%)")
print(f"   • High Risk (1): {high_risk_count:,} ({high_risk_count/len(agent_risk_scores)*100:.1f}%)")

# ============================================================================
# STEP 5: EMBEDDINGS (BATCHED PCA)
# ============================================================================

print("\n🎨 Step 5: Generating agent embeddings...")

# Use fewer components for speed, or sample data if too large
if len(X_all_selected) > 500000:
    print("   Using sampling for PCA (dataset large)...")
    sample_idx = np.random.choice(len(X_all_selected), 100000, replace=False)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_all_selected[sample_idx])
    embeddings_2d = pca.transform(X_all_selected)
else:
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(X_all_selected)

embeddings_file = Path(Config.DATA_DIR) / "agent_embeddings.npy"
np.save(embeddings_file, embeddings_2d)

embeddings_df = pd.DataFrame(embeddings_2d, columns=['embedding_1', 'embedding_2'])
embeddings_df['risk_score'] = all_edges['risk_score'].values
embeddings_df['buyerId'] = all_edges['buyerId'].values
embeddings_df['supplierId'] = all_edges['supplierId'].values

embeddings_csv = Path(Config.DATA_DIR) / "agent_embeddings.csv"
embeddings_df.to_csv(embeddings_csv, index=False)
print(f"✅ Generated embeddings: {embeddings_2d.shape}")

# ============================================================================
# STEP 6: CONTRACT PATTERN ANALYSIS (BINARY CLASSIFICATION)
# ============================================================================

print("\n📄 Step 6: Analyzing contract patterns (BINARY)...")

agent_risks_for_analyzer = agent_risk_scores[['agent_id', 'risk_score', 'risk_prediction', 'risk_category']].copy()
contract_analyzer = ContractAnalyzer(agent_risks_for_analyzer)

try:
    contract_patterns = contract_analyzer.analyze_contracts(all_edges, output_dir=Config.DATA_DIR)
    print(f"✅ Analyzed {len(contract_patterns):,} contracts")
    
    # Ensure binary classification in contract patterns
    if 'combined_risk_score' in contract_patterns.columns:
        contract_patterns['risk_prediction'] = (contract_patterns['combined_risk_score'] >= 0.5).astype(int)
        contract_patterns['risk_category'] = contract_patterns['risk_prediction'].map({
            0: 'Low Risk',
            1: 'High Risk'
        })
        contract_patterns['risk_score'] = contract_patterns['combined_risk_score']
        
        output_path = Path(Config.DATA_DIR) / 'contract_risk_patterns.csv'
        contract_patterns.to_csv(output_path, index=False)
        print(f"✅ Applied binary classification to contract patterns")
        
except Exception as e:
    print(f"⚠️  Contract analysis error: {str(e)[:100]}")
    contract_patterns = pd.DataFrame()

# ============================================================================
# STEP 7: NETWORK ANALYSIS (WITH TIMEOUT/SAMPLING)
# ============================================================================

print("\n🕸️ Step 7: Network analysis and community detection...")

# If network is too large, sample edges
MAX_EDGES_FOR_NETWORK = 100000

network_data = all_edges[['buyerId', 'supplierId']].copy()
if 'awardPrice_log' in all_edges.columns:
    network_data['awardPrice_log'] = all_edges['awardPrice_log']

if len(network_data) > MAX_EDGES_FOR_NETWORK:
    print(f"   Sampling {MAX_EDGES_FOR_NETWORK:,} edges for network (original: {len(network_data):,})...")
    network_data = network_data.sample(n=MAX_EDGES_FOR_NETWORK, random_state=42)

network_analyzer = NetworkAnalyzer(agent_risks_for_analyzer)

try:
    print("   Building network...")
    network_analyzer.build_network(network_data)
    
    print("   Detecting communities (this may take a few minutes)...")
    network_analyzer.detect_communities(method='louvain')
    
    print("   Analyzing risk propagation...")
    agent_communities, community_summary = network_analyzer.analyze_risk_propagation(
        output_dir=Config.DATA_DIR,
        fast_mode=True
    )
    
    # Apply binary classification to community data
    if not agent_communities.empty and 'risk_score' in agent_communities.columns:
        agent_communities['risk_prediction'] = (agent_communities['risk_score'] >= 0.5).astype(int)
        agent_communities['risk_category'] = agent_communities['risk_prediction'].map({
            0: 'Low Risk',
            1: 'High Risk'
        })
        # Save updated version
        output_file = Path(Config.DATA_DIR) / "agent_communities.csv"
        agent_communities.to_csv(output_file, index=False)
        print(f"✅ Applied binary classification to agent communities")
    
    print(f"✅ Detected {len(community_summary):,} communities")
except Exception as e:
    print(f"⚠️  Network analysis error: {str(e)[:100]}")
    print("   Continuing without network analysis...")
    agent_communities = pd.DataFrame()
    community_summary = pd.DataFrame()

# ============================================================================
# STEP 8: SUMMARY STATISTICS
# ============================================================================

print("\n📊 Step 8: Generating summary statistics...")

summary_stats = {
    'total_contracts': len(all_edges),
    'total_agents': len(agent_risk_scores),
    'high_risk_contracts': high_risk_count,
    'high_risk_agents': (agent_risk_scores['risk_prediction'] == 1).sum(),
    'low_risk_agents': (agent_risk_scores['risk_prediction'] == 0).sum(),
    'avg_risk_score': agent_risk_scores['risk_score'].mean(),
    'model_f1_score': metadata['test_metrics']['f1_score'],
    'model_roc_auc': metadata['test_metrics']['roc_auc'],
    'selected_features': selected_features,
    'n_communities': len(community_summary) if not community_summary.empty else 0,
    'generation_time_seconds': time.time() - start_time,
    'classification_type': 'binary',  # Added to track classification type
    'threshold': 0.5,  # Model uses 0.5 for binary classification
    'irt_threshold': IRT_THRESHOLD  # Original IRT threshold for reference
}

summary_file = Path(Config.DATA_DIR) / "dashboard_summary.pkl"
with open(summary_file, 'wb') as f:
    pickle.dump(summary_stats, f)

# ============================================================================
# VERIFICATION
# ============================================================================

elapsed_time = time.time() - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

print("\n" + "="*80)
print("✅ DATA GENERATION COMPLETE!")
print("="*80)

print(f"\n⏱️  Total Time: {minutes}m {seconds}s")

print("\n📊 Generated Files:")
print(f"   ✓ contract_risk_predictions.csv ({len(contract_predictions):,} rows)")
print(f"   ✓ agent_risk_scores.csv ({len(agent_risk_scores):,} rows)")
print(f"   ✓ agent_embeddings.csv ({len(embeddings_df):,} rows)")
if not contract_patterns.empty:
    print(f"   ✓ contract_risk_patterns.csv ({len(contract_patterns):,} rows)")
if not agent_communities.empty:
    print(f"   ✓ agent_communities.csv / community_risk_summary.csv")

print(f"\n🎯 Risk Classification Summary:")
print(f"   • Classification Type: BINARY (Low Risk / High Risk)")
print(f"   • Model Threshold: 0.5 (standard binary classification)")
print(f"   • IRT Training Threshold: {IRT_THRESHOLD:.4f} (reference)")
print(f"   • Contracts - High Risk: {high_risk_count:,} ({high_risk_count/len(all_edges)*100:.1f}%)")
print(f"   • Agents - High Risk: {summary_stats['high_risk_agents']:,} ({summary_stats['high_risk_agents']/len(agent_risk_scores)*100:.1f}%)")

print("\n🚀 Ready to launch dashboard!")
print("   Run: streamlit run app.py")
print("="*80)