"""
================================================================================
CREATE FEATURE CONFIGURATION
================================================================================
File: deployment/create_feature_config.py

Generates a JSON file documenting:
1. Features used for model training (including graph embeddings)
2. Features selected by feature selector
3. Column name mappings for dashboard
4. Feature descriptions

Run once after training: python create_feature_config.py
"""

import json
import pickle
from pathlib import Path

# Load model metadata
MODEL_DIR = Path(r"C:\Users\ASUS\Documents\FYP\deployment\models")

with open(MODEL_DIR / "lgbm_model_metadata.pkl", 'rb') as f:
    metadata = pickle.load(f)

# Define base features used in training (non-embedding)
BASE_TRAINING_FEATURES = [
    'low_transparency_risk', 'competition_x_transparency', 'composite_risk_score',
    'low_competition_risk', 'duration_anomaly', 'weighted_compliance_risk',
    'regulatory_compliance_risk', 'awardYear', 'contractDuration', 'cpv', 
    'src_degree', 'dst_degree', 'degree_diff', 'core_difference', 'pagerank_diff', 
    'dst_pagerank', 'src_pagerank', 'typeOfContract', 'onBehalf', 'fraAgreement',
    'outOfDirectives', 'gpa', 'edge_betweenness', 'both_high_influence',
    'correctionsNb', 'cancelled', 'awardPrice_log'
]

# Get all features from metadata (includes embeddings)
ALL_TRAINING_FEATURES = metadata.get('all_features', BASE_TRAINING_FEATURES)

# Separate embedding features
EMBEDDING_FEATURES = [f for f in ALL_TRAINING_FEATURES if f.startswith('emb_')]

# Column name mappings (what dashboard expects vs what data has)
COLUMN_MAPPINGS = {
    'dashboard_expects': {
        'risk_score': ['combined_risk_score', 'risk_probability', 'risk_score'],
        'risk_prediction': ['predicted_risk_label', 'risk_prediction'],
        'risk_category': ['risk_category'],
        'agent_id': ['buyerId', 'supplierId', 'agent_id'],
        'contract_id': ['lotId', 'tedCanId']
    },
    'data_files': {
        'contract_risk_patterns.csv': ['combined_risk_score', 'buyer_risk_score', 'supplier_risk_score'],
        'contract_risk_predictions.csv': ['risk_probability', 'predicted_risk_label'],
        'agent_risk_scores.csv': ['risk_score', 'risk_prediction', 'risk_category']
    }
}

# Feature descriptions (base features)
FEATURE_DESCRIPTIONS = {
    'low_transparency_risk': 'Indicator of low transparency in procurement process',
    'competition_x_transparency': 'Interaction between competition and transparency',
    'composite_risk_score': 'Overall composite risk indicator',
    'low_competition_risk': 'Indicator of insufficient competition',
    'duration_anomaly': 'Abnormal contract duration patterns',
    'weighted_compliance_risk': 'Weighted regulatory compliance risk score',
    'regulatory_compliance_risk': 'Non-compliance with regulations indicator',
    'awardYear': 'Year of contract award',
    'contractDuration': 'Duration of contract in days',
    'cpv': 'Common Procurement Vocabulary code',
    'src_degree': 'Number of connections for source node (buyer)',
    'dst_degree': 'Number of connections for target node (supplier)',
    'degree_diff': 'Difference in node degrees',
    'core_difference': 'Difference in core numbers',
    'pagerank_diff': 'Difference in PageRank scores',
    'dst_pagerank': 'PageRank score of supplier',
    'src_pagerank': 'PageRank score of buyer',
    'typeOfContract': 'Type of procurement contract',
    'onBehalf': 'Whether contract is on behalf of another entity',
    'fraAgreement': 'Framework agreement indicator',
    'outOfDirectives': 'Whether outside EU directives',
    'gpa': 'Government Procurement Agreement indicator',
    'edge_betweenness': 'Betweenness centrality of edge',
    'both_high_influence': 'Both parties have high influence',
    'correctionsNb': 'Number of corrections made',
    'cancelled': 'Contract cancellation status',
    'awardPrice_log': 'Log-transformed award price'
}

# Add descriptions for embedding features
for emb_feat in EMBEDDING_FEATURES:
    FEATURE_DESCRIPTIONS[emb_feat] = f'Graph embedding dimension ({emb_feat})'

# Create configuration - UPDATED FOR LIGHTGBM
config = {
    'model_info': {
        'model_type': 'LightGBM',  # CHANGED from RandomForest
        'test_f1_score': metadata['test_metrics']['f1_score'],
        'test_roc_auc': metadata['test_metrics']['roc_auc'],
        'test_precision': metadata['test_metrics']['precision'],
        'test_recall': metadata['test_metrics']['recall'],
        # LightGBM specific parameters
        'n_estimators': metadata['best_params'].get('n_estimators', 300),
        'max_depth': metadata['best_params'].get('max_depth', 20),
        'num_leaves': metadata['best_params'].get('num_leaves', 100),
        'learning_rate': metadata['best_params'].get('learning_rate', 0.15),
    },
    'features': {
        'base_features': BASE_TRAINING_FEATURES,
        'embedding_features': EMBEDDING_FEATURES,
        'all_training_features': ALL_TRAINING_FEATURES,
        'selected_features': metadata['selected_features'],
        'n_base': len(BASE_TRAINING_FEATURES),
        'n_embeddings': len(EMBEDDING_FEATURES),
        'n_total': len(ALL_TRAINING_FEATURES),
        'n_selected': len(metadata['selected_features']),
        'feature_selection_method': metadata.get('feature_selection_method', 'SelectKBest (f_classif)'),
        'feature_descriptions': FEATURE_DESCRIPTIONS
    },
    'column_mappings': COLUMN_MAPPINGS,
    'data_schema': {
        'contracts': {
            'required_columns': ['buyerId', 'supplierId', 'risk_score', 'risk_prediction'],
            'optional_columns': ['awardYear', 'awardPrice_log', 'cpv', 'typeOfContract']
        },
        'agents': {
            'required_columns': ['agent_id', 'risk_score', 'risk_prediction', 'risk_category'],
            'optional_columns': ['agent_type', 'contract_count', 'max_risk']
        }
    }
}

# Save configuration
output_dir = Path(__file__).parent / "data"
output_dir.mkdir(exist_ok=True)

output_file = output_dir / 'feature_config.json'
with open(output_file, 'w') as f:
    json.dump(config, f, indent=2)

print("="*80)
print("✅ FEATURE CONFIGURATION CREATED")
print("="*80)

print(f"\n📁 Saved: {output_file}")

print("\n📋 Model Information:")
print(f"   • Model Type: {config['model_info']['model_type']}")
print(f"   • F1 Score: {config['model_info']['test_f1_score']:.4f}")
print(f"   • ROC-AUC: {config['model_info']['test_roc_auc']:.4f}")
print(f"   • Learning Rate: {config['model_info']['learning_rate']}")
print(f"   • Num Leaves: {config['model_info']['num_leaves']}")

print(f"\n🎯 Features:")
print(f"   • Base Features: {len(BASE_TRAINING_FEATURES)}")
print(f"   • Embedding Features: {len(EMBEDDING_FEATURES)}")
print(f"   • Total Training Features: {len(ALL_TRAINING_FEATURES)}")
print(f"   • Selected Features: {len(metadata['selected_features'])}")

print(f"\n📊 Top 10 Selected Features:")
for i, feat in enumerate(metadata['selected_features'][:10], 1):
    desc = FEATURE_DESCRIPTIONS.get(feat, 'Graph embedding dimension')
    print(f"   {i:2d}. {feat:35s}")

if len(metadata['selected_features']) > 10:
    print(f"   ... and {len(metadata['selected_features']) - 10} more")

print("\n🔧 Column Mappings:")
print("   Dashboard expects 'risk_score' -> Data has: combined_risk_score, risk_probability")

print("\n✅ Configuration ready for dashboard use!")
print("="*80)