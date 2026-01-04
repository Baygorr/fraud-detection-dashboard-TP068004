"""
================================================================================
SERVICES: CONTRACT ANALYSIS
================================================================================
File: deployment/services/contract_analysis.py

Analyzes contract-level risk patterns by combining agent predictions
with contract data. Identifies high-risk contracts not caught by simple rules.

Updated to use BINARY CLASSIFICATION (Low Risk / High Risk) consistent with
IRT-based model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path

class ContractAnalyzer:
    def __init__(self, agent_risk_scores, threshold=0.5):
        """
        Args:
            agent_risk_scores: DataFrame with agent_id, risk_score, risk_prediction
            threshold: Risk threshold for binary classification (default: 0.5)
        """
        self.agent_risks = agent_risk_scores
        self.threshold = threshold
        print(f"📋 ContractAnalyzer initialized with binary classification (threshold={threshold})")
    
    def analyze_contracts(self, contracts_df, output_dir='data'):
        """
        Compute contract-level risk patterns using BINARY classification
        
        Args:
            contracts_df: DataFrame with contract information
                Required columns: buyerId, supplierId, awardPrice, awardYear, etc.
            output_dir: Directory to save outputs
        
        Returns:
            contract_patterns: DataFrame with contract risk analysis
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        df = contracts_df.copy()
        
        # Merge buyer risk
        buyer_risks = self.agent_risks.copy()
        buyer_risks.columns = ['buyerId', 'buyer_risk_score', 'buyer_risk_pred', 'buyer_risk_cat']
        df = df.merge(buyer_risks, on='buyerId', how='left')
        
        # Merge supplier risk
        supplier_risks = self.agent_risks.copy()
        supplier_risks.columns = ['supplierId', 'supplier_risk_score', 'supplier_risk_pred', 'supplier_risk_cat']
        df = df.merge(supplier_risks, on='supplierId', how='left')
        
        # Fill missing values
        df['buyer_risk_score'] = df['buyer_risk_score'].fillna(df['buyer_risk_score'].mean())
        df['supplier_risk_score'] = df['supplier_risk_score'].fillna(df['supplier_risk_score'].mean())
        
        # Compute combined risk score (weighted average: supplier more important)
        df['combined_risk_score'] = (
            0.3 * df['buyer_risk_score'] + 
            0.7 * df['supplier_risk_score']
        )
        
        # *** BINARY CLASSIFICATION (consistent with model training) ***
        # Risk prediction: 1 if combined_risk_score >= threshold, else 0
        df['risk_prediction'] = (df['combined_risk_score'] >= self.threshold).astype(int)
        
        # Risk category: BINARY (Low Risk / High Risk)
        df['risk_category'] = df['risk_prediction'].map({
            0: 'Low Risk',
            1: 'High Risk'
        })
        
        # Identify anomalies: high ML risk but "normal" contract attributes
        if 'awardPrice_log' in df.columns:
            df['price_zscore'] = (df['awardPrice_log'] - df['awardPrice_log'].mean()) / df['awardPrice_log'].std()
            df['is_price_anomaly'] = df['price_zscore'].abs() > 2
        else:
            df['is_price_anomaly'] = False
        
        # Flag: High ML risk but not price anomaly (false negatives of simple rules)
        df['ml_caught_not_rules'] = (
            (df['risk_prediction'] == 1) & 
            (~df['is_price_anomaly'])
        )
        
        # Temporal clustering
        if 'awardYear' in df.columns:
            # Check if awardMonth exists
            if 'awardMonth' in df.columns:
                temporal_risk = df.groupby(['awardYear', 'awardMonth'])['risk_prediction'].mean()
                df['temporal_risk_cluster'] = df.apply(
                    lambda x: temporal_risk.get((x['awardYear'], x.get('awardMonth', 1)), 0),
                    axis=1
                )
            else:
                # Use only awardYear if awardMonth doesn't exist
                temporal_risk = df.groupby('awardYear')['risk_prediction'].mean()
                df['temporal_risk_cluster'] = df['awardYear'].map(temporal_risk)
        
        # Select output columns
        output_cols = [
            'buyerId', 'supplierId', 'lotId', 'tedCanId',
            'buyer_risk_score', 'supplier_risk_score', 'combined_risk_score',
            'risk_prediction', 'risk_category',
            'ml_caught_not_rules'
        ]
        
        # Add available columns
        for col in ['awardYear', 'awardPrice_log', 'cpv', 'price_zscore', 
                    'is_price_anomaly', 'temporal_risk_cluster']:
            if col in df.columns:
                output_cols.append(col)
        
        contract_patterns = df[output_cols].copy()
        
        # Add risk_score column for dashboard compatibility
        contract_patterns['risk_score'] = contract_patterns['combined_risk_score']
        
        # Save
        output_path = output_dir / 'contract_risk_patterns.csv'
        contract_patterns.to_csv(output_path, index=False)
        print(f"✅ Saved: {output_path}")
        
        # Summary statistics (BINARY)
        print(f"\n📊 Contract Risk Analysis Summary (BINARY):")
        print(f"   Total contracts: {len(contract_patterns):,}")
        print(f"   Low Risk (0):  {(contract_patterns['risk_prediction']==0).sum():,} "
              f"({(contract_patterns['risk_prediction']==0).sum()/len(contract_patterns)*100:.1f}%)")
        print(f"   High Risk (1): {(contract_patterns['risk_prediction']==1).sum():,} "
              f"({(contract_patterns['risk_prediction']==1).sum()/len(contract_patterns)*100:.1f}%)")
        print(f"   ML-caught (not rules): {contract_patterns['ml_caught_not_rules'].sum():,}")
        
        # Show risk category distribution
        risk_cat_dist = contract_patterns['risk_category'].value_counts()
        print(f"   Risk categories: {risk_cat_dist.to_dict()}")
        
        return contract_patterns