"""
================================================================================
NETWORK ANALYSIS SERVICE - BINARY CLASSIFICATION VERSION
================================================================================
Community detection and risk propagation analysis for binary risk classification
"""

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class NetworkAnalyzer:
    """
    Analyzes procurement networks and detects fraud risk communities
    Updated for BINARY classification: High Risk (1) vs Low Risk (0)
    """
    
    def __init__(self, agent_risks):
        """
        Initialize with agent risk scores
        
        Args:
            agent_risks: DataFrame with columns ['agent_id', 'risk_score', 'risk_prediction', 'risk_category']
                        risk_prediction: 0 (Low Risk) or 1 (High Risk)
                        risk_score: probability [0, 1]
        """
        self.agent_risks = agent_risks.set_index('agent_id')
        self.G = None
        self.communities = None
        self.contracts_df = None  # Store contract data for visualization
        
        # Binary classification threshold
        self.BINARY_THRESHOLD = 0.5
        
        print(f"🕸️ NetworkAnalyzer initialized")
        print(f"   • Agents: {len(self.agent_risks):,}")
        print(f"   • Binary threshold: {self.BINARY_THRESHOLD}")
        
        # Verify risk distribution
        if 'risk_prediction' in self.agent_risks.columns:
            high_risk = (self.agent_risks['risk_prediction'] == 1).sum()
            low_risk = (self.agent_risks['risk_prediction'] == 0).sum()
            print(f"   • High Risk agents: {high_risk:,} ({high_risk/len(self.agent_risks)*100:.1f}%)")
            print(f"   • Low Risk agents: {low_risk:,} ({low_risk/len(self.agent_risks)*100:.1f}%)")
    
    def build_network(self, contracts_df):
        """
        Build network from contract data
        
        Args:
            contracts_df: DataFrame with columns ['buyerId', 'supplierId', 'awardPrice_log' (optional)]
        """
        print("\n🔨 Building network graph...")
        
        # Store contracts for later visualization
        self.contracts_df = contracts_df.copy()
        
        self.G = nx.Graph()
        
        # Add edges from contracts
        edge_data = []
        for _, row in contracts_df.iterrows():
            buyer = row['buyerId']
            supplier = row['supplierId']
            weight = row.get('awardPrice_log', 1.0)
            
            edge_data.append((buyer, supplier, {'weight': weight}))
        
        self.G.add_edges_from(edge_data)
        
        print(f"✅ Network built:")
        print(f"   • Nodes: {self.G.number_of_nodes():,}")
        print(f"   • Edges: {self.G.number_of_edges():,}")
        print(f"   • Avg degree: {sum(dict(self.G.degree()).values()) / self.G.number_of_nodes():.2f}")
    
    def detect_communities(self, method='louvain'):
        """
        Detect communities using specified method
        
        Args:
            method: 'louvain' or 'label_propagation'
        """
        print(f"\n🔍 Detecting communities using {method}...")
        
        if method == 'louvain':
            import community.community_louvain as community_louvain
            self.communities = community_louvain.best_partition(self.G)
        elif method == 'label_propagation':
            communities_generator = nx.algorithms.community.label_propagation_communities(self.G)
            communities_list = list(communities_generator)
            self.communities = {}
            for i, comm in enumerate(communities_list):
                for node in comm:
                    self.communities[node] = i
        else:
            raise ValueError(f"Unknown method: {method}")
        
        n_communities = len(set(self.communities.values()))
        print(f"✅ Detected {n_communities:,} communities")
    
    def analyze_risk_propagation(self, output_dir=None, fast_mode=True):
        """
        Analyze risk propagation through communities (BINARY version)
        
        Returns:
            agent_communities: DataFrame with agent-level community info
            community_summary: DataFrame with community-level statistics
        """
        print("\n📊 Analyzing risk propagation (BINARY classification)...")
        
        if self.communities is None:
            raise ValueError("Run detect_communities() first")
        
        # Agent-level data
        agent_data = []
        
        for agent_id in self.G.nodes():
            # Get risk score (default to 0.25 if not found - low risk)
            risk_score = self.agent_risks.loc[agent_id, 'risk_score'] if agent_id in self.agent_risks.index else 0.25
            risk_prediction = 1 if risk_score >= self.BINARY_THRESHOLD else 0
            risk_category = 'High Risk' if risk_prediction == 1 else 'Low Risk'
            
            # Network metrics
            degree = self.G.degree(agent_id)
            
            # Neighbor risk (BINARY)
            neighbors = list(self.G.neighbors(agent_id))
            if neighbors:
                neighbor_risks = []
                neighbor_predictions = []
                for neighbor in neighbors:
                    if neighbor in self.agent_risks.index:
                        neighbor_risks.append(self.agent_risks.loc[neighbor, 'risk_score'])
                        neighbor_predictions.append(
                            1 if self.agent_risks.loc[neighbor, 'risk_score'] >= self.BINARY_THRESHOLD else 0
                        )
                    else:
                        neighbor_risks.append(0.25)  # Low risk default
                        neighbor_predictions.append(0)
                
                neighbor_avg_risk = np.mean(neighbor_risks)
                neighbor_high_risk_pct = np.mean(neighbor_predictions) * 100
            else:
                neighbor_avg_risk = 0.25
                neighbor_high_risk_pct = 0.0
            
            agent_data.append({
                'agent_id': agent_id,
                'community_id': self.communities[agent_id],
                'risk_score': risk_score,
                'risk_prediction': risk_prediction,
                'risk_category': risk_category,
                'degree': degree,
                'neighbor_avg_risk': neighbor_avg_risk,
                'neighbor_high_risk_pct': neighbor_high_risk_pct
            })
        
        agent_communities = pd.DataFrame(agent_data)
        
        # Community-level aggregation (BINARY)
        print("   Computing community-level statistics...")
        
        community_stats = agent_communities.groupby('community_id').agg({
            'agent_id': 'count',
            'risk_score': ['mean', 'max', 'std'],
            'risk_prediction': ['sum', 'mean'],  # sum = count of high risk, mean = percentage
            'degree': 'mean',
            'neighbor_avg_risk': 'mean'
        }).reset_index()
        
        community_stats.columns = [
            'community_id', 'size', 
            'avg_risk_score', 'max_risk_score', 'risk_std',
            'high_risk_count', 'high_risk_pct',
            'avg_degree', 'avg_neighbor_risk'
        ]
        
        # Convert high_risk_pct to percentage
        community_stats['high_risk_pct'] = community_stats['high_risk_pct'] * 100
        
        # Add community risk category (BINARY)
        community_stats['community_risk_category'] = community_stats['avg_risk_score'].apply(
            lambda x: 'High Risk' if x >= self.BINARY_THRESHOLD else 'Low Risk'
        )
        
        # Sort by risk
        community_stats = community_stats.sort_values('avg_risk_score', ascending=False)
        
        print(f"✅ Risk propagation analysis complete")
        print(f"   • Total communities: {len(community_stats):,}")
        
        # Binary classification summary
        high_risk_communities = (community_stats['avg_risk_score'] >= self.BINARY_THRESHOLD).sum()
        low_risk_communities = len(community_stats) - high_risk_communities
        
        print(f"   • High Risk communities: {high_risk_communities:,} ({high_risk_communities/len(community_stats)*100:.1f}%)")
        print(f"   • Low Risk communities: {low_risk_communities:,} ({low_risk_communities/len(community_stats)*100:.1f}%)")
        print(f"   • Avg community size: {community_stats['size'].mean():.1f}")
        print(f"   • Largest community: {community_stats['size'].max():,}")
        
        # Save results
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            agent_file = output_dir / "agent_communities.csv"
            agent_communities.to_csv(agent_file, index=False)
            print(f"   📁 Saved: {agent_file}")
            
            summary_file = output_dir / "community_risk_summary.csv"
            community_stats.to_csv(summary_file, index=False)
            print(f"   📁 Saved: {summary_file}")
            
            # *** NEW: Save network edges for visualization ***
            if self.contracts_df is not None:
                # Add community info to contracts
                contracts_with_community = self.contracts_df.copy()
                
                # Map agents to communities
                agent_to_community = {agent: self.communities.get(agent, -1) 
                                     for agent in self.G.nodes()}
                
                contracts_with_community['buyer_community'] = contracts_with_community['buyerId'].map(agent_to_community)
                contracts_with_community['supplier_community'] = contracts_with_community['supplierId'].map(agent_to_community)
                
                # Save network edges
                edges_file = output_dir / "network_edges.csv"
                contracts_with_community.to_csv(edges_file, index=False)
                print(f"   📁 Saved: {edges_file} (for network visualization)")
        
        return agent_communities, community_stats
    
    def get_community_subgraph(self, community_id):
        """
        Extract subgraph for a specific community
        
        Args:
            community_id: Community ID to extract
            
        Returns:
            NetworkX graph of the community
        """
        nodes = [node for node, comm in self.communities.items() if comm == community_id]
        return self.G.subgraph(nodes).copy()
    
    def get_high_risk_communities(self, threshold=None):
        """
        Get communities with high average risk
        
        Args:
            threshold: Risk threshold (default: 0.5 for binary classification)
            
        Returns:
            List of community IDs with avg risk >= threshold
        """
        if threshold is None:
            threshold = self.BINARY_THRESHOLD
        
        if self.communities is None:
            raise ValueError("Run analyze_risk_propagation() first")
        
        # Compute community average risks
        community_risks = defaultdict(list)
        for agent_id, comm_id in self.communities.items():
            if agent_id in self.agent_risks.index:
                risk_score = self.agent_risks.loc[agent_id, 'risk_score']
                community_risks[comm_id].append(risk_score)
        
        high_risk_comms = []
        for comm_id, risks in community_risks.items():
            if np.mean(risks) >= threshold:
                high_risk_comms.append(comm_id)
        
        return high_risk_comms
    
    def identify_risk_clusters(self, min_size=5):
        """
        Identify tightly connected high-risk clusters (BINARY)
        
        Args:
            min_size: Minimum cluster size
            
        Returns:
            List of risk cluster dictionaries
        """
        print(f"\n🎯 Identifying high-risk clusters (min size: {min_size})...")
        
        high_risk_comms = self.get_high_risk_communities(threshold=self.BINARY_THRESHOLD)
        
        clusters = []
        for comm_id in high_risk_comms:
            subgraph = self.get_community_subgraph(comm_id)
            
            if subgraph.number_of_nodes() >= min_size:
                # Get agents in this community
                agents = list(subgraph.nodes())
                
                # Calculate cluster statistics
                risk_scores = []
                risk_predictions = []
                for agent in agents:
                    if agent in self.agent_risks.index:
                        risk_scores.append(self.agent_risks.loc[agent, 'risk_score'])
                        risk_predictions.append(
                            1 if self.agent_risks.loc[agent, 'risk_score'] >= self.BINARY_THRESHOLD else 0
                        )
                
                if risk_scores:
                    clusters.append({
                        'community_id': comm_id,
                        'size': len(agents),
                        'avg_risk_score': np.mean(risk_scores),
                        'max_risk_score': np.max(risk_scores),
                        'high_risk_count': sum(risk_predictions),
                        'high_risk_pct': np.mean(risk_predictions) * 100,
                        'density': nx.density(subgraph),
                        'avg_degree': sum(dict(subgraph.degree()).values()) / len(agents)
                    })
        
        clusters = sorted(clusters, key=lambda x: x['avg_risk_score'], reverse=True)
        
        print(f"✅ Found {len(clusters)} high-risk clusters")
        if clusters:
            print(f"   • Total agents in clusters: {sum(c['size'] for c in clusters):,}")
            print(f"   • Avg cluster size: {np.mean([c['size'] for c in clusters]):.1f}")
            print(f"   • Highest avg risk: {clusters[0]['avg_risk_score']:.4f}")
        
        return clusters