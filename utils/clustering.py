from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import streamlit as st

class ClusteringAnalysis:
    def __init__(self, df: pd.DataFrame, numeric_columns: list):
        self.df = df
        self.numeric_columns = numeric_columns
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(df[numeric_columns])
        
    def perform_kmeans(self, n_clusters: int) -> pd.Series:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return pd.Series(kmeans.fit_predict(self.X_scaled), name='cluster')
    
    def perform_dbscan(self, eps: float, min_samples: int) -> pd.Series:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        return pd.Series(dbscan.fit_predict(self.X_scaled), name='cluster')
    
    def perform_hierarchical(self, n_clusters: int) -> pd.Series:
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        return pd.Series(hierarchical.fit_predict(self.X_scaled), name='cluster')
    
    def get_cluster_statistics(self, clusters: pd.Series) -> pd.DataFrame:
        df_with_clusters = self.df.copy()
        df_with_clusters['Cluster'] = clusters
        
        stats = []
        for cluster in df_with_clusters['Cluster'].unique():
            cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
            cluster_stats = {
                'Cluster': cluster,
                'Size': len(cluster_data),
                'Percentage': len(cluster_data) / len(df_with_clusters) * 100
            }
            for col in self.numeric_columns:
                cluster_stats[f'{col}_mean'] = cluster_data[col].mean()
                cluster_stats[f'{col}_std'] = cluster_data[col].std()
            stats.append(cluster_stats)
            
        return pd.DataFrame(stats)