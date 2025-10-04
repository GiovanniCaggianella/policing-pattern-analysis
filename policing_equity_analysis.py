"""
Unsupervised ML approach to understanding patterns in police activities

Author: Giovanni Caggianella
Matriculation number: 9215236
Course: DLBDSMLUSL01
Date: October 1, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats
import warnings
import os

# Configuration
plt.style.use('seaborn-v0_8')
warnings.filterwarnings('ignore')
sns.set_palette("husl")

# File paths
DATA_PATH = "archive/Dept_49-00081/49-00081_Incident-Reports_2012_to_May_2015.csv"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class PolicingEquityAnalysis:
    """
    Complete analysis pipeline for policing equity study.
    Implements clustering to identify distinct policing patterns across districts.
    """
    
    def __init__(self):
        self.df = None
        self.crime_agg = None
        self.scaled_data = None
        self.optimal_k = None
        self.features = None
        self.clusters = None
        
    def run(self):
        """Execute the complete analysis pipeline."""
        
        self.load_data()
        self.preprocess()
        self.engineer_features()
        self.aggregate_by_district()
        self.descriptive_statistics()
        self.determine_optimal_clusters()
        self.perform_clustering()
        self.validate_with_hierarchical()
        self.dimensionality_reduction()
        self.visualize()
        
        print("\n" + "="*70)
        print("✓ Analysis completed successfully!")
        print(f"📁 Results saved in: {OUTPUT_DIR}/")
        print("="*70)
        
    def load_data(self):
        """Load and perform initial data inspection."""
        print("\n[1/10] Loading data...")
        
        self.df = pd.read_csv(DATA_PATH, low_memory=False)
        
        # Remove header row if present
        if str(self.df.iloc[0].get('INCIDENT_UNIQUE_IDENTIFIER', '')).strip() == 'IncidntNum':
            self.df = self.df.drop(0).reset_index(drop=True)
        
        print(f"  ✓ Loaded {len(self.df):,} policing incidents")
        
    def preprocess(self):
        """Clean and standardize data types."""
        print("\n[2/10] Preprocessing & cleaning...")
        
        # Geographic coordinates - convert to numeric
        self.df['LOCATION_LONGITUDE'] = pd.to_numeric(
            self.df['LOCATION_LONGITUDE'], errors='coerce'
        )
        self.df['LOCATION_LATITUDE'] = pd.to_numeric(
            self.df['LOCATION_LATITUDE'], errors='coerce'
        )
        
        # Temporal parsing
        self.df['INCIDENT_DATE'] = pd.to_datetime(
            self.df['INCIDENT_DATE'], errors='coerce'
        )
        
        time_parsed = pd.to_datetime(
            self.df['INCIDENT_TIME'], format='%H:%M', errors='coerce'
        )
        self.df['HOUR'] = time_parsed.dt.hour
        
        print(f"  ✓ Data cleaning completed")
        
    def engineer_features(self):
        """Create derived features from raw data."""
        print("\n[3/10] Feature engineering...")
        
        # Date components
        self.df['YEAR'] = self.df['INCIDENT_DATE'].dt.year
        self.df['MONTH'] = self.df['INCIDENT_DATE'].dt.month
        self.df['DAY_OF_WEEK'] = self.df['INCIDENT_DATE'].dt.dayofweek
        
        # Binary weekend indicator
        self.df['IS_WEEKEND'] = self.df['DAY_OF_WEEK'].isin([5, 6]).astype(int)
        
        print(f"  ✓ Created temporal features")
        
    def aggregate_by_district(self):
        """Aggregate incident-level data to district level."""
        print("\n[4/10] Aggregating to district level...")
        
        districts = []
        
        for district in self.df['LOCATION_DISTRICT'].dropna().unique():
            d_data = self.df[self.df['LOCATION_DISTRICT'] == district]
            crime_types = d_data['INCIDENT_REASON'].value_counts()
            total = len(d_data)
            
            # Calculate district-level metrics
            districts.append({
                'DISTRICT': district,
                'TOTAL_INCIDENTS': total,
                'INCIDENTS_PER_DAY': total / ((2015 - 2012) * 365),
                
                # Crime categorization
                'VIOLENT_CRIMES': (
                    crime_types.get('ASSAULT', 0) + 
                    crime_types.get('ROBBERY', 0)
                ),
                'PROPERTY_CRIMES': (
                    crime_types.get('LARCENY/THEFT', 0) + 
                    crime_types.get('BURGLARY', 0)
                ),
                'DRUG_CRIMES': crime_types.get('DRUG/NARCOTIC', 0),
                
                # Complexity measure
                'CRIME_DIVERSITY': len(crime_types),
                
                # Temporal patterns
                'AVG_HOUR': d_data['HOUR'].mean(),
                'WEEKEND_RATE': d_data['IS_WEEKEND'].mean() * 100,
                'NIGHT_RATE': (
                    (d_data['HOUR'] >= 22) | (d_data['HOUR'] <= 6)
                ).mean() * 100,
            })
        
        self.crime_agg = pd.DataFrame(districts)
        print(f"  ✓ Aggregated to {len(self.crime_agg)} districts")
        
    def descriptive_statistics(self):
        """Compute comprehensive descriptive statistics."""
        print("\n[5/10] Computing descriptive statistics...")
        
        # Features for analysis
        feature_cols = [
            'TOTAL_INCIDENTS', 'INCIDENTS_PER_DAY', 'VIOLENT_CRIMES',
            'PROPERTY_CRIMES', 'DRUG_CRIMES', 'CRIME_DIVERSITY',
            'AVG_HOUR', 'WEEKEND_RATE', 'NIGHT_RATE'
        ]
        
        self.stats = {}
        for col in feature_cols:
            data = self.crime_agg[col].dropna()
            self.stats[col] = {
                'mean': data.mean(),
                'std': data.std(),
                'median': data.median(),
                'min': data.min(),
                'max': data.max(),
                'cv': (data.std() / data.mean() * 100) if data.mean() != 0 else 0,
                'skewness': stats.skew(data),
            }
        
        # Crime type distribution
        self.crime_distribution = self.df['INCIDENT_REASON'].value_counts()
        
        # Temporal patterns
        self.peak_hour = self.df['HOUR'].mode()[0]
        self.weekend_pct = (self.df['IS_WEEKEND'] == 1).sum() / len(self.df) * 100
        
        print(f"  ✓ Statistics computed for {len(feature_cols)} features")
        
    def determine_optimal_clusters(self):
        """Use silhouette method to find optimal K."""
        print("\n[6/10] Determining optimal number of clusters...")
        
        # Select features for clustering
        self.features = [
            'TOTAL_INCIDENTS', 'INCIDENTS_PER_DAY', 'VIOLENT_CRIMES',
            'PROPERTY_CRIMES', 'DRUG_CRIMES', 'CRIME_DIVERSITY',
            'AVG_HOUR', 'WEEKEND_RATE', 'NIGHT_RATE'
        ]
        
        X = self.crime_agg[self.features].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(X)
        
        # Test different K values
        K_range = range(2, 8)
        silhouette_scores = []
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.scaled_data)
            score = silhouette_score(self.scaled_data, labels)
            silhouette_scores.append(score)
            print(f"  K={k}: Silhouette Score = {score:.3f}")
        
        # Store results
        self.silhouette_results = {
            'K_range': list(K_range),
            'scores': silhouette_scores
        }
        
        optimal_idx = np.argmax(silhouette_scores)
        self.optimal_k = K_range[optimal_idx]
        
        print(f"  ✓ Optimal K = {self.optimal_k} (Silhouette = {silhouette_scores[optimal_idx]:.3f})")
        
    def perform_clustering(self):
        """Apply K-Means clustering with optimal K."""
        print(f"\n[7/10] Performing K-Means clustering (K={self.optimal_k})...")
        
        # K-Means clustering
        kmeans = KMeans(
            n_clusters=self.optimal_k, 
            random_state=42, 
            n_init=10
        )
        self.crime_agg['CLUSTER'] = kmeans.fit_predict(self.scaled_data)
        
        # Store cluster centroids
        self.centroids = kmeans.cluster_centers_
        
        print(f"  ✓ Districts assigned to {self.optimal_k} clusters")
        
        # Display cluster composition
        for i in range(self.optimal_k):
            districts = self.crime_agg[self.crime_agg['CLUSTER'] == i]['DISTRICT'].tolist()
            print(f"  Cluster {i}: {', '.join(districts)} ({len(districts)} districts)")
        
    def validate_with_hierarchical(self):
        """Validate clustering using hierarchical method."""
        print("\n[8/10] Validating with hierarchical clustering...")
        
        # Agglomerative clustering with Ward linkage
        hierarchical = AgglomerativeClustering(
            n_clusters=self.optimal_k, 
            linkage='ward'
        )
        self.crime_agg['CLUSTER_HIER'] = hierarchical.fit_predict(self.scaled_data)
        
        # Create linkage matrix for dendrogram
        self.linkage_matrix = linkage(self.scaled_data, method='ward')
        
        # Calculate agreement
        agreement = (
            self.crime_agg['CLUSTER'] == self.crime_agg['CLUSTER_HIER']
        ).mean() * 100
        
        print(f"  ✓ Hierarchical clustering completed")
        print(f"  ✓ Agreement with K-Means: {agreement:.0f}%")
        
    def dimensionality_reduction(self):
        """Apply PCA for 2D and 3D visualization."""
        print("\n[9/10] Performing PCA dimensionality reduction...")
        
        # PCA to 2 components
        self.pca_2d = PCA(n_components=2)
        self.pca_data_2d = self.pca_2d.fit_transform(self.scaled_data)
        
        variance_2d = self.pca_2d.explained_variance_ratio_
        
        print(f"  ✓ 2D PCA: PC1={variance_2d[0]:.1%}, PC2={variance_2d[1]:.1%}")
        print(f"  ✓ 2D Total variance explained: {variance_2d.sum():.1%}")
        
        # PCA to 3 components
        self.pca_3d = PCA(n_components=3)
        self.pca_data_3d = self.pca_3d.fit_transform(self.scaled_data)
        
        variance_3d = self.pca_3d.explained_variance_ratio_
        
        print(f"  ✓ 3D PCA: PC1={variance_3d[0]:.1%}, PC2={variance_3d[1]:.1%}, PC3={variance_3d[2]:.1%}")
        print(f"  ✓ 3D Total variance explained: {variance_3d.sum():.1%}")
        
    def visualize(self):
        """Create comprehensive visualizations."""
        print("\n[10/10] Creating visualizations...")
        
        # 1. Categorical Analysis
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Categorical Analysis - Crime Distribution', 
                     fontsize=16, fontweight='bold')
        
        # Top crimes
        top_crimes = self.crime_distribution.head(10)
        axes[0,0].barh(range(len(top_crimes)), top_crimes.values, color='steelblue')
        axes[0,0].set_yticks(range(len(top_crimes)))
        axes[0,0].set_yticklabels([c[:35] for c in top_crimes.index], fontsize=9)
        axes[0,0].set_xlabel('Number of Incidents')
        axes[0,0].set_title('Top 10 Crime Categories')
        axes[0,0].invert_yaxis()
        
        # District comparison
        axes[0,1].bar(self.crime_agg['DISTRICT'], 
                      self.crime_agg['TOTAL_INCIDENTS'],
                      color='coral')
        axes[0,1].set_xlabel('District')
        axes[0,1].set_ylabel('Total Incidents')
        axes[0,1].set_title('Incidents by District')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Crime types by district
        crime_types = self.crime_agg[['DISTRICT', 'VIOLENT_CRIMES', 
                                       'PROPERTY_CRIMES', 'DRUG_CRIMES']].set_index('DISTRICT')
        crime_types.plot(kind='bar', stacked=True, ax=axes[1,0], 
                        color=['#e74c3c', '#3498db', '#2ecc71'])
        axes[1,0].set_xlabel('District')
        axes[1,0].set_ylabel('Number of Incidents')
        axes[1,0].set_title('Crime Composition by District')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].legend(title='Crime Type')
        
        # Crime diversity
        axes[1,1].bar(self.crime_agg['DISTRICT'], 
                      self.crime_agg['CRIME_DIVERSITY'],
                      color='purple', alpha=0.7)
        axes[1,1].set_xlabel('District')
        axes[1,1].set_ylabel('Number of Crime Types')
        axes[1,1].set_title('Crime Diversity by District')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/categorical_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Temporal Analysis
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Temporal Analysis - Crime Patterns', 
                     fontsize=16, fontweight='bold')
        
        # Hourly distribution
        hourly = self.df['HOUR'].value_counts().sort_index()
        axes[0,0].plot(hourly.index, hourly.values, 'o-', 
                      color='darkgreen', linewidth=2, markersize=6)
        axes[0,0].axvline(x=self.peak_hour, color='red', 
                         linestyle='--', label=f'Peak: {self.peak_hour}:00')
        axes[0,0].set_xlabel('Hour of Day')
        axes[0,0].set_ylabel('Number of Incidents')
        axes[0,0].set_title('Incidents by Hour')
        axes[0,0].grid(alpha=0.3)
        axes[0,0].legend()
        
        # Day of week
        daily = self.df['DAY_OF_WEEK'].value_counts().sort_index()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[0,1].bar(range(7), daily.values, color='orange', alpha=0.7)
        axes[0,1].set_xticks(range(7))
        axes[0,1].set_xticklabels(day_names)
        axes[0,1].set_ylabel('Number of Incidents')
        axes[0,1].set_title('Incidents by Day of Week')
        
        # Weekend vs Weekday rates by district
        axes[1,0].bar(self.crime_agg['DISTRICT'], 
                      self.crime_agg['WEEKEND_RATE'],
                      color='skyblue', alpha=0.7)
        axes[1,0].axhline(y=self.weekend_pct, color='red', 
                         linestyle='--', label=f'City avg: {self.weekend_pct:.1f}%')
        axes[1,0].set_xlabel('District')
        axes[1,0].set_ylabel('Weekend Rate (%)')
        axes[1,0].set_title('Weekend Incident Rate by District')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].legend()
        
        # Night rate by district
        axes[1,1].bar(self.crime_agg['DISTRICT'], 
                      self.crime_agg['NIGHT_RATE'],
                      color='navy', alpha=0.7)
        axes[1,1].set_xlabel('District')
        axes[1,1].set_ylabel('Night Rate (%)')
        axes[1,1].set_title('Night Incident Rate by District (22:00-06:00)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/temporal_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Silhouette Analysis
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.silhouette_results['K_range'], 
               self.silhouette_results['scores'], 
               'o-', linewidth=2, markersize=8, color='darkblue')
        ax.axvline(x=self.optimal_k, color='red', linestyle='--', 
                  label=f'Optimal K={self.optimal_k}')
        ax.set_xlabel('Number of Clusters (K)', fontsize=12)
        ax.set_ylabel('Silhouette Score', fontsize=12)
        ax.set_title('Silhouette Analysis for Optimal Cluster Determination', 
                    fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/silhouette_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. PCA Visualizations (2D and 3D side by side)
        fig = plt.figure(figsize=(18, 7))
        
        # 2D PCA (left)
        ax1 = fig.add_subplot(121)
        scatter_2d = ax1.scatter(self.pca_data_2d[:, 0], self.pca_data_2d[:, 1], 
                           c=self.crime_agg['CLUSTER'], 
                           cmap='viridis', s=300, alpha=0.7, 
                           edgecolors='black', linewidth=2)
        
        # Annotate districts
        for idx, row in self.crime_agg.iterrows():
            ax1.annotate(row['DISTRICT'], 
                       (self.pca_data_2d[idx, 0], self.pca_data_2d[idx, 1]),
                       fontsize=9, ha='center', fontweight='bold')
        
        ax1.set_xlabel(f'PC1 ({self.pca_2d.explained_variance_ratio_[0]:.1%} variance)', 
                     fontsize=11)
        ax1.set_ylabel(f'PC2 ({self.pca_2d.explained_variance_ratio_[1]:.1%} variance)', 
                     fontsize=11)
        ax1.set_title('2D PCA Projection', 
                    fontsize=13, fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # 3D PCA (right)
        ax2 = fig.add_subplot(122, projection='3d')
        scatter_3d = ax2.scatter(self.pca_data_3d[:, 0], 
                                self.pca_data_3d[:, 1], 
                                self.pca_data_3d[:, 2],
                                c=self.crime_agg['CLUSTER'], 
                                cmap='viridis', s=300, alpha=0.7, 
                                edgecolors='black', linewidth=2)
        
        # Annotate districts in 3D
        for idx, row in self.crime_agg.iterrows():
            ax2.text(self.pca_data_3d[idx, 0], 
                    self.pca_data_3d[idx, 1], 
                    self.pca_data_3d[idx, 2],
                    row['DISTRICT'], 
                    fontsize=8, ha='center', fontweight='bold')
        
        ax2.set_xlabel(f'PC1 ({self.pca_3d.explained_variance_ratio_[0]:.1%})', 
                      fontsize=10)
        ax2.set_ylabel(f'PC2 ({self.pca_3d.explained_variance_ratio_[1]:.1%})', 
                      fontsize=10)
        ax2.set_zlabel(f'PC3 ({self.pca_3d.explained_variance_ratio_[2]:.1%})', 
                      fontsize=10)
        ax2.set_title('3D PCA Projection', 
                     fontsize=13, fontweight='bold')
        
        # Add colorbar
        fig.colorbar(scatter_3d, ax=ax2, label='Cluster ID', pad=0.1, shrink=0.8)
        
        fig.suptitle('District Clustering - PCA Projections', 
                    fontsize=15, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/pca_clustering.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Dendrogram
        fig, ax = plt.subplots(figsize=(12, 6))
        dendrogram(self.linkage_matrix, 
                  labels=self.crime_agg['DISTRICT'].tolist(),
                  ax=ax, leaf_font_size=11, color_threshold=0)
        ax.set_title('Hierarchical Clustering Dendrogram (Ward Linkage)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('District', fontsize=12)
        ax.set_ylabel('Euclidean Distance', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/dendrogram.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Geographic Distribution
        geo = self.df.dropna(subset=['LOCATION_LONGITUDE', 'LOCATION_LATITUDE'])
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(geo['LOCATION_LONGITUDE'], geo['LOCATION_LATITUDE'], 
                  alpha=0.2, s=1, c='red')
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title('Geographic Distribution of Policing Incidents', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/geographic_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":

    analysis = PolicingEquityAnalysis()
    analysis.run()
