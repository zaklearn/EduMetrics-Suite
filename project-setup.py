import os
import sys

def create_directory_structure():
    # Création de la structure des dossiers
    directories = [
        'utils',
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def write_file(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def create_project_files():
    # Contenu des fichiers
    files = {
        'config.py': '''
import streamlit as st

SUPPORTED_LANGUAGES = {
    "fr": "Français",
    "en": "English",
    "nl": "Nederlands"
}

THEMES = {
    "light": {
        "primary": "#1f77b4",
        "secondary": "#ff7f0e",
        "background": "#ffffff"
    },
    "dark": {
        "primary": "#2d3035",
        "secondary": "#3498db",
        "background": "#1e1e1e"
    }
}

CACHE_TTL = 3600  # 1 heure
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB
''',
        
        'utils/data_validation.py': '''
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import streamlit as st

@st.cache_data
def validate_input_file(file) -> Tuple[bool, str, Dict[str, Any]]:
    try:
        if file.size > MAX_FILE_SIZE:
            return False, "Fichier trop volumineux", {}
        
        df = pd.read_excel(file)
        metadata = {
            "rows": len(df),
            "columns": len(df.columns),
            "numeric_cols": df.select_dtypes(include=[np.number]).columns.tolist(),
            "date_cols": df.select_dtypes(include=['datetime64']).columns.tolist(),
            "categorical_cols": df.select_dtypes(include=['object']).columns.tolist()
        }
        return True, "Fichier valide", metadata
    except Exception as e:
        return False, f"Erreur de validation: {str(e)}", {}
''',

        'utils/statistics.py': '''
from scipy import stats
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, Any

@st.cache_data
def perform_statistical_tests(df: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, Any]:
    results = {}
    
    # T-test
    groups = df[group_col].unique()
    if len(groups) == 2:
        group1 = df[df[group_col] == groups[0]][value_col]
        group2 = df[df[group_col] == groups[1]][value_col]
        t_stat, p_val = stats.ttest_ind(group1, group2)
        results['t_test'] = {'statistic': t_stat, 'p_value': p_val}
    
    # ANOVA
    if len(groups) > 2:
        groups_data = [group[value_col].values for _, group in df.groupby(group_col)]
        f_stat, p_val = stats.f_oneway(*groups_data)
        results['anova'] = {'statistic': f_stat, 'p_value': p_val}
    
    # Chi-square
    if df[value_col].dtype == 'category':
        observed = pd.crosstab(df[group_col], df[value_col])
        chi2, p_val, dof, expected = stats.chi2_contingency(observed)
        results['chi_square'] = {'statistic': chi2, 'p_value': p_val}
    
    return results
''',

        'utils/visualization.py': '''
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import streamlit as st

@st.cache_data
def create_interactive_visualization(df: pd.DataFrame, plot_type: str, x_col: str, y_col: str, 
                                  color_col: str = None, title: str = "") -> go.Figure:
    if plot_type == "scatter":
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                        title=title, trendline="ols")
    elif plot_type == "box":
        fig = px.box(df, x=x_col, y=y_col, color=color_col,
                    title=title)
    elif plot_type == "histogram":
        fig = px.histogram(df, x=x_col, color=color_col,
                         title=title, marginal="box")
    elif plot_type == "3d_scatter":
        fig = px.scatter_3d(df, x=x_col, y=y_col, z=color_col,
                           title=title)
    
    fig.update_layout(
        template="plotly_dark" if st.session_state.get('theme') == 'dark' else "plotly_white",
        hovermode='closest'
    )
    
    return fig

@st.cache_data
def plot_outliers(df: pd.DataFrame, column: str) -> go.Figure:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))][column]
    
    fig = go.Figure()
    fig.add_trace(go.Box(y=df[column], name=column))
    fig.add_trace(go.Scatter(y=outliers, mode='markers', name='Outliers',
                            marker=dict(color='red', size=10, symbol='x')))
    
    return fig
''',

        'utils/clustering.py': '''
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import streamlit as st

class ClusteringAnalysis:
    @st.cache_data
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
    
    @st.cache_data
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
''',

        'utils/prediction.py': '''
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pandas as pd
import numpy as np
import streamlit as st

class PredictionModel:
    @st.cache_data
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.models = {
            'regression': {
                'linear': LinearRegression(),
                'random_forest': RandomForestRegressor(random_state=42)
            },
            'classification': {
                'logistic': LogisticRegression(random_state=42),
                'random_forest': RandomForestClassifier(random_state=42)
            }
        }
    
    def prepare_data(self, target_col: str, feature_cols: list):
        X = self.df[feature_cols]
        y = self.df[target_col]
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train_model(self, model_type: str, model_name: str, X_train, y_train):
        model = self.models[model_type][model_name]
        model.fit(X_train, y_train)
        return model
    
    def get_predictions(self, model, X_test):
        return model.predict(X_test)
''',

        'utils/reporting.py': '''
import pandas as pd
from fpdf import FPDF
import base64
from io import BytesIO
import json
import streamlit as st

class Report:
    def __init__(self, df: pd.DataFrame, analysis_results: dict, figures: list):
        self.df = df
        self.results = analysis_results
        self.figures = figures
        
    def to_pdf(self) -> bytes:
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
        pdf.set_font('DejaVu', '', 12)
        
        pdf.cell(200, 10, txt="Rapport d'Analyse Statistique", ln=True, align='C')
        pdf.ln(10)
        
        if 'descriptive_stats' in self.results:
            pdf.cell(200, 10, txt="Statistiques Descriptives", ln=True)
            stats_text = str(self.results['descriptive_stats'])
            pdf.multi_cell(0, 10, txt=stats_text)
        
        for fig in self.figures:
            img_bytes = BytesIO()
            fig.write_image(img_bytes, format='png')
            img_bytes.seek(0)
            pdf.image(img_bytes, x=10, w=190)
            pdf.ln(10)
        
        return pdf.output(dest='S').encode('latin-1', 'ignore')
    
    def to_excel(self) -> BytesIO:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            self.df.to_excel(writer, sheet_name='Données', index=False)
            pd.DataFrame(self.results).to_excel(writer, sheet_name='Résultats')
        output.seek(0)
        return output
    
    def to_json(self) -> str:
        report_dict = {
            'statistics': self.results,
            'data_summary': self.df.describe().to_dict()
        }
        return json.dumps(report_dict, ensure_ascii=False, indent=2)
''',

        'requirements.txt': '''
streamlit==1.31.1
pandas==2.2.0
numpy==1.26.3
scipy==1.12.0
scikit-learn==1.4.0
plotly==5.18.0
fpdf==1.7.2
openpyxl==3.1.2
'''
    }
    
    # Écriture des fichiers
    for file_path, content in files.items():
        full_path = os.path.join(os.getcwd(), file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        write_file(full_path, content.strip())

def main():
    print("Création de la structure du projet...")
    create_directory_structure()
    print("Création des fichiers...")
    create_project_files()
    print("""
    Projet créé avec succès !
    
    Pour installer les dépendances, exécutez :
    pip install -r requirements.txt
    
    Pour lancer l'application :
    streamlit run main.py
    """)

if __name__ == "__main__":
    main()
