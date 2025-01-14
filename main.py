import streamlit as st
import pandas as pd
import numpy as np
from utils.data_validation import validate_input_file
from utils.statistics import perform_statistical_tests
from utils.visualization import create_interactive_visualization, plot_outliers
from utils.clustering import ClusteringAnalysis
from utils.prediction import PredictionModel
from utils.reporting import Report
from config import SUPPORTED_LANGUAGES, THEMES, CACHE_TTL, MAX_FILE_SIZE
import plotly.graph_objects as go

# Dictionnaire de traduction
TRANSLATIONS = {
    'fr': {
        'upload': "Télécharger un fichier",
        'analysis': "Analyses",
        'visualization': "Visualisations",
        'clustering': "Segmentation",
        'prediction': "Prédictions",
        'export': "Exporter",
        'settings': "Paramètres"
    },
    'en': {
        'upload': "Upload file",
        'analysis': "Analysis",
        'visualization': "Visualizations",
        'clustering': "Clustering",
        'prediction': "Predictions",
        'export': "Export",
        'settings': "Settings"
    }
}

def get_translation(key: str, lang: str = 'fr') -> str:
    """Récupère la traduction d'une clé pour une langue donnée"""
    return TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)

# Configuration initiale de la session
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'language' not in st.session_state:
    st.session_state.language = 'fr'

def main():
    st.set_page_config(
        page_title="Analyse de Données Avancée",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar - Configuration
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Sélection de la langue
        lang = st.selectbox("Langue / Language", 
                           options=['fr', 'en'],
                           format_func=lambda x: SUPPORTED_LANGUAGES[x])
        st.session_state.language = lang
        
        # Sélection du thème
        theme = st.selectbox("Thème",
                           options=['light', 'dark'])
        st.session_state.theme = theme
        
        # Upload de fichier
        uploaded_file = st.file_uploader(
            get_translation('upload', lang),
            type=['xlsx', 'xls', 'csv']
        )

    if uploaded_file is not None:
        # Validation et chargement des données
        with st.spinner("Validation des données..."):
            is_valid, message, metadata = validate_input_file(uploaded_file)
            
        if not is_valid:
            st.error(message)
            return
            
        # Affichage des données
        st.subheader("📋 Aperçu des données")
        df = pd.read_excel(uploaded_file)
        st.dataframe(df.head())
        
        # Menu principal
        tab1, tab2, tab3, tab4 = st.tabs([
            get_translation('analysis', lang),
            get_translation('visualization', lang),
            get_translation('clustering', lang),
            get_translation('prediction', lang)
        ])
        
        # Onglet Analyses
        with tab1:
            st.subheader("📊 Analyses Statistiques")
            
            col1, col2 = st.columns(2)
            with col1:
                analysis_type = st.selectbox(
                    "Type d'analyse",
                    ["Descriptive", "Tests Statistiques", "Corrélations"]
                )
                
            if analysis_type == "Descriptive":
                st.write(df.describe())
                
            elif analysis_type == "Tests Statistiques":
                group_col = st.selectbox("Variable de groupement", metadata['categorical_cols'])
                value_col = st.selectbox("Variable à analyser", metadata['numeric_cols'])
                
                results = perform_statistical_tests(df, group_col, value_col)
                st.write(results)
                
            elif analysis_type == "Corrélations":
                corr_matrix = df[metadata['numeric_cols']].corr()
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns
                ))
                st.plotly_chart(fig)
        
        # Onglet Visualisations
        with tab2:
            st.subheader("📈 Visualisations")
            
            plot_type = st.selectbox(
                "Type de graphique",
                ["scatter", "box", "histogram", "3d_scatter"]
            )
            
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Variable X", df.columns)
                y_col = st.selectbox("Variable Y", df.columns)
            with col2:
                color_col = st.selectbox("Variable de couleur (optionnel)", 
                                       ["Aucune"] + list(df.columns))
            
            fig = create_interactive_visualization(
                df, plot_type, x_col, y_col,
                color_col if color_col != "Aucune" else None
            )
            st.plotly_chart(fig)

        # Onglet Clustering
        with tab3:
            st.subheader("🔍 Segmentation des Données")
            
            clustering_vars = st.multiselect(
                "Sélectionnez les variables pour la segmentation",
                metadata['numeric_cols']
            )
            
            if clustering_vars:
                clustering = ClusteringAnalysis(df[clustering_vars], clustering_vars)
                
                method = st.selectbox(
                    "Méthode de segmentation",
                    ["K-Means", "DBSCAN", "Hiérarchique"]
                )
                
                if method == "K-Means":
                    n_clusters = st.slider("Nombre de segments", 2, 10, 3)
                    clusters = clustering.perform_kmeans(n_clusters)
                elif method == "DBSCAN":
                    eps = st.slider("Distance maximale (eps)", 0.1, 2.0, 0.5)
                    min_samples = st.slider("Échantillons minimum", 2, 10, 5)
                    clusters = clustering.perform_dbscan(eps, min_samples)
                else:
                    n_clusters = st.slider("Nombre de segments", 2, 10, 3)
                    clusters = clustering.perform_hierarchical(n_clusters)
                
                df['Segment'] = clusters
                stats = clustering.get_cluster_statistics(clusters)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Statistiques par segment :")
                    st.dataframe(stats)
                
                with col2:
                    if len(clustering_vars) >= 2:
                        fig = create_interactive_visualization(
                            df, "scatter",
                            clustering_vars[0], clustering_vars[1],
                            'Segment', "Visualisation des segments"
                        )
                        st.plotly_chart(fig)
        
        # Onglet Prédictions
        with tab4:
            st.subheader("🎯 Modèles Prédictifs")
            
            target_col = st.selectbox(
                "Variable à prédire",
                df.columns
            )
            
            feature_cols = st.multiselect(
                "Variables explicatives",
                [col for col in df.columns if col != target_col]
            )
            
            if target_col and feature_cols:
                predictor = PredictionModel(df)
                
                is_classification = df[target_col].dtype == 'object' or len(df[target_col].unique()) < 10
                problem_type = 'classification' if is_classification else 'regression'
                
                model_name = st.selectbox(
                    "Algorithme",
                    ['random_forest', 'linear' if problem_type == 'regression' else 'logistic']
                )
                
                X_train, X_test, y_train, y_test = predictor.prepare_data(target_col, feature_cols)
                model = predictor.train_model(problem_type, model_name, X_train, y_train)
                predictions = predictor.get_predictions(model, X_test)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Prédictions vs Réalité")
                    results_df = pd.DataFrame({
                        'Réel': y_test,
                        'Prédit': predictions
                    })
                    st.dataframe(results_df)
                
                with col2:
                    if problem_type == 'regression':
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=y_test, y=predictions,
                            mode='markers',
                            name='Prédictions'
                        ))
                        fig.add_trace(go.Scatter(
                            x=[y_test.min(), y_test.max()],
                            y=[y_test.min(), y_test.max()],
                            mode='lines',
                            name='Ligne parfaite'
                        ))
                        st.plotly_chart(fig)

        # Export des résultats
        st.sidebar.subheader("📥 Export")
        export_format = st.sidebar.selectbox(
            "Format d'export",
            ["PDF", "Excel", "JSON"]
        )
        
        if st.sidebar.button("Exporter"):
            # Préparation des données d'export selon l'onglet actif
            export_data = {}
            export_figures = []
            
            if 'analysis_type' in locals() and analysis_type == "Descriptive":
                export_data['descriptive_stats'] = df.describe()
                
            elif 'analysis_type' in locals() and analysis_type == "Tests Statistiques":
                export_data['statistical_tests'] = results if 'results' in locals() else None
                
            elif 'analysis_type' in locals() and analysis_type == "Corrélations":
                export_data['correlation_matrix'] = corr_matrix if 'corr_matrix' in locals() else None
                export_figures.append(fig) if 'fig' in locals() else None
                
            elif 'plot_type' in locals():  # Onglet Visualisations
                export_data['visualization_type'] = plot_type
                export_figures.append(fig) if 'fig' in locals() else None
                
            elif 'clustering_vars' in locals() and clustering_vars:  # Onglet Clustering
                export_data['clustering_stats'] = stats if 'stats' in locals() else None
                export_figures.append(fig) if 'fig' in locals() else None
                
            elif 'target_col' in locals() and target_col:  # Onglet Prédictions
                if 'results_df' in locals():
                    export_data['prediction_results'] = results_df
                    export_figures.append(fig) if 'fig' in locals() and problem_type == 'regression' else None

            # Export selon le format choisi
            try:
                report = Report(df, export_data, export_figures)
                
                if export_format == "PDF":
                    pdf_bytes = report.to_pdf()
                    if pdf_bytes:
                        st.sidebar.download_button(
                            "Télécharger PDF",
                            pdf_bytes,
                            "rapport.pdf",
                            "application/pdf"
                        )
                        
                elif export_format == "Excel":
                    excel_buffer = report.to_excel()
                    st.sidebar.download_button(
                        "Télécharger Excel",
                        excel_buffer,
                        "rapport.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                else:  # JSON
                    json_str = report.to_json()
                    st.sidebar.download_button(
                        "Télécharger JSON",
                        json_str,
                        "rapport.json",
                        "application/json"
                    )
                    
            except Exception as e:
                st.sidebar.error(f"Erreur lors de l'export : {str(e)}")

    else:
        st.info("Veuillez télécharger un fichier Excel pour commencer.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Une erreur s'est produite : {str(e)}")
