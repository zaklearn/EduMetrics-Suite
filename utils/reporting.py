import pandas as pd
from fpdf import FPDF
import base64
from io import BytesIO
import json
import streamlit as st
from typing import Dict, List, Optional, Union
import plotly.graph_objects as go

class Report:
    def __init__(self, df: pd.DataFrame, analysis_results: Dict, figures: List[go.Figure]):
        """
        Initialise un objet Report pour la génération de rapports.
        
        Args:
            df (pd.DataFrame): DataFrame contenant les données
            analysis_results (Dict): Résultats des analyses
            figures (List[go.Figure]): Liste des figures à inclure
        """
        self.df = df
        self.results = analysis_results
        self.figures = figures

    def _add_section_to_pdf(self, pdf: FPDF, title: str, content: str) -> None:
        """
        Ajoute une section au PDF avec un titre et un contenu.
        
        Args:
            pdf (FPDF): Objet PDF
            title (str): Titre de la section
            content (str): Contenu à ajouter
        """
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(200, 10, txt=title, ln=True)
        pdf.set_font('Arial', '', 12)
        
        # Gestion des caractères spéciaux et multilignes
        for line in content.split('\n'):
            try:
                # Encodage sécurisé pour les caractères spéciaux
                safe_line = line.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 10, txt=safe_line)
            except Exception as e:
                pdf.multi_cell(0, 10, txt=f"Erreur d'encodage: {str(e)}")

    def _add_figures_to_pdf(self, pdf: FPDF) -> None:
        """
        Ajoute les figures au PDF.
        
        Args:
            pdf (FPDF): Objet PDF
        """
        if not self.figures:
            return

        try:
            import kaleido
        except ImportError:
            pdf.ln(10)
            pdf.cell(0, 10, txt="Package 'kaleido' manquant pour l'export des figures", ln=True)
            return

        pdf.ln(10)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(200, 10, txt="Visualisations", ln=True)
        
        for i, fig in enumerate(self.figures, 1):
            try:
                img_bytes = BytesIO()
                fig.write_image(img_bytes, format='png', engine='kaleido')
                img_bytes.seek(0)
                
                # Ajout d'un titre pour la figure
                pdf.ln(5)
                pdf.set_font('Arial', '', 12)
                pdf.cell(0, 10, txt=f"Figure {i}", ln=True, align='C')
                
                # Calcul des dimensions pour centrer l'image
                page_width = pdf.w - 2 * 10  # Marges de 10mm
                img_width = min(190, page_width)
                pdf.image(img_bytes, x=(pdf.w - img_width) / 2, w=img_width)
                pdf.ln(10)
            
            except Exception as e:
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 10, txt=f"Erreur lors de l'ajout de la figure {i}: {str(e)}", ln=True)

    def to_pdf(self) -> Optional[bytes]:
        """
        Génère un rapport PDF.
        
        Returns:
            Optional[bytes]: Contenu du PDF en bytes ou None en cas d'erreur
        """
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            
            # Titre principal
            pdf.cell(200, 10, txt="Rapport d'Analyse", ln=True, align='C')
            pdf.ln(10)
            
            # Contenu selon le type de données disponibles
            if isinstance(self.results, dict):
                # Statistiques descriptives
                if 'descriptive_stats' in self.results:
                    self._add_section_to_pdf(
                        pdf, 
                        "Statistiques Descriptives",
                        str(self.results['descriptive_stats'])
                    )

                # Tests statistiques
                if 'statistical_tests' in self.results:
                    self._add_section_to_pdf(
                        pdf,
                        "Tests Statistiques",
                        str(self.results['statistical_tests'])
                    )

                # Clustering
                if 'clustering_stats' in self.results:
                    self._add_section_to_pdf(
                        pdf,
                        "Résultats du Clustering",
                        str(self.results['clustering_stats'])
                    )

                # Prédictions
                if 'prediction_results' in self.results:
                    self._add_section_to_pdf(
                        pdf,
                        "Résultats des Prédictions",
                        str(self.results['prediction_results'])
                    )

            # Ajout des figures
            self._add_figures_to_pdf(pdf)

            return pdf.output(dest='S').encode('latin-1', 'ignore')
            
        except Exception as e:
            st.error(f"Erreur lors de la génération du PDF : {str(e)}")
            return None

    def to_excel(self) -> Optional[BytesIO]:
        """
        Génère un rapport Excel.
        
        Returns:
            Optional[BytesIO]: Contenu Excel en BytesIO ou None en cas d'erreur
        """
        try:
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Onglet données
                self.df.to_excel(writer, sheet_name='Données', index=False)
                
                # Onglets résultats
                if isinstance(self.results, dict):
                    for key, value in self.results.items():
                        if isinstance(value, (pd.DataFrame, pd.Series)):
                            sheet_name = key[:31]  # Excel limite les noms à 31 caractères
                            value.to_excel(writer, sheet_name=sheet_name)
                        elif isinstance(value, dict):
                            pd.DataFrame(value).to_excel(writer, sheet_name=key[:31])
                
                # Métadonnées
                metadata = {
                    'nombre_lignes': len(self.df),
                    'nombre_colonnes': len(self.df.columns),
                    'colonnes': list(self.df.columns)
                }
                pd.DataFrame([metadata]).to_excel(writer, sheet_name='Métadonnées')
            
            output.seek(0)
            return output
            
        except Exception as e:
            st.error(f"Erreur lors de la génération Excel : {str(e)}")
            return None

    def to_json(self) -> Optional[str]:
        """
        Génère un rapport JSON.
        
        Returns:
            Optional[str]: Contenu JSON en string ou None en cas d'erreur
        """
        try:
            report_dict = {
                'metadata': {
                    'nombre_lignes': len(self.df),
                    'nombre_colonnes': len(self.df.columns),
                    'colonnes': list(self.df.columns)
                },
                'summary': self.df.describe().to_dict(),
                'analysis_results': self.results
            }
            
            # Conversion des DataFrames en dictionnaires pour JSON
            def convert_to_json_serializable(obj):
                if isinstance(obj, (pd.DataFrame, pd.Series)):
                    return obj.to_dict()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int64, np.float64)):
                    return obj.item()
                return obj

            # Parcours récursif du dictionnaire pour conversion
            def convert_dict_values(d):
                return {k: convert_to_json_serializable(v) if isinstance(v, (pd.DataFrame, pd.Series, np.ndarray))
                        else v if not isinstance(v, dict)
                        else convert_dict_values(v)
                        for k, v in d.items()}

            json_ready_dict = convert_dict_values(report_dict)
            return json.dumps(json_ready_dict, ensure_ascii=False, indent=2)
            
        except Exception as e:
            st.error(f"Erreur lors de la génération JSON : {str(e)}")
            return None