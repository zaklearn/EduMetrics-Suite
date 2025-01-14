import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import streamlit as st
from config import MAX_FILE_SIZE  # Ajout de cet import

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