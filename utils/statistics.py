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