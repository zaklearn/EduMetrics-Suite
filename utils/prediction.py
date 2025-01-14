from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pandas as pd
import numpy as np
import streamlit as st

class PredictionModel:
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