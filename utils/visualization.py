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