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