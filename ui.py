from pathlib import Path
import streamlit as st
from utils import run_pipeline

ROOT=Path(__file__).resolve().parent

@st.cache_resource(show_spinner='Building Parcl buyer intelligence…')
def get_data(): return run_pipeline()

def load_theme():
    with open(ROOT/'style.css',encoding='utf-8') as f: st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)

def sidebar_filters(transactions):
    with st.sidebar:
        st.markdown('## 🔎 Market Filters')
        countries=st.multiselect('Country',sorted(transactions.country.dropna().unique()),default=sorted(transactions.country.dropna().unique()))
        regions=st.multiselect('Region',sorted(transactions.region.dropna().unique()),default=sorted(transactions.region.dropna().unique()))
        purposes=st.multiselect('Acquisition Purpose',sorted(transactions.acquisition_purpose.dropna().unique()),default=sorted(transactions.acquisition_purpose.dropna().unique()))
        buyer_types=st.multiselect('Client Type',sorted(transactions.client_type.dropna().unique()),default=sorted(transactions.client_type.dropna().unique()))
    f=transactions[transactions.country.isin(countries)&transactions.region.isin(regions)&transactions.acquisition_purpose.isin(purposes)&transactions.client_type.isin(buyer_types)].copy()
    return f

def header(title,subtitle):
    st.markdown(f"<div class='hero'><h1>{title}</h1><p>{subtitle}</p></div>",unsafe_allow_html=True)

def kpi_card(label,value,delta=None):
    st.markdown(f"<div class='card'><div class='small-label'>{label}</div><div class='big-number'>{value}</div>{f'<div class=muted>{delta}</div>' if delta else ''}</div>",unsafe_allow_html=True)
