import streamlit as st
from ui import load_theme, get_data, header
from charts import inertia_curve, k_curve_plot, model_comparison

st.set_page_config(page_title='ML Model Insights',page_icon='📈',layout='wide')
load_theme(); d=get_data(); m=d['metrics']; header('📈 ML Model Insights','Transparent model selection using clustering quality metrics and business coverage constraints.')

c1,c2,c3,c4,c5=st.columns(5)
for col,label,val in zip([c1,c2,c3,c4,c5],['Winner','Operating K','Metric-optimal K','Silhouette','DB Index'],[m['best_algorithm'],m['operating_k'],m['metric_best_k'],f"{m['silhouette']:.3f}",f"{m['davies_bouldin']:.3f}"]):
    with col: col.metric(label,val)

l,r=st.columns(2)
with l: st.plotly_chart(inertia_curve(d['X']),use_container_width=True)
with r: st.plotly_chart(k_curve_plot(d['k_curve']),use_container_width=True)
st.plotly_chart(model_comparison(d['comparison']),use_container_width=True)

st.subheader('Model comparison')
st.dataframe(d['comparison'][['algorithm','k','n_clusters','silhouette','davies_bouldin','calinski_harabasz','noise_share','eligible_for_final','composite_rank','final_rank']].round(4),use_container_width=True,hide_index=True)

st.subheader('Why K-Means is the operating model')
st.markdown("""
<div class='insight'><b>Step 1 — Metric-optimal K:</b> Silhouette reaches its maximum at K=2 on this dataset.</div>
<div class='insight'><b>Step 2 — Business requirement:</b> the requested solution needs four actionable buyer personas, so K=4 is used for the operating model.</div>
<div class='insight'><b>Step 3 — Algorithm selection at K=4:</b> K-Means, Agglomerative and GMM are compared on Silhouette, Davies-Bouldin and Calinski-Harabasz. DBSCAN is retained as an outlier benchmark but is not eligible as the final segmentation when it labels more than 10% of buyers as noise.</div>
<div class='insight'><b>Result:</b> K-Means is the best full-coverage K=4 model on the supplied data and produces four stable, easy-to-explain buyer segments for the dashboard.</div>
""",unsafe_allow_html=True)
