import streamlit as st
from ui import load_theme, get_data, header, kpi_card

st.set_page_config(page_title='Parcl AI Market Intelligence',page_icon='🏙️',layout='wide',initial_sidebar_state='expanded')
load_theme(); d=get_data(); m=d['metrics']
header('🏙️ PARCL AI MARKET INTELLIGENCE','Machine Learning Based Buyer Segmentation & Investment Profiling')

cols=st.columns(5)
for c,label,val in zip(cols,['Unique Buyers','Transactions','Investment Value','Best Model','Silhouette'],[f"{d['client_df'].client_id.nunique():,}",f"{len(d['transactions']):,}",f"${d['transactions'].sale_price.sum()/1e6:,.2f}M",m['best_algorithm'],f"{m['silhouette']:.3f}"]):
    with c: kpi_card(label,val)

st.markdown("<div class='section-title'>How this intelligence model works</div>",unsafe_allow_html=True)
st.markdown("<div class='insight'>Transactions are aggregated to <b>client level</b> before clustering, preventing frequent purchasers from being over-weighted. K=2 is the purely metric-optimal K by Silhouette, while K=4 is the requested business operating model; among full-coverage algorithms at K=4, <b>K-Means</b> is the winner.</div>",unsafe_allow_html=True)

c1,c2=st.columns(2)
with c1:
    st.subheader('🏆 Model decision')
    st.dataframe(d['comparison'][['algorithm','k','n_clusters','silhouette','davies_bouldin','calinski_harabasz','noise_share','eligible_for_final','composite_rank','final_rank']].round(4),use_container_width=True,hide_index=True)
with c2:
    st.subheader('🎯 Buyer segments')
    st.dataframe(d['profile'].round(2),use_container_width=True)

with st.sidebar:
    st.markdown('### Parcl AI')
    st.caption('Five analytical views are available in the navigation panel.')
    st.divider()
    st.info(f"**Winner:** {m['best_algorithm']}\n\n**Operating K:** {m['operating_k']}\n\n**Metric-optimal K:** {m['metric_best_k']}\n\n**Silhouette:** {m['silhouette']:.3f}")
