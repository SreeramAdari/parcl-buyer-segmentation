import streamlit as st
from ui import load_theme, get_data, header, sidebar_filters
from charts import cluster_distribution, pca_scatter, cluster_profile_heatmap, cluster_price, cluster_age, radar

st.set_page_config(page_title='AI Buyer Segments',page_icon='🤖',layout='wide')
load_theme(); d=get_data(); t=sidebar_filters(d['transactions']); ids=set(t.client_id.dropna()); c=d['client_df'][d['client_df'].client_id.isin(ids)].copy()
header('🤖 AI Buyer Segments','Four actionable buyer personas discovered from demographics, financing and transaction behavior.')

persona=st.selectbox('Explore buyer persona',['All']+sorted(c.BuyerPersona.unique().tolist()))
view=c if persona=='All' else c[c.BuyerPersona==persona]

l,r=st.columns(2)
with l: st.plotly_chart(cluster_distribution(c),use_container_width=True)
with r:
    p=d['pca_df'].copy(); p=p[p.client_id.isin(ids)]
    st.plotly_chart(pca_scatter(p),use_container_width=True)
st.plotly_chart(cluster_profile_heatmap(d['profile']),use_container_width=True)
l,r=st.columns(2)
with l: st.plotly_chart(cluster_price(c),use_container_width=True)
with r: st.plotly_chart(cluster_age(c),use_container_width=True)
st.plotly_chart(radar(d['profile']),use_container_width=True)

st.subheader('Persona explorer')
show=['client_id','client_type','country','region','acquisition_purpose','loan_applied','age','transaction_count','avg_investment','total_investment','BuyerPersona']
st.dataframe(view[show].sort_values('avg_investment',ascending=False),use_container_width=True,hide_index=True)
st.download_button('⬇️ Download selected segment',view.to_csv(index=False).encode(), 'selected_buyer_segment.csv','text/csv')

st.subheader('Persona playbook')
strategies={
'Luxury Investors':'Premium launches, concierge service, exclusive inventory and high-touch advisory.',
'First-Time Buyers':'EMI calculators, financing partnerships, affordable inventory and guided education.',
'Corporate Buyers':'Dedicated account managers, commercial bundles and portfolio pricing.',
'Global Investors':'Cross-border campaigns, ROI reports, international market updates and multilingual support.'}
for name,g in c.groupby('BuyerPersona'):
    st.markdown(f"<div class='persona'><h3>{name}</h3><p><b>{len(g):,}</b> buyers &nbsp; | &nbsp; Avg investment <b>${g.avg_investment.mean():,.0f}</b> &nbsp; | &nbsp; Avg age <b>{g.age.mean():.1f}</b></p><p>{strategies.get(name,'Use the segment profile to build targeted messaging.')}</p></div>",unsafe_allow_html=True)
