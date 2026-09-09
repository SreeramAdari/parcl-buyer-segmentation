import streamlit as st
from ui import load_theme, get_data, header, sidebar_filters
from charts import world_map, country_bubble, region_heatmap, satisfaction_by_region

st.set_page_config(page_title='Geographic Intelligence',page_icon='🌍',layout='wide')
load_theme(); d=get_data(); t=sidebar_filters(d['transactions'])
header('🌍 Geographic Buyer Intelligence','Locate markets with the strongest buyer volume, investment value and satisfaction.')

st.plotly_chart(world_map(t),use_container_width=True)
l,r=st.columns(2)
with l: st.plotly_chart(country_bubble(t),use_container_width=True)
with r: st.plotly_chart(satisfaction_by_region(t),use_container_width=True)
st.plotly_chart(region_heatmap(t),use_container_width=True)

summary=t.groupby('country').agg(Buyers=('client_id','nunique'),Transactions=('listing_id','count'),TotalInvestment=('sale_price','sum'),AvgInvestment=('sale_price','mean'),Satisfaction=('satisfaction_score','mean')).sort_values('TotalInvestment',ascending=False).reset_index()
st.subheader('Country leaderboard')
st.dataframe(summary.head(15).round(2),use_container_width=True,hide_index=True)
