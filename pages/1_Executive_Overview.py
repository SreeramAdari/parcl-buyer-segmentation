import streamlit as st
from ui import load_theme, get_data, header, sidebar_filters, kpi_card
from charts import buyer_type, purpose, monthly_transactions, monthly_value, price_area, top_countries, property_mix

st.set_page_config(page_title='Executive Overview',page_icon='📊',layout='wide')
load_theme(); d=get_data(); t=sidebar_filters(d['transactions']); c=d['client_df'].copy()
header('📊 Executive Overview','The market at a glance — buyer mix, investment momentum and property demand.')

# Translate transaction filters to client IDs so persona metrics remain consistent.
ids=set(t.client_id.dropna())
fc=c[c.client_id.isin(ids)]
cols=st.columns(5)
for col,label,val in zip(cols,['Unique Buyers','Transactions','Investment Value','Avg Property Price','Loan Rate'],[f'{fc.client_id.nunique():,}',f'{len(t):,}',f'${t.sale_price.sum()/1e6:,.2f}M',f'${t.sale_price.mean():,.0f}',f'{fc.loan_binary.mean()*100:.1f}%']):
    with col: kpi_card(label,val)

l,r=st.columns(2)
with l: st.plotly_chart(buyer_type(t),use_container_width=True)
with r: st.plotly_chart(purpose(t),use_container_width=True)
l,r=st.columns(2)
with l: st.plotly_chart(monthly_transactions(t),use_container_width=True)
with r: st.plotly_chart(monthly_value(t),use_container_width=True)
st.plotly_chart(price_area(t),use_container_width=True)
l,r=st.columns(2)
with l: st.plotly_chart(top_countries(t),use_container_width=True)
with r: st.plotly_chart(property_mix(t),use_container_width=True)
