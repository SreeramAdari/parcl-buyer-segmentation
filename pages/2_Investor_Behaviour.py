import streamlit as st
from ui import load_theme, get_data, header, sidebar_filters, kpi_card
from charts import referral, loan_purpose, property_mix, price_area, loan_by_persona, company_by_persona

st.set_page_config(page_title='Investor Behaviour',page_icon='💰',layout='wide')
load_theme(); d=get_data(); t=sidebar_filters(d['transactions']); ids=set(t.client_id.dropna()); c=d['client_df'][d['client_df'].client_id.isin(ids)].copy()
header('💰 Investor Behaviour','Understand financing, property choice and acquisition channels across the buyer base.')

cols=st.columns(4)
metrics=[('Investment Purpose',f'{c.investment_binary.mean()*100:.1f}%'),('Repeat Buyers',f'{c.repeat_buyer.mean()*100:.1f}%'),('Corporate Share',f'{c.company_binary.mean()*100:.1f}%'),('Avg Investment',f'${c.avg_investment.mean():,.0f}')]
for col,(label,val) in zip(cols,metrics):
    with col: kpi_card(label,val)

l,r=st.columns(2)
with l: st.plotly_chart(loan_purpose(t),use_container_width=True)
with r: st.plotly_chart(referral(t),use_container_width=True)
st.plotly_chart(price_area(t),use_container_width=True)
l,r=st.columns(2)
with l: st.plotly_chart(loan_by_persona(c),use_container_width=True)
with r: st.plotly_chart(company_by_persona(c),use_container_width=True)
st.plotly_chart(property_mix(t),use_container_width=True)

st.subheader('Buyer behavior by financing')
summary=c.groupby('loan_applied').agg(Buyers=('client_id','count'),AvgInvestment=('avg_investment','mean'),AvgAge=('age','mean'),Satisfaction=('satisfaction_score','mean')).reset_index()
st.dataframe(summary.round(2),use_container_width=True,hide_index=True)
