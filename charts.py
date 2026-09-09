import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

TEMPLATE='plotly_dark'
FONT={'color':'#E5E7EB','family':'Inter, Segoe UI, sans-serif'}
GRID='rgba(148,163,184,.10)'

def base(fig):
    fig.update_layout(template=TEMPLATE,font=FONT,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',margin=dict(l=30,r=20,t=60,b=35),hoverlabel=dict(bgcolor='#111827'))
    fig.update_xaxes(showgrid=True,gridcolor=GRID,zeroline=False)
    fig.update_yaxes(showgrid=True,gridcolor=GRID,zeroline=False)
    return fig

def buyer_type(df): return base(px.pie(df,names='client_type',hole=.62,title='Buyer Mix'))
def purpose(df): return base(px.pie(df,names='acquisition_purpose',hole=.62,title='Acquisition Purpose'))
def monthly_transactions(df):
    g=df.groupby('year_month').size().reset_index(name='Transactions').sort_values('year_month')
    return base(px.area(g,x='year_month',y='Transactions',markers=True,title='Transaction Momentum'))
def monthly_value(df):
    g=df.groupby('year_month')['sale_price'].sum().reset_index().sort_values('year_month')
    return base(px.area(g,x='year_month',y='sale_price',markers=True,title='Monthly Investment Value'))
def price_area(df): return base(px.scatter(df,x='floor_area_sqft',y='sale_price',color='BuyerPersona' if 'BuyerPersona' in df else 'acquisition_purpose',size='satisfaction_score',hover_data=['country','region','client_type'],title='Investment Value vs Property Size',opacity=.75))
def top_countries(df,n=10):
    g=df.groupby('country').agg(Buyers=('client_id','count'),Investment=('sale_price','sum'),AvgPrice=('sale_price','mean')).reset_index().sort_values('Buyers',ascending=False).head(n)
    return base(px.bar(g,y='country',x='Buyers',orientation='h',color='AvgPrice',color_continuous_scale='Blues',title='Top Buyer Markets'))
def property_mix(df):
    g=df.groupby('unit_category').agg(Transactions=('listing_id','count'),Investment=('sale_price','sum')).reset_index()
    return base(px.bar(g,x='unit_category',y='Transactions',color='Investment',color_continuous_scale='Purples',title='Property Category Mix'))
def referral(df):
    g=df.groupby('referral_channel').agg(Buyers=('client_id','count'),AvgPrice=('sale_price','mean')).reset_index().sort_values('Buyers',ascending=False)
    return base(px.bar(g,y='referral_channel',x='Buyers',orientation='h',color='AvgPrice',color_continuous_scale='Teal',title='Acquisition Channels'))
def loan_purpose(df): return base(px.histogram(df,x='acquisition_purpose',color='loan_applied',barmode='group',title='Financing by Acquisition Purpose'))
def satisfaction_by_region(df):
    g=df.groupby('region').satisfaction_score.mean().reset_index().sort_values('satisfaction_score',ascending=False).head(15)
    return base(px.bar(g,x='satisfaction_score',y='region',orientation='h',color='satisfaction_score',color_continuous_scale='Viridis',title='Top Regions by Satisfaction'))
def world_map(df):
    g=df.groupby('country').agg(Buyers=('client_id','count'),Investment=('sale_price','sum'),AvgInvestment=('sale_price','mean')).reset_index()
    return base(px.choropleth(g,locations='country',locationmode='country names',color='Buyers',hover_data=['Investment','AvgInvestment'],color_continuous_scale='Blues',title='Global Buyer Distribution'))
def country_bubble(df):
    g=df.groupby('country').agg(Buyers=('client_id','count'),AvgInvestment=('sale_price','mean'),Satisfaction=('satisfaction_score','mean')).reset_index()
    return base(px.scatter(g,x='Buyers',y='AvgInvestment',size='Buyers',color='Satisfaction',hover_name='country',color_continuous_scale='Viridis',title='Market Opportunity: Volume vs Value'))
def region_heatmap(df):
    g=df.pivot_table(index='region',columns='acquisition_purpose',values='sale_price',aggfunc='mean')
    return base(px.imshow(g,text_auto='.0f',aspect='auto',color_continuous_scale='Blues',title='Average Investment by Region & Purpose'))
def cluster_distribution(df): return base(px.pie(df,names='BuyerPersona',hole=.62,title='AI Buyer Segment Mix'))
def pca_scatter(pca_df): return base(px.scatter(pca_df,x='PC1',y='PC2',color='BuyerPersona',hover_data=['client_id'],title='Buyer Segments in PCA Space'))
def cluster_profile_heatmap(profile):
    x=profile.copy()
    for c in x.columns:
        if x[c].nunique()>1: x[c]=(x[c]-x[c].min())/(x[c].max()-x[c].min())
    return base(px.imshow(x.T,text_auto='.2f',aspect='auto',color_continuous_scale='Viridis',title='Normalized Segment Profile'))
def cluster_price(df): return base(px.box(df,x='BuyerPersona',y='avg_investment',points=False,title='Investment Distribution by Buyer Persona'))
def cluster_age(df): return base(px.violin(df,x='BuyerPersona',y='age',box=True,title='Age Profile by Buyer Persona'))
def model_comparison(tab):
    g=tab.melt(id_vars='algorithm',value_vars=['silhouette','davies_bouldin'],var_name='metric',value_name='score')
    return base(px.bar(g,x='algorithm',y='score',color='metric',barmode='group',title='Model Quality Comparison'))
def k_curve_plot(tab):
    return base(px.line(tab,x='k',y='silhouette',markers=True,title='Silhouette Score by K'))
def inertia_curve(X):
    from sklearn.cluster import KMeans
    rows=[]
    for k in range(2,9): rows.append({'k':k,'inertia':KMeans(n_clusters=k,random_state=42,n_init=30).fit(X).inertia_})
    return base(px.line(pd.DataFrame(rows),x='k',y='inertia',markers=True,title='Elbow Curve (Inertia)'))
def radar(profile):
    cols=[c for c in ['avg_age','avg_investment','avg_area','loan_rate','company_rate','investment_rate','satisfaction'] if c in profile.columns]
    z=profile[cols].copy()
    for c in cols:
        if z[c].nunique()>1: z[c]=(z[c]-z[c].min())/(z[c].max()-z[c].min())
    fig=go.Figure()
    for idx,row in z.iterrows(): fig.add_trace(go.Scatterpolar(r=row.values,theta=cols,fill='toself',name=f'Cluster {idx}'))
    fig.update_layout(template=TEMPLATE,title='Buyer Persona Radar',paper_bgcolor='rgba(0,0,0,0)',font=FONT,polar=dict(bgcolor='rgba(0,0,0,0)',radialaxis=dict(visible=True,gridcolor=GRID)))
    return fig

def loan_by_persona(client_df):
    g=client_df.groupby('BuyerPersona').loan_binary.mean().mul(100).reset_index(name='LoanRate')
    return base(px.bar(g,x='BuyerPersona',y='LoanRate',title='Loan Dependency by Buyer Persona',text_auto='.1f'))

def company_by_persona(client_df):
    g=client_df.groupby('BuyerPersona').company_binary.mean().mul(100).reset_index(name='CompanyRate')
    return base(px.bar(g,x='BuyerPersona',y='CompanyRate',title='Corporate Buyer Share by Persona',text_auto='.1f'))
