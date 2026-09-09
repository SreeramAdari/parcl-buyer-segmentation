from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

ROOT = Path(__file__).resolve().parent

def safe_mode(s, default='Unknown'):
    m = s.dropna().mode()
    return m.iloc[0] if len(m) else default

def money_to_float(s):
    return pd.to_numeric(s.astype(str).str.replace(r'[^0-9.\-]', '', regex=True), errors='coerce')

def load_dataset(clients_path=None, properties_path=None):
    clients_path = Path(clients_path or ROOT/'data'/'clients.csv')
    properties_path = Path(properties_path or ROOT/'data'/'properties.csv')
    clients = pd.read_csv(clients_path)
    props = pd.read_csv(properties_path)
    clients = clients.drop_duplicates(subset=['client_id']).copy()
    props = props.drop_duplicates().copy()

    # Handles mixed values such as 6/18/1943 and 05-11-1968.
    clients['date_of_birth'] = pd.to_datetime(clients['date_of_birth'], format='mixed', errors='coerce')
    props['transaction_date'] = pd.to_datetime(props['transaction_date'], format='mixed', errors='coerce')
    props['sale_price'] = money_to_float(props['sale_price'])
    props['floor_area_sqft'] = pd.to_numeric(props['floor_area_sqft'], errors='coerce')
    clients['satisfaction_score'] = pd.to_numeric(clients['satisfaction_score'], errors='coerce')

    for col in ['client_type','gender','country','region','acquisition_purpose','referral_channel']:
        clients[col] = clients[col].astype(str).str.strip().replace({'nan':'Unknown'})
    loan = clients['loan_applied'].astype(str).str.strip().str.upper()
    clients['loan_applied'] = loan.replace({'YES':'Yes','NO':'No','Y':'Yes','N':'No','TRUE':'Yes','FALSE':'No','NAN':'Unknown'})
    props['listing_status'] = props['listing_status'].astype(str).str.strip()
    return clients, props

def build_transaction_data(clients, props):
    tx = props.merge(clients, left_on='client_ref', right_on='client_id', how='left')
    tx['age'] = ((pd.Timestamp.today().normalize()-tx['date_of_birth']).dt.days/365.2425).round(1)
    tx.loc[~tx['age'].between(18,100), 'age'] = np.nan
    tx['price_segment'] = pd.qcut(tx['sale_price'], 4, labels=['Budget','Mid','Premium','Luxury'], duplicates='drop')
    tx['size_segment'] = pd.cut(tx['floor_area_sqft'], [0,700,1200,1800,np.inf], labels=['Compact','Medium','Large','Ultra Large'])
    tx['year'] = tx['transaction_date'].dt.year
    tx['month_name'] = tx['transaction_date'].dt.strftime('%b')
    tx['month_num'] = tx['transaction_date'].dt.month
    tx['year_month'] = tx['transaction_date'].dt.to_period('M').astype(str)
    return tx

def build_client_features(clients, tx):
    today = pd.Timestamp.today().normalize()
    c = clients.copy()
    c['age'] = ((today-c['date_of_birth']).dt.days/365.2425).round(1)
    c.loc[~c['age'].between(18,100), 'age'] = np.nan
    valid = tx.dropna(subset=['client_id']).copy()
    agg = valid.groupby('client_id').agg(
        transaction_count=('listing_id','count'), total_investment=('sale_price','sum'),
        avg_investment=('sale_price','mean'), median_investment=('sale_price','median'),
        total_area_sqft=('floor_area_sqft','sum'), avg_area_sqft=('floor_area_sqft','mean'),
        max_investment=('sale_price','max'), property_types=('unit_category','nunique'),
        first_transaction=('transaction_date','min'), last_transaction=('transaction_date','max')
    ).reset_index()
    office_share = valid.assign(is_office=(valid['unit_category'].astype(str).str.lower()=='office').astype(int)).groupby('client_id')['is_office'].mean().rename('office_share')
    investment_share = valid.assign(is_investment=(valid['acquisition_purpose'].astype(str).str.lower()=='investment').astype(int)).groupby('client_id')['is_investment'].mean().rename('investment_share')
    avg_satisfaction = valid.groupby('client_id')['satisfaction_score'].mean().rename('avg_satisfaction')
    avg_ppsf = (valid['sale_price']/valid['floor_area_sqft']).replace([np.inf,-np.inf],np.nan).groupby(valid['client_id']).mean().rename('avg_price_per_sqft')

    f = c.merge(agg,on='client_id',how='left').merge(office_share,on='client_id',how='left').merge(investment_share,on='client_id',how='left').merge(avg_satisfaction,on='client_id',how='left').merge(avg_ppsf,on='client_id',how='left')
    for col in ['transaction_count','total_investment','avg_investment','median_investment','total_area_sqft','avg_area_sqft','max_investment','property_types','office_share','investment_share','avg_satisfaction','avg_price_per_sqft']:
        if col in f.columns: f[col]=pd.to_numeric(f[col],errors='coerce')
    f['loan_binary']=(f['loan_applied']=='Yes').astype(int)
    f['company_binary']=(f['client_type'].str.lower()=='company').astype(int)
    f['home_binary']=(f['acquisition_purpose'].str.lower()=='home').astype(int)
    f['investment_binary']=(f['acquisition_purpose'].str.lower()=='investment').astype(int)
    f['repeat_buyer']=(f['transaction_count'].fillna(0)>1).astype(int)
    f['recency_days']=(today-f['last_transaction']).dt.days
    f['tenure_days']=(f['last_transaction']-f['first_transaction']).dt.days
    for col in ['total_investment','avg_investment','median_investment','total_area_sqft','max_investment','avg_price_per_sqft']:
        f['log_'+col]=np.log1p(f[col].clip(lower=0))
    for col in f.select_dtypes(include=[np.number]).columns:
        f[col]=f[col].replace([np.inf,-np.inf],np.nan)
        f[col]=f[col].fillna(f[col].median())
    for col in f.select_dtypes(include=['object']).columns:
        f[col]=f[col].fillna(safe_mode(f[col]))
    return f

def prepare_features(client_df):
    numeric=['age','satisfaction_score','transaction_count','repeat_buyer','loan_binary','company_binary','home_binary','investment_binary','office_share','investment_share','property_types','avg_area_sqft','log_total_investment','log_avg_investment','log_median_investment','log_total_area_sqft','log_max_investment','log_avg_price_per_sqft']
    categorical=['gender','country','region','acquisition_purpose','referral_channel']
    numeric=[c for c in numeric if c in client_df.columns]
    categorical=[c for c in categorical if c in client_df.columns]
    pre=ColumnTransformer([('num',StandardScaler(),numeric),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False,drop='if_binary'),categorical)])
    X=pre.fit_transform(client_df[numeric+categorical])
    return X, pre, list(pre.get_feature_names_out())

def score_labels(X, labels):
    labels=np.asarray(labels)
    mask=(labels!=-1)
    Xe=X[mask] if -1 in set(labels) else X
    ye=labels[mask] if -1 in set(labels) else labels
    if len(np.unique(ye))<2: return None
    return {'silhouette':float(silhouette_score(Xe,ye)),'davies_bouldin':float(davies_bouldin_score(Xe,ye)),'calinski_harabasz':float(calinski_harabasz_score(Xe,ye)),'n_clusters':int(len(np.unique(ye))),'noise_share':float(1-mask.mean())}

def find_best_k(X, k_min=2, k_max=8):
    rows=[]
    for k in range(k_min,min(k_max,len(X)-1)+1):
        m=KMeans(n_clusters=k,random_state=42,n_init=30); y=m.fit_predict(X); s=score_labels(X,y); s['k']=k; rows.append(s)
    tab=pd.DataFrame(rows).sort_values(['silhouette','davies_bouldin'],ascending=[False,True]).reset_index(drop=True)
    return tab,int(tab.iloc[0]['k'])

def optimize_dbscan(X):
    rows=[]
    for eps in [0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.4,2.8,3.2]:
        for ms in [8,12,16]:
            m=DBSCAN(eps=eps,min_samples=ms); y=m.fit_predict(X); s=score_labels(X,y)
            if s and s['noise_share']<0.5:
                s.update(eps=eps,min_samples=ms); rows.append(s)
    if not rows: return pd.DataFrame(),None
    tab=pd.DataFrame(rows).sort_values(['silhouette','davies_bouldin'],ascending=[False,True]).reset_index(drop=True)
    return tab,tab.iloc[0].to_dict()

def compare_algorithms(X,best_k):
    rows=[]
    specs={
        'K-Means':KMeans(n_clusters=best_k,random_state=42,n_init=30),
        'Agglomerative':AgglomerativeClustering(n_clusters=best_k,linkage='ward'),
        'Gaussian Mixture':GaussianMixture(n_components=best_k,covariance_type='full',random_state=42,n_init=5)
    }
    for name,m in specs.items():
        y=m.fit_predict(X); s=score_labels(X,y); s.update(algorithm=name,k=best_k,bic=(float(m.bic(X)) if name=='Gaussian Mixture' else np.nan)); rows.append(s)
    dbtab,db=optimize_dbscan(X)
    if db:
        rows.append({'algorithm':'DBSCAN','k':int(db['n_clusters']),'n_clusters':db['n_clusters'],'silhouette':db['silhouette'],'davies_bouldin':db['davies_bouldin'],'calinski_harabasz':db['calinski_harabasz'],'noise_share':db['noise_share'],'eps':db['eps'],'min_samples':db['min_samples'],'bic':np.nan})
    tab=pd.DataFrame(rows)
    # DBSCAN is treated as an outlier detector; a model that labels a large share as noise is not suitable for a full buyer-segmentation operating model.
    tab['eligible_for_final'] = ~((tab['algorithm']=='DBSCAN') & (tab['noise_share']>0.10))
    tab['rank_silhouette']=tab['silhouette'].rank(ascending=False,method='min')
    tab['rank_db']=tab['davies_bouldin'].rank(ascending=True,method='min')
    tab['rank_ch']=tab['calinski_harabasz'].rank(ascending=False,method='min')
    tab['composite_rank']=tab[['rank_silhouette','rank_db','rank_ch']].mean(axis=1)
    tab['final_rank'] = tab['composite_rank'].where(tab['eligible_for_final'], np.inf)
    return tab.sort_values(['final_rank','silhouette','davies_bouldin'],ascending=[True,False,True]).reset_index(drop=True),dbtab

def fit_winner(X, comparison):
    w=comparison.iloc[0]; name=w['algorithm']
    if name=='K-Means': model=KMeans(n_clusters=int(w['k']),random_state=42,n_init=30)
    elif name=='Agglomerative': model=AgglomerativeClustering(n_clusters=int(w['k']),linkage='ward')
    elif name=='Gaussian Mixture': model=GaussianMixture(n_components=int(w['k']),covariance_type='full',random_state=42,n_init=5)
    else: model=DBSCAN(eps=float(w['eps']),min_samples=int(w['min_samples']))
    return model,model.fit_predict(X)

def assign_personas(client_df):
    d=client_df.copy()
    p=d.groupby('cluster').agg(buyers=('client_id','count'),avg_age=('age','mean'),avg_investment=('avg_investment','mean'),loan_rate=('loan_binary','mean'),company_rate=('company_binary','mean'),investment_rate=('investment_binary','mean'),satisfaction=('satisfaction_score','mean'),avg_area=('avg_area_sqft','mean'))
    if len(p)!=4:
        labels={i:f'Segment {j+1}' for j,i in enumerate(p.index)}
        labels[p.avg_investment.idxmax()]='High-Value Investors'
        labels[p.avg_age.idxmin()]='Younger / Financing-Led Buyers'
        return d.assign(BuyerPersona=d.cluster.map(labels)),p,labels
    labels={}
    luxury=p.avg_investment.idxmax(); labels[luxury]='Luxury Investors'
    remaining=[i for i in p.index if i not in labels]
    corporate=p.loc[remaining,'company_rate'].idxmax(); labels[corporate]='Corporate Buyers'
    remaining=[i for i in p.index if i not in labels]
    score=p.loc[remaining,'loan_rate'].rank(pct=True)+(-p.loc[remaining,'avg_age']).rank(pct=True)+p.loc[remaining,'investment_rate'].rank(pct=True)
    first=score.idxmax(); labels[first]='First-Time Buyers'
    for i in p.index:
        if i not in labels: labels[i]='Global Investors'
    return d.assign(BuyerPersona=d.cluster.map(labels)),p,labels

def run_pipeline(clients_path=None,properties_path=None):
    clients,props=load_dataset(clients_path,properties_path)
    tx=build_transaction_data(clients,props)
    cf=build_client_features(clients,tx)
    X,pre,features=prepare_features(cf)
    k_curve,metric_best_k=find_best_k(X)
    # For the business deliverable we use the requested four actionable personas.
    # Algorithm selection is then performed at k=4, with DBSCAN excluded from the final winner if it marks >10% as noise.
    operating_k=4 if len(cf)>=4 else metric_best_k
    comparison,db_curve=compare_algorithms(X,operating_k)
    model,labels=fit_winner(X,comparison)
    cf=cf.assign(cluster=labels)
    cf,profile,persona_map=assign_personas(cf)
    pca=PCA(n_components=2,random_state=42); coords=pca.fit_transform(X)
    pca_df=pd.DataFrame({'PC1':coords[:,0],'PC2':coords[:,1],'cluster':cf.cluster.values,'BuyerPersona':cf.BuyerPersona.values,'client_id':cf.client_id.values})
    metrics=score_labels(X,labels); metrics.update(best_algorithm=comparison.iloc[0]['algorithm'],best_k=operating_k,metric_best_k=metric_best_k,operating_k=operating_k)
    return {'clients':clients,'properties':props,'transactions':tx,'client_df':cf,'X':X,'preprocessor':pre,'feature_names':features,'k_curve':k_curve,'best_k':operating_k,'metric_best_k':metric_best_k,'comparison':comparison,'db_curve':db_curve,'model':model,'labels':labels,'profile':profile,'persona_map':persona_map,'pca_df':pca_df,'pca':pca,'metrics':metrics}

def save_bundle(data,path=None):
    path=Path(path or ROOT/'models'/'parcl_clustering_bundle.joblib'); path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(data,path); return path
