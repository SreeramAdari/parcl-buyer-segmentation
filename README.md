# Parcl AI Market Intelligence

Machine-learning-based buyer segmentation and investment profiling for the supplied Parcl-style real-estate datasets.

## Data-aware fixes
- Mixed date parsing with `format="mixed"` and `errors="coerce"`.
- Currency strings converted safely to numeric values.
- Clustering is performed at **client level** after aggregating transactions, so repeat purchases do not over-weight a buyer.
- Four actionable personas are used for the operating model.
- K=2 is reported as the metric-optimal K, while K=4 is the required business operating model.
- K-Means/Agglomerative/GMM are compared at K=4; DBSCAN is treated as an outlier benchmark and excluded from the final winner if noise exceeds 10%.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

Place `clients.csv` and `properties.csv` in `data/` (already included in the ZIP).
