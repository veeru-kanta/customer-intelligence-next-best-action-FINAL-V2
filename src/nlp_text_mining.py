
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

df=pd.read_csv("data/processed/model_data.csv")
vec=TfidfVectorizer(stop_words="english",ngram_range=(1,2),min_df=5,max_features=3000)
X=vec.fit_transform(df["feedback"].fillna(""))

k=7
km=KMeans(n_clusters=k,random_state=42,n_init=20)
df["feedback_cluster"]=km.fit_predict(X)

terms=vec.get_feature_names_out()
order=km.cluster_centers_.argsort()[:,::-1]
rows=[]
for c in range(k):
    top=[terms[i] for i in order[c,:8]]
    rows.append({"cluster":c,"top_terms":", ".join(top),
                 "records":int((df.feedback_cluster==c).sum())})
    print(c, rows[-1])

pd.DataFrame(rows).to_csv("reports/nlp_clusters.csv",index=False)
df.to_csv("data/processed/feedback_clusters.csv",index=False)
