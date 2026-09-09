
import json
from pathlib import Path
import pandas as pd
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/"data/processed/model_data.csv")

target="churn"
drop=["customer_id","feedback",target]
X=df.drop(columns=drop)
y=df[target]

cat=["sentiment","feedback_theme","category","channel"]
num=[c for c in X.columns if c not in cat]

pre=ColumnTransformer([
    ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),
                     ("scale",StandardScaler())]),num),
    ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                     ("ohe",OneHotEncoder(handle_unknown="ignore"))]),cat)
])

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=.2,stratify=y,random_state=42
)

models={
"LogisticRegression":LogisticRegression(max_iter=1200,class_weight="balanced"),
"RandomForest":RandomForestClassifier(n_estimators=350,max_depth=12,
                                       min_samples_leaf=3,class_weight="balanced",
                                       random_state=42,n_jobs=-1)
}

if MLFLOW_AVAILABLE:
    mlflow.set_tracking_uri("file:"+str(ROOT/"mlruns"))
results=[]
for name, model in models.items():
    pipe=Pipeline([("preprocess",pre),("model",model)])
    if MLFLOW_AVAILABLE:
        run_ctx = mlflow.start_run(run_name=name)
    else:
        from contextlib import nullcontext
        run_ctx = nullcontext()
    with run_ctx:
        pipe.fit(X_train,y_train)
        prob=pipe.predict_proba(X_test)[:,1]
        pred=(prob>=.5).astype(int)
        metrics={
            "roc_auc":roc_auc_score(y_test,prob),
            "precision":precision_score(y_test,pred,zero_division=0),
            "recall":recall_score(y_test,pred,zero_division=0),
            "f1":f1_score(y_test,pred,zero_division=0)
        }
        if MLFLOW_AVAILABLE:
            mlflow.log_param("model",name)
            mlflow.log_metrics(metrics)
        results.append({"model":name,**metrics})
        print(name, {k:round(v,4) for k,v in metrics.items()})
        if name=="RandomForest":
            import joblib
            joblib.dump(pipe,ROOT/"models/churn_pipeline.joblib")

pd.DataFrame(results).to_csv(ROOT/"reports/model_results.csv",index=False)
print("Best model:",max(results,key=lambda x:x["roc_auc"]))
