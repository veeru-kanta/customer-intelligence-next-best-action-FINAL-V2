
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/customer_behavior.csv"
OUT = ROOT / "data/processed/model_data.csv"

def main():
    df = pd.read_csv(RAW)
    required = ["customer_id","orders","avg_order_value","days_since_purchase",
                "email_open_rate","support_tickets","rating","feedback","churn"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df["customer_id"].isna().any():
        raise ValueError("customer_id contains nulls")
    if not df["rating"].between(1,5).all():
        raise ValueError("rating outside 1-5")
    if not df["churn"].isin([0,1]).all():
        raise ValueError("churn must be binary")
    df.to_csv(OUT,index=False)
    print(f"Saved {len(df):,} records -> {OUT}")

if __name__ == "__main__":
    main()
