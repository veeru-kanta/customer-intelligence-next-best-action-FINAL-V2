
import pandas as pd
import numpy as np

df=pd.read_csv("data/processed/model_data.csv")

# Transparent prioritization score for the portfolio demonstration.
# Production version would replace this with calibrated model probabilities
# and empirically estimated treatment uplift.
risk=(
    .40*(df.days_since_purchase/180) +
    .25*(df.support_tickets/df.support_tickets.max()) +
    .20*(1-df.email_open_rate) +
    .15*(1-(df.rating-1)/4)
).clip(0,1)

df["priority_score"]=risk
df["recommended_action"]=np.select(
    [
        (risk>=.70)&(df.rating<3),
        (risk>=.70),
        (risk<.70)&(df.orders>=6)
    ],
    ["Service recovery","Retention offer","Cross-sell"],
    default="Engagement reminder"
)

# Illustrative constrained campaign: select top 15% by priority score.
threshold=df.priority_score.quantile(.85)
selected=df[df.priority_score>=threshold].copy()
selected=selected.sort_values("priority_score",ascending=False)
selected.to_csv("data/processed/next_best_actions.csv",index=False)

impact_share=selected.revenue.sum()/df.revenue.sum()
print(f"Selected: {len(selected):,} / {len(df):,} customers")
print(f"Selected revenue share: {impact_share:.2%}")
print(selected[["customer_id","priority_score","recommended_action"]].head(10))
