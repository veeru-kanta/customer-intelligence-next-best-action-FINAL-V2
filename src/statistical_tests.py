
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency

df = pd.read_csv("data/processed/model_data.csv")

# H0: mean recency is equal between churned and retained customers.
churned = df.loc[df.churn == 1, "days_since_purchase"]
retained = df.loc[df.churn == 0, "days_since_purchase"]
t_stat, p_value = ttest_ind(churned, retained, equal_var=False)

# H0: churn is independent of customer channel.
table = pd.crosstab(df["channel"], df["churn"])
chi2, chi_p, dof, expected = chi2_contingency(table)

print("Welch t-test: recency vs churn")
print(f"t-statistic={t_stat:.4f}, p-value={p_value:.6g}, significant={p_value < 0.05}")
print("\nChi-square test: channel vs churn")
print(f"chi2={chi2:.4f}, p-value={chi_p:.6g}, significant={chi_p < 0.05}")
