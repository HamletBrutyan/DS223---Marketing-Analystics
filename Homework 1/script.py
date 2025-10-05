# script.py — Step 4: Fit Bass Model for Nissan Leaf
# ====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json

path = "data/Dataset1.xlsx"

df_full = pd.read_excel(path, sheet_name="PEV Sales Final 2019", header=None)
header_row_idx = df_full.index[df_full.apply(lambda r: str(r[1]).strip().lower() == "vehicle", axis=1)].tolist()[0]
header = df_full.iloc[header_row_idx].tolist()
data = df_full.iloc[header_row_idx+1:].copy()
data.columns = header
data = data.dropna(how="all")

leaf_row = data[data["Vehicle"].astype(str).str.strip().str.lower() == "nissan leaf"].copy()
year_cols = []
for col in data.columns:
    try:
        year_cols.append(int(float(col)))
    except:
        pass
year_cols = sorted(set(year_cols))

leaf_series = leaf_row[[*year_cols]].T.reset_index()
leaf_series.columns = ["year", "sales_units"]
leaf_series["year"] = leaf_series["year"].astype(int)
leaf_series["sales_units"] = pd.to_numeric(leaf_series["sales_units"], errors="coerce").fillna(0).astype(int)
leaf_series = leaf_series.sort_values("year").reset_index(drop=True)

# === Bass Model ===
def bass_adopters(t, p, q, M):
    t = np.asarray(t, dtype=float)
    num = (p + q)**2 * np.exp(-(p + q) * t)
    den = p * (1 + (q/p) * np.exp(-(p + q) * t))**2
    return M * (num / den)

leaf_df = leaf_series.copy()
leaf_df["t"] = np.arange(1, len(leaf_df) + 1)
leaf_df["cum_units"] = leaf_df["sales_units"].cumsum()

m0 = leaf_df["sales_units"].sum() * 1.2
p0, q0 = 0.03, 0.5
bounds = ([1e-6, 1e-6, max(leaf_df["sales_units"])], [1.0, 3.0, 1e9])

popt, pcov = curve_fit(bass_adopters, leaf_df["t"], leaf_df["sales_units"], p0=[p0, q0, m0], bounds=bounds, maxfev=200000)
p_est, q_est, M_est = popt
t_star = np.log(q_est / p_est) / (p_est + q_est)

# Save fitted data
leaf_df["pred_sales"] = bass_adopters(leaf_df["t"], p_est, q_est, M_est)
leaf_df.to_csv("leaf_fit_results.csv", index=False)

# Save parameters for next script
params = {
    "p": float(p_est),
    "q": float(q_est),
    "M": float(M_est),
    "t_peak": float(t_star)
}
with open("bass_params.json", "w") as f:
    json.dump(params, f, indent=4)

# Print summary
print("=== Bass Model Results (Nissan Leaf, 2011–2019) ===")
print(json.dumps(params, indent=4))

# Plot
plt.figure(figsize=(7,4))
plt.bar(leaf_df["year"], leaf_df["sales_units"], label="Actual sales")
plt.plot(leaf_df["year"], leaf_df["pred_sales"], "r-o", label="Bass fit")
plt.title("Nissan Leaf Annual Sales vs. Bass Model Fit (U.S.)")
plt.xlabel("Year")
plt.ylabel("Units Sold")
plt.legend()
plt.tight_layout()
plt.savefig("bass_fit_leaf.png", dpi=200)
plt.close()