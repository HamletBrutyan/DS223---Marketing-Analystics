# sript2.py — Steps 5–7: Forecast VW ID. Buzz
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# === Load fitted Bass parameters ===
with open("data/bass_params.json", "r") as f:
    params = json.load(f)

p_est = params["p"]
q_est = params["q"]
M_est = params["M"]

M_target = M_est 

def bass_adopters(t, p, q, M):
    num = (p + q)**2 * np.exp(-(p + q) * t)
    den = p * (1 + (q/p) * np.exp(-(p + q) * t))**2
    return M * (num / den)

# Forecast 10 years
years_forecast = np.arange(1, 11)
pred_new = bass_adopters(years_forecast, p_est, q_est, M_target)
pred_cum = np.cumsum(pred_new)

idbuzz_df = pd.DataFrame({
    "Year_since_launch": years_forecast,
    "Pred_new_adopters": np.round(pred_new).astype(int),
    "Cumulative_adopters": np.round(pred_cum).astype(int)
})

# Save forecast
idbuzz_df.to_csv("data/idbuzz_forecast.csv", index=False)

print("=== Forecast – Volkswagen ID. Buzz (U.S.) ===")
print(f"Using parameters from Nissan Leaf:\np={p_est:.4f}, q={q_est:.4f}, M_target={M_target:,.0f}\n")
print(idbuzz_df)

# Plot forecast
plt.figure(figsize=(7,4))
plt.plot(idbuzz_df["Year_since_launch"], idbuzz_df["Pred_new_adopters"], "o-", label="Predicted new adopters")
plt.title("Volkswagen ID. Buzz – Forecasted Diffusion (U.S.)")
plt.xlabel("Years since launch")
plt.ylabel("New adopters (units)")
plt.legend()
plt.tight_layout()
plt.savefig("idbuzz_forecast.png", dpi=200)
plt.close()
