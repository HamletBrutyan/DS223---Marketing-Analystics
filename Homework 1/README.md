# Bass Diffusion Model Analysis — Volkswagen ID. Buzz

### Course: Marketing Analytics  
### Author: Hamlet Brutyan  
### Date: 05/10/2025  

---

## 🧠 Project Overview
This project applies the **Bass Diffusion Model** to forecast the market adoption of the **Volkswagen ID. Buzz**, one of *TIME’s Best Inventions of 2024*.  
Using the **Nissan Leaf (2011–2019)** as a historical analogue, the study estimates innovation (p) and imitation (q) parameters from real U.S. electric vehicle (EV) sales data, then transfers these parameters to forecast adoption of the new EV over a 10-year period.

The analysis combines quantitative modeling, diffusion theory, and visualization to explain innovation spread in the EV market.

---

## 📊 Methodology Summary

| Step | Description |
|------|--------------|
| **1. Choose an Innovation** | Selected the Volkswagen ID. Buzz (2024) as the target product. |
| **2. Select an Analogue** | Used the Nissan Leaf (2011) as the historical analogue for parameter estimation. |
| **3. Gather Data** | U.S. Department of Energy AFDC dataset — *Plug-in Electric Vehicle Sales by Model (2011–2024)*. |
| **4. Estimate Bass Model** | Fitted Bass model parameters (p, q, M) via nonlinear least squares (`scipy.optimize.curve_fit`). |
| **5. Forecast Diffusion** | Simulated 10-year adoption for the ID. Buzz using fitted parameters. |
| **6. Scope** | United States market (annual adopters). |
| **7. Estimate Adopters by Period** | Generated predicted yearly and cumulative adopters using Bass model outputs. |

---

## 🧩 Repository Structure

```
Homework 1/
│
├── data/
│   ├── Dataset1.xlsx         # Original dataset (DOE AFDC)
│   ├── leaf_fit_results.csv                 # Bass model fit results (Nissan Leaf)
│   ├── idbuzz_forecast.csv                  # Forecasted adoption (ID. Buzz)
│   ├── bass_params.json                     # Saved Bass model parameters
│
├── img/
│   ├── bass_fit_leaf.png                    # Model fit visualization
│   └── idbuzz_forecast.png                  # Forecast visualization
│
├── report/
│   ├── report_source.Rmd                    # R Markdown analysis report
│   └── report.pdf                    # Final knitted report (submission)
│
├── script.py                                # Step 4 – Bass model fitting
├── script2.py                                # Steps 5–7 – Forecast and visualization
└── README.md                                # Project overview and reproduction guide
```

---

## ⚙️ How to Reproduce

1. **Install Python dependencies**
   ```bash
   pip install numpy pandas matplotlib scipy
   ```

2. **Run the scripts in order:**
   ```bash
   python script.py     # Fits Bass model and saves parameters + figures
   python sript2.py     # Loads parameters and forecasts ID. Buzz adoption
   ```

3. **Open the report**
   - The R Markdown file (`report_source.Rmd`) compiles all analysis steps (1–7) with text, code, and figures.  
   - Knit it to PDF or HTML to generate the final report.

---

## 📈 Key Results

| Parameter | Symbol | Estimate |
|------------|---------|-----------|
| Coefficient of innovation | p | 0.0488 |
| Coefficient of imitation | q | 0.4536 |
| Market potential | M | 156,841 units |
| Peak adoption time | t* | ≈ 4.44 years |

**Forecast Summary (Volkswagen ID. Buzz):**
- Peak adoption around **year 4–5**  
- ~145,000 total adopters in 10 years (U.S. market)  
- Adoption curve consistent with diffusion theory (p < q)

---

## 📚 References

- **TIME (2024).** *Volkswagen ID. Buzz.*  
  [https://time.com/collection/best-inventions-2024/7094609/volkswagen-id-buzz/](https://time.com/collection/best-inventions-2024/7094609/volkswagen-id-buzz/)
- **U.S. Department of Energy – AFDC.** *Plug-in Electric Vehicle Sales by Model (2011–2024).*  
  [https://afdc.energy.gov/data/10567](https://afdc.energy.gov/data/10567)


## 🏁 Summary
This project demonstrates the application of the **Bass diffusion model** for forecasting innovation adoption using real-world EV data.  
By combining empirical data, modeling, and clear documentation, it provides a reproducible and insightful look into how new technologies spread in consumer markets.
