# Homework 3  
**Author:** *Hamlet Brutyan*  
**Course:** DS223 - Marketing Analytics   
**Date:** 15-11-2025  

## Overview

This project analyzes customer churn using Accelerated Failure Time (AFT) survival models. The analysis compares multiple AFT distributions (Weibull, Exponential, Log-Normal, and Log-Logistic) to predict customer retention and calculate Customer Lifetime Value (CLV).

## Files

- `homework3_survival_analysis.Rmd` - Main R Markdown file with complete analysis
- `telco.csv` - Dataset containing customer information and churn data
- `README.md` - This file

## Requirements

### R Packages

Install the following R packages:

```r
install.packages(c("survival", "survminer", "dplyr", "ggplot2", 
                   "gridExtra", "knitr", "kableExtra"))
```

### LaTeX Packages (for PDF output)

If you encounter LaTeX errors when knitting to PDF, you may need to install missing LaTeX packages. The document requires the following LaTeX packages:

- `booktabs`
- `longtable`
- `array`
- `multirow`
- `wrapfig`
- `float`
- `colortbl`
- `pdflscape`
- `tabu`
- `threeparttable`
- `threeparttablex`
- `makecell`
- `xcolor`

**Alternative:** If you cannot install LaTeX packages, you can knit to HTML instead, which doesn't require LaTeX packages.

## Usage

1. Open `homework3_survival_analysis.Rmd` in RStudio
2. Ensure all required R packages are installed
3. Ensure `telco.csv` is in the same directory
4. Click "Knit" to generate the report

## Output

The analysis includes:

- Comparison of four AFT models (Weibull, Exponential, Log-Normal, Log-Logistic)
- Model selection based on AIC and Concordance Index
- Visualization of all survival curves
- Feature selection (significant features only)
- Final model with reduced features
- Customer Lifetime Value (CLV) calculation
- CLV analysis by segments
- Retention budget analysis
- Comprehensive report with findings and recommendations

## Notes

- The document is configured to output both PDF and HTML formats
- For PDF output, ensure all LaTeX packages are installed
- HTML output doesn't require LaTeX packages and may be easier to generate

