# No Nonsense Linear Regression Guide

A comprehensive guide to linear regression for Data Science interview prep and practical application.

## Two Parts, One Goal

**1. [Linear Regression Primer](Linear_Regression_Primer.md)** — The reference document. Covers everything about linear regression that comes up in DS interviews: OLS derivation, assumptions, hypothesis testing, regularization, diagnostics, and common interview questions with answers. Mixed rigor — full math where it matters, intuition where it helps.

**2. [Working Notebook](notebooks/linear_regression_nlp.ipynb)** — A hands-on example using real CFPB consumer complaint data. Demonstrates primer concepts in practice: feature engineering from text, model building, assumption checking, regularization comparison (OLS vs Ridge vs Lasso), and coefficient interpretation. Each section links back to the relevant primer section.

## What the Primer Covers

- The linear model (simple and multiple regression, matrix form)
- OLS estimation (normal equations, Gauss-Markov theorem)
- Assumptions (LINE) — what they are, how to detect violations, how to fix them
- Hypothesis testing (t-test, F-test, partial F-test, confidence vs prediction intervals)
- Model evaluation (R², adjusted R², RMSE, MAE, AIC/BIC, cross-validation)
- Multicollinearity (VIF, remedies)
- Feature engineering (categorical variables, interactions, log transforms)
- Regularization (Ridge, Lasso, Elastic Net, bias-variance tradeoff)
- Diagnostics (residual plots, Cook's distance, leverage, robust standard errors)
- Common interview questions with suggested answers

## What the Notebook Demonstrates

Using the CFPB Consumer Complaints dataset:
- Extracting NLP features (text stats, readability, sentiment, TF-IDF) from complaint narratives
- Building and evaluating an OLS model on real-world messy data
- Why R² = 0.02 is normal and okay with real text data
- Checking assumptions: residual plots, Q-Q plots, VIF, homoscedasticity
- Robust standard errors (HC3) and how they change significance
- Cook's distance and leverage analysis
- Regularization comparison: OLS vs Ridge vs Lasso vs Elastic Net
- Log-transforming the target for skewed data

## Quick Start

```bash
# Clone
git clone https://github.com/arifyali/no-nonsense-linear-regression.git
cd no-nonsense-linear-regression

# Install dependencies
pip install -r requirements.txt

# Download CFPB data (see data/README.md for instructions)
# Then prepare the sample:
python data/prepare_data.py

# Open the notebook
jupyter notebook notebooks/linear_regression_nlp.ipynb
```

## The Dataset

[CFPB Consumer Complaints Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) — 2M+ real consumer complaints about financial products. Free, public, and messy in all the right ways.

## Project Structure

```
├── README.md
├── Linear_Regression_Primer.md     # Theory & interview reference
├── notebooks/
│   └── linear_regression_nlp.ipynb # Working example
├── data/
│   ├── README.md                   # Download instructions
│   └── prepare_data.py             # Sampling script
├── requirements.txt
└── LICENSE
```

## Who This Is For

Data scientists prepping for interviews who want a comprehensive linear regression reference paired with a real-world working example. No toy datasets, no hand-waving, no R² of 0.99 that should make you suspicious.

## Inspired By

[Mustafa Yousir's No Nonsense Experimental Design](https://github.com/mustafaysir/no_nonsense_experimental_design) — same spirit, different topic.

## License

MIT
