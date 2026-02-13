# No Nonsense Linear Regression: NLP Edition

**Can the words in a consumer complaint predict how long a company takes to respond?**

This guide uses real CFPB consumer complaint data to show how linear regression works with NLP features — not on a toy dataset, not with hand-waving, and not with an R² of 0.99 that should make you suspicious.

You'll extract text features (sentiment, readability, TF-IDF), build a regression model, interpret the coefficients, and learn what to do when your assumptions inevitably break. The kind of stuff you actually need on the job.

## What You'll Learn

- How to engineer meaningful features from raw text data
- Why your R² will be low on real data (and why that's okay)
- How to interpret standardized coefficients with mixed feature types
- What assumption violations look like in practice — and practical fixes
- The difference between statistical and practical significance
- When linear regression makes sense for text data (and when it doesn't)

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
├── notebooks/
│   └── linear_regression_nlp.ipynb   # The guide — start here
├── data/
│   ├── README.md                     # Download instructions
│   └── prepare_data.py               # Sampling script
├── requirements.txt
└── LICENSE
```

## Who This Is For

Data scientists who know what linear regression is but want to see it applied to real text data with honest results. If you're tired of tutorials that cherry-pick clean datasets to get impressive metrics, this is for you.

## License

MIT
