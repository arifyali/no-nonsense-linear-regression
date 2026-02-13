# Data

## CFPB Consumer Complaints Database

This guide uses the Consumer Financial Protection Bureau's public complaints database.

### Download Instructions

1. Go to: https://www.consumerfinance.gov/data-research/consumer-complaints/
2. Click **"Download the data"** → select **CSV** format
3. Save the zip file in this `data/` directory (e.g., `complaints.csv.zip`)

The full dataset is ~1.6GB zipped and contains 2M+ records. That's expected.

### Prepare the Sample

Once you have the zip (or extracted CSV) in `data/`, run the prep script. It reads directly from the zip — no need to unzip manually:

```bash
python data/prepare_data.py
```

This will:
- Filter to complaints with narrative text
- Compute response time (days between received and sent to company)
- Sample 50,000 records
- Save to `data/complaints_sample.csv`

### What's in the Data

Key columns we use:
- `Consumer complaint narrative` — the actual complaint text
- `Date received` — when the CFPB received the complaint
- `Date sent to company` — when it was forwarded to the company
- `Product` — financial product category
- `Company` — company name
- `Company response to consumer` — how the company responded

### Don't Commit the Data

The CSV files are large. They're in `.gitignore` for a reason.
