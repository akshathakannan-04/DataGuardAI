# DataGuard AI — Proof-Carrying Data Analyst

HackNex 2026 Internal Qualifier — HNX26PSI08

## What it does
DataGuard AI accepts a CSV dataset, checks its quality, answers supported natural-language data questions, performs the actual calculation with Pandas, and displays runnable verification code.

It is designed around the problem statement requirement that numerical answers should be independently verifiable.

## MVP
- CSV upload
- Data quality report
- Missing-value detection
- Duplicate detection
- Basic outlier detection
- Natural-language question handling
- Safe Pandas calculations
- Verification code
- Charts for grouped results
- Refusal when a question cannot be safely answered

## Optional LLM
If an OpenAI API key is configured in `.streamlit/secrets.toml`, questions not handled by the built-in intent engine are sent to an LLM only to map the question to a small, predefined set of safe operations. The calculation itself is still performed locally with Pandas.

## Run locally

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## API key (optional)

Create:

`.streamlit/secrets.toml`

and add:

```toml
OPENAI_API_KEY = "YOUR_API_KEY"
```

Do NOT commit this file to GitHub.

## Demo
Upload `sample_data.csv` and ask:

- What is the average revenue?
- Which city has the highest revenue?
- What is the total revenue?
- How many customers are there?
- What columns are in the dataset?

## Architecture

User CSV
  -> Pandas
  -> Data quality checks
  -> Question understanding
  -> Safe intent
  -> Pandas calculation
  -> Verified result + evidence + runnable code

## Scope note
Minimum viable solution: CSV + safe data questions + verification + quality report.

Stretch goals:
- Multiple CSV files
- Cross-table joins
- Contradiction detection
- Currency/date consistency checks
- More analytical operations
- Deployment

## External resources
- Streamlit
- Pandas
- OpenAI API (optional)
