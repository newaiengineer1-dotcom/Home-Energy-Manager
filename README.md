# ⚡ Home Energy Manager — Auto-Calibrating

LESCO Protected Consumer Optimizer (September 2026).

## How it works

1. Paste your **last 12 months of kWh** and/or **PKR bills** into the sidebar.
2. The app **auto-derives** the effective tariff:
   - Detects the protected threshold from the largest jump in PKR/unit.
   - Computes average rates below and above the threshold.
3. No need to enter tariffs, rates, or thresholds manually.

If no history is provided, sensible Sep 2026 LESCO defaults are used.

## Quick Start

```bash
pip install -r requirements.txt
# Optional: add GROQ_API_KEY to .streamlit/secrets.toml
streamlit run app.py
