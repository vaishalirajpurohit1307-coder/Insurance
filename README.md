
# Streamlit HR Attrition & Insurance Dashboard (single-folder package)

Files included (root of zip):
- `kpp.sy` : Main Streamlit app file (named `.sy` per request). If Streamlit Cloud expects `.py`, rename to `kpp.py` or set the main file in Streamlit settings.
- `utils.py` : Helper functions for loading, cleaning, preprocessing and modeling.
- `requirements.txt` : Packages (no pinned versions) to install on Streamlit Cloud.
- `README.md` : This file.

## How to deploy
1. Upload all files to a GitHub repository root (do not create folders).
2. On Streamlit Cloud, connect the GitHub repo and set the main file to `kpp.py` or `kpp.sy` (rename if necessary).
3. Ensure `Insurance.csv` is present in the repo root for default dataset loading OR use the Upload & Predict tab to supply a dataset.

## Notes on data handling
- Null values: numeric -> mean, categorical -> mode (or 'Unknown').
- Target detection: looks for columns containing 'attrit', 'attrition', 'label' or 'target'. Rename accordingly if needed.
- Filters: auto-detects job-role-like and satisfaction-like columns if present.

## Quick tips
- If your dataset uses different column names, rename them to commonly used names: `Attrition`, `JobRole`, `Satisfaction`, `Income`, `Age`.
- `kpp.sy` may need to be renamed to `kpp.py` for Streamlit Cloud to recognize it as a Python script.
