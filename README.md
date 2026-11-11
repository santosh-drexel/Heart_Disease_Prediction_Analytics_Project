# Heart Disease Classification — Analytics & Machine Learning

An interview-friendly Jupyter project that explores the `heart.csv` dataset, compares four classification models, and documents the preprocessing and evaluation workflow. Refactored from `heartAttackPredict_v4_Updated.ipynb` into two reusable Python helper modules and one main notebook.

> **Scope:** The model predicts the dataset's recorded binary `output` label. It is **not** a clinically validated heart-attack predictor, diagnostic tool, or symptom detector. Do not use it to make medical decisions.

## Project files

```text
heart-analysis/
├── heart_analysis_main.ipynb  # Run this notebook from top to bottom
├── helpers_a.py               # Data loading, exploration, plots, outliers
├── helpers_b.py               # Preprocessing, ML models, tuning, evaluation
├── heart.csv                  # Input dataset
├── requirements.txt           # Python dependencies
└── README.md                  # Setup and usage instructions
```

## Requirements

- Python 3.10+ recommended
- VS Code with **Python** and **Jupyter** extensions, or JupyterLab
- Python packages listed in `requirements.txt`

## Quick start — VS Code (Windows PowerShell)

1. Extract the ZIP into a folder, e.g. `heart-analysis`, and open that **folder** in VS Code.
2. Open a VS Code terminal (**Terminal → New Terminal**) and run:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   If PowerShell blocks activation, use the interpreter directly instead:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Open `heart_analysis_main.ipynb`.
4. Click **Select Kernel** (top right), and choose the `.venv` Python environment. If the kernel is not listed, use **Select Another Kernel → Python Environments**.
5. Make sure the notebook is running from the project folder so `heart.csv`, `helpers_a.py`, and `helpers_b.py` are found. The notebook sets `DATA_PATH = "heart.csv"`.
6. Click **Run All** or execute the cells in order, from top to bottom.

### macOS / Linux terminal

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then open the notebook in VS Code or JupyterLab, select `.venv`, and run all cells.

### Alternative: existing Conda environment

Activate your existing environment, install the dependencies, and select that same environment as the notebook kernel:

```bash
conda activate YOUR_ENV_NAME
python -m pip install -r requirements.txt
```

## Notebook walkthrough

| Section | What to expect |
|---|---|
| 1. Imports and configuration | Load helper functions; configure `DATA_PATH`, `RANDOM_STATE`, and `TEST_SIZE` |
| 2. Load and inspect | Dataset dimensions, missing-value and unique-value summaries |
| 3. Exploratory analysis | Target distribution, numeric distributions, target-based density plots, and categorical summaries |
| Correlation heatmaps | Heatmap with numeric correlation coefficients; second heatmap with custom `NC` / `LC` labels |
| 4. Outlier review | IQR-flagged observations and inspection of unusual categories |
| 5. Train/test split | Stratified 80/20 split before learned preprocessing |
| Winsorization | Training-derived upper-tail caps for selected variables; before/after boxplots and histograms |
| 6. Model comparison | Logistic Regression, Decision Tree, SVM, and Random Forest, with cross-validation and holdout metrics |
| 7. Hyperparameter tuning | Logistic Regression `GridSearchCV` on training data |
| 8. Diagnostic plots | Diagnostic visualizations for the tuned classifier |
| 9. Interview talking points | Summary of design choices, metrics, and limitations |

**Winsorization:** The revised notebook learns upper-tail clipping thresholds from training data (99th percentile) instead of manually deleting a row or calculating thresholds using the full dataset. This is designed to avoid leaking test-set information into preprocessing. The original notebook's results may therefore differ.

**Heatmap labels:** `NC` denotes absolute correlation ≤ 0.05; `LC` denotes absolute correlation < 0.20 (after the `NC` rule). The separate numbered heatmap displays correlation coefficients.

## Model evaluation and interpretation

The project compares classification models using accuracy, precision, recall, F1-score, and related diagnostics. For the positive class, **recall** measures how many actual positive labels the model identifies; false negatives are particularly important to discuss in a health-related context. Choose a final model based on a documented selection criterion and validation results, rather than accuracy alone.

**Important:** This Version 2 notebook compares models and tunes Logistic Regression, but it does **not** yet contain a complete interactive new-patient input/prediction interface or a persisted production model. Do not claim that feature is implemented until it is added and tested.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'helpers_a'` | Open the extracted project **folder** in VS Code; keep helper files next to the notebook; restart the kernel |
| `FileNotFoundError: heart.csv` | Confirm `heart.csv` is in the notebook's working directory, or update `DATA_PATH` to the correct path |
| Missing `sklearn`, `seaborn`, or other package | Run `python -m pip install -r requirements.txt` in the same environment selected as the notebook kernel |
| Wrong kernel / environment | Use **Select Kernel** in VS Code and choose the environment where dependencies were installed |
| Plots not appearing | Run the notebook cells in order; restart the kernel and **Run All** if state is inconsistent |
| Different metrics from original notebook | The refactor changes leakage-prone preprocessing, train/test splitting, and outlier handling; compare methodologies before comparing scores |

## Interview walkthrough (2–3 minutes)

1. **Business/analytics objective:** Classify the recorded `output` label using patient measurements and categorical features.
2. **Data understanding:** Inspect data quality, distributions, class balance, correlations, and outliers.
3. **Preprocessing:** Split first; fit clipping, imputation, encoding, and scaling only within the training pipeline.
4. **Modeling:** Compare four algorithms with cross-validation and holdout evaluation.
5. **Evaluation:** Explain accuracy vs. recall/precision/F1, false negatives, and how the preferred model would be selected.
6. **Limitations:** Discuss sample size, data provenance, validation, generalizability, and why clinical deployment would require substantially more evidence.

## Reproducibility notes

- The notebook uses `RANDOM_STATE = 42` and `TEST_SIZE = 0.20`.
- The helper files are imported by the notebook; run the notebook, **not** the helper modules individually.
- The input dataset must retain the column names expected by the helpers.
- Outputs and performance metrics may vary with package versions or code changes.
