# GP-109: Can an AI Explanation Be Misused?

**Testing Whether SHAP and LIME Explanations Help Someone Cheat a Clinical Risk Model**

B.Tech Final Year Project — Delhi Technological University, Dept. of CSE

## Team
| Name | Roll Number |
|------|-------------|
| Akash Singh | 23/CS/036 |
| Aditya Jha | 23/CS/025 |
| Akshat Kumar | 23/CS/039 |
| Ishan Singh (Jr.) | 25/CS/168 |

**Guide**: Miss. Anjali Bansal

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Download NHANES data
python scripts/download_nhanes.py

# Prepare data (Stage 3)
python scripts/prepare_data.py

# Run trial experiment (Stage 5)
python scripts/run_trial.py

# Run final experiment (Stage 7 — after pre-registration!)
python scripts/run_final.py
```

## Project Structure
```
BTP/
├── src/                    # Source modules
│   ├── config.py           # Seeds, paths, experiment parameters
│   ├── data_loader.py      # NHANES loading and preprocessing
│   ├── gap_analysis.py     # Reported-vs-measured gap analysis
│   ├── models.py           # Model training (LR, RF)
│   ├── explanations.py     # SHAP & LIME generation
│   ├── adversarial.py      # Adversarial simulation engine
│   ├── metrics.py          # Equivalence metric & statistical tests
│   └── visualization.py    # Publication-quality plots
├── scripts/                # Executable scripts
├── tests/                  # pytest test suite
├── data/raw/               # Downloaded NHANES files (gitignored)
├── data/processed/         # Cleaned datasets
├── results/figures/        # Generated plots
├── results/tables/         # Generated tables
├── roadmap/                # Stage-by-stage implementation guides
└── docs/                   # Column mapping, analysis plan
```

## Reproducibility
All random seeds are fixed in `src/config.py`. Results should be identical across machines.
