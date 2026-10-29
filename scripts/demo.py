"""
scripts/demo.py
===============
Interactive viva demo — walks through the full pipeline for a
single patient, showing:
  1. Original diabetes risk score
  2. SHAP explanation (top features)
  3. Explanation-guided attack result
  4. Blind trial-and-error comparison (N = 1, 5, 10, 20, 50)
  5. Equivalence N* for this patient

Usage:
    python scripts/demo.py                 # default patient #42
    python scripts/demo.py --patient 100   # specific index
    python scripts/demo.py --list          # show available patients

Design notes:
  - Uses 2017-2018 held-out test data (same as final experiment)
  - Config: reported_rf_shap (primary pre-registered config)
  - Lie budget: B(height)=7.29cm, B(weight)=8.96kg
  - Seed fixed to 42 for reproducibility
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pickle

from src.config import set_all_seeds, FINAL_CONFIG, FEATURE_NAMES_REPORTED
from src.models import DiabetesModel
from src.explanations import ExplanationGenerator
from src.adversarial import AdversarialSimulator
from src.metrics import compute_equivalence

set_all_seeds()

# ── Banner ──────────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════════════╗
║   GP-109 BTP — Can an AI Explanation Be Misused?            ║
║   Live Viva Demo  |  DTU CSE  |  2026-27                    ║
╚══════════════════════════════════════════════════════════════╝
"""

# ── Constants (from pre-registered FINAL_CONFIG) ─────────────────
LIE_BUDGET = {'self_height_cm': 7.29, 'self_weight_kg': 8.96}
PRIMARY_CONFIG = 'reported_rf_shap'
N_MILESTONES = [1, 5, 10, 20, 50, 100]

def sep(char='─', n=62):
    print(char * n)

def load_artefacts():
    """Load the held-out test data and trained model."""
    pkl_path = 'data/processed/nhanes_2017_2018_prepared.pkl'
    if not os.path.exists(pkl_path):
        print("ERROR: holdout data not found. Run scripts/prepare_data.py first.")
        sys.exit(1)

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Build model using reported features (Model B)
    model = DiabetesModel(model_type='rf',
                          feature_set='reported',
                          n_estimators=100,
                          max_depth=10,
                          random_state=123)

    # Load from saved 2015-2016 training data
    train_pkl = 'data/processed/nhanes_2015_2016_prepared.pkl'
    with open(train_pkl, 'rb') as f:
        train_data = pickle.load(f)

    model.train(train_data['X_reported'], train_data['y'])

    X_test = data['X_reported']
    y_test = data['y']
    feat_names = data.get('feature_names_reported', FEATURE_NAMES_REPORTED)
    return model, X_test, y_test, feat_names

def run_demo(patient_idx=42, verbose=True):
    print(BANNER)
    model, X_test, y_test, feat_names = load_artefacts()

    n_patients = len(X_test)
    if patient_idx >= n_patients:
        print(f"  Patient index {patient_idx} out of range (0–{n_patients-1})")
        sys.exit(1)

    x = X_test[patient_idx]
    true_label = y_test[patient_idx]

    # ── Step 1: Original score ────────────────────────────────────
    sep('═')
    print(f"  STEP 1 — Patient #{patient_idx} Profile")
    sep()
    original_score = model.predict_proba(x.reshape(1, -1))[0, 1]
    print(f"  True diabetes status  : {'DIABETIC ✓' if true_label else 'Non-diabetic'}")
    print(f"  Original risk score   : {original_score:.4f}  (0=low risk, 1=high risk)")
    print()
    print("  Feature values:")
    for i, (name, val) in enumerate(zip(feat_names, x)):
        print(f"    {name:30s}: {val:.2f}")

    # ── Step 2: SHAP explanation ──────────────────────────────────
    sep('─')
    print(f"  STEP 2 — SHAP Explanation (what the model sees)")
    sep()
    explainer = ExplanationGenerator(model)
    shap_attrs = explainer.get_shap_explanation(x)

    sorted_attrs = sorted(shap_attrs.items(), key=lambda kv: abs(kv[1]), reverse=True)
    print(f"  {'Feature':30s}  {'SHAP value':>12}  Impact")
    sep('-')
    for feat, val in sorted_attrs[:7]:
        direction = '↑ increases risk' if val > 0 else '↓ decreases risk'
        gameable  = ' ← GAMEABLE' if feat in LIE_BUDGET else ''
        print(f"  {feat:30s}  {val:+12.4f}  {direction}{gameable}")

    # Identify the gameable feature the explanation targets
    gameable_attrs = {k: v for k, v in shap_attrs.items() if k in LIE_BUDGET}
    target_feat    = max(gameable_attrs, key=lambda k: abs(gameable_attrs[k]))
    target_val     = gameable_attrs[target_feat]
    budget         = LIE_BUDGET[target_feat]
    direction_sign = -1 if target_val > 0 else +1  # move against the attribution

    # ── Step 3: Explanation-guided attack ─────────────────────────
    sep('─')
    print(f"  STEP 3 — Explanation-Guided Attack")
    sep()
    simulator = AdversarialSimulator(model, LIE_BUDGET)
    exp_result = simulator.explanation_guided_attack(x, method='shap')

    print(f"  Target feature  : {target_feat}")
    print(f"  Original value  : {x[feat_names.index(target_feat)]:.2f}")
    adjusted_val = x[feat_names.index(target_feat)] + direction_sign * budget
    print(f"  Adjusted value  : {adjusted_val:.2f}  (changed by {direction_sign * budget:+.2f})")
    print(f"  New score       : {exp_result['new_score']:.4f}")
    print(f"  Score reduction : {exp_result['score_reduction']:+.4f}  ◄ explanation advantage")

    # ── Step 4: Blind trial-and-error ─────────────────────────────
    sep('─')
    print(f"  STEP 4 — Blind Trial-and-Error (for comparison)")
    sep()
    blind_result = simulator.blind_attack_multi_n(x, N_MILESTONES)

    print(f"  {'N attempts':>12}  {'Best score':>12}  {'Reduction':>12}")
    sep('-')
    for n in N_MILESTONES:
        if n in blind_result['reductions']:
            red = blind_result['reductions'][n]
            score = original_score - red
            marker = ' ◄ matches explanation' if abs(red - exp_result['score_reduction']) < 0.001 else ''
            print(f"  {n:>12}  {score:>12.4f}  {red:>+12.4f}{marker}")

    # ── Step 5: Equivalence N* for this patient ───────────────────
    sep('─')
    print(f"  STEP 5 — Equivalence N* for Patient #{patient_idx}")
    sep()
    blind_curve = [(n, blind_result['reductions'][n])
                   for n in N_MILESTONES if n in blind_result['reductions']]
    n_vals   = [b[0] for b in blind_curve]
    r_vals   = [b[1] for b in blind_curve]
    n_star   = compute_equivalence(exp_result['score_reduction'], n_vals, r_vals)

    if n_star <= 1:
        verdict = "🟢 Explanation gives NO advantage over 1 blind attempt"
    elif n_star <= 5:
        verdict = "🟡 Explanation worth a few blind attempts (LOW threat)"
    elif n_star <= 20:
        verdict = "🟠 Explanation worth moderate blind attempts (MODERATE threat)"
    else:
        verdict = "🔴 Explanation very valuable (HIGH threat)"

    print(f"  N* = {n_star:.1f}  →  {verdict}")
    print()
    sep('═')
    print(f"  POPULATION RESULT (held-out 2017-2018, N=884 patients):")
    print(f"  Primary config (reported_rf_shap):")
    print(f"    Mean N*  = 12.43  [95% CI: 10.54–14.53]")
    print(f"    Median   = 0.0   (63.8% of patients: explanation useless)")
    print(f"    17.2%    of patients: explanation worth 10+ blind attempts")
    sep('═')
    print()

    return n_star

def list_patients():
    _, X_test, y_test, feat_names = load_artefacts()
    print(f"\n  Available patients: 0 – {len(X_test)-1}")
    print(f"  Diabetic patients: {y_test.sum()} / {len(y_test)}")
    print(f"  Feature names: {feat_names}")

# ── Entry point ──────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BTP Viva Demo')
    parser.add_argument('--patient', type=int, default=42,
                        help='Patient index in test set (default: 42)')
    parser.add_argument('--list', action='store_true',
                        help='List available patients')
    args = parser.parse_args()

    if args.list:
        list_patients()
    else:
        run_demo(patient_idx=args.patient)
