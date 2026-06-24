"""
migrate_results.py
──────────────────
One-time script to move old flat Results/*.pkl files into the new
per-experiment subfolder layout:

    Results/<experiment>/candidate.pkl   ← was Results/training_*.pkl
    Results/<experiment>/svm.pkl         ← was Results/<subset_name>.pkl

Run once from the repo root:
    python migrate_results.py

Old files are kept with a .bak extension so nothing is deleted.
"""

import os
import pickle
import shutil

RESULTS_DIR = 'Results'

# Map old flat filenames → (experiment_name, artifact)
# Update these if your naming conventions differ.
MIGRATION_MAP = {
    # candidate files  (have keys: candidate, genes, g_best, K_quantum, …)
    'training_optuna_random_100':       ('optuna_random_100',  'candidate'),
    'training_optuna_global_100':       ('optuna_global_100',  'candidate'),
    'training_optuna_stratified_100':   ('optuna_strat_100',   'candidate'),
    'training_optuna_optuna_random_100':('optuna_random_100',  'candidate'),  # duplicate with bad name
    'training_mealpy_random_30ep_20pop_1pm': ('mealpy_random', 'candidate'),

    # SVM results  (have keys: accuracy, best_C, best_cv_accuracy, y_pred)
    'optuna_random':      ('optuna_random_100',  'svm'),
    'optuna_global':      ('optuna_global_100',  'svm'),
    'optuna_stratified':  ('optuna_strat_100',   'svm'),
    'Classical_baseline': ('Classical_baseline', 'svm'),
}


def _detect_artifact(data: dict) -> str:
    """Auto-detect whether a pkl is a candidate or SVM result."""
    if 'candidate' in data or 'g_best' in data:
        return 'candidate'
    if 'accuracy' in data or 'y_pred' in data:
        return 'svm'
    return 'unknown'


def migrate():
    flat_pkls = [
        f[:-4] for f in os.listdir(RESULTS_DIR)
        if f.endswith('.pkl') and os.path.isfile(os.path.join(RESULTS_DIR, f))
    ]

    if not flat_pkls:
        print("No flat .pkl files found in Results/ — nothing to migrate.")
        return

    print(f"Found {len(flat_pkls)} flat pkl file(s) to migrate.\n")

    for stem in flat_pkls:
        old_path = os.path.join(RESULTS_DIR, f'{stem}.pkl')

        if stem in MIGRATION_MAP:
            experiment, artifact = MIGRATION_MAP[stem]
        else:
            # Auto-detect from content
            with open(old_path, 'rb') as f:
                data = pickle.load(f)
            artifact   = _detect_artifact(data)
            experiment = stem  # keep old name as experiment name
            print(f"  [auto] '{stem}' → experiment='{experiment}', artifact={artifact}")

        dest_dir  = os.path.join(RESULTS_DIR, experiment)
        dest_file = os.path.join(dest_dir, f'{artifact}.pkl')
        bak_path  = old_path + '.bak'

        os.makedirs(dest_dir, exist_ok=True)

        if os.path.exists(dest_file):
            print(f"  SKIP  '{stem}' → {dest_file} already exists")
            continue

        shutil.copy2(old_path, dest_file)
        os.rename(old_path, bak_path)
        print(f"  OK    '{stem}.pkl' → {dest_file}  (original kept as .bak)")

    print("\nMigration complete.")
    print("Review the mapping at the top of this script if any experiment names look wrong.")
    print("Delete .bak files once you have verified the new structure.")


if __name__ == '__main__':
    migrate()