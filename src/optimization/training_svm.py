import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

from ..data.saving import save_svm_results
from config.config import N_TRIALS, N_EPOCH, N_POPSIZE, N_PM

def run_svm_prediction(
    name: str,
    K_train: np.ndarray, y_train: np.ndarray,
    K_test: np.ndarray,  y_test: np.ndarray,
    c_range: dict | None = None,
    cv: int = 5,
) -> dict:
    """
    Runs SVM prediction and saves results to Results/{name}/svm.pkl.
 
    Saves K_test alongside predictions so that re-running with different
    hyperparameters skips the quantum kernel recomputation.
 
    Args:
        name: Experiment name (used as the folder name).
        K_train: Precomputed kernel matrix for training, shape (n_train, n_train).
        y_train: Training labels, shape (n_train,).
        K_test: Precomputed cross-kernel matrix, shape (n_test, n_train).
        y_test: Test labels, shape (n_test,).
        c_range: GridSearchCV parameter grid for C. Defaults to [0.1…1000].
        cv: Number of cross-validation folds.
 
    Returns:
        dict with keys: name, accuracy, best_C, best_cv_accuracy,
                        y_pred, y_test, K_test, report, timestamp.
    """
    if c_range is None:
        c_range = {'C': [0.1, 1, 10, 100, 1000]}
 
    # Grid search over C
    grid = GridSearchCV(
        SVC(kernel='precomputed', class_weight='balanced'), c_range, cv=cv,
        scoring='balanced_accuracy', refit=True,
    )
    grid.fit(K_train, y_train)
 
    # Prediction
    y_pred = grid.best_estimator_.predict(K_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
 
    results = {
        'name': name,
        'accuracy': accuracy,
        'best_C': grid.best_params_['C'],
        'best_cv_accuracy': grid.best_score_,
        'y_pred': y_pred,
        'y_test': y_test,   # saved so metrics can be recomputed without re-running
        'K_test': K_test,   # saved so SVM can be re-run without recomputing kernel
        'report': report,
    }
    save_svm_results(name, results)
 
    # Console report
    if 'optuna' in name:
        header = f"REPORT: {name} | {N_TRIALS} trials"
    elif 'mealpy' in name:
        header = f"REPORT: {name} | {N_EPOCH} epochs | {N_POPSIZE} pop | {N_PM} pm"
    else:
        header = f"REPORT: {name}"
 
    print(f"\n{header}")
    print(f"Best C            : {results['best_C']}")
    print(f"Best CV accuracy  : {results['best_cv_accuracy']:.4f}")
    print(f"Test accuracy     : {accuracy:.4f}")
    print(report)
 
    return results