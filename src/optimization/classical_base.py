import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

from config.config import N_QUBITS
from ..data.saving import save_svm_results

def classical_baseline(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,  y_test: np.ndarray,
    n_qubits: int = N_QUBITS,
    c_range: dict | None = None,
    cv: int = 5,
    name: str = 'Classical_baseline',
) -> dict:
    """
    Trains an RBF-kernel SVM as a classical baseline and saves results.
 
    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        n_qubits: Used to set the RBF gamma (keeps it consistent with QK).
        c_range: GridSearchCV parameter grid for C.
        cv: Cross-validation folds.
        name: Experiment name / save key.
 
    Returns:
        dict with keys: name, accuracy, best_C, best_cv_accuracy,
                        y_pred, y_test, K_test, report, timestamp.
    """
    gamma = 1.0 / (n_qubits * X_train.var())
    K_train = rbf_kernel(X_train, gamma=gamma)
    K_test = rbf_kernel(X_test, X_train, gamma=gamma)
 
    if c_range is None:
        c_range = {'C': [0.1, 1, 10, 100, 1000]}
 
    grid = GridSearchCV(
        SVC(kernel='precomputed', class_weight='balanced'), c_range, cv=cv,
        scoring='balanced_accuracy', refit=True,
    )
    grid.fit(K_train, y_train)
 
    y_pred   = grid.best_estimator_.predict(K_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred)
 
    results = {
        'name': name,
        'accuracy': accuracy,
        'best_C': grid.best_params_['C'],
        'best_cv_accuracy': grid.best_score_,
        'y_pred': y_pred,
        'y_test': y_test,
        'K_test': K_test,
        'report': report,
    }
    save_svm_results(name, results)
 
    print(f"\nREPORT: {name}")
    print(f"Best C            : {results['best_C']}")
    print(f"Best CV accuracy  : {results['best_cv_accuracy']:.4f}")
    print(f"Test accuracy     : {accuracy:.4f}")
    print(report)
 
    return results
