import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

from config.config import N_QUBITS
from ..data.saving import save_svm_results

def classical_baseline(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    n_qubits: int = N_QUBITS, c_range: dict | None = None, cv: int = 5,
    subset_name: str = 'Classical_baseline'
) -> dict:
    # Define kernels
    gamma = 1.0 / (n_qubits * X_train.var())
    K_train = rbf_kernel(X_train, gamma=gamma)
    K_test  = rbf_kernel(X_test, X_train, gamma=gamma)  # No debería de tener distinta gamma?

    # GridSearchCV over C
    if c_range is None:
        c_range = {'C': [0.1, 1, 10, 100, 1000]}
    grid = GridSearchCV(SVC(kernel='precomputed'), c_range, cv=cv, scoring='accuracy', refit=True)
    grid.fit(K_train, y_train)

    # Prediction
    y_pred = grid.best_estimator_.predict(K_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save results 
    results = {
        'subset_name': subset_name,
        'accuracy': accuracy,
        'best_C': grid.best_params_['C'],
        'best_cv_accuracy': grid.best_score_,
        'y_pred': y_pred,
    }
    save_svm_results(subset_name, results)

    # Report
    print(f"REPORT: {subset_name}")
    print("Best C:", results['best_C'])
    print("Best CV accuracy:", results['best_cv_accuracy'])
    print(classification_report(y_test, y_pred))

    return results

