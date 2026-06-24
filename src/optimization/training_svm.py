import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

from ..data.saving import save_svm_results
from config.config import N_TRIALS, N_EPOCH, N_POPSIZE, N_PM

def run_svm_prediction(
    subset_name: str,
    K_train: np.ndarray, y_train: np.ndarray,
    K_test: np.ndarray, y_test: np.ndarray,
    c_range: dict | None = None, cv: int = 5
) -> dict:
    """
    Runs SVM prediction.
    Returns:
        results (dict): with 'subset_name', 'accuracy', 'best_C', 'best_cv_accuracy', 'y_pred'
    """
    # GridsearchCV over C
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
    if 'optuna' in subset_name:
        print(f"REPORT: {subset_name} | {N_TRIALS} iterations")
        print("Best C: ", results['best_C'])
        print("Best CV accuracy: ", results['best_cv_accuracy'])
        print(classification_report(y_test, y_pred))
    elif 'mealpy' in subset_name:
        print(f"REPORT: {subset_name} | {N_EPOCH} epochs | {N_POPSIZE} pop size | {N_PM} pm")
        print("Best C: ", results['best_C'])
        print("Best CV accuracy: ", results['best_cv_accuracy'])
        print(classification_report(y_test, y_pred))
    else:
        print(f"REPORT: {subset_name}")
        print("Best C: ", results['best_C'])
        print("Best CV accuracy: ", results['best_cv_accuracy'])
        print(classification_report(y_test, y_pred))

    return results