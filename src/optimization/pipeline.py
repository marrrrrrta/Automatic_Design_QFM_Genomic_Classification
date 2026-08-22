import numpy as np
 
from config.experiments import ExperimentConfig
from .training_candidate import train_candidate_optuna, train_candidate_mealpy
from .training_candidate import train_candidate_optuna_haar, train_candidate_mealpy_haar
from .training_candidate import train_candidate_optuna_opt3
from .training_candidate import train_candidate_optuna_kta, train_candidate_mealpy_kta

from .training_svm import run_svm_prediction
from ..data.saving import (
    candidate_exists, svm_exists,
    load_parameter_train, load_svm_results,
)
from ..data.subsets import subset_random, subset_nystrom_global, subset_nystrom_stratified

# -------- OPTIONS ------------------------------------------------------------------------

_SUBSET_METHODS = {
    'random':              subset_random,
    'nystrom_global':      subset_nystrom_global,
    'nystrom_stratified':  subset_nystrom_stratified,
    'none':                None,
}
 
_TRAIN_FUNCTIONS = {
    ('optuna', 'geometric'): train_candidate_optuna,
    ('mealpy', 'geometric'): train_candidate_mealpy,
    ('optuna', 'haar'):      train_candidate_optuna_haar,
    ('mealpy', 'haar'):      train_candidate_mealpy_haar,
    ('optuna', 'angles'):    train_candidate_optuna_opt3,
    ('optuna', 'kta'):       train_candidate_optuna_kta,
    ('mealpy', 'kta'):       train_candidate_mealpy_kta,
}

# -------- PIPELINE ------------------------------------------------------------------------

def get_candidate(
    config: ExperimentConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
):
    """
    Returns the best candidate for this experiment, loading from disk if
    it has already been trained.
 
    Args:
        config (ExperimentConfig):  Experiment configuration.
        X_train (np.ndarray): Full training feature matrix.
        y_train (np.ndarray): Full training labels.
 
    Returns:
        candidate (Candidate): Best found circuit candidate.
        K_q  (np.ndarray): Quantum kernel on the optimisation subset.
        K_c  (np.ndarray): Classical kernel on the optimisation subset.
        X_sub (np.ndarray): Optimisation subset features.
        y_sub (np.ndarray): Optimisation subset labels.
    """
    load_name = config.load_from or config.name
 
    if candidate_exists(load_name):
        print(f"Loading cached candidate for '{load_name}'")
        data = load_parameter_train(load_name)
        return (
            data['candidate'],
            data['K_quantum'],
            data['K_classical'],
            data['X_subset'],
            data['y_subset'],
        )
 
    # Build subset
    if _SUBSET_METHODS[config.subset_method] is None:
        X_sub, y_sub = X_train, y_train
    else:
        subset_fn = _SUBSET_METHODS[config.subset_method]
        _, X_sub, y_sub = subset_fn(X_train, y_train, **config.subset_kwargs)
 
    # Train
    train_fn = _TRAIN_FUNCTIONS[(config.optimizer, config.variant)]
    _, candidate, K_q, K_c = train_fn(config.name, X_sub, y_sub, **config.optimizer_kwargs)
 
    return candidate, K_q, K_c, X_sub, y_sub


def get_svm_results(
    config: ExperimentConfig,
    K_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    K_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    c_range: dict | None = None,
    cv: int = 5,
) -> dict:
    """
    Returns SVM results for this experiment, loading from disk if they have
    already been computed.  Simply delete Results/{name}/svm.pkl to trigger
    a fresh run.
 
    K_test and y_test are only required when no cached results exist.
    Saving K_test inside the pkl means a re-run with different C values does
    not require recomputing the quantum kernel.
 
    Args:
        config:  Experiment configuration.
        K_train: Precomputed kernel on the training subset, shape (n_sub, n_sub).
                 Required only when no cache exists.
        y_train: Labels for the training subset. Required only when no cache exists.
        K_test:  Cross-kernel between test set and training subset, shape (n_test, n_sub).
                 Required only when no cache exists.
        y_test:  Test labels. Required only when no cache exists.
        c_range: GridSearchCV parameter grid (optional).
        cv:      Cross-validation folds (default 5).
 
    Returns:
        dict with keys: name, accuracy, best_C, best_cv_accuracy,
                        y_pred, y_test, K_test, report, timestamp.
    """
    if svm_exists(config.name):
        print(f"Loading cached SVM results for '{config.name}'")
        return load_svm_results(config.name)
 
    if K_train is None or K_test is None or y_train is None or y_test is None:
        raise ValueError(
            f"No cached SVM found for '{config.name}'. "
            "Provide K_train, K_test, y_train, and y_test to run it."
        )
 
    return run_svm_prediction(
        config.name,
        K_train, y_train,
        K_test,  y_test,
        c_range = c_range,
        cv = cv,
    )
