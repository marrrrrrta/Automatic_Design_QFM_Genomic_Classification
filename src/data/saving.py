import os
import pickle
import numpy as np
from datetime import datetime

from config.config import RESULTS_DIR
from ..circuit.candidates import Candidate


# -------- Internal helpers ------------------------------------------

def _experiment_dir(name: str) -> str:
    """Returns the experiment subdirectory path, creating it if needed."""
    path = os.path.join(RESULTS_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path

def _candidate_path(name: str) -> str:
    return os.path.join(RESULTS_DIR, name, 'candidate.pkl')
 
def _svm_path(name: str) -> str:
    return os.path.join(RESULTS_DIR, name, 'svm.pkl')
 
def _full_kernel_path(name: str) -> str:
    return os.path.join(RESULTS_DIR, name, 'full_kernel.pkl')


# -------- Existance checkers ----------------------------------------

def candidate_exists(name: str) -> bool:
    """True if a saved candidate is found for this experiment name."""
    return os.path.isfile(_candidate_path(name))
 
def svm_exists(name: str) -> bool:
    """True if saved SVM results are found for this experiment name."""
    return os.path.isfile(_svm_path(name))
 
def full_kernel_exists(name: str) -> bool:
    """True if a saved full kernel is found for this experiment name."""
    return os.path.isfile(_full_kernel_path(name))


# -------- SAVING ----------------------------------------------------

def save_parameter_train(
    name: str, 
    candidate: Candidate, 
    g_best: float, 
    K_quantum: np.ndarray, K_classical: np.ndarray, 
    X_subset: np.ndarray, y_subset: np.ndarray
) -> None:
    """
    Saves candidate training results to Results/{name}/candidate.pkl.
 
    Stores the candidate object, its gene info, the best geometric difference
    score, subset kernel matrices, and the training subset itself.
    """
    _experiment_dir(name)
    data = {
        'candidate':   candidate,
        'genes':       [gene.get_gene_info() for gene in candidate.genes],
        'g_best':      g_best,
        'K_quantum':   K_quantum,
        'K_classical': K_classical,
        'X_subset':    X_subset,
        'y_subset':    y_subset,
        'timestamp':   datetime.now().isoformat(),
    }
    path = _candidate_path(name)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved candidate  → {path}")
    
def save_full_kernel(
    name: str,
    K_quantum: np.ndarray,
    K_classical: np.ndarray,
    g_full: float,
) -> None:
    """
    Saves the full-dataset kernel matrices to Results/{name}/full_kernel.pkl.
 
    Used when the kernel is evaluated on the complete training set rather than
    the optimisation subset, e.g. for post-hoc analysis or ablation studies.
    """
    _experiment_dir(name)
    data = {
        'K_quantum':   K_quantum,
        'K_classical': K_classical,
        'g_full':      g_full,
        'timestamp':   datetime.now().isoformat(),
    }
    path = _full_kernel_path(name)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved full kernel → {path}")

def save_svm_results(name: str, results: dict) -> None:
    """
    Saves SVM results to Results/{name}/svm.pkl.
 
    The results dict should contain at minimum:
        accuracy, best_C, best_cv_accuracy, y_pred, y_test, K_test, report
 
    K_test (the cross-kernel matrix) is saved alongside predictions so that
    re-running the SVM with different hyperparameters does not require
    recomputing expensive quantum kernel evaluations.
    """
    _experiment_dir(name)
    results['timestamp'] = datetime.now().isoformat()
    path = _svm_path(name)
    with open(path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved SVM        → {path}")

# -------- LOADING ----------------------------------------------------

def load_parameter_train(name: str) -> dict:
    """
    Loads candidate training results from Results/{name}/candidate.pkl.
    """
    path = _candidate_path(name)
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_full_kernel(name: str) -> dict:
    """
    Loads full-dataset kernel matrices from Results/{name}/full_kernel.pkl.
    """
    path = _full_kernel_path(name)
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_svm_results(name: str) -> dict:
    """
    Loads SVM results from Results/{name}/svm.pkl.
    """
    path = _svm_path(name)
    with open(path, 'rb') as f:
        return pickle.load(f)