import pickle
import numpy as np

from ..circuit.candidates import Candidate

def save_parameter_train(name: str, candidate: Candidate, g_best: float, K_quantum: np.ndarray, K_classical: np.ndarray, X_subset: np.ndarray, y_subset: np.ndarray):
    """
    Saves the results from parameter training to find the best candidate
    """
    data = {
        'genes':      [gene.get_gene_info() for gene in candidate.genes],
        'g_best':     g_best,
        'K_quantum':  K_quantum,
        'K_classical':K_classical,
        'X_subset':   X_subset,
        'y_subset':   y_subset,
    }
    
    path = f'Results/{name}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved → {path}")

def load_parameter_train(name: str):
    """
    Loads the results from parameter training of a previous run
    """    
    with open(f'Results/{name}.pkl', 'rb') as f:
        return pickle.load(f)
    
def save_full_kernel(name: str, K_quantum: np.ndarray, K_classical: np.ndarray, g_full: float):
    """
    Saves the results from (...)
    """
    data = {
        'K_quantum':   K_quantum,
        'K_classical': K_classical,
        'g_full':      g_full,
    }
    with open(f'Results/{name}.pkl', 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved → Results/{name}.pkl")

def load_full_kernel(name: str):
    with open(f'Results/{name}.pkl', 'rb') as f:
        return pickle.load(f)