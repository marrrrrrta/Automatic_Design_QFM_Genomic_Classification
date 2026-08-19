from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import optuna
from typing import Callable

from config.config import N_QUBITS, N_LAYERS, N_HAAR_SAMPLES, N_HAAR_BINS
from ..circuit.candidates import OptunaCandidate, MealpyCandidate
from ..kernel.quantum_kernel import quantum_kernel_matrix
from ..kernel.metrics import geometric_difference
from ..kernel.expressibility import expressibility

from ..kernel.metrics import geometric_difference

def objective_optuna_generator(X: np.ndarray
) -> Callable[[optuna.Trial], float]:
    """
    Returns an Optuna objective over a given subset

    Args:
        X: (n, d) subset of training points
    """    
    # Redefine gamma and classical kernel
    gamma = 1.0 / (N_QUBITS * X.var())
    K_classical = rbf_kernel(X, gamma=gamma)

    def objective(trial):
        """
        Objective function for Optuna optimization.
        Optuna will suggest one gate (categorical) and one angle (float) per gene
        """    
        # Create a new candidate with its set of genes
        gates, angles = [], []
        for i in range(N_QUBITS * N_LAYERS):
            gate = trial.suggest_categorical(f"gate_{i}", ["H", "CNOT", "RX", "RY", "RZ", "I"])
            gates.append(gate)

            angle = trial.suggest_float(f"angle_{i}", 0, 2 * np.pi)
            angles.append(angle)
        
        # Define the candidate
        candidate = OptunaCandidate(N_QUBITS, N_LAYERS, gates, angles)

        # Compute quantum kernel matrix
        K_quantum = quantum_kernel_matrix(X, candidate)

        # Compute geometric difference
        g = geometric_difference(K_classical, K_quantum)

        return g
    return objective

def objective_mealpy_generator(X: np.ndarray
) -> Callable[[np.ndarray], float]:
    """
    Returns a Mealpy objective over a given subset

    Args:
        X: (n, d) subset of training points
    """    
    # Redefine gamma and classical kernel
    gamma = 1.0 / (N_QUBITS * X.var())
    K_classical = rbf_kernel(X, gamma=gamma)

    def objective(solution):
        """
        Objective function for Mealpy optimization.
        Mealpy will suggest one bitstring candidate (binary), and returns g for a certain solution.
        """    
        # Define the candidate
        candidate = MealpyCandidate(N_QUBITS, N_LAYERS, solution)

        # Compute quantum kernel matrix
        K_quantum = quantum_kernel_matrix(X, candidate)

        # Compute geometric difference
        g = geometric_difference(K_classical, K_quantum)

        return g
    return objective


## OPTION 2: HAAR DISTRIBUTION
def objective_optuna_generator_haar(
    X: np.ndarray, sample_mode: str = 'data'
) -> Callable[[optuna.Trial], float]:
    """
    Optuna objective scored by expressibility (KL divergence vs. the Haar fidelity distribution) instead of geometric difference.
    Lower KL = closer to Haar-random = more expressible.
    CHANGED***
    """
    gamma = 1.0 / (N_QUBITS * X.var())
    K_classical = rbf_kernel(X, gamma=gamma)

    def objective(trial):
        gates, angles = [], []
        for i in range(N_QUBITS * N_LAYERS):
            gate = trial.suggest_categorical(f"gate_{i}", ["H", "CNOT", "RX", "RY", "RZ", "I"])
            gates.append(gate)
            angle = trial.suggest_float(f"angle_{i}", 0, 2 * np.pi)
            angles.append(angle)

        candidate = OptunaCandidate(N_QUBITS, N_LAYERS, gates, angles)
        K_quantum = quantum_kernel_matrix(X, candidate)
        g = geometric_difference(K_classical, K_quantum)

        # optional: log KL for this trial without letting it drive the search
        kl = expressibility(candidate, N_QUBITS, n_samples=N_HAAR_SAMPLES,
                             n_bins=N_HAAR_BINS, sample_mode=sample_mode, X=X)
        trial.set_user_attr("kl_haar", kl)

        return g
    return objective


def objective_mealpy_generator_haar(
    X: np.ndarray, sample_mode: str = 'data'
) -> Callable[[np.ndarray], float]:
    """
    Mealpy objective: fitness = KL divergence vs. Haar.
    """
    def objective(solution):
        # Define the candidate
        candidate = MealpyCandidate(N_QUBITS, N_LAYERS, solution)

        # Compute expressibility
        kl = expressibility(
            candidate, N_QUBITS,
            n_samples=N_HAAR_SAMPLES, n_bins=N_HAAR_BINS,
            sample_mode=sample_mode, X=X,
        )
        return kl
    return objective