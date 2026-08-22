from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import optuna
import pennylane as qml
from typing import Callable

from config.config import N_QUBITS, N_LAYERS
from ..circuit.candidates import OptunaCandidate, MealpyCandidate
from ..kernel.quantum_kernel import quantum_kernel_matrix
from ..kernel.metrics import geometric_difference, kernel_target_alignment
from ..kernel.haar import haar_random_unitary


# –––– OPTION 1: RANDOM DISTRIBUTION (RD) ––––––––––––––––

def objective_optuna_generator_RD(X: np.ndarray
) -> Callable[[optuna.Trial], float]:
    """
    Returns an Optuna objective over a given dataset.
    Uses a random distribution to sample candidates from the search space.
    Uses geometric difference as the objective function.
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

def objective_mealpy_generator_RD(X: np.ndarray
) -> Callable[[np.ndarray], float]:
    """
    Returns a Mealpy objective over a given dataset
    Uses a random distribution to sample candidates from the search space.
    Uses geometric difference as the objective function.
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


# –––– OPTION 2: HAAR DISTRIBUTION (HD) ––––––––––––––––––

def _haar_angle() -> float:
    """One rotation angle sampled from Haar-random single-qubit unitary.
    From 'The Haar Measure' tutorial

    NOTE: qml.Rot(phi, theta, omega) = RZ(omega) · RY(theta) · RZ(phi)
          I extracted theta, which is from [0,π), but i'm not sure yet
    """
    mat = haar_random_unitary(2)
    op = qml.ops.one_qubit_decomposition(mat, wire=0, rotations="rot")
    return float(op[0].parameters[1]) 

def objective_optuna_generator_HD(
    X: np.ndarray
) -> Callable[[optuna.Trial], float]:
    """
    Returns an Optuna objective over a given subset
    Uses the Haar distribution to sample angles from.
    Uses geometric difference as the objective function.
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
            # Now, sampled from the Haar distribution, not optuna suggest
            # NOTE: this way optuna never learns. the input is always random
            angles.append(_haar_angle())

        # save angles used
        trial.set_user_attr("angles", angles)
        
        # Define the candidate
        candidate = OptunaCandidate(N_QUBITS, N_LAYERS, gates, angles)

        # Compute quantum kernel matrix
        K_quantum = quantum_kernel_matrix(X, candidate)

        # Compute geometric difference
        g = geometric_difference(K_classical, K_quantum)

        return g
    return objective

def objective_mealpy_generator_HD(
    X: np.ndarray
) -> Callable[[np.ndarray], float]:
    """
    Returns a Mealpy objective over a given subset
    Uses the Haar distribution to sample angles from.
    Uses geometric difference as the objective function.
    Args:
        X: (n, d) subset of training points
    """
    ...


# –––– OPTION 3: HD + ANGLE TRAINING –––––––––––––––––––––

def objective_optuna_angles_generator(
        X: np.ndarray, y: np.ndarray, gates: list
) -> Callable[[optuna.Trial], float]:
    """
    Gate structure is fixed (from stage 1). Only angles are optimized,
    against kernel-target alignment on the training subset.
    Uses geometric difference as the objective function.
    Args:
        X: (n, d) subset of training points
        y: labels for X
        gates: fixed gate list, one per gene
    """
    def objective(trial):
        angles = [trial.suggest_float(f"angle_{i}", 0, 2 * np.pi) for i in range(len(gates))]
        candidate = OptunaCandidate(N_QUBITS, N_LAYERS, gates, angles)
        K_quantum = quantum_kernel_matrix(X, candidate)
        return kernel_target_alignment(K_quantum, y)
    return objective


# –––– OPTION 4: KTA –––––––––––––––––––––––––––––––––––––

def objective_optuna_KTA_generator(X, y):
    """
    Returns an Optuna objective over a given dataset.
    Uses a random distribution to sample candidates from the search space.
    The objective function is now Kernel Target Alignment.
    """
    def objective(trial):
        gates, angles = [], []
        for i in range(N_QUBITS * N_LAYERS):
            gates.append(trial.suggest_categorical(f"gate_{i}", ["H", "CNOT", "RX", "RY", "RZ", "I"]))
            angles.append(trial.suggest_float(f"angle_{i}", 0, 2 * np.pi))
        candidate = OptunaCandidate(N_QUBITS, N_LAYERS, gates, angles)
        K_quantum = quantum_kernel_matrix(X, candidate)
        return kernel_target_alignment(K_quantum, y)
    return objective


def objective_mealpy_generator_KTA(X, y):
    """
    Returns a Mealpy objective over a given dataset.
    Uses a random distribution to sample candidates from the search space.
    The objective function is now Kernel Target Alignment.
    """
    def objective(solution):
        candidate = MealpyCandidate(N_QUBITS, N_LAYERS, solution)
        K_quantum = quantum_kernel_matrix(X, candidate)
        return kernel_target_alignment(K_quantum, y)
    return objective