import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import optuna
from mealpy import BinaryVar, GA

from config.config import N_QUBITS, N_LAYERS, N_TRIALS, N_EPOCH, N_POPSIZE, N_PM
from .objectives import objective_optuna_generator_RD, objective_mealpy_generator_RD
from .objectives import objective_optuna_generator_HD, objective_mealpy_generator_HD
from ..circuit.candidates import OptunaCandidate, MealpyCandidate
from ..data.saving import save_parameter_train
from ..kernel.quantum_kernel import quantum_kernel_matrix


# –––– OPTION 1: RANDOM DISTRIBUTION (RD) ––––––––––––––––

def train_candidate_optuna(
    subset_type: str, 
    X_subset: np.ndarray, y_subset: np.ndarray, 
    n_qubits: int = N_QUBITS, n_trials: int = N_TRIALS, n_layers: int = N_LAYERS
) -> tuple[float, OptunaCandidate, np.ndarray, np.ndarray]:
 
    # Train
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_optuna_generator_RD(X_subset), n_trials=n_trials)
 
    # Extract best candidate
    best_gates = [study.best_params[f"gate_{i}"] for i in range(n_qubits * n_layers)]
    best_angles = [study.best_params[f"angle_{i}"] for i in range(n_qubits * n_layers)]
    best_candidate = OptunaCandidate(n_qubits, n_layers, best_gates, best_angles)
 
    # Quantum and classical kernels
    K_quantum_best = quantum_kernel_matrix(X_subset, best_candidate)
    K_classical = rbf_kernel(X_subset, gamma=1.0 / (n_qubits * X_subset.var()))
 
    # Save results
    save_parameter_train(subset_type, best_candidate, study.best_value, K_quantum_best, K_classical, X_subset, y_subset)
 
    return study.best_value, best_candidate, K_quantum_best, K_classical

def train_candidate_mealpy(
    subset_type: str, 
    X_subset: np.ndarray, y_subset: np.ndarray, 
    n_qubits:int = N_QUBITS, n_layers:int = N_LAYERS, 
    n_epoch: int = N_EPOCH, n_pop_size: int = N_POPSIZE, n_pm: float = N_PM
):
 
    # Problem dictionary
    problem = {
        'obj_func': objective_mealpy_generator_RD(X_subset),
        'bounds': BinaryVar(n_vars = n_qubits * n_layers * 5),
        'minmax': 'max'
    }
 
    # Run optimizer
    optimizer = GA.BaseGA(epoch = n_epoch, pop_size = n_pop_size, pm = n_pm)
    optimizer.solve(problem)
 
    # Extract best candidate
    best_bitstring = optimizer.g_best.solution
    best_candidate = MealpyCandidate(n_qubits, n_layers, best_bitstring)
 
    # New quantum and classical kernels
    K_quantum_best = quantum_kernel_matrix(X_subset, best_candidate)
    K_classical = rbf_kernel(X_subset, gamma= 1.0 / (n_qubits * X_subset.var()))
 
    # Save results
    save_parameter_train(subset_type, best_candidate, optimizer.g_best.target.fitness, K_quantum_best, K_classical, X_subset, y_subset)
 
    return optimizer.g_best.target.fitness, best_candidate, K_quantum_best, K_classical



# –––– OPTION 2: HAAR DISTRIBUTION (HD) ––––––––––––––––––

def train_candidate_optuna_haar(
    subset_type: str,
    X_subset: np.ndarray, y_subset: np.ndarray,
    n_qubits: int = N_QUBITS, n_trials: int = N_TRIALS, n_layers: int = N_LAYERS,
) -> tuple[float, OptunaCandidate, np.ndarray, np.ndarray]:

    # Train
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_optuna_generator_HD(X_subset), n_trials=n_trials)

    # Extract best candidate
    # Note: angles are NOT in study.best_params because they're sampled from Haar distribution
    # during objective evaluation, not suggested to Optuna. Regenerate them here.
    from .objectives import _haar_angle
    best_gates = [study.best_params[f"gate_{i}"] for i in range(n_qubits * n_layers)]
    best_angles = [_haar_angle() for i in range(n_qubits * n_layers)]
    best_candidate = OptunaCandidate(n_qubits, n_layers, best_gates, best_angles)

    # Quantum and Classical kernels
    K_quantum_best = quantum_kernel_matrix(X_subset, best_candidate)
    K_classical = rbf_kernel(X_subset, gamma=1.0 / (n_qubits * X_subset.var()))


    # NOTE: the saved "g_best" now holds a KL divergence, not a geometric difference — relabel accordingly wherever it's plotted downstream.
    save_parameter_train(subset_type, best_candidate, study.best_value, K_quantum_best, K_classical, X_subset, y_subset)

    return study.best_value, best_candidate, K_quantum_best, K_classical


def train_candidate_mealpy_haar(
    subset_type: str,
    X_subset: np.ndarray, y_subset: np.ndarray,
    n_qubits: int = N_QUBITS, n_layers: int = N_LAYERS,
    n_epoch: int = N_EPOCH, n_pop_size: int = N_POPSIZE, n_pm: float = N_PM,
    sample_mode: str = 'data',
):
    ...