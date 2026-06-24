import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import optuna
from mealpy import BinaryVar, GA

from config.config import N_QUBITS, N_LAYERS, N_TRIALS, N_EPOCH, N_POP_SIZE, N_PM
from .objectives import objective_optuna_generator, objective_mealpy_generator
from ..circuit.candidates import OptunaCandidate, MealpyCandidate
from ..data.saving import save_parameter_train
from ..kernel.quantum_kernel import quantum_kernel_matrix

def train_candidate_optuna(
    subset_type: str, 
    X_subset: np.ndarray, y_subset: np.ndarray, 
    n_qubits: int = N_QUBITS, n_trials: int = N_TRIALS, n_layers: int = N_LAYERS
) -> tuple[float, OptunaCandidate, np.ndarray, np.ndarray]:

    # Train
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_optuna_generator(X_subset), n_trials=n_trials)

    # Extract best candidate
    best_gates = [study.best_params[f"gate_{i}"] for i in range(n_qubits * n_layers)]
    best_angles = [study.best_params[f"angle_{i}"] for i in range(n_qubits * n_layers)]
    best_candidate = OptunaCandidate(n_qubits, n_layers, best_gates, best_angles)

    # Quantum and classical kernels
    K_quantum_best = quantum_kernel_matrix(X_subset, best_candidate)
    K_classical = rbf_kernel(X_subset, gamma=1.0 / (n_qubits * X_subset.var()))

    # Save results
    save_parameter_train(f'training_optuna_{subset_type}_{n_trials}', best_candidate, study.best_value, K_quantum_best, K_classical, X_subset, y_subset)

    return study.best_value, best_candidate, K_quantum_best, K_classical

def train_candidate_mealpy(
    subset_type: str, 
    X_subset: np.ndarray, y_subset: np.ndarray, 
    n_qubits:int = N_QUBITS, n_layers:int = N_LAYERS, 
    n_epoch: int = N_EPOCH, n_pop_size: int = N_POP_SIZE, n_pm: float = N_PM
):

    # Problem dictionary
    problem = {
        'obj_func': objective_mealpy_generator((X_subset)),
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
    pm_str = str(n_pm).split('.')[1]
    save_parameter_train(f'training_mealpy_{subset_type}_{n_epoch}ep_{n_pop_size}pop_{pm_str}pm', best_candidate, optimizer.g_best.target.fitness, K_quantum_best, K_classical, X_subset, y_subset)

    return optimizer.g_best.target.fitness, best_candidate, K_quantum_best, K_classical