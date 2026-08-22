import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import optuna
from mealpy import BinaryVar, GA

from config.config import N_QUBITS, N_LAYERS, N_TRIALS, N_EPOCH, N_POPSIZE, N_PM
from .objectives import objective_optuna_generator_RD, objective_mealpy_generator_RD
from .objectives import objective_optuna_generator_HD, objective_mealpy_generator_HD
from .objectives import objective_optuna_angles_generator
from .objectives import objective_optuna_KTA_generator, objective_mealpy_generator_KTA
from ..circuit.candidates import OptunaCandidate, MealpyCandidate
from ..data.saving import save_parameter_train
from ..kernel.quantum_kernel import quantum_kernel_matrix


# –––– OPTION 1: RANDOM DISTRIBUTION (RD) ––––––––––––––––

def train_candidate_optuna(
    dataset_type: str, 
    X_train: np.ndarray, y_train: np.ndarray, 
    n_qubits: int = N_QUBITS, n_trials: int = N_TRIALS, n_layers: int = N_LAYERS
) -> tuple[float, OptunaCandidate, np.ndarray, np.ndarray]:
 
    # Train
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_optuna_generator_RD(X_train), n_trials=n_trials)
 
    # Extract best candidate
    best_gates = [study.best_params[f"gate_{i}"] for i in range(n_qubits * n_layers)]
    best_angles = [study.best_params[f"angle_{i}"] for i in range(n_qubits * n_layers)]
    best_candidate = OptunaCandidate(n_qubits, n_layers, best_gates, best_angles)
 
    # Quantum and classical kernels
    K_quantum_best = quantum_kernel_matrix(X_train, best_candidate)
    K_classical = rbf_kernel(X_train, gamma=1.0 / (n_qubits * X_train.var()))
 
    # Save results
    save_parameter_train(dataset_type, best_candidate, study.best_value, K_quantum_best, K_classical, X_train, y_train)
 
    return study.best_value, best_candidate, K_quantum_best, K_classical

def train_candidate_mealpy(
    dataset_type: str, 
    X_train: np.ndarray, y_train: np.ndarray, 
    n_qubits:int = N_QUBITS, n_layers:int = N_LAYERS, 
    n_epoch: int = N_EPOCH, n_pop_size: int = N_POPSIZE, n_pm: float = N_PM
):
 
    # Problem dictionary
    problem = {
        'obj_func': objective_mealpy_generator_RD(X_train),
        'bounds': BinaryVar(n_vars = n_qubits * n_layers * 5),
        'minmax': 'max'
    }
 
    # Run optimizer
    optimizer = GA.BaseGA(epoch = n_epoch, pop_size = n_pop_size, pm = n_pm)
    optimizer.solve(problem)
 
    # Extract best candidate
    best_bitstring = optimizer.g_best.solution # pyright: ignore[reportOptionalMemberAccess]
    best_candidate = MealpyCandidate(n_qubits, n_layers, best_bitstring)
 
    # New quantum and classical kernels
    K_quantum_best = quantum_kernel_matrix(X_train, best_candidate)
    K_classical = rbf_kernel(X_train, gamma= 1.0 / (n_qubits * X_train.var()))
 
    # Save results
    save_parameter_train(dataset_type, best_candidate, optimizer.g_best.target.fitness, K_quantum_best, K_classical, X_train, y_train) # pyright: ignore[reportOptionalMemberAccess, reportArgumentType]
 
    return optimizer.g_best.target.fitness, best_candidate, K_quantum_best, K_classical # pyright: ignore[reportOptionalMemberAccess]


# –––– OPTION 2: HAAR DISTRIBUTION (HD) ––––––––––––––––––

def train_candidate_optuna_haar(
    dataset_type: str,
    X_train: np.ndarray, y_train: np.ndarray,
    n_qubits: int = N_QUBITS, n_trials: int = N_TRIALS, n_layers: int = N_LAYERS,
) -> tuple[float, OptunaCandidate, np.ndarray, np.ndarray]:

    # Train
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_optuna_generator_HD(X_train), n_trials=n_trials)

    # Extract best candidate
    best_gates = [study.best_params[f"gate_{i}"] for i in range(n_qubits * n_layers)]
    best_angles = study.best_trial.user_attrs["angles"] 
    best_candidate = OptunaCandidate(n_qubits, n_layers, best_gates, best_angles)

    # Quantum and Classical kernels
    K_quantum_best = quantum_kernel_matrix(X_train, best_candidate)
    K_classical = rbf_kernel(X_train, gamma=1.0 / (n_qubits * X_train.var()))
    
    save_parameter_train(dataset_type, best_candidate, study.best_value, K_quantum_best, K_classical, X_train, y_train)

    return study.best_value, best_candidate, K_quantum_best, K_classical

def train_candidate_mealpy_haar(
    dataset_type: str,
    X_train: np.ndarray, y_train: np.ndarray,
    n_qubits: int = N_QUBITS, n_layers: int = N_LAYERS,
    n_epoch: int = N_EPOCH, n_pop_size: int = N_POPSIZE, n_pm: float = N_PM,
    sample_mode: str = 'data',
):
    ...

# –––– OPTION 3: HD + ANGLE TRAINING –––––––––––––––––––––

def train_candidate_optuna_angles(
    dataset_type: str, gates: list,
    X_train: np.ndarray, y_train: np.ndarray,
    n_qubits: int = N_QUBITS, n_trials: int = N_TRIALS, n_layers: int = N_LAYERS,
) -> tuple[float, OptunaCandidate, np.ndarray, np.ndarray]:
    """From a fixed gate structure, find the best angles"""

    # create study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_optuna_angles_generator(X_train, y_train, gates), n_trials=n_trials)

    best_angles = [study.best_params[f"angle_{i}"] for i in range(n_qubits * n_layers)]
    best_candidate = OptunaCandidate(n_qubits, n_layers, gates, best_angles)

    K_quantum_best = quantum_kernel_matrix(X_train, best_candidate)
    K_classical = rbf_kernel(X_train, gamma=1.0 / (n_qubits * X_train.var()))

    save_parameter_train(dataset_type, best_candidate, study.best_value, K_quantum_best, K_classical, X_train, y_train)
    return study.best_value, best_candidate, K_quantum_best, K_classical

def train_candidate_optuna_opt3(
    dataset_type: str,
    X_train: np.ndarray, y_train: np.ndarray,
    n_qubits: int = N_QUBITS, n_layers: int = N_LAYERS,
    stage1_trials: int = N_TRIALS, stage2_trials: int = N_TRIALS
) -> tuple[float, OptunaCandidate, np.ndarray, np.ndarray]:
    """
    Two stages:
        1. Finds gate structure using Haar-sampled angles
        2. Finds the best angles via kernel-target alignment
    """

    # STAGE 1: finding circuit structure
    _, stage1_candidate, _, _ = train_candidate_optuna_haar(
        f"{dataset_type}_stage1", X_train, y_train,
        n_qubits=n_qubits, n_trials=stage1_trials, n_layers=n_layers,
    )
    gates = [gene.gate for gene in stage1_candidate.genes]

    # STAGE 2: finding angles
    return train_candidate_optuna_angles(
        dataset_type, gates, X_train, y_train,
        n_qubits=n_qubits, n_trials=stage2_trials, n_layers=n_layers,
    )

# –––– OPTION 4: KTA –––––––––––––––––––––––––––––––––––––

def train_candidate_optuna_kta(subset_type, X_subset, y_subset, n_qubits=N_QUBITS, n_trials=N_TRIALS, n_layers=N_LAYERS):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_optuna_KTA_generator(X_subset, y_subset), n_trials=n_trials)
    best_gates = [study.best_params[f"gate_{i}"] for i in range(n_qubits * n_layers)]
    best_angles = [study.best_params[f"angle_{i}"] for i in range(n_qubits * n_layers)]
    best_candidate = OptunaCandidate(n_qubits, n_layers, best_gates, best_angles)
    K_quantum_best = quantum_kernel_matrix(X_subset, best_candidate)
    K_classical = rbf_kernel(X_subset, gamma=1.0 / (n_qubits * X_subset.var()))
    save_parameter_train(subset_type, best_candidate, study.best_value, K_quantum_best, K_classical, X_subset, y_subset)
    return study.best_value, best_candidate, K_quantum_best, K_classical

def train_candidate_mealpy_kta(subset_type, X_subset, y_subset, n_qubits=N_QUBITS, n_layers=N_LAYERS, n_epoch=N_EPOCH, n_pop_size=N_POPSIZE, n_pm=N_PM):
    problem = {
        'obj_func': objective_mealpy_generator_KTA(X_subset, y_subset),
        'bounds': BinaryVar(n_vars=n_qubits * n_layers * 5),
        'minmax': 'max'
    }
    optimizer = GA.BaseGA(epoch=n_epoch, pop_size=n_pop_size, pm=n_pm)
    optimizer.solve(problem)
    best_bitstring = optimizer.g_best.solution # pyright: ignore[reportOptionalMemberAccess]
    best_candidate = MealpyCandidate(n_qubits, n_layers, best_bitstring)
    K_quantum_best = quantum_kernel_matrix(X_subset, best_candidate)
    K_classical = rbf_kernel(X_subset, gamma=1.0 / (n_qubits * X_subset.var()))
    save_parameter_train(subset_type, best_candidate, optimizer.g_best.target.fitness, K_quantum_best, K_classical, X_subset, y_subset) # pyright: ignore[reportOptionalMemberAccess, reportArgumentType]
    return optimizer.g_best.target.fitness, best_candidate, K_quantum_best, K_classical # pyright: ignore[reportOptionalMemberAccess]