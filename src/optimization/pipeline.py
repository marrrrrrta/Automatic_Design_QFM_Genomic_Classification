import numpy as np

from config.experiments import ExperimentConfig
from .training_candidate import train_candidate_optuna, train_candidate_mealpy
from ..data.saving import load_parameter_train
from ..data.subsets import subset_random, subset_nystrom_global, subset_nystrom_stratified

def get_candidate(
        config: ExperimentConfig, 
        X_train: np.ndarray, y_train: np.ndarray
):
    """
    Decides whether to train or load a candidate from a previous run

    Args:
        config (ExperimentConfig): experiment configuration
        X_train (np.ndarray): training set
        y_train (np.ndarray): target set

    Returns:
        candidate (Candidate): best candidate
        K_q (np.ndarray): quantum kernel with best candidate
        K_c (np.ndarray): classical kernel 
        X_sub (np.ndarray): features training subset
        y_sub (np.ndarray): target training subset
    """    
    # If there's loading information, return it
    if config.load_from:
        data = load_parameter_train(config.load_from)
        return data['candidate'], data['K_quantum'], data['K_classical'], data['X_subset'], data['y_subset']

    # If not, train it
    subset_info = {
        'random': subset_random,
        'nystrom_global': subset_nystrom_global,
        'nystrom_stratified': subset_nystrom_stratified,
    }[config.subset_method]
    _, X_sub, y_sub = subset_info(X_train, y_train, **config.subset_kwargs)

    train_functions = {'optuna': train_candidate_optuna, 'mealpy': train_candidate_mealpy}[config.optimizer]
    _, candidate, K_q, K_c = train_functions(config.name, X_sub, y_sub, **config.optimizer_kwargs)

    return candidate, K_q, K_c, X_sub, y_sub

