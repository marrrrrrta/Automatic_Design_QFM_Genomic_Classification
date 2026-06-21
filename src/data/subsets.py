import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

def subset_random(X_train: np.ndarray, y_train: np.ndarray, n_sample: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a random subset
    """    
    rng = np.random.default_rng(seed)
    index = rng.choice(len(X_train), n_sample, replace=False)
    return index, X_train[index], y_train[index]


def nystrom_landmark_selection(X_train: np.ndarray, n_landmarks: int, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedy column-pivoting Nyström landmark selection.
    Returns: selected indices, selected X points
    """   
    # Initialize variables
    N = len(X_train)
    K = rbf_kernel(X_train, gamma=gamma)    # N×N kernel
    residual = np.diag(K).copy()             
    G = np.zeros((N, n_landmarks))          # Cholesky factor columns
    selected = []

    for step in range(n_landmarks):
        # Choose the least explained point
        j = int(np.argmax(residual))
        selected.append(j)
        point = residual[j]

        # New Cholesky column
        G[:, step] = (K[:, j] - G[:, :step] @ G[j, :step]) / np.sqrt(point)

        # Reduce residual by the variance explained by this new landmark
        residual -= G[:, step] ** 2
        residual  = np.maximum(residual, 0.0)

    return np.array(selected), X_train[np.array(selected)]

def subset_nystrom_global(X_train: np.ndarray, y_train: np.ndarray, n_qubits: int, n_landmarks: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a subset using the Nyström Landmark selection method
    Returns: selected indices, selected X and y points
    """
    # Redefine gamma with the subset used
    gamma = 1.0 / (n_qubits * X_train.var())

    # Obtain selected datapoints and indices
    global_idx, X_train_global = nystrom_landmark_selection(X_train, n_landmarks, gamma)

    # Recalculate y points
    y_train_global = y_train[global_idx]

    return global_idx, X_train_global, y_train_global

def subset_nystrom_stratified(X_train: np.ndarray, y_train: np.ndarray, n_qubits: int, n_pos: int, n_neg: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a subset using the Nyström Landmark selection method.
    Now the process is done separately for a certain number of points of either option.
    Returns: selected indices, selected X and y points
    """
    # Recompute gamma
    gamma = 1.0 / (n_qubits * X_train.var())

    # Indices
    idx_neg = np.where(y_train == -1)[0]
    idx_pos = np.where(y_train == +1)[0]    

    # Nystrom selection separately
    lm_neg, _ = nystrom_landmark_selection(X_train[idx_neg], n_landmarks=n_pos, gamma=gamma)
    lm_pos, _ = nystrom_landmark_selection(X_train[idx_pos], n_landmarks=n_neg, gamma=gamma)

    # Join them
    stratified_idx = np.concatenate([idx_neg[lm_neg], idx_pos[lm_pos]])

    return stratified_idx, X_train[stratified_idx], y_train[stratified_idx]