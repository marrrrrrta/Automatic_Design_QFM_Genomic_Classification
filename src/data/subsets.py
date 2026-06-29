import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

def subset_random(
    X_train: np.ndarray, y_train: np.ndarray, 
    n_sample: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a random subset

    Args:
        X_train (np.ndarray): features training set
        y_train (np.ndarray): labels training set
        n_sample (int): number of samples to select
        seed (int | None): random seed for reproducibility

    Returns:
        selected indices, selected X and y points
    """    
    rng = np.random.default_rng(seed)
    index = rng.choice(len(X_train), n_sample, replace=False)
    return index, X_train[index], y_train[index]


def nystrom_landmark_selection(
    X_train: np.ndarray, 
    n_landmarks: int, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedy column-pivoting Nyström landmark selection.

    Args:
        X_train (np.ndarray): features training set
        n_landmarks (int): number of landmarks to select
        gamma (float): RBF kernel parameter

    Returns:
        selected indices, selected X points
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


def subset_nystrom_global(
    X_train: np.ndarray, y_train: np.ndarray, 
    n_qubits: int, n_landmarks: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a subset using the Nyström Landmark selection method

    Args:
        X_train (np.ndarray): features training set
        y_train (np.ndarray): labels training set
        n_qubits (int): number of qubits used in the quantum kernel
        n_landmarks (int): number of landmarks to select

    Returns:
        selected indices, selected X and y points
    """
    # Redefine gamma with the subset used
    gamma = 1.0 / (n_qubits * X_train.var())

    # Obtain selected datapoints and indices
    global_idx, X_train_global = nystrom_landmark_selection(X_train, n_landmarks, gamma)

    # Recalculate y points
    y_train_global = y_train[global_idx]

    return global_idx, X_train_global, y_train_global


def subset_nystrom_stratified(
    X_train: np.ndarray, y_train: np.ndarray, 
    n_qubits: int, n_per_class: int | dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Nyström Landmark selection method (stratified)
    Landmark selection is performed independently within each class, then the results are 
    concatenated. This ensures all classes are represented in the optimisation subset regardless of class imbalance.

    Args:
        X_train (np.ndarray): features training set
        y_train (np.ndarray): labels training set
        n_qubits (int): number of qubits used in the quantum kernel
        n_per_class (int | dict): number of landmarks to select per class. 
                int: same number for all classes
                dict: specify number of landmarks for each class, e.g. {-1: 10, +1: 20}
    Returns:
        selected indices, selected X and y points
    """    
    # Recompute gamma
    gamma = 1.0 / (n_qubits * X_train.var())

    # Find classes of the subset
    classes = np.unique(y_train)
    indices = []

    # Converts n_per_class in dict if an int is passed
    if isinstance(n_per_class, int):
        n_per_class = {cls: n_per_class for cls in classes}

    for cls in np.unique(y_train):            # iterates over every distinct class label (however many there are)
        # indices where this class is
        idx_cls = np.where(y_train == cls)[0]

        # get the landmark selection for every class separately. use only the X_train points that form that class
        landmarks, _ = nystrom_landmark_selection(X_train[idx_cls], n_per_class[cls], gamma)
        indices.append(idx_cls[landmarks])
    
    idx = np.concatenate(indices)
    return idx, X_train[idx], y_train[idx]