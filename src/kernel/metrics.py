import numpy as np
from scipy.linalg import sqrtm

def geometric_difference(K_classical, K_quantum, eps=1e-7):
    """
    Compute the geometric difference function, defined as:

        g = sqrt( ||sqrt(K_q) * inv(K_c) * sqrt(K_q) ||_infty )

    Args:
        K_classical (_type_): classical Gram matrix
        K_quantum (_type_): quantum Gram matrix
        eps (_type_, optional): _description_. Defaults to 1e-7.
    """    
    # Number of training samples
    N = K_classical.shape[0]

    # Kernel matrices (with a small diagonal to avoid numerical singularity)
    Kc_inv = np.linalg.inv(K_classical + eps*np.eye(N)) 
    sqrt_Kq = sqrtm(K_quantum + eps * np.eye(N)).real

    # Compute g
    M = sqrt_Kq @ Kc_inv @ sqrt_Kq

    # The infinity norm of a symmetric matrix is its largest eigenvalue
    # eigenvalsh looks for eigenvalues in symmetric matrices
    eigenval_M = np.linalg.eigvalsh(M)
    max_eigenval_M = np.max(eigenval_M)

    return np.sqrt(np.maximum(max_eigenval_M, 0.0))


def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """
    Measures how well K's similarity structure matches the class labels y.
    - Higher = better alignment with same-class vs different-class pairs
    - Used to train the gate's angles
    """
    y = np.asarray(y).reshape(-1, 1)
    yyT = y @ y.T
    return float(np.sum(K * yyT) / (np.linalg.norm(K) * np.linalg.norm(yyT)))