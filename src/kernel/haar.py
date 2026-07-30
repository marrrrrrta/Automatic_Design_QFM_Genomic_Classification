import numpy as np
from numpy.linalg import qr

def haar_random_unitary(N: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generates a Haar-random matrix using the QR decomposition (Mezzadri, 2006).
    (From 'Understanding Haar measure' tutorial)
    """
    # Generates NxN matrix with complex numbers (a+ib) where a and b are normally distributed with mean 0 and variance 1
    Z = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))

    # Compute QR decomposition
    Q, R = qr(Z)

    # Compute diagonal matrix
    Lambda = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(N)])

    # Compute Q' = Q * Lambda (which is Haar-random)
    return np.dot(Q, Lambda)


def sample_haar_fidelities(n_qubits: int, n_samples: int = 2000, seed=None) -> np.ndarray:
    """
    Fidelities between pairs of states prepared by independent, literal
    Haar-random unitaries (reference distribution).
    """
    rng = np.random.default_rng(seed)
    N = 2 ** n_qubits
    fids = np.empty(n_samples)
    for i in range(n_samples):
        # Generate a random unitary matrix and get the first column of each
        psi_a = haar_random_unitary(N, rng)[:, 0]
        psi_b = haar_random_unitary(N, rng)[:, 0]
        # Compute ⟨ψ_b|ψ_a⟩
        fids[i] = np.abs(np.vdot(psi_b, psi_a)) ** 2
    return fids