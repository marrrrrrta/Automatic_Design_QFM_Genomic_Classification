import numpy as np
import pennylane as qp
from pprint import pprint
from numpy.linalg import qr

def haar_random_unitary(N: int) -> np.ndarray:
    """
    Generates a Haar-random matrix using the QR decomposition.
    (From 'Understanding Haar measure' tutorial)
    """
    # 1. Generates NxN matrix with complex numbers (a+ib) where a and b are normally distributed with mean 0 and variance 1
    A, B = np.random.normal(size=(N, N)), np.random.normal(size=(N, N))
    Z = A + 1j * B

    # 2. Compute QR decomposition (Q:orthogonal, R:triangular)
    Q, R = qr(Z)

    # 3. Compute diagonal matrix
    Lambda = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(N)])

    # 4. Compute Q' = Q * Lambda (which is Haar-random)
    return np.dot(Q, Lambda)


U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # Hadamard
decomp = qp.ops.one_qubit_decomposition(U, 0, rotations='rot')
print("for rotations='rot', with |+>")
pprint(decomp)

U = haar_random_unitary(2)
decomp = qp.ops.one_qubit_decomposition(U, 0, rotations='rot')
print("for rotations='rot', with Haar")
pprint(decomp)