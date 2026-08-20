import numpy as np
import pennylane as qml
from pennylane import numpy as np
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


def sample_haar_fidelities(n_qubits: int, n_samples: int = 2000) -> np.ndarray:
    """
    Fidelities between pairs of states sampled from the Haar distribution.
    # TODO: (REVISE)
    """
    N = 2 ** n_qubits
    fids = np.empty(n_samples)
    for i in range(n_samples):
        # Generate a random unitary matrix and get the first column of each
        psi_a = haar_random_unitary(N)[:, 0]
        psi_b = haar_random_unitary(N)[:, 0]
        # Compute ⟨ψ_b|ψ_a⟩
        fids[i] = np.abs(np.vdot(psi_b, psi_a)) ** 2
    return fids


# test

if __name__ == "__main__":

    dev = qml.device("default.qubit", wires=1)
    mat = haar_random_unitary(2)

    # take unitary matrix -> parametrized rotations
    # is a list containing an operator
    op = qml.ops.one_qubit_decomposition(mat, wire=0, rotations="rot") # pyright: ignore[reportAttributeAccessIssue]

    # obtain parameters
    params = op[0].parameters
    params = np.array([params[0], params[1], params[2]], requires_grad=True)

    # circuit. measure the expectation value of the polyx operator
    @qml.qnode(dev)
    def haar_circuit(params):
        qml.Rot(*params, wires=0)
        return qml.expval(qml.PauliX(0))

    # optimization
    opt = qml.GradientDescentOptimizer(0.1)

    for _ in range(100):
        params, loss = opt.step_and_cost(haar_circuit, params)
        print (params, loss) 

    # the optimizer rotates the qubit's state towards the -1 eigenstate of PauliX
