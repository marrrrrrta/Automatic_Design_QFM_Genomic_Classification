import numpy as np
import pennylane as qml

from ..circuit.candidates import Candidate
from ..circuit.building import build_circuit
from config.config import N_QUBITS

# setup quantum circuit
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev)

def quantum_kernel_circuit(x1: np.ndarray, x2: np.ndarray, candidate: Candidate):
    build_circuit(candidate, x1)
    qml.adjoint(build_circuit)(candidate, x2)
    return qml.probs(wires=range(candidate.n_qubits))

def quantum_kernel(x1: np.ndarray, x2: np.ndarray, candidate:Candidate
) -> float:
    """
    Computes the quantum kernel element given two datapoints x1 and x2.

    Args:
        x1 (1D array): one data point, shape (n_qubits,)
        x2 (1D array): one data point, shape (n_qubits,)
        candidate (Candidate): proposed candidate
    Returns:
        float: fidelity of the circuit
    """    
    return float(quantum_kernel_circuit(x1, x2, candidate)[0])

def quantum_kernel_matrix(X: np.ndarray, candidate:Candidate
) -> np.ndarray:
    """
    Computes the Quantum Kernel Matrix

    Args:
        X (list/array): classical datapoints
        candidate (Candidate): proposed candidate

    Returns:
        array: quantum kernel matrix
    """    
    # Initialize dimensions and skeleton QK matrix
    N = len(X)
    K_quantum = np.zeros((N, N))

    for i in range(N):
        for j in range(i, N):
            value = quantum_kernel(X[i], X[j], candidate)
            K_quantum[i, j] = value
            K_quantum[j, i] = value
    
    return K_quantum

def quantum_kernel_matrix_cross(X1: np.ndarray, X2: np.ndarray, candidate: Candidate
) -> np.ndarray:
    """
    Computes the rectangular cross-kernel matrix K(X1, X2). 
    X1=test set, X2=training landmarks.
    Returns shape (len(X1), len(X2)).
    """
    N1, N2 = len(X1), len(X2)
    K = np.zeros((N1, N2))
    for i in range(N1):
        for j in range(N2):
            K[i, j] = quantum_kernel(X1[i], X2[j], candidate)
    return K