import numpy as np
import pennylane as qml

from .candidates import Candidate

def build_circuit(candidate:Candidate, x: np.ndarray):
    """
    Generates a quantum circuit based on one proposed candidate
    Args:
        candidate (Candidate): proposed candidate.
        x (np.ndarray): input features.
    """    
    # Counter to track which gene we are reading
    gene_idx = 0

    for layer in range(candidate.n_layers):
        for qubit in range(candidate.n_qubits):
            # Obtain info from candidate -> gene -> gate + angle
            gene = candidate.get_ith_gene(gene_idx)
            info = gene.get_gene_info()

            gate = info['gate']
            angle = info['angle']

            # Get the feature value from the candidate
            feature_val = x[qubit]

            # Generate the circuit
            if gate == "H":
                qml.Hadamard(wires=qubit)
            elif gate == "CNOT":
                qml.CNOT(wires=[qubit, (qubit + 1) % candidate.n_qubits])
            elif gate == "RX":
                qml.RX(angle * feature_val, wires=qubit)
            elif gate == "RY":
                qml.RY(angle * feature_val, wires=qubit)
            elif gate == "RZ":
                qml.RZ(angle * feature_val, wires=qubit)
            else:  # gate == "I", do nothing
                pass

            # Add counter
            gene_idx += 1

## OPTION 2: HAAR DISTRIBUTION

def build_circuit_params(candidate: Candidate, theta: np.ndarray):
    """
    Same gate layout as build_circuit, but each rotation gate uses a
    free parameter theta[i] directly (no data encoding).
    Used for expressibility sampling, which only cares about structure.
    """
    gene_idx = 0
    for layer in range(candidate.n_layers):
        for qubit in range(candidate.n_qubits):
            gate = candidate.get_ith_gene(gene_idx).get_gene_info()['gate']
            if gate == "H":
                qml.Hadamard(wires=qubit)
            elif gate == "CNOT":
                qml.CNOT(wires=[qubit, (qubit + 1) % candidate.n_qubits])
            elif gate in ("RX", "RY", "RZ"):
                getattr(qml, gate)(theta[gene_idx], wires=qubit)
            gene_idx += 1

# src/circuit/building.py — addition
def circuit_unitary(candidate, theta, n_qubits) -> np.ndarray:
    """
    Returns the explicit N x N unitary matrix implemented by the
    candidate's structure for a given parameter vector theta.
    NOTE: (circuit -> unitary matrix)
          for circuit <- unitary matrix, use qml.QubitUnitary(U, wires=...)
    """
    return qml.matrix(build_circuit_params, wire_order=range(n_qubits))(candidate, theta)   # type: ignore[reportArgumentType]

def sample_circuit_fidelities_matrix(candidate, n_qubits, n_samples=2000, seed=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_params = candidate.n_qubits * candidate.n_layers
    fids = np.empty(n_samples)
    for i in range(n_samples):
        ta = rng.uniform(0, 2*np.pi, n_params)
        tb = rng.uniform(0, 2*np.pi, n_params)
        psi_a = circuit_unitary(candidate, ta, n_qubits)[:, 0]
        psi_b = circuit_unitary(candidate, tb, n_qubits)[:, 0]
        fids[i] = np.abs(np.vdot(psi_b, psi_a)) ** 2
    return fids