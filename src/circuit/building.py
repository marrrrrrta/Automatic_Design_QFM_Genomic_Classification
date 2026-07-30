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