from .candidates import Candidate
import pennylane as qml

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