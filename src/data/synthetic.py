import numpy as np
import pennylane as qml
from scipy.stats import unitary_group

from config.config import N_QUBITS, N_LAYERS
from src.utils.visuals import graph_dataset
from src.circuit.candidates import OptunaCandidate
from src.circuit.building import build_circuit

dim = 2 ** N_QUBITS
dev = qml.device("default.qubit", wires=N_QUBITS)

# –––– Havlíček et al. 2019 –––––––––––––––––––––––––––––––

def parity_diagonal(n):
    """Gives the diagonal of ZxZx...xZ, n times"""
    bits = np.arange(2 ** n)
    parity = np.array([bin(b).count("1") % 2 for b in bits])
    return 1 - 2 * parity

def Havlicek_synthetic_dataset(n_per_class: dict |int = 20, delta=0.3, seed=None):
    """Generates a synthetic dataset based on Havlíček et al. 2019"""

    def feature_map(x):
        """Encodes classical data into a quantum state"""
        # add superposition (H, so it's sensitive to phase) and feature information
        for i in range(N_QUBITS):
            # fig 1.b
            qml.Hadamard(wires=i)
            qml.RZ(2 * x[i], wires=i)  # multiplied by 2 to contrarrest qml's 1/2

        # 2-qubit interaction encodes the  product feature (φij(x)= (π-xi)(π-xj)) into the state.
        # makes it more difficult for classical methods
        for i in range(N_QUBITS):
            for j in range(i+1, N_QUBITS):
                # fig 1.c
                qml.CNOT(wires=[i, j])
                qml.RZ(2 * (np.pi - x[i]) * (np.pi - x[j]), wires=j)
                qml.CNOT(wires=[i, j])

    @qml.qnode(dev)
    def get_state(x):
        """Runs the circuit and returns the statevector"""
        feature_map(x)
        return qml.state()

    # support for even or uneven classes
    if isinstance(n_per_class, int):
        n_per_class = {1: n_per_class, -1: n_per_class}

    # set random unitaries
    rng = np.random.default_rng(seed)
    V = unitary_group.rvs(dim, random_state=seed)             # random unitary
    O = V.conj().T @ np.diag(parity_diagonal(N_QUBITS)) @ V   # V+ f V

    # generate labels
    X, y, counts = [], [], {1: 0, -1: 0}
    while counts[1] < n_per_class[1] or counts[-1] < n_per_class[-1]:
        x = rng.uniform(0, 2 * np.pi, size=N_QUBITS)
        val = np.real(np.conj(get_state(x)) @ O @ get_state(x))  # <Φ|O|Φ>

        # classifier
        if val >= delta and counts[1] < n_per_class[1]:
            X.append(x); y.append(1); counts[1] += 1
        elif val <= -delta and counts[-1] < n_per_class[-1]:
            X.append(x); y.append(-1); counts[-1] += 1

    return np.array(X), np.array(y)

# TODO: what would happen if we rotate on another axis?


# –––– Havlicek modified ––––––––––––––––––––––––––––––––––
# replaces how the information is encoded in the dataset (feature_map())
# with how we encode it (candidate circuit)

def ModHav_synthetic_dataset(
    n_per_class:dict |int = 20, n_qubits = N_QUBITS, n_layers = N_LAYERS, 
    delta = 0.3, seed: int | None = None
):
    GATES = ("H", "CNOT", "RX", "RY", "RZ", "I")
    ANGLES = (np.pi, np.pi / 2, np.pi / 4, np.pi / 8)

    rng = np.random.default_rng(seed)
    dim = 2 ** n_qubits
    dev = qml.device("default.qubit", wires=n_qubits)

    # support for even or uneven classes
    if isinstance(n_per_class, int):
        n_per_class = {1: n_per_class, -1: n_per_class}

    # define random candidate (w/ optuna or mealpy, indistinct)
    gates = list(rng.choice(GATES, size=n_qubits * n_layers))
    angles = list(rng.choice(ANGLES, size=n_qubits * n_layers))
    enc_candidate = OptunaCandidate(n_qubits, n_layers, gates, angles)

    @qml.qnode(dev)
    def get_state(x):
        build_circuit(enc_candidate, x)
        return qml.state()

    # set random unitaries
    V = unitary_group.rvs(dim, random_state=seed)
    O = V.conj().T @ np.diag(parity_diagonal(n_qubits)) @ V

    # generate labels
    X, y, counts = [], [], {1: 0, -1: 0}
    while counts[1] < n_per_class[1] or counts[-1] < n_per_class[-1]:
        x = rng.uniform(0, 2 * np.pi, size=N_QUBITS)
        val = np.real(np.conj(get_state(x)) @ O @ get_state(x))  # <Φ|O|Φ>
    
        # classifier
        if val >= delta and counts[1] < n_per_class[1]:
            X.append(x); y.append(1); counts[1] += 1
        elif val <= -delta and counts[-1] < n_per_class[-1]:
            X.append(x); y.append(-1); counts[-1] += 1
    
    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = Havlicek_synthetic_dataset(seed=45)
    graph_dataset("Havlicek, even", X, y)

    X, y = Havlicek_synthetic_dataset(n_per_class={1:10, -1:30}, seed=45)
    graph_dataset("Havlicek, uneven", X, y)