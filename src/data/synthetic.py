import numpy as np
import pennylane as qml
from scipy.stats import unitary_group

from config.config import N_QUBITS, N_LAYERS
from src.utils.visuals import graph_dataset
from src.circuit.candidates import OptunaCandidate
from src.circuit.building import build_circuit
from sklearn.metrics.pairwise import rbf_kernel
from src.kernel.metrics import geometric_difference


# –––– Havlíček et al. 2019 –––––––––––––––––––––––––––––––

def parity_diagonal(n):
    """Gives the diagonal of ZxZx...xZ, n times"""
    bits = np.arange(2 ** n)
    parity = np.array([bin(b).count("1") % 2 for b in bits])
    return 1 - 2 * parity

def Havlicek_synthetic_dataset(
    n_per_class: dict |int = 20, delta=0.3, seed=None,
    n_qubits: int = N_QUBITS,
):
    """Generates a synthetic dataset based on Havlíček et al. 2019"""

    def feature_map(x):
        """Encodes classical data into a quantum state"""
        # add superposition (H, so it's sensitive to phase) and feature information
        for i in range(n_qubits):
            # fig 1.b
            qml.Hadamard(wires=i)
            qml.RZ(2 * x[i], wires=i)  # multiplied by 2 to contrarrest qml's 1/2

        # 2-qubit interaction encodes the  product feature (φij(x)= (π-xi)(π-xj)) into the state.
        # makes it more difficult for classical methods
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                # fig 1.c
                qml.CNOT(wires=[i, j])
                qml.RZ(2 * (np.pi - x[i]) * (np.pi - x[j]), wires=j)
                qml.CNOT(wires=[i, j])

    dim = 2 ** n_qubits
    dev = qml.device("default.qubit", wires=n_qubits)
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
    attempts, max_attempts = 0, 20000
    X, y, counts = [], [], {1: 0, -1: 0}
    while counts[1] < n_per_class[1] or counts[-1] < n_per_class[-1]:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"delta={delta} too high, only got {counts} after {attempts} tries")
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
    delta = 0.3, seed: int | None = None, enc_candidate: OptunaCandidate | None = None
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
    if enc_candidate is None:
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
    attempts, max_attempts = 0, 20000
    X, y, counts = [], [], {1: 0, -1: 0}
    while counts[1] < n_per_class[1] or counts[-1] < n_per_class[-1]:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"delta={delta} too high, only got {counts} after {attempts} tries")
        x = rng.uniform(0, 2 * np.pi, size=N_QUBITS)
        val = np.real(np.conj(get_state(x)) @ O @ get_state(x))  # <Φ|O|Φ>
    
        # classifier
        if val >= delta and counts[1] < n_per_class[1]:
            X.append(x); y.append(1); counts[1] += 1
        elif val <= -delta and counts[-1] < n_per_class[-1]:
            X.append(x); y.append(-1); counts[-1] += 1
    
    return np.array(X), np.array(y)


# –––– Claude –––––––––––––––––––––––––––––––––––––––––––––
def KernelGap_synthetic_dataset(
    n_per_class: dict | int = 20, n_qubits = N_QUBITS, n_layers = N_LAYERS,
    datapoints: int | None = None, seed: int | None = None, skipfactor: int = 4,
    gen_candidate: OptunaCandidate | None = None
):
    """
    Tailored dataset.
    """
    GATES = ("H", "CNOT", "RX", "RY", "RZ", "I")
    ANGLES = (np.pi, np.pi / 2, np.pi / 4, np.pi / 8)

    rng = np.random.default_rng(seed)
    dev = qml.device("default.qubit", wires=n_qubits)

    if isinstance(n_per_class, int):
        n_per_class = {1: n_per_class, -1: n_per_class}
    if datapoints is None:
        datapoints = 8 * (n_per_class[1] + n_per_class[-1])

    # Generate random candidate if none is provided
    if gen_candidate is None:
        gates = list(rng.choice(GATES, size=n_qubits * n_layers))
        angles = list(rng.choice(ANGLES, size=n_qubits * n_layers))
        gen_candidate = OptunaCandidate(n_qubits, n_layers, gates, angles)

    @qml.qnode(dev)
    def get_state(x):
        build_circuit(gen_candidate, x)
        return qml.state()

    # sample datapoints uniformly in [0, 2π]^n_qubits (no labels)
    X_pool = rng.uniform(0, 2 * np.pi, size=(datapoints, n_qubits)) # type: ignore
    states = np.array([get_state(x) for x in X_pool])

    # compute gram matrix (NxN positive semi-definite)
    K_q = np.abs(states.conj() @ states.T) ** 2

    # eigendecompose. the largest eigenvalue is the one where the qfm spreads data out the most
    # easiest to draw a boundary around
    eigvals, eigvecs = np.linalg.eigh(K_q)
    v = eigvecs[:, -1] - np.median(eigvecs[:, -1])

    gap = eigvals[-1] - eigvals[-2]
    if gap < 0.05 * eigvals[-1]:
        import warnings
        warnings.warn(f"Leading eigenvalue gap is small ({gap:.4f}); try a different seed.")

    # labels set along v axis, splits the values in two classes
    # larger skip = harder task
    skip = datapoints // skipfactor   # pyright: ignore[reportOptionalOperand] 
    keep_pos = np.argsort(-v)[skip : skip + n_per_class[1]]
    keep_neg = np.argsort(v)[skip : skip + n_per_class[-1]]

    X = np.concatenate([X_pool[keep_pos], X_pool[keep_neg]])
    y = np.concatenate([np.ones(len(keep_pos)), -np.ones(len(keep_neg))])
    return X, y


def find_hard_generating_candidate(
    n_candidates: int = 20, datapoints: int = 100,
    n_qubits=N_QUBITS, n_layers=N_LAYERS, seed: int | None = None
):
    """
    Draws n_candidates random circuits from the ADQFM gate family and returns 
    the one whose fidelity kernel has the largest geometric difference from a 
    classical RBF kernel. Used to tailor KernelGap dataset.
    """
    GATES = ("H", "CNOT", "RX", "RY", "RZ", "I")
    ANGLES = (np.pi, np.pi / 2, np.pi / 4, np.pi / 8)

    rng = np.random.default_rng(seed)
    dev = qml.device("default.qubit", wires=n_qubits)

    # fixed pool shared across all candidates -> fair comparison
    X_pool = rng.uniform(0, 2 * np.pi, size=(datapoints, n_qubits))
    gamma = 1.0 / (n_qubits * X_pool.var())
    K_classical = rbf_kernel(X_pool, gamma=gamma)

    best_g, best_candidate = -np.inf, None

    for _ in range(n_candidates):
        gates = list(rng.choice(GATES, size=n_qubits * n_layers))
        angles = list(rng.choice(ANGLES, size=n_qubits * n_layers))
        candidate = OptunaCandidate(n_qubits, n_layers, gates, angles)

        @qml.qnode(dev)
        def get_state(x, candidate=candidate):
            build_circuit(candidate, x)
            return qml.state()

        states = np.array([get_state(x) for x in X_pool])
        K_quantum = np.abs(states.conj() @ states.T) ** 2

        g = geometric_difference(K_classical, K_quantum)
        if g > best_g:
            best_g, best_candidate = g, candidate

    print(f"Best of {n_candidates}: g = {best_g:.4f}")
    return best_candidate, best_g

if __name__ == "__main__":
    X, y = KernelGap_synthetic_dataset(seed=45)
    graph_dataset("KernelGap, even", X, y)

    X, y = KernelGap_synthetic_dataset(n_per_class={1:10, -1:30}, seed=45)
    graph_dataset("KernelGap, uneven", X, y)