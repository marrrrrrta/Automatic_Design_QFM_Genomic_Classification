import numpy as np
import pennylane as qml

from ..circuit.building import build_circuit_params
from .quantum_kernel import quantum_kernel
from .haar import sample_haar_fidelities

def _fidelity_circuit(n_qubits):
    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev)

    def circuit(theta_a, theta_b, candidate):
        build_circuit_params(candidate, theta_a)
        qml.adjoint(build_circuit_params)(candidate, theta_b)  # type: ignore[reportCallIssue]
        return qml.probs(wires=range(n_qubits))
    return circuit

def sample_fidelities(
    candidate, n_qubits, 
    n_samples: int = 2000, 
    seed: int | None = None, 
    sample_mode: str = 'uniform',   # 'uniform' | 'data'
    X: np.ndarray | None = None
) -> np.ndarray:
    """
    Estimates the circuit's fidelity distribution by repeatedly running the same 
    fixed gate structure with fresh random parameters. Note: the Haar reference needs a distribution over fidelities. 

        * 'uniform' — draws random parameters ~ U(0, 2π), ignoring the candidate
        * 'data' - draws random pairs of real data points from X and reuses the candidate's actual (angle, feature) encoding
    """
    rng = np.random.default_rng(seed)
    fids = np.empty(n_samples)

    if sample_mode == 'uniform':
        circuit = _fidelity_circuit(n_qubits)
        n_params = candidate.n_qubits * candidate.n_layers
    
        for i in range(n_samples):
            ta = rng.uniform(0, 2*np.pi, n_params)
            tb = rng.uniform(0, 2*np.pi, n_params)
            fids[i] = circuit(ta, tb, candidate)[0]
    elif sample_mode == 'data':
        N = len(X)
        for i in range(n_samples):
            ia, ib = rng.choice(N, size=2, replace=False)   # avoid trivial fidelity=1 pairs
            fids[i] = quantum_kernel(X[ia], X[ib], candidate)
    return fids


def expressibility(
    candidate, n_qubits, n_samples=2000, n_bins=75, seed=None,
    sample_mode='uniform', X=None
):
    fids_circuit = sample_fidelities(candidate, n_qubits, n_samples, seed, sample_mode, X)
    fids_haar    = sample_haar_fidelities(n_qubits, n_samples, seed)   # literal Haar samples

    bin_edges = np.linspace(0, 1, n_bins + 1)
    p_circuit, _ = np.histogram(fids_circuit, bins=bin_edges)
    p_circuit = p_circuit / p_circuit.sum()

    p_haar, _ = np.histogram(fids_haar, bins=bin_edges)
    p_haar = p_haar / p_haar.sum()

    eps = 1e-10
    kl = np.sum(p_circuit * np.log((p_circuit + eps) / (p_haar + eps)))
    return kl