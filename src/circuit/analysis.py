import numpy as np
from collections import Counter

from ..data.saving import candidate_exists, load_parameter_train


def collect_gate_grids(
    experiment_names: list[str], n_qubits: int, n_layers: int
) -> dict[str, np.ndarray]:
    """
    Loads saved candidates and arranges each one's gate types into a
    (n_layers, n_qubits) grid, matching build_circuit's loop order
    (layer outer, qubit inner).
    """
    grids = {}
    for name in experiment_names:
        if not candidate_exists(name):
            print(f"Skipping '{name}': no saved candidate found.")
            continue
        genes = load_parameter_train(name)['genes']
        grid = np.empty((n_layers, n_qubits), dtype=object)
        for i, gene in enumerate(genes):
            layer, qubit = divmod(i, n_qubits)
            grid[layer, qubit] = gene['gate']
        grids[name] = grid
    return grids


def gate_consensus(
    grids: dict[str, np.ndarray], n_qubits: int, n_layers: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each (layer, qubit) position, finds the most common gate type
    across all given experiments and how strongly they agree.

    Returns:
        consensus_gate: (n_layers, n_qubits) array, most common gate string per cell
        agreement:      (n_layers, n_qubits) array, agreement fraction in [0, 1]
    """
    consensus_gate = np.empty((n_layers, n_qubits), dtype=object)
    agreement = np.zeros((n_layers, n_qubits))
    n_experiments = len(grids)

    for layer in range(n_layers):
        for qubit in range(n_qubits):
            votes = [grid[layer, qubit] for grid in grids.values()]
            gate, count = Counter(votes).most_common(1)[0]
            consensus_gate[layer, qubit] = gate
            agreement[layer, qubit] = count / n_experiments

    return consensus_gate, agreement