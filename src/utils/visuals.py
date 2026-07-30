import matplotlib.pyplot as plt
import pennylane as qml
import numpy as np
import os
import matplotlib.patches as patches

from src.circuit.analysis import collect_gate_grids, gate_consensus
from src.data.saving import candidate_exists, load_parameter_train
from src.circuit.building import build_circuit
from src.circuit.candidates import Candidate

RESULTS_DIR = 'Results'
BASELINE = 'Classical_baseline'

# -------- DATA VISUALIZATION ------------------------------------------------

def graph_subset(
    subset_name: str, 
    X_train_full: np.ndarray, X_train_subset: np.ndarray, 
    y_train_full: np.ndarray, y_train_subset:np.ndarray
):
    """
    Graph of the selected landmarks of a subset by a specific method
    """
    counts = {cls: (y_train_subset == cls).sum() for cls in np.unique(y_train_subset)}
    count_str = '  |  '.join(f'Class {cls}: {n}' for cls, n in counts.items())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X_train_full[:, 0], X_train_full[:, 1],
            c=y_train_full, cmap='coolwarm', alpha=0.30, s=20, label='All training points')
    ax.scatter(X_train_subset[:, 0], X_train_subset[:, 1],
            c=y_train_subset, cmap='coolwarm',
            edgecolors='black', linewidths=1.2, s=90, zorder=5, label=f'{subset_name} points')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'{subset_name} Subset\n{count_str}', fontsize=10)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/graph_subset_{subset_name}.png')
    plt.show()

# -------- KERNEL VISUALIZATION ------------------------------------------------

def graph_kernels(
    classical_name: str, quantum_name: str,
    K_classic: np.ndarray, K_quantum: np.ndarray
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(K_classic, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title(classical_name)
    im2 = axes[1].imshow(K_quantum, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title(quantum_name)
    plt.colorbar(im2, ax=axes[1], label="Similarity")
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/graph_{classical_name}_{quantum_name}.png')
    plt.show()

# -------- QUANTUM CIRCUIT VISUALIZATION ----------------------------------------

def draw_kernel_circuit(
    candidate: Candidate, x1: np.ndarray, x2: np.ndarray,
    name: str | None = None, title: str | None = None,
    decimals: int = 2, style: str | None = None, print_genes: bool = False,
):
    """
    Draws the full fidelity kernel circuit U(x2)†U(x1)|0> exactly as it's
    executed during training. Saves to Results/{name}/kernel_circuit.png if name is given.
    """
    dev = qml.device("default.qubit", wires=candidate.n_qubits)

    @qml.qnode(dev)
    def circuit(x1, x2, candidate):
        build_circuit(candidate, x1)
        qml.adjoint(build_circuit)(candidate, x2)
        return qml.probs(wires=range(candidate.n_qubits))

    fig, ax = qml.draw_mpl(circuit, decimals=decimals, style=style)(x1, x2, candidate)
    title = title or name
    if title:
        (ax[0] if isinstance(ax, (list, np.ndarray)) else ax).set_title(title, fontsize=12, pad=12)
    if print_genes:
        print_gene_layout(candidate, label=title)
    if name:
        os.makedirs(f'{RESULTS_DIR}/{name}', exist_ok=True)
        fig.savefig(f'{RESULTS_DIR}/{name}/kernel_circuit.png', dpi=150, bbox_inches='tight')
    return fig, ax

def print_gene_layout(candidate: Candidate, label: str | None = None):
    """
    Prints (layer, qubit, gate, angle) for every gene, in build_circuit's
    raster order. Useful because Identity genes draw as nothing in qml.draw_mpl,
    and a CNOT's target wire can pick up an extra box that doesn't map cleanly
    to a single visual column.
    """
    header = f"Gene layout — {label}" if label else "Gene layout"
    print(f"\n{header}")
    print(f"{'gene':<6}{'layer':<7}{'qubit':<7}{'gate':<7}angle")
    for i, gene in enumerate(candidate.genes):
        layer, qubit = divmod(i, candidate.n_qubits)
        info = gene.get_gene_info()
        angle = info.get('angle', '—')
        angle_str = f"{angle:.3f}" if isinstance(angle, (int, float)) else angle
        print(f"g{i+1:<5}{layer:<7}{qubit:<7}{info['gate']:<7}{angle_str}")

def draw_feature_map(
    candidate: Candidate, x: np.ndarray,
    name: str | None = None, title: str | None = None,
    decimals: int = 2, style: str | None = None, print_genes: bool = True,
):
    """
    Draws the feature map U(x)|0> for one candidate using PennyLane's
    native matplotlib drawer. Saves to Results/{name}/feature_map.png if name is given.
    """
    dev = qml.device("default.qubit", wires=candidate.n_qubits)

    @qml.qnode(dev)
    def circuit(x, candidate):
        build_circuit(candidate, x)
        return qml.state()

    fig, ax = qml.draw_mpl(circuit, decimals=decimals, style=style)(x, candidate)
    title = title or name
    if title:
        (ax[0] if isinstance(ax, (list, np.ndarray)) else ax).set_title(title, fontsize=12, pad=12)
    if print_genes:
        print_gene_layout(candidate, label=title)
    if name:
        os.makedirs(f'{RESULTS_DIR}/{name}', exist_ok=True)
        fig.savefig(f'{RESULTS_DIR}/{name}/feature_map.png', dpi=150, bbox_inches='tight')
    return fig, ax


GATE_COLORS = {
    'H': 'lightsteelblue', 'CNOT': 'coral',
    'RX': 'mediumseagreen', 'RY': 'mediumseagreen', 'RZ': 'mediumseagreen',
    'I': 'lightgray',
}

def _draw_grid(ax, n_qubits, n_layers, cell_fn):
    """Shared grid scaffolding; cell_fn(ax, layer, qubit, y) draws one cell."""
    for layer in range(n_layers):
        for qubit in range(n_qubits):
            y = n_qubits - 1 - qubit
            cell_fn(ax, layer, qubit, y)
    ax.set_xlim(0, n_layers); ax.set_ylim(0, n_qubits)
    ax.set_xticks(np.arange(n_layers) + 0.5)
    ax.set_xticklabels([f"layer {i}" for i in range(n_layers)])
    ax.set_yticks(np.arange(n_qubits) + 0.5)
    ax.set_yticklabels([f"q{n_qubits - 1 - i}" for i in range(n_qubits)])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def plot_gate_consensus(
    consensus_gate: np.ndarray, agreement: np.ndarray,
    n_qubits: int, n_layers: int,
    title: str = 'Gate consensus across experiments',
    save_path: str | None = None,
):
    """
    Plots the most common gate type per (layer, qubit) position across a
    set of experiments. Cell opacity encodes agreement strength —
    faint = contested position, solid = strong consensus.
    """
    fig, ax = plt.subplots(figsize=(n_layers * 1.6, n_qubits * 1.1))

    def draw_cell(ax, layer, qubit, y):
        gate = consensus_gate[layer, qubit]
        agree = agreement[layer, qubit]
        color = GATE_COLORS.get(gate, 'white')
        alpha = 0.3 + 0.7 * agree
        rect = patches.FancyBboxPatch(
            (layer + 0.05, y + 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=color, edgecolor='black', linewidth=0.8, alpha=alpha,
        )
        ax.add_patch(rect)
        ax.text(layer + 0.5, y + 0.58, gate, ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(layer + 0.5, y + 0.30, f"{agree:.0%} agree", ha='center', va='center', fontsize=7.5, color='dimgray')

    _draw_grid(ax, n_qubits, n_layers, draw_cell)
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig, ax


def plot_gate_diff(
    name_a: str, name_b: str, n_qubits: int, n_layers: int,
    save_path: str | None = None,
):
    """
    Compares two saved candidates position-by-position. Green = same gate
    type at that position, coral = diverged.
    """
    grids = collect_gate_grids([name_a, name_b], n_qubits, n_layers)
    grid_a, grid_b = grids[name_a], grids[name_b]
    fig, ax = plt.subplots(figsize=(n_layers * 1.6, n_qubits * 1.1))

    def draw_cell(ax, layer, qubit, y):
        ga, gb = grid_a[layer, qubit], grid_b[layer, qubit]
        same = ga == gb
        color = 'mediumseagreen' if same else 'coral'
        label = ga if same else f"{ga} / {gb}"
        rect = patches.FancyBboxPatch(
            (layer + 0.05, y + 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=color, edgecolor='black', linewidth=0.8, alpha=0.55,
        )
        ax.add_patch(rect)
        ax.text(layer + 0.5, y + 0.5, label, ha='center', va='center', fontsize=9)

    _draw_grid(ax, n_qubits, n_layers, draw_cell)
    n_same = sum(grid_a[l, q] == grid_b[l, q] for l in range(n_layers) for q in range(n_qubits))
    ax.set_title(f"{name_a}  vs  {name_b}   ({n_same}/{n_layers*n_qubits} positions match)", fontsize=10.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig, ax

# -------- RESULTS VISUALIZATION ------------------------------------------------

def plot_accuracy(summary: dict, columns) -> None:
    """
    Grouped bar chart: test accuracy vs CV accuracy for every experiment.
    Args:
        summary (dict): data to plot
        options (array): choose which columns to plot. available: ['test', 'CV']
    """
    names     = list(summary.keys())
    test_acc  = [s['accuracy'] for s in summary.values()]
    cv_acc    = [s['cv_acc']   for s in summary.values()]
    baseline  = summary[BASELINE]['accuracy']
 
    x, w = np.arange(len(names)), 0.38
    fig, ax = plt.subplots(figsize=(13, 5))
    if 'test' in columns:
        b1 = ax.bar(x - w/2, test_acc, w, label='Test accuracy', color='steelblue')
        ax.bar_label(b1, fmt='%.3f', fontsize=7.5, padding=2)
    if 'CV' in columns:
        b2 = ax.bar(x + w/2, cv_acc,   w, label='CV accuracy',   color='coral', alpha=0.85)
        ax.bar_label(b2, fmt='%.3f', fontsize=7.5, padding=2)
    ax.axhline(baseline, color='grey', linestyle='--', linewidth=1, label=f'Baseline ({baseline:.3f})')
 
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=28, ha='right', fontsize=9)
    ax.set_ylim(max(0, min(test_acc + cv_acc) - 0.08), 1.05)
    ax.set_ylabel('Accuracy')
    ax.set_title('Experiment accuracies')
    ax.legend(fontsize=9)
 
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/accuracy_comparison.png', dpi=150)
    plt.show()
    print(f"Saved → {RESULTS_DIR}/accuracy_comparison.png")
 
 
def plot_g_best(summary: dict) -> None:
    """Horizontal bar chart of geometric difference (g_best) — quantum experiments only."""
    quantum = {k: v for k, v in summary.items() if not np.isnan(v['g_best'])}
    if not quantum:
        return
 
    names  = list(quantum.keys())
    values = [v['g_best'] for v in quantum.values()]
 
    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.55)))
    bars = ax.barh(names, values, color='mediumseagreen', alpha=0.85)
    ax.bar_label(bars, fmt='%.2f', padding=4, fontsize=8)
    ax.set_xlabel('g_best  (geometric difference vs RBF)')
    ax.set_title('Quantum kernel expressibility per experiment')
    ax.invert_yaxis()
 
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/g_best_comparison.png', dpi=150)
    plt.show()
    print(f"Saved → {RESULTS_DIR}/g_best_comparison.png")
 
 
def plot_kernel_heatmaps(summary: dict, max_show: int = 3) -> None:
    """
    Side-by-side kernel heatmaps (classical vs quantum) for the first
    `max_show` quantum experiments. Loads K_quantum and K_classical
    directly from the saved candidate pkls.
    """
    quantum_names = [k for k in summary if k != BASELINE][:max_show]
    n = len(quantum_names)
    if n == 0:
        return
 
    fig, axes = plt.subplots(n, 2, figsize=(8, 3.8 * n), squeeze=False)
 
    for i, name in enumerate(quantum_names):
        if not candidate_exists(name):
            continue
        data = load_parameter_train(name)
        K_c, K_q = data['K_classical'], data['K_quantum']
 
        for j, (K, title) in enumerate([(K_c, 'Classical (RBF)'), (K_q, 'Quantum')]):
            im = axes[i][j].imshow(K, cmap='viridis', vmin=0, vmax=1)
            axes[i][j].set_title(f'{name}\n{title}', fontsize=8.5)
            axes[i][j].axis('off')
            plt.colorbar(im, ax=axes[i][j], fraction=0.046, pad=0.04)
 
    plt.suptitle('Kernel matrices — training subset', fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/kernel_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {RESULTS_DIR}/kernel_heatmaps.png")

## OPTION 2

def plot_expressibility(summary: dict) -> None:
    """Horizontal bar chart of expressibility (KL divergence vs. Haar) —
    Haar-variant experiments only. Lower KL = more expressive/Haar-like."""
    haar = {k: v for k, v in summary.items() if not np.isnan(v.get('kl_haar', float('nan')))}
    if not haar:
        return

    names  = list(haar.keys())
    values = [v['kl_haar'] for v in haar.values()]

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.55)))
    bars = ax.barh(names, values, color='mediumpurple', alpha=0.85)
    ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=8)
    ax.set_xlabel('KL divergence vs. Haar  (lower = more expressive)')
    ax.set_title('Circuit expressibility per experiment')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/expressibility_comparison.png', dpi=150)
    plt.show()
    print(f"Saved → {RESULTS_DIR}/expressibility_comparison.png")