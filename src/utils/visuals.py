import matplotlib.pyplot as plt
import pennylane as qml
import numpy as np
import os
import matplotlib.patches as patches

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

def graph_dataset(
    dataset_name: str,
    X: np.ndarray, y: np.ndarray,
    X_train : np.ndarray | None = None, y_train: np.ndarray | None = None,
    save: str | None = None,
):
    """
    Graph of a dataset, in the same style the subsets are plotted
    """
    counts = {cls: (y == cls).sum() for cls in np.unique(y)}
    count_str = '  |  '.join(f'Class {cls}: {n}' for cls, n in counts.items())

    _, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X[:, 0], X[:, 1],
            c=y, cmap='coolwarm', alpha=0.30, s=20, label='All points')
    if X_train is not None and y_train is not None:
        ax.scatter(X_train[:, 0], X_train[:, 1],
                c=y_train, cmap='coolwarm',
                edgecolors='black', linewidths=1.2, s=90, zorder=5, label='Training points')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'{dataset_name} Dataset\n{count_str}', fontsize=10)
    ax.legend()
    plt.tight_layout()
    if save:
        plt.savefig(f'{RESULTS_DIR}/graph_dataset_{dataset_name}.png')
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

# -------- 6. VISUALIZATIONS ------------------------------------------------

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
    ax.set_title('Geometric difference per experiment')
    ax.invert_yaxis()
 
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/g_best_comparison.png', dpi=150)
    plt.show()
    print(f"Saved → {RESULTS_DIR}/g_best_comparison.png")
 
def plot_kernel_heatmaps(summary: dict) -> None:
    """
    Side-by-side kernel heatmaps (classical vs quantum).
    Loads K_quantum and K_classical directly from the saved candidate pkls.
    """
    quantum_names = [k for k in summary if k != BASELINE]
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
 
    plt.suptitle('Kernel matrices', fontsize=11, y=1.01)
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