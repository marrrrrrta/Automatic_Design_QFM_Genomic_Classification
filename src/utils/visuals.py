import matplotlib.pyplot as plt
import numpy as np

from src.data.saving import candidate_exists, load_parameter_train

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
    plt.savefig(f'{RESULTS_DIR}/graph_subset_{subset_name}.png')
    plt.tight_layout()
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
    plt.savefig(f'{RESULTS_DIR}/graph_{classical_name}_{quantum_name}.png')
    plt.tight_layout()
    plt.show()


# -------- VISUALIZATION OF RESULTS ------------------------------------------------

def plot_accuracy(summary: dict) -> None:
    """Grouped bar chart: test accuracy vs CV accuracy for every experiment."""
    names     = list(summary.keys())
    test_acc  = [s['accuracy'] for s in summary.values()]
    cv_acc    = [s['cv_acc']   for s in summary.values()]
    baseline  = summary[BASELINE]['accuracy']
 
    x, w = np.arange(len(names)), 0.38
    fig, ax = plt.subplots(figsize=(13, 5))
 
    b1 = ax.bar(x - w/2, test_acc, w, label='Test accuracy', color='steelblue')
    b2 = ax.bar(x + w/2, cv_acc,   w, label='CV accuracy',   color='coral', alpha=0.85)
    ax.axhline(baseline, color='grey', linestyle='--', linewidth=1, label=f'Baseline ({baseline:.3f})')
 
    ax.bar_label(b1, fmt='%.3f', fontsize=7.5, padding=2)
    ax.bar_label(b2, fmt='%.3f', fontsize=7.5, padding=2)
 
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=28, ha='right', fontsize=9)
    ax.set_ylim(max(0, min(test_acc + cv_acc) - 0.08), 1.05)
    ax.set_ylabel('Accuracy')
    ax.set_title('Experiment comparison — test vs CV accuracy')
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