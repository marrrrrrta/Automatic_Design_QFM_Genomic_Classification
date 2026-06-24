import matplotlib.pyplot as plt
import numpy as np

def graph_subset(
    subset_name: str, 
    X_train_full: np.ndarray, X_train_subset: np.ndarray, 
    y_train_full: np.ndarray, y_train_subset:np.ndarray
):
    """Graph of the selected landmarks of a subset by a specific method"""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X_train_full[:, 0], X_train_full[:, 1],
            c=y_train_full, cmap='coolwarm', alpha=0.30, s=20, label='All training points')
    ax.scatter(X_train_subset[:, 0], X_train_subset[:, 1],
            c=y_train_subset, cmap='coolwarm',
            edgecolors='black', linewidths=1.2, s=90, zorder=5, label=f'{subset_name} points')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'{subset_name} Subset')
    ax.legend()
    plt.tight_layout()
    plt.show()

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
    plt.savefig(f'Results/graph_{classical_name}_{quantum_name}.png')
    plt.tight_layout()
    plt.show()

def graph_accuracy(

):
    