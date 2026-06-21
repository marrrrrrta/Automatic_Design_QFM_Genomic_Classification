import matplotlib.pyplot as plt
import numpy as np

def graph_subset(name: str, X_train_full: np.ndarray, X_train_subset: np.ndarray, y_train_full: np.ndarray, y_train_subset:np.ndarray):
    """Graph of the selected landmarks of a subset by a specific method"""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X_train_full[:, 0], X_train_full[:, 1],
            c=y_train_full, cmap='coolwarm', alpha=0.30, s=20, label='All training points')
    ax.scatter(X_train_subset[:, 0], X_train_subset[:, 1],
            c=y_train_subset, cmap='coolwarm',
            edgecolors='black', linewidths=1.2, s=90, zorder=5, label=f'{name} points')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'{name} Subset')
    ax.legend()
    plt.tight_layout()
    plt.show()