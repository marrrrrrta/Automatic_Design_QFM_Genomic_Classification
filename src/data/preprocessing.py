import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def preprocess(
    X: np.ndarray, y: np.ndarray, n_qubits: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess a dataset by converting labels to {-1, 1}, splitting the data into training and test sets, applying PCA to reduce the features to ``n_qubits`` components, and scaling the features to the range [-1, 1].

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features)
        y (np.ndarray): Binary labels of shape (n_samples,) with values in {0, 1}
        n_qubits (int): Number of PCA components to retain for dimensionality reduction

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
    """    
    # LABELS (Y)
    y = 2 * y - 1  

    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # FEATURES (X)
    # PCA to reduce the number of features
    pca = PCA(n_components=n_qubits).fit(X_train)
    X_train = pca.transform(X_train)
    X_test  = pca.transform(X_test)

    # Scaling the features to [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test