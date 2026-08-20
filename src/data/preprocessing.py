import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from config.config import TTS_SEED

def preprocess(
    X: np.ndarray, y: np.ndarray, 
    n_qubits: int, tts_seed: int | None = TTS_SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocesses a dataset for quantum kernel SVM: encodes labels, splits, applies PCA, and scales features to [-1, 1].

    Label encoding:
        Binary     → {-1, +1}   (standard SVM convention)
        Multi-class → {0, 1, …, n-1} 

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features)
        y (np.ndarray): Binary labels of shape (n_samples,) with values in {0, 1}
        n_qubits (int): Number of PCA components to retain for dimensionality reduction

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
    """    
    # LABEL ENCODING (Y)
    le = LabelEncoder()
    y = le.fit_transform(y) # pyright: ignore[reportAssignmentType]
    n_classes = len(le.classes_)

    if n_classes == 2:
        # Binary case
        y = 2 * y - 1

    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=tts_seed, stratify=y)

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


def preprocess_synthetic(X, y, tts_seed=TTS_SEED):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=tts_seed, stratify=y
    )
    return X_train, X_test, y_train, y_test