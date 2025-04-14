import numpy as np

def impute_knn(data_with_nan: np.ndarray, k) -> np.ndarray:
    """
    Imputes missing values in a 2D matrix using KNN with Euclidean Distance.

    Args:
        data_with_nan: matrix with np.nan in missing positions
        k: number of neighbors used to imputate

    Returns:
        Matrix with predicted values for NaN 
    """
    if not isinstance(data_with_nan, np.ndarray):
        raise TypeError("Input data must be a NumPy array.")
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    # NaN copy
    data = data_with_nan.copy()
    # Number of rows and columns
    n_genes, n_samples = data.shape
    # Real values
    imputed = data.copy()

    # Look for NaN in matrix
    for i in range(n_genes):
        for j in range(n_samples):
            if np.isnan(data[i, j]):
                distances = []

                # 1. Look for potential neighbors
                for r in range(n_genes):
                    # Is value different / non-NaN
                    if r == i or np.isnan(data[r, j]):
                        continue
                    not_nan_i = ~np.isnan(data[i])           # gen(i) has data 
                    not_nan_other = ~np.isnan(data[r])       # gen(r) has data
                    pos_candidates = not_nan_i & not_nan_other     

                    if np.sum(pos_candidates) == 0:
                        continue

                    # 2. Euclidean distance for candidates 
                    vector_i = data[i, pos_candidates]
                    vector_other = data[r, pos_candidates]

                    sq_diff = (vector_i - vector_other) ** 2
                    dist = np.sqrt(np.sum(sq_diff))

                    distances.append((dist, data[r, j]))

                # 3. Find the closest neighbors / "best" candidates 
                if distances:
                    distances.sort(key=lambda x: x[0])
                    nearest_values = [val for _, val in distances[:k]]
                    # Predict NaN with mean of these values
                    imputed[i, j] = np.mean(nearest_values)
                else:
                    imputed[i, j] = np.nan

    return imputed
