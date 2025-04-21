#!/usr/bin/env python3
import random 
import numpy as np


class KNN_tester:
    def __init__(self, filename: str):
        """
        Initializes the KNN_tester with the filename of the data. Loads the data.
        """
        self.filename = filename
        self.raw_data = self.load_data()
        self.data_with_nan = None
        self.imputed_data = None

    def load_data(self) -> np.ndarray:
        """
        Loads the tab-separated data from the specified file into a NumPy array.
        The first column (probe IDs) is skipped.
        """
        try:
            return np.genfromtxt(self.filename, delimiter='\t', skip_header=1, dtype=float, encoding=None)
        except FileNotFoundError:
            print(f"Error: File not found at {self.filename}")
        except Exception as e:
            print(f"An error occurred during data loading: {e}")

    def introduce_nan_randomly(self, nan_percentage: float):
        """
        Introduces NaN values randomly into the loaded data.

        Args:
            nan_percentage: The percentage of elements to replace with NaN (e.g., 0.1 for 10%).
        """
        if not (0 <= nan_percentage <= 1):
            raise ValueError("nan_percentage must be between 0 and 1.")

        n_elements = self.raw_data.size
        n_nan = int(n_elements * nan_percentage)

        flat_data = self.raw_data.flatten()
        indices = random.sample(range(n_elements), n_nan)
        flat_data[indices] = np.nan
        self.data_with_nan = flat_data.reshape(self.raw_data.shape).copy()
        
    def mean_impute(self):
        """
        Imputes missing values in self.data_with_nan using column-wise mean.
        Stores the result in self.imputed_data.
        """
        if self.data_with_nan is None:
            raise ValueError("Data with NaNs has not been initialized. Call introduce_nan_randomly first.")

        data = self.data_with_nan.copy()
        # Compute column means ignoring NaNs
        col_means = np.nanmean(data, axis=0)
        # Find indices where NaNs are located
        inds = np.where(np.isnan(data))
        # Replace NaNs with corresponding column mean
        data[inds] = np.take(col_means, inds[1])

        self.imputed_data = data

    def impute_data_knn(self, k: int):
        """
        Imputes the NaN values in self.data_with_nan using KNN and stores result in self.imputed_data.

        Args:
            k: number of neighbors used to impute
        """
        if self.data_with_nan is None:
            raise ValueError("Data with NaNs has not been initialized. Call introduce_nan_randomly first.")

        if not isinstance(self.data_with_nan, np.ndarray):
            raise TypeError("Input data must be a NumPy array.")

        if k <= 0:
            raise ValueError("k must be a positive integer.")

        data = self.data_with_nan.copy()
        n_genes, n_samples = data.shape
        imputed = data.copy()

        for i in range(n_genes):
            for j in range(n_samples):
                if np.isnan(data[i, j]):
                    distances = []

                    for r in range(n_genes):
                        if r == i or np.isnan(data[r, j]):
                            continue

                        not_nan_i = ~np.isnan(data[i])
                        not_nan_other = ~np.isnan(data[r])
                        pos_candidates = not_nan_i & not_nan_other

                        if np.sum(pos_candidates) == 0:
                            continue

                        vector_i = data[i, pos_candidates]
                        vector_other = data[r, pos_candidates]

                        sq_diff = (vector_i - vector_other) ** 2
                        dist = np.sqrt(np.sum(sq_diff))

                        distances.append((dist, data[r, j]))

                    if distances:
                        distances.sort(key=lambda x: x[0])
                        nearest_values = [val for _, val in distances[:k]]
                        imputed[i, j] = np.mean(nearest_values)
                    else:
                        imputed[i, j] = np.nan

        self.imputed_data = imputed

    def get_test_data(self) -> np.ndarray:
        """
        Returns the test data with NaNs.
        """
        return self.data_with_nan

    def get_raw_data(self) -> np.ndarray:
        """ 
        Returns the original raw data.
        """
        return self.raw_data

    def get_imputed_data(self) -> np.ndarray:
        """
        Returns the data after KNN imputation.
        """
        return self.imputed_data
