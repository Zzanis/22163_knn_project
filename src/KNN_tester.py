#!/usr/bin/env python3
import random
import numpy as np


class KNN_tester:
    def __init__(self, filename: str):
        """
        Initializes the KNN_tester with the filename of the data. Loads the data.
        """
        self.filename = filename
        self.data_with_nan = None
        self.imputed_data = None
        self.log_imputed_data = None
        
        self.raw_data = self.load_data()
        self.log_data = self.log_transform()

    def load_data(self) -> np.ndarray:
        """
        Loads the tab-separated data from the specified file into a NumPy array.
        The first column (probe IDs) is skipped.
        """
        try:
            return np.genfromtxt(self.filename, delimiter='\t', skip_header=1, dtype=float, encoding=None)
        except FileNotFoundError:
            print(f"Error: File not found at {self.filename}")
            return None
        except Exception as e:
            print(f"An error occurred during data loading: {e}")
            return None
    
    def log_transform(self):
        """
        Applies log2 transformation to raw data and stores it in self.log_data.
        """
        if self.raw_data is None:
            raise ValueError("Raw data must be loaded before log-transform.")
        return np.log2(self.raw_data + 1e-6)

    def introduce_nan_randomly(self, nan_percentage: float):
        """
        Introduces NaN values randomly into the loaded data.

        Args:
            nan_percentage: The percentage of elements to replace with NaN (e.g., 0.1 for 10%).
        """
        if not (0 <= nan_percentage <= 1):
            raise ValueError("nan_percentage must be between 0 and 1.")

        n_elements = self.log_data.size
        n_nan = int(n_elements * nan_percentage)

        flat_data = self.log_data.flatten()
        indices = random.sample(range(n_elements), n_nan)
        flat_data[indices] = np.nan
        self.data_with_nan = flat_data.reshape(self.log_data.shape).copy()
        

        
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
        
        self.log_imputed_data = data
        self.imputed_data = (2 ** data) - 1e-6

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
                    # print('row ', i)
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

        # self.imputed_data = imputed
        self.log_imputed_data = imputed
        self.imputed_data = (2 ** imputed) - 1e-6

    def evaluate_imputation(self):
        """
        Evaluates imputation method by comparing imputed values to original values.
        
        Returns:
            tuple: (RMSE, PCC) - Root Mean Square Error and Pearson Correlation Coefficient
        """
        if self.raw_data is None or self.data_with_nan is None or self.imputed_data is None:
            raise ValueError("Original data, data with NaNs, and imputed data must all be available.")
            
        # NaNs positions
        mask = np.isnan(self.data_with_nan)
        
        # Original data at positions where NaNs were introduced
        real_values = self.log_data[mask]
        
        # Imputed values at those same positions
        predicted_values = self.log_imputed_data[mask]
        
        # RMSE formula
        rmse_mean = np.mean((real_values - predicted_values) ** 2)
        rmse = np.sqrt(rmse_mean)
        
        # PCC formula
        # More than one value present and variated values
        if len(real_values) > 1 and np.std(predicted_values) > 0 and np.std(real_values) > 0:
            # Pearson correlation coefficient
            pcc = np.corrcoef(real_values, predicted_values)[0, 1]
        else:
            pcc = np.nan
            
        return rmse, pcc

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
    
    def save_imputed_data(self, output_file: str):
        """
        Saves the imputed data to a tab-separated file.
        
        Args:
            output_file: Path to save the imputed data
        """
        if self.imputed_data is None:
            raise ValueError("No imputed data available to save.")
            
        try:
            np.savetxt(output_file, self.imputed_data, delimiter='\t')
            return True
        except Exception as e:
            print(f"Error saving imputed data: {e}")
            return False


# If run directly, show usage information
if __name__ == "__main__":
    print("This is a module file containing the KNN_tester class.")
    print("To use it for testing, run generate_test.py instead.")
    print("\nExample usage:")
    print("  generate_test.py -file data.txt -missing 0.05 -k 10")
    print("  generate_test.py -file data.txt -missing 0.05 -k 10 -m mean")
    print("  generate_test.py -file data.txt -missing 0.05 -k 10 -o imputed_data.txt")