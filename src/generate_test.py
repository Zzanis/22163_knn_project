#!/usr/bin/env python3
import numpy as np 
import random

class KNN_tester:
    def __init__(self, filename: str):
        """
        Initializes the KNN_tester with the filename of the data. Loads the data.
        """
        self.filename = filename
        self.raw_data = self.load_data()
        self.data_with_nan = None
        self.imputed_data = None

    def load_data(self) -> bool:
        """
        Loads the tab-separated data from the specified file into a NumPy array.
        The first column (probe IDs) is skipped.
        Returns True if loading is successful, False otherwise.
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
        n_nan = int(n_elements * nan_percentage) # how many NaN's to add

        # flatten the matrix and randomly pick indices to replace with NaN
        flat_data = self.raw_data.flatten()
        indices = random.sample(range(n_elements), n_nan)
        # introduce the NaN
        flat_data[indices] = np.nan
        # reshape to original dims
        self.data_with_nan = flat_data.reshape(self.raw_data.shape).copy()

    def get_test_data(self) -> np.ndarray:
        """
        Returns the test data.
        """
        return self.data_with_nan
    
    def get_raw_data(self) -> np.ndarray:
        """ 
        Returns the raw data.
        """
        return self.raw_data