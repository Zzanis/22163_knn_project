import numpy as np
import pytest
from your_module import impute_knn  # Import your function

def test_impute_knn_basic():
    """Test basic functionality of KNN imputation with a simple dataset."""
    # Create a simple test matrix with known NaN values
    test_data = np.array([
        [1.0, 2.0, np.nan, 4.0],
        [2.0, np.nan, 6.0, 8.0],
        [1.5, 2.5, 3.5, 4.5],
        [1.2, 2.2, 3.2, 4.2]
    ])
    
    # Expected result after imputation with k=2
    # Missing values should be imputed based on 2 nearest neighbors
    k = 2
    result = impute_knn(test_data, k)
    
    # Check that the result is not None and has the same shape
    assert result is not None
    assert result.shape == test_data.shape
    
    # Verify non-NaN values remain unchanged
    mask = ~np.isnan(test_data)
    np.testing.assert_array_almost_equal(result[mask], test_data[mask])
    
    # Verify NaN values were imputed (not NaN anymore)
    assert not np.isnan(result).any()
    
    # The expected values should be calculated based on nearest neighbors
    # For example, test_data[0, 2] should be approximately the mean of values
    # from its 2 nearest neighbors in the same column
    
    # This is a simple check - in real world you might want to manually verify
    # the exact expected values for the NaN positions
    
def test_impute_knn_all_nan_column():
    """Test handling of columns with all NaN values."""
    # Create a matrix with one column containing all NaNs
    test_data = np.array([
        [1.0, np.nan, 3.0],
        [2.0, np.nan, 6.0],
        [3.0, np.nan, 9.0]
    ])
    
    k = 2
    result = impute_knn(test_data, k)
    
    # Check that the all-NaN column remains NaN
    assert np.isnan(result[:, 1]).all()
    
    # Check that other values are preserved
    assert result[0, 0] == 1.0
    assert result[2, 2] == 9.0

def test_impute_knn_error_handling():
    """Test error handling for invalid inputs."""
    # Test with non-numpy array
    with pytest.raises(TypeError):
        impute_knn([[1, 2], [3, 4]], 2)
    
    # Test with invalid k value
    with pytest.raises(ValueError):
        impute_knn(np.array([[1, 2], [3, 4]]), 0)
    
    # Test with negative k value
    with pytest.raises(ValueError):
        impute_knn(np.array([[1, 2], [3, 4]]), -1)

def test_impute_knn_exact_values():
    """Test specific imputation values with a controlled example."""
    # Create data where we know exactly what values should be imputed
    test_data = np.array([
        [1.0, 1.0, np.nan],  # This row is similar to row 1
        [1.1, 1.1, 5.0],     # Nearest neighbor to row 0
        [2.0, 2.0, 10.0],    # Different row
        [10.0, 10.0, 20.0]   # Very different row
    ])
    
    k = 1
    result = impute_knn(test_data, k)
    
    # With k=1, the missing value should be imputed as 5.0 (from row 1)
    assert np.isclose(result[0, 2], 5.0)
    
    # Try with k=2
    k = 2
    result = impute_knn(test_data, k)
    
    # With k=2, the missing value should be imputed as mean of 5.0 and 10.0
    assert np.isclose(result[0, 2], 7.5)