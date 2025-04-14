import numpy as np 

def evaluate_imputation(original: np.ndarray, with_nan: np.ndarray,imputed: np.ndarray):
    """
    Evaluates imputation method comparing to original values of NaN

    RMSE (Root Mean Square Error)
    PCC (Pearson Correlation Coefficient)
    """
    if not all(isinstance(arr, np.ndarray) for arr in [original, with_nan, imputed]):
        raise TypeError("All inputs must be NumPy arrays.")

    if original.shape != with_nan.shape or original.shape != imputed.shape:
        raise ValueError("All input arrays must have the same shape.")
    
    # NaNs positions
    mask = np.isnan(with_nan)
    # Original data
    real_values = original[mask]
    # Imputed values 
    predicted_values = imputed[mask]

    # RMSE formula
    rmse_mean = np.mean((real_values-predicted_values) **2)
    rmse = np.sqrt(rmse_mean)

    # PCC formula
    # More than one value present and variated values
    if len(real_values)>1 and np.std(predicted_values) > 0 and np.std(real_values) > 0:
        #Is the library calling okay or should we do the formula manually?
        pcc = np.corrcoef(real_values, predicted_values)[0, 1]
    else:
        pcc = np.nan 

    return rmse, pcc