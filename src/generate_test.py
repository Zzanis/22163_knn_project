#!/usr/bin/env python3
import argparse
import random
import numpy as np
import sys
import os
from KNN_tester import KNN_tester


def main():
    parser = argparse.ArgumentParser(description='Generate and evaluate KNN imputation test sets')
    parser.add_argument('-file', required=True, help='Input data file path')
    parser.add_argument('-missing', type=float, default=0.05, 
                        help='Percentage of missing values to introduce (0-1, default: 0.05)')
    parser.add_argument('-k', type=int, default=5, 
                        help='Number of neighbors for KNN imputation (default: 5)')
    parser.add_argument('-o', '--output', help='Output file for imputed data')
    parser.add_argument('-m', '--method', choices=['mean', 'knn'], default='knn',
                        help='Imputation method (default: knn)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found")
        return 1
        
    if not (0 < args.missing < 1):
        print(f"Error: Missing percentage must be between 0 and 1, got {args.missing}")
        return 1
        
    if args.k < 1:
        print(f"Error: k must be a positive integer, got {args.k}")
        return 1
    
    print(f"Loading data from: {args.file}")
    print(f"Introducing {args.missing:.1%} missing values")
    print(f"Using imputation method: {args.method}" + (f" with k={args.k}" if args.method == 'knn' else ""))
    
    # Initialize KNN tester
    tester = KNN_tester(args.file)
    
    if tester.raw_data is None:
        print("Failed to load data")
        return 1
        
    print(f"Data loaded with shape: {tester.raw_data.shape}")
    
    # Introduce NaNs
    tester.introduce_nan_randomly(args.missing)
    nan_count = np.sum(np.isnan(tester.data_with_nan))
    total_elements = tester.data_with_nan.size
    print(f"Introduced {nan_count} NaN values ({nan_count/total_elements:.2%} of data)")
    
    # Perform imputation
    if args.method == 'mean':
        print("Performing mean imputation...")
        tester.mean_impute()
    else:  # knn
        print(f"Performing KNN imputation with k={args.k}...")
        tester.impute_data_knn(args.k)
    
    # Evaluate imputation
    rmse, pcc = tester.evaluate_imputation()
    print(f"\nImputation Results:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  PCC: {pcc:.6f}")
    
    # Save imputed data if output file specified
    if args.output:
        tester.save_imputed_data(args.output)
        print(f"Imputed data saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())