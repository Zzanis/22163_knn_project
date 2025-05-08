# 🧬 Microarray Imputation Toolkit

This repository provides tools for compiling and imputing microarray gene expression data using mean or K-Nearest Neighbors (KNN) approaches. It includes log-transformation to enhance imputation accuracy and robustness.

---

## 📂 Compiling the Full Dataset

To compile the full dataset, follow these steps:

1. Navigate to the project directory in your terminal:
   ```bash
   cd /path/to/project
   ```

2. Run the following command:
   ```bash
   python3 src/data_proc.py target/full_dataset.txt source
   ```

### 📝 Arguments
- **First Argument:** Path where the compiled dataset will be saved (e.g., `target/full_dataset.txt`).
- **Second Argument:** Path to the directory containing raw input data files (e.g., `source`).

---

### 📊 Dataset Format

The compiled full dataset has the following format:

- **First Column:** `Probe_ID`, which uniquely identifies each probe.
- **Subsequent Columns:** Raw microarray signal values, one column per subject.

> 🔍 **Note:** The compiled dataset contains **raw signal values**.  
> Log₂ transformation is applied automatically later during imputation to stabilize variance and improve KNN performance.

---

## 🧪 Using the `KNN_tester` Module

The `KNN_tester` class handles:

- Loading tab-separated datasets
- Introducing missing values
- Performing imputation (KNN or mean)
- Evaluating performance
- Saving results

Log₂ transformation is **mandatory** and automatically applied prior to imputation.

---

### 🚀 Running an Imputation Test (CLI)

Use the provided command-line interface to test the imputation process:

```bash
python3 src/generate_test.py -file target/full_dataset.txt -missing 0.05 -k 3 -m knn -o imputed_output.txt
```

### 🧾 CLI Arguments

| Flag             | Description |
|------------------|-------------|
| `-file`          | Path to the input dataset file |
| `-missing`       | Percentage of data to replace with NaNs (e.g., `0.05` for 5%) |
| `-k`             | Number of neighbors to use for KNN imputation |
| `-m`, `--method` | Imputation method: `knn` (default) or `mean` |
| `-o`, `--output` | (Optional) Output file path for saving imputed data |

---

### 🔍 Log Transformation Details

All data is transformed using the following prior to imputation:

```
log_value = log2(raw_value + ε), where ε = 1e-6
```

After imputation, inverse transformation is applied:

```
raw_value = 2 ** log_value - ε
```

This process ensures stability during KNN distance calculation and evaluation.

---

## 📈 Evaluation Metrics

Once imputation is completed, the following metrics are reported:

- **RMSE** – Root Mean Square Error
- **PCC** – Pearson Correlation Coefficient

These are computed **only on the entries where values were artificially made missing**.

---

## ✅ Example Output

```bash
Loading data from: target/full_dataset.txt
Introducing 5.0% missing values
Using imputation method: knn with k=10
Data loaded with shape: (1000, 50)
Introduced 2500 NaN values (5.00% of data)
Performing KNN imputation with k=10...

Imputation Results (log scale):
  RMSE: 0.182945
  PCC:  0.973422

Imputed data saved to: imputed_output.txt
```

---

## 📁 Project Structure

```
project/
├── src/
│   ├── data_proc.py         # Compiles the full dataset
│   ├── generate_test.py     # CLI interface for testing imputation
│   └── KNN_tester.py        # Main class for imputation and evaluation
├── target/
│   └── full_dataset.txt     # Output from data compilation
├── source/                  # Directory of raw data files
└── README.md                # This documentation
```

---


