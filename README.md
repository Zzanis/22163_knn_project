# 📂 Compiling the Full Dataset

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
1. **First Argument:** The path where the compiled dataset will be saved (e.g., `target/full_dataset.txt`).
2. **Second Argument:** The path to the raw data files (e.g., `source`).

---

### 📊 Dataset Format
The compiled full dataset has the following format:
- **First Column:** `Probe_ID`, which uniquely identifies each probe.
- **Subsequent Columns:** Raw microarray signal values, one column per subject.

This structure allows for efficient data processing and analysis.



