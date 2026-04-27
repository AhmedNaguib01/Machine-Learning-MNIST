# MNIST Binary Classification - Notebooks Guide

## 📓 Complete Pipeline Notebooks

Each notebook contains the **complete end-to-end pipeline**:
- Data loading from MNIST
- Binary class filtering
- Data preprocessing (normalization, flattening, standardization)
- PCA dimensionality reduction
- Model training
- Evaluation with metrics
- Visualizations

**No dependencies between notebooks!** Each one is fully self-contained.

---

## 🎯 Available Notebooks

### 1️⃣ Logistic Regression
**File**: `01_logistic_regression_complete.ipynb`

**What it does**:
- Complete pipeline from data loading to evaluation
- Implements Logistic Regression from scratch
- Gradient descent optimization
- Training loss visualization
- Confusion matrix and metrics

**Time**: ~2-3 minutes

---

### 2️⃣ K-Nearest Neighbors
**File**: `02_knn_complete.ipynb`

**What it does**:
- Complete pipeline from data loading to evaluation
- Implements KNN from scratch
- Euclidean distance calculation
- Majority voting
- Progress bars for predictions

**Time**: ~3-5 minutes (KNN is slower)

---

### 3️⃣ Decision Tree
**File**: `03_decision_tree_complete.ipynb`

**What it does**:
- Complete pipeline from data loading to evaluation
- Implements Decision Tree from scratch
- Gini impurity calculation
- Recursive tree building
- Confusion matrix and metrics

**Time**: ~2-3 minutes

---

## 🚀 How to Use

### Start Jupyter
```bash
jupyter notebook
# or
jupyter lab
```

### Run Any Notebook
- **No specific order required!** Each notebook is independent
- Open any notebook you want to explore
- Press `Shift + Enter` to run each cell
- Outputs display immediately below each cell

---

## ⚙️ Configuration

Each notebook has a configuration cell at the top where you can change:

```python
# Binary classification
DIGIT_A = 0          # Change to any digit (0-9)
DIGIT_B = 1          # Change to any digit (0-9)

# PCA
PCA_COMPONENTS = 50  # Number of PCA components

# Model-specific hyperparameters
# (varies by notebook)
```

---

## 📊 What You'll See

Each notebook displays:
- ✅ Sample MNIST images
- ✅ Data statistics
- ✅ Training progress
- ✅ Performance metrics (Accuracy, Precision, Recall, F1)
- ✅ Confusion matrix heatmap
- ✅ Classification report
- ✅ Final summary

---

## 💡 Tips

1. **Run independently**: No need to run all notebooks
2. **Experiment freely**: Change parameters and re-run
3. **Compare results**: Run multiple notebooks to compare models
4. **Save outputs**: Notebooks save outputs automatically
5. **Export**: Download as HTML/PDF for sharing

---

## 🎯 Quick Start

```bash
# 1. Install dependencies
pip install -r ../requirements.txt

# 2. Start Jupyter
jupyter notebook

# 3. Open any notebook and run!
```

---

## 📈 Expected Results (Digits 0 vs 1)

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~99% |
| K-Nearest Neighbors | ~98% |
| Decision Tree | ~96% |

---

## 🔧 Troubleshooting

**Jupyter not found?**
```bash
pip install jupyter jupyterlab
```

**Import errors?**
```bash
pip install -r ../requirements.txt
```

**Data not found?**
- Make sure `../mnist.npz/` folder exists
- Check that .npy files are present

**KNN too slow?**
- This is normal for KNN
- Wait patiently (3-5 minutes)
- Or reduce training set size in the notebook

---

**Ready to start? Open any notebook and begin! 🚀**
