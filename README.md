# MNIST Binary Image Classification Pipeline

A complete, modular implementation of binary image classification using MNIST dataset with three machine learning algorithms implemented from scratch. **Now organized with Jupyter notebooks for interactive exploration!**

## 🎯 Project Overview

This project implements Phase 1 of a Machine Learning university project, featuring:
- **Binary classification** of MNIST digits (configurable)
- **Three ML algorithms from scratch**: Logistic Regression, K-Nearest Neighbors, Decision Tree
- **Interactive Jupyter notebooks** with outputs displayed under each cell
- **Organized folder structure** for easy navigation
- **Complete pipeline**: Data loading → Preprocessing → Feature engineering → Training → Evaluation → Visualization

## 📁 Project Structure

```
mnist-binary-classification/
│
├── notebooks/                                # 📓 Jupyter Notebooks (Complete Pipelines)
│   ├── 01_logistic_regression_complete.ipynb  # Complete LR pipeline
│   ├── 02_knn_complete.ipynb                  # Complete KNN pipeline
│   ├── 03_decision_tree_complete.ipynb        # Complete DT pipeline
│   └── README.md                               # Notebook guide
│

│
├── mnist.npz/                          # 📦 Raw MNIST Dataset
│   ├── x_train.npy
│   ├── y_train.npy
│   ├── x_test.npy
│   └── y_test.npy
│
├── docs/                               # 📚 Documentation
│   ├── PROJECT_STRUCTURE.md           # Detailed structure info
│   ├── USAGE_GUIDE.md                 # Comprehensive usage guide
│   ├── QUICK_REFERENCE.md             # Quick reference card
│   └── PIPELINE_FLOW.md               # Visual pipeline diagrams
│

│
├── .gitignore                          # Git ignore file
├── README.md                           # This file
├── GETTING_STARTED.md                  # Setup guide
├── requirements.txt                    # Python dependencies
└── test_modules.py                     # Module testing script
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Jupyter
```bash
jupyter notebook
# or
jupyter lab
```

### 3. Run Any Notebook
Open any notebook - they're all independent!
- `notebooks/01_logistic_regression_complete.ipynb`
- `notebooks/02_knn_complete.ipynb`
- `notebooks/03_decision_tree_complete.ipynb`

**Each notebook contains the complete pipeline from data loading to evaluation!**

## ✨ Features

### 📓 Jupyter Notebooks
- **Interactive execution**: Run code cell by cell
- **Inline outputs**: See results immediately under each cell
- **Rich visualizations**: Plots displayed directly in notebooks
- **Easy experimentation**: Modify parameters and re-run
- **Well-documented**: Markdown cells explain each step

### 🔬 Data Processing
- ✅ Load MNIST from .npy files
- ✅ Binary class filtering (configurable digits)
- ✅ Image normalization (0-255 → 0-1)
- ✅ Flattening (28×28 → 784)
- ✅ Feature standardization (StandardScaler)
- ✅ PCA dimensionality reduction
- ✅ Train/Validation/Test split (70/15/15)

### 🤖 Machine Learning Algorithms (From Scratch)
1. **Logistic Regression**
   - Gradient descent optimization
   - Sigmoid activation
   - Binary cross-entropy loss
   - Training loss visualization

2. **K-Nearest Neighbors**
   - Euclidean distance
   - Majority voting
   - K-value experimentation
   - Performance comparison

3. **Decision Tree**
   - Gini impurity
   - Recursive tree building
   - Depth experimentation
   - Overfitting analysis

### 📊 Evaluation & Visualization
- ✅ Accuracy, Precision, Recall, F1 Score
- ✅ Confusion matrices for all models
- ✅ Model comparison charts
- ✅ PCA explained variance plots
- ✅ Radar charts
- ✅ Error analysis
- ✅ Comprehensive dashboard

## 🎛️ Configuration

Edit `src/config.py` or modify directly in notebooks:

```python
# Binary classification
DIGIT_A = 0
DIGIT_B = 1

# PCA
PCA_COMPONENTS = 50

# Logistic Regression
LR_LEARNING_RATE = 0.01
LR_ITERATIONS = 1000

# K-Nearest Neighbors
KNN_K = 5

# Decision Tree
DT_MAX_DEPTH = 10
DT_MIN_SAMPLES_SPLIT = 2
```

## 📊 Expected Results

For digits 0 vs 1 with default configuration:

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~99% | ~99% | ~99% | ~99% |
| K-Nearest Neighbors | ~98% | ~98% | ~98% | ~98% |
| Decision Tree | ~96% | ~96% | ~96% | ~96% |

## 📖 Documentation

- **[Quick Reference](docs/QUICK_REFERENCE.md)** - Quick commands and tips
- **[Usage Guide](docs/USAGE_GUIDE.md)** - Comprehensive examples
- **[Project Structure](docs/PROJECT_STRUCTURE.md)** - Detailed module descriptions
- **[Pipeline Flow](docs/PIPELINE_FLOW.md)** - Visual pipeline diagrams

## 🎓 Notebook Workflow

### Notebook 1: Data Preprocessing
- Load MNIST dataset
- Visualize sample images
- Filter binary classes
- Split data (train/val/test)
- Normalize and flatten
- Apply PCA
- Save preprocessed data

### Notebook 2: Logistic Regression
- Load preprocessed data
- Implement Logistic Regression from scratch
- Train with gradient descent
- Visualize training loss
- Evaluate on train/val/test
- Display confusion matrix
- Save trained model

### Notebook 3: K-Nearest Neighbors
- Load preprocessed data
- Implement KNN from scratch
- Train (store data)
- Evaluate on val/test
- Experiment with different K values
- Compare K performance
- Save trained model

### Notebook 4: Decision Tree
- Load preprocessed data
- Implement Decision Tree from scratch
- Train with Gini impurity
- Evaluate on train/val/test
- Experiment with different depths
- Analyze overfitting
- Save trained model

### Notebook 5: Model Comparison
- Load all trained models
- Compare all metrics
- Generate comparison charts
- Create confusion matrices
- Build radar charts
- Error analysis
- Comprehensive dashboard

## 💡 Usage Examples

### Change Classification Digits
```python
# In notebook or config.py
DIGIT_A = 3
DIGIT_B = 8
```

### Adjust PCA Components
```python
PCA_COMPONENTS = 100  # Use more features
```

### Experiment with Hyperparameters
```python
# Logistic Regression
LR_LEARNING_RATE = 0.05
LR_ITERATIONS = 2000

# KNN
KNN_K = 7

# Decision Tree
DT_MAX_DEPTH = 15
```

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Jupyter not found | `pip install jupyter` or `pip install jupyterlab` |
| KNN too slow | Reduce training size or PCA components |
| LR not converging | Increase iterations or adjust learning rate |
| DT overfitting | Reduce max_depth |
| Import errors | Run `pip install -r requirements.txt` |
| Data not found | Check `mnist.npz/` directory |

## 📦 Dependencies

```
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
tqdm>=4.62.0
```

**Note**: All ML algorithms are implemented from scratch. Scikit-learn is only used for preprocessing (StandardScaler, PCA) and displaying metrics.

## 🎓 Academic Requirements Met

✅ Binary image classification  
✅ MNIST dataset from .npy files  
✅ Three ML algorithms from scratch  
✅ Complete preprocessing pipeline  
✅ Feature engineering (PCA)  
✅ Train/Validation/Test split  
✅ Comprehensive evaluation metrics  
✅ Professional visualizations  
✅ Interactive notebooks with inline outputs  
✅ Clean, modular code  
✅ Well-documented  
✅ Reproducible (random seed)  
✅ Organized folder structure  

## 🌟 Key Advantages of Notebook Structure

1. **Interactive Learning**: Execute code step-by-step
2. **Immediate Feedback**: See outputs right away
3. **Easy Debugging**: Test and modify individual cells
4. **Rich Documentation**: Mix code, text, and visualizations
5. **Reproducible**: Save outputs with the notebook
6. **Shareable**: Easy to share with instructors/teammates
7. **Professional**: Industry-standard data science workflow

## 📝 License

This is an academic project for educational purposes.

## 👥 Authors

ML Project Team - Phase 1 Implementation

## 🙏 Acknowledgments

- MNIST dataset by Yann LeCun
- Scikit-learn for preprocessing utilities
- Matplotlib and Seaborn for visualization tools
- Jupyter Project for interactive computing

---

**Ready to start?** Open `notebooks/01_data_preprocessing.ipynb` and begin your journey! 🚀
