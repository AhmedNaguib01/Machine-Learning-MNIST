# MNIST Classification Pipeline - Complete Implementation

A comprehensive, modular implementation of both binary and multiclass image classification using MNIST dataset with multiple machine learning algorithms implemented from scratch. **Organized with interactive Jupyter notebooks for exploration and learning!**

## 🎯 Project Overview

This project implements a complete Machine Learning pipeline in two phases:

### Phase 1: Binary Classification
- **Binary classification** of MNIST digits (configurable digit pairs)
- **Three ML algorithms from scratch**: Logistic Regression, K-Nearest Neighbors, Decision Tree
- **Interactive Jupyter notebooks** with outputs displayed under each cell
- **Complete pipeline**: Data loading → Preprocessing → Feature engineering → Training → Evaluation → Visualization

### Phase 2: Multiclass Classification (10-Class MNIST)
- **10-class classification** of all MNIST digits (0-9)
- **Five advanced ML algorithms from scratch**: 
  - K-Nearest Neighbors (multiclass)
  - Linear SVM (One-vs-Rest)
  - Logistic Regression (One-vs-Rest)
  - Random Forest Ensemble
  - Gradient Boosting Ensemble
- **Advanced feature engineering**: PCA, HOG features, standardization
- **Hyperparameter tuning** with cross-validation
- **Learning curves** and comprehensive evaluation

## 📁 Project Structure

```
mnist-classification-pipeline/
│
├── notebooks/                                # 📓 Jupyter Notebooks (Complete Pipelines)
│   ├── Phase 1/                             # Binary Classification (2-class)
│   │   ├── knn_binary.ipynb                 # KNN binary classification
│   │   ├── linear_svm_binary.ipynb          # Linear SVM binary classification
│   │   └── logistic_regression_binary.ipynb # Logistic Regression binary classification
│   │
│   ├── Phase 2/                             # Multiclass Classification (10-class)
│   │   ├── knn_multiclass.ipynb             # KNN multiclass (k=5, tuned)
│   │   ├── linear_svm_multiclass.ipynb      # Linear SVM One-vs-Rest
│   │   ├── logistic_regression_multiclass.ipynb # Logistic Regression One-vs-Rest
│   │   ├── random_forest_ensemble.ipynb     # Random Forest (30 trees)
│   │   └── gradient_boosting_ensemble.ipynb # Gradient Boosting (30 estimators)
│   │
│   ├── ml_utils.py                          # Shared ML utilities and algorithms
│   └── __pycache__/                         # Python cache files
│
├── mnist.npz/                               # 📦 Raw MNIST Dataset
│   ├── x_train.npy                          # Training images (60,000 samples)
│   ├── y_train.npy                          # Training labels
│   ├── x_test.npy                           # Test images (10,000 samples)
│   └── y_test.npy                           # Test labels
│
├── .gitignore                               # Git ignore file
├── README.md                                # This file
└── requirements.txt                         # Python dependencies
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

**Phase 1 (Binary Classification):**
- `notebooks/Phase 1/knn_binary.ipynb`
- `notebooks/Phase 1/linear_svm_binary.ipynb`
- `notebooks/Phase 1/logistic_regression_binary.ipynb`

**Phase 2 (10-Class Classification):**
- `notebooks/Phase 2/knn_multiclass.ipynb`
- `notebooks/Phase 2/linear_svm_multiclass.ipynb`
- `notebooks/Phase 2/logistic_regression_multiclass.ipynb`
- `notebooks/Phase 2/random_forest_ensemble.ipynb`
- `notebooks/Phase 2/gradient_boosting_ensemble.ipynb`

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

#### Phase 1: Binary Classification
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

3. **Linear SVM**
   - Hinge loss optimization
   - Gradient descent
   - Binary classification
   - Margin maximization

#### Phase 2: Multiclass Classification (10-Class MNIST)
1. **K-Nearest Neighbors (Multiclass)**
   - Euclidean distance computation
   - Majority voting for 10 classes
   - Cross-validation tuning (k=1,3,5,7,9)
   - **Best Performance: 95% accuracy** with k=3

2. **Linear SVM (One-vs-Rest)**
   - Hinge loss with L2 regularization
   - One-vs-Rest strategy for multiclass
   - Gradient descent optimization
   - **Best Performance: 98% accuracy** with HOG features

3. **Logistic Regression (One-vs-Rest)**
   - Softmax activation for multiclass
   - Cross-entropy loss
   - One-vs-Rest binary classifiers
   - **Best Performance: 95% accuracy** with HOG features

4. **Random Forest Ensemble**
   - 30 decision trees with bootstrap sampling
   - Gini impurity splitting criterion
   - Feature bagging (sqrt features per tree)
   - **Best Performance: 96% accuracy** with HOG features

5. **Gradient Boosting Ensemble**
   - 30 weak learners (depth-3 trees)
   - Sequential boosting with residual fitting
   - Learning rate: 0.1
   - **Best Performance: 85% accuracy** (challenging for from-scratch implementation)

### 📊 Evaluation & Visualization
- ✅ Accuracy, Precision, Recall, F1 Score for all classes
- ✅ Confusion matrices for all models
- ✅ Model comparison charts and performance analysis
- ✅ Cross-validation with k-fold splitting
- ✅ Hyperparameter tuning with grid search
- ✅ Learning curves for bias-variance analysis
- ✅ Feature engineering comparison (PCA vs HOG vs Flatten)
- ✅ Comprehensive evaluation reports
- ✅ Error analysis and misclassification visualization

## 🎛️ Configuration

### Phase 1 (Binary Classification)
Edit configuration directly in notebooks:

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

# Linear SVM
SVM_LEARNING_RATE = 0.001
SVM_LAMBDA = 0.01
SVM_ITERATIONS = 200
```

### Phase 2 (10-Class Classification)
Configuration for multiclass algorithms:

```python
# Dataset
NUM_CLASSES = 10
TRAIN_SIZE, VAL_SIZE, TEST_SIZE = 0.70, 0.15, 0.15

# Feature Engineering
PCA_VARIANCE = 0.95  # Keep 95% of variance

# Cross-Validation
K_FOLDS = 5

# Algorithm-Specific Parameters
KNN_K_GRID = [1, 3, 5, 7, 9]
DEFAULT_K = 5

LR_GRID = [0.001, 0.01, 0.05]
ITER_GRID = [300, 500, 1000]

SVM_LR = 0.001
SVM_LAMBDA = 0.01
SVM_ITER = 200

RF_N_ESTIMATORS = 30
RF_MAX_DEPTH = 12
RF_MAX_FEATURES = 'sqrt'

GB_N_ESTIMATORS = 30
GB_MAX_DEPTH = 3
GB_LEARNING_RATE = 0.1
```

## 📊 Expected Results

### Phase 1: Binary Classification (digits 0 vs 1)

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~99% | ~99% | ~99% | ~99% |
| K-Nearest Neighbors | ~98% | ~98% | ~98% | ~98% |
| Linear SVM | ~99% | ~99% | ~99% | ~99% |

### Phase 2: 10-Class Classification (all digits 0-9)

| Model | Best Feature | Test Accuracy | Notes |
|-------|-------------|---------------|-------|
| **Linear SVM** | HOG Features | **98%** | Best overall performer |
| **Random Forest** | HOG Features | **96%** | Robust ensemble method |
| **KNN (k=3)** | PCA Features | **95%** | Simple but effective |
| **Logistic Regression** | HOG Features | **95%** | Good baseline |
| **Gradient Boosting** | HOG Features | **85%** | Complex from-scratch implementation |

**Key Insights:**
- **HOG features** consistently outperform PCA and flattened features
- **Linear SVM** achieves the highest accuracy (98%) on 10-class MNIST
- **Ensemble methods** (Random Forest) provide good robustness
- **Feature engineering** is crucial for performance (HOG > PCA > Flatten)

## 📖 Documentation

- **[Quick Reference](docs/QUICK_REFERENCE.md)** - Quick commands and tips
- **[Usage Guide](docs/USAGE_GUIDE.md)** - Comprehensive examples
- **[Project Structure](docs/PROJECT_STRUCTURE.md)** - Detailed module descriptions
- **[Pipeline Flow](docs/PIPELINE_FLOW.md)** - Visual pipeline diagrams

## 🎓 Notebook Workflow

### Phase 1: Binary Classification Notebooks
Each notebook follows this structure:
1. **Data Loading & Preprocessing**
   - Load MNIST dataset from .npy files
   - Filter for binary classification (configurable digits)
   - Normalize pixel values (0-255 → 0-1)
   - Train/validation/test split (70/15/15)

2. **Feature Engineering**
   - Flatten images (28×28 → 784 features)
   - Standardization with StandardScaler
   - PCA dimensionality reduction

3. **Algorithm Implementation**
   - From-scratch implementation of ML algorithm
   - Training with appropriate optimization
   - Hyperparameter experimentation

4. **Evaluation & Visualization**
   - Performance metrics calculation
   - Confusion matrix visualization
   - Learning curves and analysis

### Phase 2: 10-Class Classification Notebooks
Each Phase 2 notebook includes:

1. **Complete Data Pipeline**
   - Load full MNIST dataset (70,000 samples)
   - Balance classes (use minimum class count)
   - 70/15/15 train/validation/test split
   - Multiple feature engineering approaches

2. **Feature Engineering Comparison**
   - **PCA Features**: Dimensionality reduction (95% variance)
   - **Flattened Features**: Raw pixel standardization
   - **HOG Features**: Histogram of Oriented Gradients

3. **Algorithm Training & Tuning**
   - Baseline model training
   - Cross-validation hyperparameter tuning
   - Performance comparison across feature types

4. **Comprehensive Evaluation**
   - Classification reports for all 10 classes
   - Confusion matrices with heatmaps
   - Learning curves for bias-variance analysis
   - Misclassification analysis

5. **Results Summary**
   - Best configuration identification
   - Performance comparison tables
   - Feature importance analysis

## 💡 Usage Examples

### Phase 1: Binary Classification
```python
# Change classification digits
DIGIT_A = 3
DIGIT_B = 8

# Adjust PCA components
PCA_COMPONENTS = 100

# Experiment with hyperparameters
LR_LEARNING_RATE = 0.05
KNN_K = 7
SVM_LEARNING_RATE = 0.001
```

### Phase 2: Multiclass Experiments
```python
# Try different PCA variance thresholds
PCA_VARIANCE = 0.90  # Keep 90% of variance

# Experiment with ensemble parameters
RF_N_ESTIMATORS = 50  # More trees
RF_MAX_DEPTH = 15     # Deeper trees

GB_N_ESTIMATORS = 50  # More boosting rounds
GB_LEARNING_RATE = 0.05  # Slower learning

# Cross-validation grids
KNN_K_GRID = [1, 3, 5, 7, 9, 11]  # Extended K range
LR_GRID = [0.001, 0.01, 0.05, 0.1]  # More learning rates
```

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Jupyter not found | `pip install jupyter` or `pip install jupyterlab` |
| KNN too slow | Reduce training size or use PCA features |
| LR not converging | Increase iterations or adjust learning rate |
| SVM not converging | Adjust learning rate or regularization |
| RF overfitting | Reduce max_depth or increase min_samples_split |
| GB poor performance | Increase n_estimators or adjust learning_rate |
| Import errors | Run `pip install -r requirements.txt` |
| Data not found | Check `mnist.npz/` directory exists |
| Memory issues | Use PCA features instead of flattened features |
| Slow training | Start with smaller datasets or fewer estimators |

## 📦 Dependencies

```
numpy>=1.21.0
scikit-learn>=1.0.0      # Only for preprocessing (StandardScaler, PCA) and metrics
matplotlib>=3.4.0        # Visualization
seaborn>=0.11.0          # Statistical plots
jupyter>=1.0.0           # Interactive notebooks
tqdm>=4.62.0             # Progress bars
scikit-image>=0.18.0     # HOG feature extraction
```

**Note**: All ML algorithms are implemented from scratch. Scikit-learn is only used for:
- Preprocessing utilities (StandardScaler, PCA)
- HOG feature extraction (scikit-image)
- Evaluation metrics display
- The core ML logic is entirely custom-built

## 🎓 Academic Requirements Met

### Phase 1: Binary Classification
✅ Binary image classification  
✅ MNIST dataset from .npy files  
✅ Three ML algorithms from scratch  
✅ Complete preprocessing pipeline  
✅ Feature engineering (PCA)  
✅ Train/Validation/Test split  
✅ Comprehensive evaluation metrics  
✅ Professional visualizations  

### Phase 2: Multiclass Classification
✅ 10-class MNIST classification  
✅ Five advanced ML algorithms from scratch  
✅ Ensemble methods (Random Forest, Gradient Boosting)  
✅ Advanced feature engineering (PCA, HOG, Standardization)  
✅ Cross-validation and hyperparameter tuning  
✅ Learning curves and bias-variance analysis  
✅ Comprehensive performance comparison  
✅ Feature engineering impact analysis  

### Technical Excellence
✅ Interactive notebooks with inline outputs  
✅ Clean, modular code organization  
✅ Well-documented with markdown explanations  
✅ Reproducible results (random seed)  
✅ Professional-grade visualizations  
✅ Industry-standard data science workflow  
✅ Complete error analysis and model interpretation  

## 🌟 Key Advantages

### Comprehensive Learning Path
1. **Progressive Complexity**: Start with binary classification, advance to multiclass
2. **Multiple Algorithms**: Compare different ML approaches and their strengths
3. **Feature Engineering**: Understand the impact of different feature representations
4. **Ensemble Methods**: Learn advanced techniques like Random Forest and Boosting

### Interactive Development
1. **Jupyter Notebooks**: Execute code step-by-step with immediate feedback
2. **Rich Visualizations**: See results, plots, and metrics inline
3. **Easy Experimentation**: Modify parameters and re-run individual cells
4. **Professional Workflow**: Industry-standard data science practices

### Educational Value
1. **From-Scratch Implementation**: Understand algorithms at the fundamental level
2. **Hyperparameter Tuning**: Learn systematic approach to model optimization
3. **Cross-Validation**: Proper model evaluation and selection techniques
4. **Performance Analysis**: Comprehensive evaluation with multiple metrics

### Real-World Relevance
1. **Complete Pipeline**: End-to-end ML project structure
2. **Multiple Feature Types**: PCA, HOG, and raw features comparison
3. **Scalable Code**: Modular design for easy extension
4. **Reproducible Results**: Proper random seeding and documentation

## 📝 License

This is an academic project for educational purposes.

## 👥 Authors

ML Project Team - Phase 1 Implementation

## 🙏 Acknowledgments

- **MNIST Dataset** by Yann LeCun and Corinna Cortes
- **Scikit-learn** for preprocessing utilities and evaluation metrics
- **Scikit-image** for HOG feature extraction
- **Matplotlib & Seaborn** for comprehensive visualization tools
- **Jupyter Project** for interactive computing environment
- **NumPy** for efficient numerical computations

---

**Ready to start?** 

**For Binary Classification:** Open `notebooks/Phase 1/knn_binary.ipynb` and begin with simple 2-class problems! 

**For Advanced Multiclass:** Jump to `notebooks/Phase 2/linear_svm_multiclass.ipynb` for the best-performing algorithm!

🚀 **Happy Learning!**
