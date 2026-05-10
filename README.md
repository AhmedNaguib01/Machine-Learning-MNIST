# MNIST Classification Pipeline - Complete Implementation

A comprehensive, modular implementation of both binary and multiclass image classification using MNIST dataset with multiple machine learning algorithms implemented from scratch. **Organized with interactive Jupyter notebooks for exploration and learning!**

## 🎯 Project Overview

This project implements a complete Machine Learning pipeline in two phases:

### Phase 1: Binary Classification
- **Binary classification** of MNIST digits (configurable digit pairs)
- **Three ML algorithms from scratch**: Logistic Regression, K-Nearest Neighbors, Linear SVM
- **Interactive Jupyter notebooks** with outputs displayed under each cell
- **Complete pipeline**: Data loading → Preprocessing → Feature engineering → Training → Evaluation → Visualization

### Phase 2: Multiclass Classification (10-Class MNIST)
- **10-class classification** of all MNIST digits (0-9)
- **Advanced Logistic Regression** with One-vs-Rest strategy
- **Regularization techniques**: L1 (Lasso), L2 (Ridge), ElasticNet
- **Bias-Variance analysis** for model understanding
- **Advanced feature engineering**: PCA, HOG features, standardization
- **Hyperparameter tuning** with cross-validation
- **Learning curves** and comprehensive evaluation

## 📁 Project Structure

```
Machine-Learning-MNIST/
│
├── notebooks/                                # 📓 Jupyter Notebooks (Complete Pipelines)
│   ├── Phase 1/                             # Binary Classification (2-class)
│   │   ├── knn_binary.ipynb                 # KNN binary classification
│   │   ├── linear_svm_binary.ipynb          # Linear SVM binary classification
│   │   └── logistic_regression_binary.ipynb # Logistic Regression binary classification
│   │
│   ├── Phase 2/                             # Multiclass Classification (10-class)
│   │   └── logistic_regression_multiclass.ipynb # Complete multiclass pipeline with:
│   │                                        #   - One-vs-Rest Logistic Regression
│   │                                        #   - L1/L2/ElasticNet Regularization
│   │                                        #   - Bias-Variance Analysis
│   │                                        #   - PCA & HOG Feature Engineering
│   │                                        #   - Hyperparameter Tuning
│   │                                        #   - Learning Curves
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
- `notebooks/Phase 2/logistic_regression_multiclass.ipynb` - **Complete multiclass pipeline!**

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
- ✅ Class balancing (use minimum class count)
- ✅ Image normalization (0-255 → 0-1)
- ✅ Flattening (28×28 → 784)
- ✅ Feature standardization (StandardScaler)
- ✅ PCA dimensionality reduction (95% variance)
- ✅ HOG feature extraction
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

**Logistic Regression (One-vs-Rest) - Complete Implementation**

The Phase 2 notebook includes a comprehensive multiclass classification pipeline:

1. **Core Algorithm**
   - One-vs-Rest strategy for 10-class classification
   - Softmax-based probability computation
   - Cross-entropy loss optimization
   - Gradient descent with configurable learning rate

2. **Regularization Techniques** (Section 10)
   - **L2 Regularization (Ridge)**: Penalizes large weights, prevents overfitting
   - **L1 Regularization (Lasso)**: Encourages sparsity, feature selection
   - **ElasticNet**: Combines L1 and L2 for balanced regularization
   - Comparison of all regularization methods

3. **Bias-Variance Analysis** (Section 11)
   - Bootstrap-based decomposition
   - Bias² computation (underfitting measure)
   - Variance computation (overfitting measure)
   - Total error analysis
   - Visual comparison across regularization methods
   - Interpretation guidelines

4. **Feature Engineering**
   - **PCA Features**: Dimensionality reduction (95% variance retained)
   - **HOG Features**: Histogram of Oriented Gradients for edge detection
   - **Flattened Features**: Raw pixel standardization
   - Performance comparison across all feature types

5. **Hyperparameter Tuning**
   - K-fold cross-validation (k=5)
   - Grid search over learning rates [0.001, 0.01, 0.05]
   - Grid search over iterations [300, 500, 1000]
   - Systematic best parameter selection

6. **Learning Curves**
   - Training vs validation accuracy
   - Sample size impact analysis
   - Overfitting/underfitting detection
   - Model convergence visualization

### 📊 Evaluation & Visualization
- ✅ Accuracy, Precision, Recall, F1 Score for all classes
- ✅ Confusion matrices with heatmaps
- ✅ Classification reports (per-class metrics)
- ✅ Model comparison charts
- ✅ Cross-validation performance tracking
- ✅ Hyperparameter tuning results
- ✅ Learning curves for bias-variance analysis
- ✅ Feature engineering comparison (PCA vs HOG vs Flatten)
- ✅ Regularization impact visualization
- ✅ Bias-variance tradeoff plots
- ✅ Misclassified samples visualization
- ✅ Comprehensive evaluation reports

## 🛠️ ML Utilities (`ml_utils.py`)

The `ml_utils.py` module contains all shared utilities and algorithms:

### Data Processing
- **`train_test_split()`**: Stratified train/test splitting
- **`StandardScaler`**: Feature standardization (zero mean, unit variance)

### Model Evaluation
- **`confusion_matrix()`**: Compute confusion matrix
- **`classification_report()`**: Detailed per-class metrics
- **`compute_accuracy()`**: Simple accuracy calculation
- **`evaluate_model()`**: Complete evaluation with visualization

### Cross-Validation & Tuning
- **`k_fold_split()`**: K-fold cross-validation splitting
- **`cross_validate()`**: Cross-validation with any model
- **`plot_learning_curve()`**: Learning curve visualization

### Advanced Algorithms
- **`LogisticRegressionRegularized`**: Logistic regression with L1/L2/ElasticNet
  - Configurable penalty type ('l1', 'l2', 'elasticnet')
  - Regularization strength parameter (C)
  - One-vs-Rest multiclass support
  
- **`bias_variance_decomposition()`**: Bootstrap-based bias-variance analysis
  - Computes bias², variance, and total error
  - Configurable bootstrap samples (default: 50)
  - Works with any model class
  
- **`plot_bias_variance()`**: Visualize bias-variance tradeoff
  - Comparative bar charts
  - Multiple models comparison
  - Clear interpretation

### Visualization
- **`show_misclassified()`**: Display misclassified samples with predictions

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
Configuration for multiclass logistic regression:

```python
# Dataset
NUM_CLASSES = 10
TRAIN_SIZE, VAL_SIZE, TEST_SIZE = 0.70, 0.15, 0.15

# Feature Engineering
PCA_VARIANCE = 0.95  # Keep 95% of variance

# Cross-Validation
K_FOLDS = 5
LC_FRACTIONS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]  # Learning curve points

# Hyperparameter Grids
LR_GRID = [0.001, 0.01, 0.05]
ITER_GRID = [300, 500, 1000]

# Default Parameters
DEFAULT_LR = 0.01
DEFAULT_ITER = 500

# Regularization
REGULARIZATION_C = 1.0  # Inverse of regularization strength

# Bias-Variance Analysis
N_BOOTSTRAP = 30  # Number of bootstrap samples
```

## 📊 Expected Results

### Phase 1: Binary Classification (digits 0 vs 1)

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~99% | ~99% | ~99% | ~99% |
| K-Nearest Neighbors | ~98% | ~98% | ~98% | ~98% |
| Linear SVM | ~99% | ~99% | ~99% | ~99% |

### Phase 2: 10-Class Classification (all digits 0-9)

**Logistic Regression (One-vs-Rest) Performance:**

| Feature Type | Test Accuracy | Notes |
|-------------|---------------|-------|
| **HOG Features** | **~95%** | Best performance, captures edge information |
| **PCA Features** | **~92%** | Good dimensionality reduction |
| **Flattened Features** | **~90%** | Baseline performance |

**Regularization Impact:**

| Method | Bias² | Variance | Total Error | Notes |
|--------|-------|----------|-------------|-------|
| **No Regularization** | Low | High | Medium | May overfit |
| **L2 (Ridge)** | Medium | Low | **Lowest** | Best generalization |
| **L1 (Lasso)** | Medium | Low | Low | Feature selection |
| **ElasticNet** | Medium | Low | Low | Balanced approach |

**Key Insights:**
- **HOG features** consistently outperform PCA and flattened features
- **L2 regularization** typically provides the best bias-variance tradeoff
- **Regularization** reduces variance (overfitting) at the cost of slight bias increase
- **Cross-validation** is essential for proper hyperparameter selection
- **Learning curves** help identify if more data would improve performance

## 📖 Notebook Workflow

### Phase 1: Binary Classification Notebooks
Each notebook follows this structure:
1. **Setup & Imports** - Load libraries and set random seed
2. **Configuration** - Set hyperparameters and digit pairs
3. **Data Loading & Preprocessing** - Load, filter, normalize, split
4. **Feature Engineering** - Flatten, standardize, PCA
5. **Algorithm Implementation** - From-scratch ML algorithm
6. **Training** - Fit model with visualization
7. **Evaluation** - Metrics, confusion matrix, analysis

### Phase 2: Multiclass Classification Notebook

The `logistic_regression_multiclass.ipynb` notebook includes:

**Section 1-3: Setup & Data Preparation**
- Library imports and configuration
- MNIST loading (70,000 samples)
- Class balancing and train/val/test split (70/15/15)
- Dataset visualization

**Section 4: Feature Extraction**
- **4a. PCA Features**: Dimensionality reduction (95% variance)
- **4b. HOG Features**: Histogram of Oriented Gradients
- Feature visualization and comparison

**Section 6: Logistic Regression Model**
- One-vs-Rest implementation
- Sigmoid activation and cross-entropy loss
- Gradient descent optimization

**Section 7: Baseline Training**
- Training on PCA features
- Initial performance evaluation
- Confusion matrix visualization

**Section 8: Hyperparameter Tuning**
- K-fold cross-validation (k=5)
- Grid search over learning rates and iterations
- Best parameter selection

**Section 9: Learning Curve**
- Training vs validation accuracy
- Sample size impact analysis
- Overfitting/underfitting detection

**Section 10: Regularization (L1/L2)** ⭐ NEW
- **10a. L2 Regularization (Ridge)**: Weight penalty for smoothness
- **10b. L1 Regularization (Lasso)**: Sparsity-inducing penalty
- **10c. ElasticNet**: Combined L1 and L2
- **10d. Comparison**: Performance across all methods

**Section 11: Bias-Variance Analysis** ⭐ NEW
- **11a. Compute Decomposition**: Bootstrap-based analysis
- **11b. Display Results**: Bias², variance, and error metrics
- **11c. Visualize Tradeoff**: Comparative bar charts
- **11d. Interpretation**: Understanding the results

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

# Adjust regularization strength
REGULARIZATION_C = 0.5  # Stronger regularization (smaller C = stronger penalty)

# Experiment with learning rates
LR_GRID = [0.001, 0.005, 0.01, 0.05, 0.1]

# More bootstrap samples for bias-variance
N_BOOTSTRAP = 50  # More accurate but slower

# Extended learning curve analysis
LC_FRACTIONS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
```

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Jupyter not found | `pip install jupyter` or `pip install jupyterlab` |
| Import errors | Run `pip install -r requirements.txt` |
| Data not found | Check `mnist.npz/` directory exists with .npy files |
| Memory issues | Use PCA features instead of flattened features |
| Slow training | Reduce iterations or use smaller dataset |
| LR not converging | Increase iterations or adjust learning rate |
| Bias-variance slow | Reduce `N_BOOTSTRAP` from 50 to 20-30 |
| HOG extraction slow | Use PCA features for faster experimentation |
| Regularization not helping | Try different C values (0.1, 1.0, 10.0) |

## 📦 Dependencies

```
numpy>=1.21.0            # Numerical computations
scikit-learn>=1.0.0      # Preprocessing (StandardScaler, PCA) and metrics
matplotlib>=3.4.0        # Visualization
seaborn>=0.11.0          # Statistical plots
jupyter>=1.0.0           # Interactive notebooks
jupyterlab>=3.0.0        # Modern notebook interface
scikit-image>=0.18.0     # HOG feature extraction
tqdm>=4.62.0             # Progress bars
```

**Note**: All ML algorithms are implemented from scratch. External libraries are only used for:
- **Preprocessing**: StandardScaler, PCA (scikit-learn)
- **Feature extraction**: HOG (scikit-image)
- **Visualization**: Matplotlib, Seaborn
- **Metrics display**: Classification report formatting
- **Core ML logic is entirely custom-built** ✅

## 🎓 Academic Requirements Met

### Phase 1: Binary Classification ✅
✅ Binary image classification  
✅ MNIST dataset from .npy files  
✅ Three ML algorithms from scratch  
✅ Complete preprocessing pipeline  
✅ Feature engineering (PCA)  
✅ Train/Validation/Test split  
✅ Comprehensive evaluation metrics  
✅ Professional visualizations  

### Phase 2: Multiclass Classification ✅
✅ 10-class MNIST classification  
✅ Logistic Regression (One-vs-Rest) from scratch  
✅ **Regularization techniques** (L1, L2, ElasticNet)  
✅ **Bias-Variance decomposition** and analysis  
✅ Advanced feature engineering (PCA, HOG, Standardization)  
✅ Cross-validation and hyperparameter tuning  
✅ Learning curves for model diagnosis  
✅ Comprehensive performance comparison  
✅ Feature engineering impact analysis  
✅ Misclassification analysis  

### Technical Excellence ✅
✅ Interactive notebooks with inline outputs  
✅ Clean, modular code organization  
✅ Well-documented with markdown explanations  
✅ Reproducible results (random seed)  
✅ Professional-grade visualizations  
✅ Industry-standard data science workflow  
✅ Complete error analysis and model interpretation  
✅ Reusable utility functions in `ml_utils.py`  
✅ Bootstrap-based statistical analysis  

## 🌟 Key Advantages

### Comprehensive Learning Path
1. **Progressive Complexity**: Start with binary classification, advance to multiclass
2. **Multiple Algorithms**: Compare different ML approaches (Phase 1)
3. **Advanced Techniques**: Regularization and bias-variance analysis (Phase 2)
4. **Feature Engineering**: Understand the impact of different feature representations

### Interactive Development
1. **Jupyter Notebooks**: Execute code step-by-step with immediate feedback
2. **Rich Visualizations**: See results, plots, and metrics inline
3. **Easy Experimentation**: Modify parameters and re-run individual cells
4. **Professional Workflow**: Industry-standard data science practices

### Educational Value
1. **From-Scratch Implementation**: Understand algorithms at the fundamental level
2. **Regularization**: Learn to prevent overfitting with L1/L2/ElasticNet
3. **Bias-Variance Tradeoff**: Deep understanding of model behavior
4. **Hyperparameter Tuning**: Systematic approach to model optimization
5. **Cross-Validation**: Proper model evaluation and selection techniques
6. **Performance Analysis**: Comprehensive evaluation with multiple metrics

### Real-World Relevance
1. **Complete Pipeline**: End-to-end ML project structure
2. **Multiple Feature Types**: PCA, HOG, and raw features comparison
3. **Scalable Code**: Modular design for easy extension
4. **Reproducible Results**: Proper random seeding and documentation
5. **Statistical Rigor**: Bootstrap-based analysis for robust conclusions

## 🔬 Advanced Topics Covered

### Regularization
- **Purpose**: Prevent overfitting by penalizing complex models
- **L1 (Lasso)**: Encourages sparsity, useful for feature selection
- **L2 (Ridge)**: Smooth weight distribution, better generalization
- **ElasticNet**: Combines benefits of L1 and L2
- **Implementation**: Custom gradient computation with penalty terms

### Bias-Variance Tradeoff
- **Bias**: Error from overly simplistic assumptions (underfitting)
- **Variance**: Error from sensitivity to training data (overfitting)
- **Decomposition**: Bootstrap sampling to separate bias and variance
- **Analysis**: Visual comparison across different regularization methods
- **Interpretation**: Guidelines for model selection and improvement

### Feature Engineering
- **PCA**: Dimensionality reduction while preserving variance
- **HOG**: Edge and gradient information for image classification
- **Comparison**: Systematic evaluation of feature impact on performance

## 📝 License

This is an academic project for educational purposes.

## 👥 Authors

ML Project Team - Complete Implementation (Phase 1 & Phase 2)

## 🙏 Acknowledgments

- **MNIST Dataset** by Yann LeCun and Corinna Cortes
- **Scikit-learn** for preprocessing utilities and evaluation metrics
- **Scikit-image** for HOG feature extraction
- **Matplotlib & Seaborn** for comprehensive visualization tools
- **Jupyter Project** for interactive computing environment
- **NumPy** for efficient numerical computations

---

## 🚀 Getting Started

### For Beginners
**Start with Phase 1:** Open `notebooks/Phase 1/logistic_regression_binary.ipynb` to learn binary classification fundamentals!

### For Advanced Users
**Jump to Phase 2:** Open `notebooks/Phase 2/logistic_regression_multiclass.ipynb` for the complete multiclass pipeline with regularization and bias-variance analysis!

### Quick Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Start Jupyter Notebook
jupyter notebook

# Or start JupyterLab (modern interface)
jupyter lab
```

---

**🎓 Happy Learning!** 

*This project demonstrates a complete machine learning workflow from data loading to advanced model analysis, suitable for both learning and academic projects.*
