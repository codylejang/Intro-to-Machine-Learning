
# UCLA Intro to Machine Learning (ECE 146) – Course Projects

This repository contains assignments completed for **ECE 146: Introduction to Machine Learning** at UCLA. The projects explore foundational machine learning algorithms through direct implementation in Python, with a strong emphasis on understanding the "under the hood" math and mechanics behind each model.

## Homework 2 – Logistic Regression & Decision Trees
**File:** `Sp24-ECE146-HW2.ipynb`

### Implementations:
- **Logistic Regression:**  
  Implemented using gradient descent and evaluated using accuracy metrics.
- **Decision Tree Classifier:**  
  Constructed a recursive decision tree with Gini impurity as the splitting criterion.
- **Cross-validation:**  
  Used `StratifiedShuffleSplit` to validate model performance.
- **Evaluation:**  
  Measured model accuracy and generated ROC curves for binary classification tasks.

### Dataset:
**File:** `adult_subsample.csv`

## Homework 3 – Model Selection & Twitter Data Classification
**File:** `HW3_release.ipynb`

### Implementations:
- **Text Preprocessing & Feature Engineering:**  
  Tokenized tweets, removed punctuation, and converted text to numerical features (e.g., bag-of-words).
- **Model Training & Comparison:**  
  Trained models including Decision Trees, Linear SVMs, and Random Forest Classifiers.
- **Cross-validation:**  
  Employed `StratifiedKFold` for performance evaluation.
- **Performance Metrics:**  
  Analyzed AUC, ROC curves, loss, and accuracy across classifiers.

### Dataset:
- Real-world Twitter dataset used for sentiment or text classification.

## Homework 4 – Neural Networks & Optimization with FashionMNIST
**File:** `Sp23-ECE146-HW4.ipynb`

### Implementations:
- **Feedforward Neural Network:**  
  Manually implemented 2-layer neural net with adjustable hidden layers.
- **Backpropagation:**  
  Derived and coded the gradient computations for weight updates.
- **Optimizers:**  
  Implemented gradient descent with variable learning rates and optional momentum.
- **Dimensionality Reduction & Clustering:**  
  Used **PCA** for visualization and **KMeans** for exploratory analysis.
- **Performance Tracking:**  
  Visualized training loss and accuracy progression across epochs.

### Dataset:
- **FashionMNIST** image dataset (used for multiclass classification).

## Getting Started

To run the notebooks:

```bash
git clone https://github.com/yourusername/UCLA-ML-ECE146.git
cd UCLA-ML-ECE146

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook
```

## Author

**Cody Lejang**  
B.S. in Cognitive Science, Specialization in Computing, minor in Data Science Engineering – UCLA  
Interested in the intersection of machine learning, psychology, and data analytics.

