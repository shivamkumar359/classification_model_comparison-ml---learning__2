# Classification Model Comparison

A beginner-friendly machine learning project that compares multiple classification algorithms on a synthetic binary-classification dataset. The notebook demonstrates a complete baseline workflow: data generation, train/test splitting, model training, accuracy evaluation, and visual comparison of results.

## Project Objective

The objective is to understand how different classifiers behave when trained on the same dataset. Rather than assuming one algorithm is always best, this project evaluates several models under the same experimental conditions and compares their test-set accuracy.

## Workflow

1. Generate a synthetic binary-classification dataset with Scikit-learn.
2. Split the data into training and testing sets.
3. Train multiple classification algorithms.
4. Evaluate each model using test accuracy.
5. Store the results in a Pandas DataFrame.
6. Visualize model performance using a Seaborn bar chart.

## Dataset

The dataset is created with `sklearn.datasets.make_classification`.

| Property | Value |
|---|---:|
| Samples | 1,000 |
| Features | 5 |
| Classes | 2 |
| Redundant features | 0 |
| Split | 80% training / 20% testing |

The target classes are approximately balanced. `stratify=Y` is used in the split so that the training and test sets retain similar class proportions.

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)
```

## Models Compared

The notebook evaluates the following classifiers:

- K-Nearest Neighbors
- Linear Support Vector Machine
- Polynomial Support Vector Machine
- RBF Support Vector Machine
- Gaussian Process Classifier
- Gradient Boosting Classifier
- Decision Tree Classifier
- Extra Trees Classifier
- Random Forest Classifier
- Multi-Layer Perceptron Neural Network
- AdaBoost Classifier
- Gaussian Naive Bayes
- Quadratic Discriminant Analysis
- Stochastic Gradient Descent Classifier

## Technologies Used

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Scikit-learn
- Seaborn
- Matplotlib

## Installation

Install the required packages:

```bash
pip install numpy pandas scikit-learn seaborn matplotlib jupyter
```

## Running the Project

1. Clone or download this repository.
2. Open the project directory in a terminal.
3. Start Jupyter Notebook:

```bash
jupyter notebook
```

4. Open `classification_model_comparison_beginner.ipynb`.
5. Run all cells from top to bottom.

## Evaluation

Each classifier is fitted using the training data and evaluated using the held-out test data.

```python
clf.fit(X_train, Y_train)
test_accuracy = clf.score(X_test, Y_test)
```

The notebook saves model names and scores in a DataFrame, then presents a horizontal bar chart. Different colors are applied to the bars so models are easier to distinguish visually.

> Results will vary when the generated dataset, split seed, model settings, or library version changes. The purpose is to compare models under the same conditions, not to claim a universally best model.

## Learning Outcomes

After completing this notebook, you should understand:

- The distinction between features (`X`) and labels (`Y`)
- Why data is divided into training and testing sets
- How `random_state` supports reproducible experiments
- Why `stratify` is useful for classification datasets
- How to train and evaluate Scikit-learn classifiers
- Why different algorithms can produce different results
- How to compare model performance in a structured DataFrame
- How to visualize model metrics with Seaborn

## Recommended Next Improvements

This project intentionally focuses on the core workflow. The following additions are recommended as the next learning steps:

1. Compare training accuracy and test accuracy to identify overfitting.
2. Add a confusion matrix and classification report for selected models.
3. Apply feature scaling with `StandardScaler` for KNN, SVM, MLP, and SGD models.
4. Use cross-validation instead of relying on one train/test split.
5. Tune hyperparameters with `GridSearchCV` or `RandomizedSearchCV`.
6. Repeat the workflow on a real dataset such as Breast Cancer, Titanic, Customer Churn, or Loan Approval data.
7. Compare accuracy with precision, recall, F1-score, and ROC-AUC when classes are imbalanced.

## Project Structure

```text
.
├── classification_model_comparison_beginner.ipynb
└── README.md
```

## Notes for Beginners

- A higher test score is useful, but it is not the only measure of a good model.
- Avoid choosing a model based solely on one run; model performance can change with a different split.
- Keep the test set separate from training to measure how well a model generalizes to unseen data.
- Change one parameter at a time during experiments so you can understand the effect of that change.

## License

This repository is intended for educational and learning purposes.
