# Sci-Kit Learn

## Introduction

Sci-Kit Learn is probably the most popular library in Python for classical machine learning.

It is built on top of NumPy, SciPy, and matplotlib. It contains a number of efficient tools for machine learning and 
statistical modeling including classification, regression, clustering and dimensionality 
reduction. It is designed to interoperate with the Python numerical and scientific libraries 
NumPy and SciPy.

!!!note
    Sci-Kit Learn is not designed to create neural networks, for which the most popular libraries are
    PyTorch and TensorFlow + Keras.

## Installation

To install Sci-Kit Learn, we can use `pip`:

```bash
pip install scikit-learn
```

or, alternatively, `poetry`:

```bash
poetry add scikit-learn
```

## Basic Usage

Sci-Kit Learn is a very large library, so we will only cover the basics here. For more information,
check the [official documentation](https://scikit-learn.org/stable/).

The most common words in Sci-Kit Learn are:

- **Estimator**: An estimator is any object that learns from data. It may be a classification, regression or clustering
  algorithm or a transformer that extracts or filters useful features from raw data.
- **Transformer**: A transformer is an estimator that transforms data. For example, an imputer is a transformer
  that replaces missing values, and a scaler is a transformer that scales data.
- **Predictor**: A predictor is an estimator that predicts values based on data. For example, a classifier is a
  predictor that predicts classes based on data.
- **Pipeline**: A pipeline is a sequence of transformers and predictors. Pipelines are very useful to encapsulate
  the preprocessing and modeling steps in a machine learning workflow.
- **Model selection**: Model selection is the process of choosing the best model for a given dataset. Sci-Kit Learn
  provides a number of tools to perform model selection, including cross-validation and grid search.
- **Hyperparameter**: A hyperparameter is a parameter that is not learned by the model, but that is set before
  training. For example, the number of neighbors in a KNN classifier is a hyperparameter.
- **Grid search**: Grid search is a technique to find the best hyperparameters for a given model. It consists of
  trying all possible combinations of hyperparameters and choosing the one that performs best.
- **Cross-validation**: Cross-validation is a technique to evaluate a model. It consists of splitting the data into
  training and test sets, training the model on the training set, and evaluating it on the test set. This process is
  repeated several times, and the results are averaged to obtain a more robust evaluation.
- **Metrics**: Metrics are functions that measure the performance of a model. For example, the accuracy is a metric
  that measures the proportion of correct predictions made by a classifier.
- **Classification**: Classification is the task of predicting a class from a set of classes. For example, predicting
  whether an email is spam or not is a classification task.
- **Regression**: Regression is the task of predicting a continuous value. For example, predicting the price of a
  house is a regression task.
- **Clustering**: Clustering is the task of grouping data points into clusters. For example, grouping customers into
  clusters based on their purchase history is a clustering task.
- **Dimensionality reduction**: Dimensionality reduction is the task of reducing the number of features in a dataset.
  For example, reducing the number of features in an image to the most important ones is a dimensionality reduction
  task.
- **Ensemble**: An ensemble is a model that combines the predictions of several models. For example, a random forest
  is an ensemble of decision trees.

The Sci-Kit Learn API is designed to be as simple as possible. It consists of a number of classes that implement
different algorithms, and a number of functions that implement utilities. The classes are called estimators, and
the functions are called helpers.

### Estimators

Estimators are classes that implement machine learning algorithms. They have two main methods: `fit` and `predict`.

The `fit` method is used to train the model. It takes as input the training data and the training labels, and
trains the model. For example, to train a KNN classifier, we can do the following:

```python
from sklearn.neighbors import KNeighborsClassifier

X_train = ...
y_train = ...

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
```

The `predict` method is used to make predictions. It takes as input the data to predict, and returns the predictions.
For example, to predict the labels of a test set, we can do the following:

```python
X_test = ...

y_pred = knn.predict(X_test)
```





### Datasets

Sci-Kit Learn contains a number of datasets that can be used to test machine learning algorithms.
To load a dataset, we can use the `load_*` functions:

```python
from sklearn.datasets import load_iris

iris = load_iris()
```

The `iris` object is a dictionary-like object that contains the data, the target, and other information
about the dataset:

```python
>>> iris.keys()
dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
```

The `data` key contains the data, which is a NumPy array:

```python
print(iris['data'])

# Output:
array([[5.1, 3.5, 1.4, 0.2],
       [4.9, 3. , 1.4, 0.2],
       [4.7, 3.2, 1.3, 0.2],
       [4.6, 3.1, 1.5, 0.2],
       [5. , 3.6, 1.4, 0.2],
       [5.4, 3.9, 1.7, 0.4],
       [4.6, 3.4, 1.4, 0.3],
       [5. , 3.4, 1.5, 0.2],
       [4.4, 2.9, 1.4, 0.2],
       [4.9, 3.1, 1.5, 0.1],
       [5.4, 3.7, 1.5, 0.2],
       [4.8, 3.4, 1.6, 0.2],
       [4.8, 3. , 1.4, 0.1],
       [4.3, 3. , 1.1, 0.1],
       [5.8, 4. , 1.2, 0.2],
       [5.7, 4.4, 1.5, 0.4],
       [5.4, 3.9, 1.3, 0.4],
       [5.1, 3.5, 1.4, 0.3],
       [5.7, 3.8, 1.7, 0.3],
       [5.1, 3.8, 1.5, 0.3],
       [5.4, 3.4, 1.7, 0.2],
       [5.1, 3.7, 1.5, 0.4],
       [4.6, 3.6, 1. , 0.2],
       [5.1, 3.3, 1.7, 0.5],
       [
```

The `target` key contains the target, which is also a NumPy array:

```python
print(iris['target'])

# Output:
array([

    
