# Introduction to Scikit Learn

## Overview

**Scikit Learn** is probably the most popular Python library for classical machine learning. 
Scikit-Learn is characterized by a clean, uniform, and streamlined API, as well as by very 
useful and complete online documentation. A benefit of this uniformity is that once you understand the basic 
use and syntax of Scikit-Learn for one type of model, switching to a new model or algorithm is very straightforward.

Scikit learn is built on top of NumPy, SciPy, and Matplotlib. It contains tools for classical machine learning and
statistical modeling, including classification, regression, clustering and dimensionality 
reduction. In addition, it also provides a selection of sample datasets,
most of which are fairly small and well-known.

!!!note
    Scikit Learn is not designed to create neural networks / work with deep learning. For this 
    the most popular libraries are PyTorch or the tuple TensorFlow + Keras.

!!!note  
    Scikit Learn is a very large library, so we will only cover the basics here. For more information,
    check the [official documentation](https://scikit-learn.org/stable/).

!!!note
    Pandas is not the default data structure in Scikit Learn (which, instead, uses NumPy arrays). However,
    by changing one of the arguments of the load functions, we can use Pandas DataFrames without any problems. 

## Installation

To install Scikit Learn, we can use `pip`:

```bash
pip install scikit-learn
```

or, alternatively, `poetry`:

```bash
poetry add scikit-learn
```

## Datasets

Machine learning is about creating models from data: for that reason, we'll start by discussing how data can be 
represented in order to be understood by the computer. The best way to think about data within Scikit-Learn is 
in terms of tables of data.

### Loading data

A basic table is a two-dimensional grid of data, in which the rows represent individual elements of the 
dataset, and the columns represent quantities related to each of these elements. 

!!!note
    Scikit Learn contains several datasets that can be used to test machine learning algorithms.
    To load a dataset, we can use the `load_*` functions, where `*` is the name of the dataset (e.g. `load_iris`).

For example, consider the Iris dataset, famously analyzed by Ronald Fisher in 1936:

```python
from sklearn import datasets

iris = datasets.load_iris(as_frame=True)
iris_df = iris.frame

print(iris_df.head())

# Output:
    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target target_names
0                 5.1               3.5                1.4               0.2       0       setosa
1                 4.9               3.0                1.4               0.2       0       setosa
2                 4.7               3.2                1.3               0.2       0       setosa
3                 4.6               3.1                1.5               0.2       0       setosa
4                 5.0               3.6                1.4               0.2       0       setosa
```

!!!note
    The `as_frame=True` argument tells Scikit Learn to return a Pandas DataFrame instead of a NumPy array.
    This is useful because we can use the column names to refer to the columns, instead of using the column indices.

After calling the `load_iris()` function, we get the data from the Iris dataset. There are typically two
ways to return the data:

* As `X`, `y` arrays. Here `X` is the data (independent variables) and `y` is the target (dependent variables).
  This option can be chosen by setting `return_X_y=True` when calling the load method.

    !!!note
        The `X` and `y` names are very general and, in the context of machine learning, almost always refer to the
        data and target variables, respectively (not only in Scikit Learn, but in other libraries as well). 

* The other option is a `Bunch` object, a type ofdictionary (which is what we used in the previous
  example). It contains the data, the target, and other information about the dataset:

    ```python
    print(iris.keys())
  
    # Output:
    dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
    ```

    The values of this dictionary are, by default, NumPy arrays, except if we used `as_frame=True` when calling
    the load method, in which case they are Pandas DataFrames:

    * `data`: the data, which is a NumPy array or a Pandas DataFrame. It contains the features of the dataset (i.e,
    the independent variables). In this table, each column is a feature, and each row is an observation.
    * `target`: the target, which is a NumPy array or a Pandas DataFrame. It contains the labels of the dataset (i.e,
      the dependent variable).
    * `frame`: the data and target combined into a single DataFrame.
    * `target_names`: the names of the labels (in this case, the names of the flower species).
    * `DESCR`: the description of the dataset.
    * `feature_names`: the names of the features (in this case, the names of the flower measurements).
    * `filename`: the path to the file containing the dataset.

### Supervised vs Unsupervised problems

When using scikit-learn, we will typically work with tables following this `data` and `target` keys convention.
This is always the case when we are working with **supervised learning** machine learning problems, where we have a
target variable that we want to predict. 

!!!note
    **Supervised problems** are, at a very high level, interpolation problems. We have a set of points, and we want to
    find a function $f(X)$ that passes through all of them, so that $f(X) \sim y$. 

    More formally, we want to minimize the error (known as the loss or cost function), 

    $$
    \min_{f} \mathcal{L}(f(X), y)
    $$
    
    By doing this, we hope that if we use the function in a new point $X'$, we will get a value $f(X')$ that is close to
    the true value of its associated $y'$.

In the Iris dataset, the target variable is the flower species, and the features are the flower measurements, so
if were to use this dataset for a machine learning problem, we would try to construct a model that predicts the 
flower species (dependent variable) from the flower measurements (independent variables).

When we are working with **unsupervised learning** machine learning problems we don't have a target variable
that we want to predict. In such cases, we will usually work with tables following the `data` *only* key convention, 
since we don't have a target variable. 

## Scikit-Learn's API

Scikit-Learn's API is designed around the following principles:

* **Consistency**: All objects share a common interface drawn from a limited set of methods, with consistent
  documentation. This means that we can learn one estimator API (i.e., methods) and other estimators will work the
  same way.
* **Inspection**: All parameter values are exposed as public attributes. This means that we can inspect
  the parameters of an estimator by looking at its attributes.
* **Limited object hierarchy**: Only algorithms are represented by Python classes; datasets are represented in
  standard formats (NumPy arrays, Pandas DataFrames, SciPy sparse matrices) and parameter names use standard Python
  strings.
* **Composition**: Many machine learning tasks can be expressed as sequences of more fundamental algorithms, and
  Scikit-Learn makes use of this wherever possible. 
* **Sensible defaults**: When models require user-specified parameters, the library defines an appropriate default
  value.

In practice, these principles make Scikit-Learn very easy to use, once the basic principles are understood.