# Estimators

## Estimators' API

**Estimators** are classes from the library that implement machine learning algorithms (i.e., objects that "learn"
from data). In the context of Supervised Learning, an estimator is a Python object that implements (at least) the 
following methods:

* `fit(X, y)`: fit the model using `X` as training data and `y` as target values.
* `predict(X)`: predict the target values of `X` using the trained model.

The two most important tasks that machine learning algorithms can perform are probably 
**classification** and **regression**:

* If the prediction task is to classify the observations in a set of finite labels, or in other words to “name” 
the objects observed, the task is said to be a _classification_ task. In this case, the target variable is said to be
a _categorical_ variable (i.e., a variable that can take on one of a limited, and usually fixed, number of possible
values). 
* If, on the other hand, the target variable is a continuous target variable, it is said to be a 
_regression_ task.

### The `fit` method

The `fit` method is used to train the model, and is the first method that we should call when using an estimator (after
our data is ready, of course). It takes as input the training data and the training labels, and
trains the model. For example, to train a KNN classifier, we can do the following:

```python
from sklearn.neighbors import KNeighborsClassifier

X_train = ...
y_train = ...

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
```

!!!note
    A K-Nearest Neightbors is an algorithm that **does not fit** a model to the data. Instead, the `fit` method 
    memorizes the training data in an internal data structure that is efficient to query. 
    
### The `predict` method

The `predict` method is used to make predictions. It takes as input the data to predict, and returns the predictions.
For example, to predict the labels of a test set, we can do the following:

```python
X_test = ...

y_pred = knn.predict(X_test)
```

!!!note
    The previous examples serve as an illustration that in Scikit Learn, the order is always the same: 
    first we create the estimator, then we fit it to the data, then we use it to make predictions.


## Common ML algorithms in SKLearn

Scikit Learn provides implementations of a large number of machine learning algorithms in the form of 
the aforementioned **estimators**. 

### Regression models

Regression is the task of predicting a continuous value. For example, predicting the price of a house is a
regression task, or predicting the height of a person. Scikit Learn provides a number of regression models that
can be used to solve regression problems.

The following are some of the most common regression models; for a complete list, see the
[Scikit Learn documentation](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning).

#### Common regression models

* **Linear regression**: Linear regression is a linear model that assumes a linear relationship between the
  dependent variable and the model parameters (this is not always equivalent to assuming a linear relationship
    between the dependent and independent variables). It is implemented by the `LinearRegression` class in the
  `linear_model` module.

    ```python
    from sklearn.linear_model import LinearRegression

    X_train = ...
    y_train = ...

    model = LinearRegression()
    model.fit(X_train, y_train)
    ```
  
* **Random forest regression**: Random forest regression is an ensemble model that fits a number of decision tree
  classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control
  over-fitting. It is implemented by the `RandomForestRegressor` class in the `ensemble` module.

    ```python
    from sklearn.ensemble import RandomForestRegressor

    X_train = ...
    y_train = ...

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    ```

    !!!note 
        The `RandomForestRegressor` class from Scikit Learn is probably the most popular implementation of random
        forest regression. The library also provides boosted tree regression models, such as `GradientBoostingRegressor`,
        but these are not as popular as other implementations, such as XGBoost or CatBoost.

### Classification

Classification is the task of predicting a class from a set of classes. For example, predicting whether an email is
spam or not is a classification task, or predicting whether a person has a disease or not. Scikit Learn provides a
number of classification models that can be used to solve classification problems.

#### Common classification models

* **Logistic regression**: Logistic regression is a linear model that assumes a linear relationship between the
  log-odds of the dependent variable and the independent variables (actually, it is an affine transformation of the
  log-odds, since there can be a constant term summed to the log-odds). 

    It is used to model the probability of a certain class or event existing such as pass/fail, win/lose, etc. 
    and also to predict multiclass problems (using the softmax function). 
    It is implemented by the `LogisticRegression` class in the `linear_model` module.

    ```python
    from sklearn.linear_model import LogisticRegression

    X_train = ...
    y_train = ...

    model = LogisticRegression()
    model.fit(X_train, y_train)
    ```
    
    !!!note
        Despite its name, which comes that the logistic regression is a type of Generalized Linear Model (GLM), 
        once trained, the logistic regression is used as a classification algorithm, not a regression algorithm. 
        
* **K-Nearest Neighbors**: K-Nearest Neighbors is a non-parametric method used for classification and regression. 
  It is a lazy learning algorithm that does not fit any model to the data, and instead memorizes the training data
  to make predictions (by checking the class of the K nearest neighbors of the new data point). 
  It is implemented by the `KNeighborsClassifier` class in the `neighbors` module.

    ```python
    from sklearn.neighbors import KNeighborsClassifier

    X_train = ...
    y_train = ...

    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    ```

    !!!note
        In the presence of a lot of features, KNN can be very slow, since it has to compute the distance between the
        new data point and all the training data points. 

* **Random forest classification**: Random forest classification is an ensemble model that fits a number of decision
    tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and
    control over-fitting. It is implemented by the `RandomForestClassifier` class in the `ensemble` module.
    
    ```python
    from sklearn.ensemble import RandomForestClassifier

    X_train = ...
    y_train = ...

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    ```

### Clustering

Clustering is the task of grouping data points into clusters. For example, grouping customers into clusters based on
their purchase history is a clustering task. Scikit Learn provides a number of clustering algorithms that can be used
to solve clustering problems.

#### Common clustering models

* **K-Means**: K-Means is a clustering algorithm that partitions the data into K clusters. It is implemented by the
  `KMeans` class in the `cluster` module.

    ```python
    from sklearn.cluster import KMeans

    X_train = ...

    model = KMeans(n_clusters=3)
    model.fit(X_train)
    ```
  
* **HD-BSCAN**: HD-BSCAN is a clustering algorithm that partitions the data into clusters. It is implemented by the
  `HDBSCAN` class in the `cluster` module.

    ```python
    from sklearn.cluster import HDBSCAN

    X_train = ...

    model = HDBSCAN()
    model.fit(X_train)
    ```

### Dimensionality reduction

Dimensionality reduction is the task of reducing the number of features in a dataset. For example, image 
compression algorithms make heavy use of dimensionality reduction, since images have a very large 
number of features (i.e., pixels), but many of these features are correlated with each other, and thus
can be removed without losing much information. Scikit Learn provides several dimensionality reduction algorithms.

#### Common dimensionality reduction models

* **Principal Component Analysis**: Principal Component Analysis (PCA) is a dimensionality reduction algorithm that
  transforms the data into a lower-dimensional space. It is implemented by the `PCA` class in the `decomposition` module.

    ```python
    from sklearn.decomposition import PCA

    X_train = ...

    model = PCA(n_components=2)
    model.fit(X_train)
    ```

!!!note
    Not all dimensionality reduction algorithms can be found in SKLearn. UMAP, for example, is a very popular
    dimensionality reduction algorithm that is not implemented in the library.