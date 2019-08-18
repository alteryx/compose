<p align="center">
    <img style="margin: 50px;" width=50% src="docs/source/images/compose.png" alt="Compose" />
</p>

[![CircleCI](https://circleci.com/gh/FeatureLabs/compose/tree/master.svg?style=shield)](https://circleci.com/gh/FeatureLabs/compose-ml/tree/master)
[![codecov](https://codecov.io/gh/FeatureLabs/compose-ml/branch/master/graph/badge.svg)](https://codecov.io/gh/FeatureLabs/compose-ml)
[![Documentation Status](https://readthedocs.org/projects/composeml/badge/?version=latest)](http://docs.compose.ml/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/composeml.svg?maxAge=2592000)](https://badge.fury.io/py/composeml)
[![StackOverflow](http://img.shields.io/badge/questions-on_stackoverflow-blue.svg)](https://stackoverflow.com/questions/tagged/composeml)

[Compose](http://compose.ml) is a python library for automated prediction engineering. An end user defines an outcome of interest over the data by writing a *"labeling function"*. Compose will automatically search and extract historical training examples to train machine learning examples, balance them across time, entities and label categories to reduce biases in learning process. See the [documentation](http://docs.compose.ml) for more information.

Its result is then provided to the automatic feature engineering tools Featuretools and subsequently to AutoML/ML libraries to develop a model. This automation for the very early stage of ML pipeline process allows our end user to easily define a task and solve it. The workflow of an applied machine learning engineer then becomes:

## Installation
Compose can be installed by running the following command.
```shell
pip install composeml
```

## Example
In this example, we will generate labels on a mock dataset of transactions. For each customer, we want to label whether the total purchase amount over the next hour of transactions will exceed $1200. Additionally, we want to predict one hour in advance.

### Load Data
With the package installed, we load in the data. To get an idea on how the transactions looks, we preview the data frame.

```python
import composeml as cp

df = cp.datasets.transactions()
df[df.columns[:7]].head()
```
```
   transaction_id  session_id    transaction_time  product_id  amount  customer_id   device
0             298           1 2014-01-01 00:00:00           5  127.64            2  desktop
1              10           1 2014-01-01 00:09:45           5   57.39            2  desktop
2             495           1 2014-01-01 00:14:05           5   69.45            2  desktop
3             460          10 2014-01-01 02:33:50           5  123.19            2   tablet
4             302          10 2014-01-01 02:37:05           5   64.47            2   tablet
```


## Feature Labs
<a href="https://www.featurelabs.com/">
    <img src="http://www.featurelabs.com/wp-content/uploads/2017/12/logo.png" alt="Featuretools" />
</a>

Compose is an open source project created by [Feature Labs](https://www.featurelabs.com/). To see the other open source projects we're working on visit Feature Labs [Open Source](https://www.featurelabs.com/open). If building impactful data science pipelines is important to you or your business, please [get in touch](https://www.featurelabs.com/contact/).

### Contact

Any questions can be directed to help@featurelabs.com
