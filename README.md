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
In this example, we will generate labels on a mock dataset of transactions. For each customer, we want to label whether the total purchase amount over the next hour of transactions will exceed $100. Additionally, we want to predict one hour in advance.

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
### Create Labeling Function
To get started, we define the labeling function that will return the total purchase amount given a hour of transactions.

```python
def total_spent(df):
    total = df['amount'].sum()
    return total
```
### Construct Label Maker
With the labeling function, we create the [`LabelMaker`](http://docs.compose.ml/en/latest/generated/composeml.LabelMaker.html#composeml-labelmaker) for our prediction problem. To process one hour of transactions for each customer, we set the  `target_entity` to the customer ID and the `window_size` to one hour.

```python
label_maker = cp.LabelMaker(
    target_entity="customer_id",
    time_index="transaction_time",
    labeling_function=total_spent,
    window_size="1h",
)
```
### Search Labels
Next, we automatically search and extract the labels by using [`LabelMaker.search`](http://docs.compose.ml/en/latest/generated/methods/composeml.LabelMaker.search.html#composeml.LabelMaker.search).

```python
labels = label_maker.search(
    df,
    minimum_data="1h",
    num_examples_per_instance=25,
    gap=1,
    verbose=True,
)
labels.head()
```
```
          customer_id         cutoff_time  total_spent
label_id
0                   1 2014-01-01 04:13:51        65.11
1                   1 2014-01-03 15:41:34       101.08
2                   1 2014-01-05 11:46:10        16.78
3                   1 2014-01-06 09:54:58       108.16
4                   1 2014-01-08 08:54:02        48.33
```

### Transform Labels
With the generated [`LabelTimes`](https://docs.compose.ml/en/latest/generated/composeml.LabelTimes.html#composeml.LabelTimes), we will apply specific transforms for our prediction problem. To make the labels binary, a threshold is applied for amounts exceeding $100.

```python
labels = labels.threshold(100)
labels.head()
```
```
          customer_id         cutoff_time  total_spent
label_id
0                   1 2014-01-01 04:13:51        False
1                   1 2014-01-03 15:41:34         True
2                   1 2014-01-05 11:46:10        False
3                   1 2014-01-06 09:54:58         True
4                   1 2014-01-08 08:54:02        False
```

Additionally, the label times are shifted 1 hour earlier for predicting in advance.

```python
labels = labels.apply_lead('1h')
labels.head()
```
```
          customer_id         cutoff_time  total_spent
label_id
0                   1 2014-01-01 03:13:51        False
1                   1 2014-01-03 14:41:34         True
2                   1 2014-01-05 10:46:10        False
3                   1 2014-01-06 08:54:58         True
4                   1 2014-01-08 07:54:02        False
```

### Describe Labels

After transforming the labels, we can use [`LabelTimes.describe`](https://docs.compose.ml/en/latest/generated/methods/composeml.LabelTimes.describe.html#composeml.LabelTimes.describe) to print out the distribution with the settings and transforms that were used to make these labels. This is useful as a reference for understanding how the labels were generated from raw data. Also, the label distribution is helpful for determining if we have imbalanced labels.

```python
labels.describe()
```
```
Label Distribution
------------------
False      75
True       50
Total:    125


Settings
--------
num_examples_per_instance    25
minimum_data                 1h
window_size                  1h
gap                           1


Transforms
----------
1. threshold
  - value:    100

2. apply_lead
  - value:    1h
```

## Feature Labs
<a href="https://www.featurelabs.com/">
    <img src="http://www.featurelabs.com/wp-content/uploads/2017/12/logo.png" alt="Featuretools" />
</a>

Compose is an open source project created by [Feature Labs](https://www.featurelabs.com/). To see the other open source projects we're working on visit Feature Labs [Open Source](https://www.featurelabs.com/open). If building impactful data science pipelines is important to you or your business, please [get in touch](https://www.featurelabs.com/contact/).

### Contact

Any questions can be directed to help@featurelabs.com
