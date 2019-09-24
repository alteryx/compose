<p align="center">
    <img width=50% src="https://raw.githubusercontent.com/FeatureLabs/compose/master/docs/source/images/compose.png" alt="Compose" />
</p>
<br>
<br>

[![CircleCI](https://circleci.com/gh/FeatureLabs/compose/tree/master.svg?style=shield)](https://circleci.com/gh/FeatureLabs/compose/tree/master)
[![codecov](https://codecov.io/gh/FeatureLabs/compose/branch/master/graph/badge.svg)](https://codecov.io/gh/FeatureLabs/compose)
[![Documentation Status](https://readthedocs.org/projects/composeml/badge/?version=latest)](https://compose.featurelabs.com/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/composeml.svg?maxAge=2592000)](https://badge.fury.io/py/composeml)
[![StackOverflow](http://img.shields.io/badge/questions-on_stackoverflow-blue.svg)](https://stackoverflow.com/questions/tagged/composeml)

[Compose](https://compose.featurelabs.com) is a python library for automated prediction engineering. An end user defines an outcome of interest over the data by writing a *"labeling function"*. Compose will automatically search and extract historical training examples to train machine learning examples, balance them across time, entities and label categories to reduce biases in learning process. See the [documentation](https://compose.featurelabs.com) for more information.

Its result is then provided to the automatic feature engineering tools Featuretools and subsequently to AutoML/ML libraries to develop a model. This automation for the very early stage of ML pipeline process allows our end user to easily define a task and solve it. The workflow of an applied machine learning engineer then becomes:

<br>
<p align="center">
    <img width=90% src="https://raw.githubusercontent.com/FeatureLabs/compose/master/docs/source/images/workflow.png" alt="Compose" />
</p>
<br>

## Installation
Compose can be installed by running the following command.
```shell
pip install composeml
```

## Example
In this example, we will generate labels on a mock dataset of transactions. For each customer, we want to label whether the total purchase amount over the next hour of transactions will exceed $300. Additionally, we want to predict one hour in advance.

### Load Data
With the package installed, we load in the data. To get an idea on how the transactions looks, we preview the data frame.

```python
import composeml as cp

df = cp.demos.load_transactions()

df[df.columns[:7]].head()
```

<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>transaction_id</th>
      <th>session_id</th>
      <th>transaction_time</th>
      <th>product_id</th>
      <th>amount</th>
      <th>customer_id</th>
      <th>device</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>298</td>
      <td>1</td>
      <td>2014-01-01 00:00:00</td>
      <td>5</td>
      <td>127.64</td>
      <td>2</td>
      <td>desktop</td>
    </tr>
    <tr>
      <td>10</td>
      <td>1</td>
      <td>2014-01-01 00:09:45</td>
      <td>5</td>
      <td>57.39</td>
      <td>2</td>
      <td>desktop</td>
    </tr>
    <tr>
      <td>495</td>
      <td>1</td>
      <td>2014-01-01 00:14:05</td>
      <td>5</td>
      <td>69.45</td>
      <td>2</td>
      <td>desktop</td>
    </tr>
    <tr>
      <td>460</td>
      <td>10</td>
      <td>2014-01-01 02:33:50</td>
      <td>5</td>
      <td>123.19</td>
      <td>2</td>
      <td>tablet</td>
    </tr>
    <tr>
      <td>302</td>
      <td>10</td>
      <td>2014-01-01 02:37:05</td>
      <td>5</td>
      <td>64.47</td>
      <td>2</td>
      <td>tablet</td>
    </tr>
  </tbody>
</table>

### Create Labeling Function
To get started, we define the labeling function that will return the total purchase amount given a hour of transactions.

```python
def total_spent(df):
    total = df['amount'].sum()
    return total
```

### Construct Label Maker
With the labeling function, we create the [`LabelMaker`](https://compose.featurelabs.com/en/latest/generated/composeml.LabelMaker.html#composeml.LabelMaker) for our prediction problem. To process one hour of transactions for each customer, we set the  `target_entity` to the customer ID and the `window_size` to one hour.

```python
label_maker = cp.LabelMaker(
    target_entity="customer_id",
    time_index="transaction_time",
    labeling_function=total_spent,
    window_size="1h",
)
```

### Search Labels
Next, we automatically search and extract the labels by using [`LabelMaker.search`](https://compose.featurelabs.com/en/latest/generated/methods/composeml.LabelMaker.search.html#composeml.LabelMaker.search). For more details on how the label maker works, see [Main Concepts](https://compose.featurelabs.com/en/latest/main_concepts.html).

```python
labels = label_maker.search(
    df.sort_values('transaction_time'),
    num_examples_per_instance=-1,
    gap=1,
    verbose=True,
)

labels.head()
```
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>customer_id</th>
      <th>cutoff_time</th>
      <th>total_spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>2014-01-01 00:45:30</td>
      <td>914.73</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2014-01-01 00:46:35</td>
      <td>806.62</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2014-01-01 00:47:40</td>
      <td>694.09</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2014-01-01 00:52:00</td>
      <td>687.80</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2014-01-01 00:53:05</td>
      <td>656.43</td>
    </tr>
  </tbody>
</table>

### Transform Labels
With the generated [`LabelTimes`](https://compose.featurelabs.com/en/latest/generated/composeml.LabelTimes.html#composeml.LabelTimes), we will apply specific transforms for our prediction problem. To make the labels binary, a threshold is applied for amounts exceeding $300.

```python
labels = labels.threshold(300)

labels.head()
```

<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>customer_id</th>
      <th>cutoff_time</th>
      <th>total_spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>2014-01-01 00:45:30</td>
      <td>True</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2014-01-01 00:46:35</td>
      <td>True</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2014-01-01 00:47:40</td>
      <td>True</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2014-01-01 00:52:00</td>
      <td>True</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2014-01-01 00:53:05</td>
      <td>True</td>
    </tr>
  </tbody>
</table>

Additionally, the label times are shifted one hour earlier for predicting in advance.

```python
labels = labels.apply_lead('1h')

labels.head()
```

<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>customer_id</th>
      <th>cutoff_time</th>
      <th>total_spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>2013-12-31 23:45:30</td>
      <td>True</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2013-12-31 23:46:35</td>
      <td>True</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2013-12-31 23:47:40</td>
      <td>True</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2013-12-31 23:52:00</td>
      <td>True</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2013-12-31 23:53:05</td>
      <td>True</td>
    </tr>
  </tbody>
</table>

### Describe Labels

After transforming the labels, we can use [`LabelTimes.describe`](https://compose.featurelabs.com/en/latest/generated/methods/composeml.LabelTimes.describe.html#composeml.LabelTimes.describe) to print out the distribution with the settings and transforms that were used to make these labels. This is useful as a reference for understanding how the labels were generated from raw data. Also, the label distribution is helpful for determining if we have imbalanced labels.

```python
labels.describe()
```

```
Label Distribution
------------------
False      56
True       44
Total:    100


Settings
--------
num_examples_per_instance        -1
minimum_data                   None
window_size                  <Hour>
gap                               1


Transforms
----------
1. threshold
  - value:    300

2. apply_lead
  - value:    1h
```

## Testing & Development
The Feature Labs community welcomes pull requests. Instructions for testing and development are available here.

## Support
The Feature Labs open source community is happy to provide support to users of Compose. Project support can be found in four places depending on the type of question:

1. For usage questions, use [Stack Overflow](https://stackoverflow.com/questions/tagged/composeml) with the `composeml` tag.
2. For bugs, issues, or feature requests start a Github [issue](https://github.com/FeatureLabs/compose/issues).
3. For discussion regarding development on the core library, use [Slack](https://featuretools.slack.com/messages/CKP6D0KUP).
4. For everything else, the core developers can be reached by email at [help@featurelabs.com](mailto:help@featurelabs.com).

## Citing Compose
Compose is built upon a newly defined part of the machine learning process - prediction engineering. If you use Compose please consider citing this paper:
James Max Kanter, Gillespie, Owen, Kalyan Veeramachaneni. [Label, Segment,Featurize: a cross domain framework for prediction engineering.](https://dai.lids.mit.edu/wp-content/uploads/2017/10/Pred_eng1.pdf) IEEE DSAA 2016.

BibTeX entry:

```
@inproceedings{kanter2016label,
  title={Label, segment, featurize: a cross domain framework for prediction engineering},
  author={Kanter, James Max and Gillespie, Owen and Veeramachaneni, Kalyan},
  booktitle={2016 IEEE International Conference on Data Science and Advanced Analytics (DSAA)},
  pages={430--439},
  year={2016},
  organization={IEEE}
}
```

## Acknowledgements 
Compose open source has been developed by Feature Labs engineering team. The open source development has been supported in part by DARPA's Data driven discovery of models program (D3M). 

## Feature Labs
<a href="https://www.featurelabs.com/">
    <img src="http://www.featurelabs.com/wp-content/uploads/2017/12/logo.png" alt="Featuretools" />
</a>

Compose is an open source project created by Feature Labs. We developed Compose to enable flexible definition of the machine learning task. Read more about our rationale behind automating and developing this stage of the machine learning process here.

To see the other open source projects we're working on visit Feature Labs [Open Source](https://www.featurelabs.com/open). If building impactful data science pipelines is important to you or your business, please [get in touch](https://www.featurelabs.com/contact/).
