<p align="center"><img width=50% src="https://raw.githubusercontent.com/alteryx/compose/main/docs/source/images/compose.png" alt="Compose" /></p>
<p align="center"><i>"Build better training examples in a fraction of the time."</i></p>
<p align="center">
    <a href="https://github.com/alteryx/compose/actions?query=workflow%3ATests" target="_blank">
        <img src="https://github.com/alteryx/compose/workflows/Tests/badge.svg" alt="Tests" />
    </a>
    <a href="https://codecov.io/gh/alteryx/compose">
        <img src="https://codecov.io/gh/alteryx/compose/branch/main/graph/badge.svg?token=mDz4ueTUEO"/>
    </a>
    <a href="https://compose.alteryx.com/en/stable/?badge=stable" target="_blank">
        <img src="https://readthedocs.com/projects/feature-labs-inc-compose/badge/?version=stable&token=5c3ace685cdb6e10eb67828a4dc74d09b20bb842980c8ee9eb4e9ed168d05b00"
            alt="ReadTheDocs" />
    </a>
    <a href="https://badge.fury.io/py/composeml" target="_blank">
        <img src="https://badge.fury.io/py/composeml.svg?maxAge=2592000" alt="PyPI Version" />
    </a>
    <a href="https://stackoverflow.com/questions/tagged/compose-ml" target="_blank">
        <img src="https://img.shields.io/badge/questions-on_stackoverflow-blue.svg?" alt="StackOverflow" />
    </a>
    <a href="https://pepy.tech/project/composeml" target="_blank">
        <img src="https://pepy.tech/badge/composeml/month" alt="PyPI Downloads" />
    </a>
</p>
<hr>

[Compose](https://compose.alteryx.com) is a machine learning tool for automated prediction engineering. It allows you to structure prediction problems and generate labels for supervised learning. An end user defines an outcome of interest by writing a *labeling function*, then runs a search to automatically extract training examples from historical data. Its result is then provided to [Featuretools](https://docs.featuretools.com/) for automated feature engineering and subsequently to [EvalML](https://evalml.alteryx.com/) for automated machine learning. The workflow of an applied machine learning engineer then becomes:

<br><p align="center"><img width=90% src="https://raw.githubusercontent.com/alteryx/compose/main/docs/source/images/workflow.png" alt="Compose" /></p><br>

By automating the early stage of the machine learning pipeline, our end user can easily define a task and solve it. See the [documentation](https://compose.alteryx.com) for more information.

## Installation
Install with pip

```
python -m pip install composeml
```

or from the Conda-forge channel on [conda](https://anaconda.org/conda-forge/composeml):

```
conda install -c conda-forge composeml
```

### Add-ons

**Update checker** - Receive automatic notifications of new Compose releases

```
python -m pip install "composeml[update_checker]"
```

## Example
> Will a customer spend more than 300 in the next hour of transactions?

In this example, we automatically generate new training examples from a historical dataset of transactions.

```python
import composeml as cp
df = cp.demos.load_transactions()
df = df[df.columns[:7]]
df.head()
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

First, we represent the prediction problem with a labeling function and a label maker.

```python
def total_spent(ds):
    return ds['amount'].sum()

label_maker = cp.LabelMaker(
    target_dataframe_name="customer_id",
    time_index="transaction_time",
    labeling_function=total_spent,
    window_size="1h",
)
```

Then, we run a search to automatically generate the training examples.

```python
label_times = label_maker.search(
    df.sort_values('transaction_time'),
    num_examples_per_instance=2,
    minimum_data='2014-01-01',
    drop_empty=False,
    verbose=False,
)

label_times = label_times.threshold(300)
label_times.head()
```

<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>customer_id</th>
      <th>time</th>
      <th>total_spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>2014-01-01 00:00:00</td>
      <td>True</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2014-01-01 01:00:00</td>
      <td>True</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2014-01-01 00:00:00</td>
      <td>False</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2014-01-01 01:00:00</td>
      <td>False</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2014-01-01 00:00:00</td>
      <td>False</td>
    </tr>
  </tbody>
</table>

We now have labels that are ready to use in [Featuretools](https://docs.featuretools.com/) to generate features.

## Support

The Innovation Labs open source community is happy to provide support to users of Compose. Project support can be found in three places depending on the type of question:

1. For usage questions, use [Stack Overflow](https://stackoverflow.com/questions/tagged/compose-ml) with the `composeml` tag.
2. For bugs, issues, or feature requests start a Github [issue](https://github.com/alteryx/compose/issues/new).
3. For discussion regarding development on the core library, use [Slack](https://join.slack.com/t/alteryx-oss/shared_invite/zt-182tyvuxv-NzIn6eiCEf8TBziuKp0bNA).
4. For everything else, the core developers can be reached by email at open_source_support@alteryx.com

## Citing Compose
Compose is built upon a newly defined part of the machine learning process — prediction engineering. If you use Compose, please consider citing this paper:
James Max Kanter, Gillespie, Owen, Kalyan Veeramachaneni. [Label, Segment,Featurize: a cross domain framework for prediction engineering.](https://dai.lids.mit.edu/wp-content/uploads/2017/10/Pred_eng1.pdf) IEEE DSAA 2016.

BibTeX entry:

```bibtex
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

The open source development has been supported in part by DARPA's Data driven discovery of models program (D3M). 

## Alteryx

**Compose** is an open source project maintained by [Alteryx](https://www.alteryx.com). We developed Compose to enable flexible definition of the machine learning task. To see the other open source projects we’re working on visit [Alteryx Open Source](https://www.alteryx.com/open-source). If building impactful data science pipelines is important to you or your business, please get in touch.

<p align="center">
  <a href="https://www.alteryx.com/open-source">
    <img src="https://alteryx-oss-web-images.s3.amazonaws.com/OpenSource_Logo-01.png" alt="Alteryx Open Source" width="800"/>
  </a>
</p>
