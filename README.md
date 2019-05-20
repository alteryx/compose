# compose

Compose is a Python library for prediction engineering.

Compose enables you to systematically define prediction problems and automatically extracting historical training examples to train machine learning algorithms


# Example

Imagine you have a table of transactions from different customers. It has columns for `transaction_id`, `timestamp`, `department`, `customer_id`, `transaction_type`, `amount`.


You want to extract label times for each customer where the label is total purchase amount over the next 5 transactions that are purchases.

We also want to take those label times and transform them into binary labels if the total was above X.

We also want to take those label times and shift the time 1 hour earlier, so we can predict in advanced.


