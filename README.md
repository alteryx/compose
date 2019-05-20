# compose

Compose is a Python library for prediction engineering.

Compose enables you to systematically define prediction problems and automatically extracting historical training examples to train machine learning algorithms


# Example

Imagine you have a table of sensor data from different machines. It has columns for `reading_id`, `timestamp`, `machine_setting`, `machine_id`, `voltage`, and `current`


You want to extract label times for each machine where the label is average sensor reading for voltage is the next 5 observations.

We also want to take those label times and transform them into binary labels if the average was above X.

We also want to take those label times and shift the time 1 hour earlier, so we can predict in advanced.


