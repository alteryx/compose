# %matplotlib inline
import composeml as cp
import pandas as pd

PARTITION = '100'
BASE_DIR = 's3://customer-churn-spark/'
PARTITION_DIR = BASE_DIR + 'p' + PARTITION
transactions = f'{PARTITION_DIR}/transactions.csv'

# +
transactions = pd.read_csv(
    transactions,
    parse_dates=['transaction_date', 'membership_expire_date'],
    infer_datetime_format=True,
)

transactions.head()
# -

month_begin = pd.offsets.MonthBegin()
expire_date = transactions['membership_expire_date']
transactions['lead_time'] = expire_date.apply(month_begin.rollback)
transactions[['msno', 'lead_time', 'membership_expire_date']].head()


def inactive_membership(transactions, window):
    transactions = transactions.sort_values('transaction_date')
    
    if len(transactions) == 1:
        cutoff_time, end_time = window
        elapsed_inactive = end_time - transactions['membership_expire_date'].iloc[0]
        return elapsed_inactive

    membership_expire_date = transactions['membership_expire_date'].iloc[0]
    next_transaction_date = transactions['transaction_date'].iloc[1]
    elapsed_inactive = next_transaction_date - membership_expire_date

    return elapsed_inactive


label_maker = cp.LabelMaker(
    target_entity='msno',
    time_index='lead_time',
    labeling_function=inactive_membership,
    window_size='100d',
)

# +
now = pd.Timestamp.now()

label_times = label_maker.search(
    transactions,
    minimum_data=0,
    num_examples_per_instance=2,
    gap=1,
    verbose=True,
)

label_times.head()

# +
min_bin = label_times.inactive_membership.min()
zero = pd.Timedelta('0d')
one_month = pd.Timedelta('31d')
max_bin = label_times.inactive_membership.max()
bins = [min_bin, zero, one_month, max_bin]

labels = ['active', 'inactive', 'churn']
is_churn = label_times.bin(bins, labels=labels)

is_churn.head()
# -

is_churn.describe()
is_churn.plot.distribution()
