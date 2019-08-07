# # Predict Next Purchase

# %matplotlib inline
import composeml as cp
import data

# ## Load Data

df = data.load_orders('data', nrows=int(1e+5))
df.head()

# ## Generate Labels
#
# ### Create Labeling Function


def bought_product(df, product_name):
    was_purchased = df.product_name.eq(product_name).any()
    return was_purchased


# ### Construct Label Maker

lm = cp.LabelMaker(
    target_entity='user_id',
    time_index='order_time',
    labeling_function=bought_product,
    window_size='4w',
)

# ### Search Labels

lt = lm.search(
    df,
    product_name='Banana',
    num_examples_per_instance=10,
    verbose=True,
)
lt.head()

# ### Describe Labels

lt.describe()
lt.plot.distribution();
