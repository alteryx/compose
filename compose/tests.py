import pandas as pd
import label_by_window as cp
import numpy as np


np.random.seed = 10
df = pd.DataFrame({
    "id": range(20),
    "datetime": pd.date_range(start="Jan 1, 2019", end="Jan 2, 2019", periods=20),
    "user_id": [0, 1, 2, 3, 4] * 4,
    "value": np.linspace(0, 20, num=20) * np.linspace(-1, 1, num=20),
    "event_type": ["a", "b", "a", "c"] * 5
})



def test_label_by_instance():
    def total_spendng(user_df, number_events):
        return pd.DataFrame({
          "user_id": user_df.iloc[0]["user_id"],
          "time": user_df.iloc[number_events-1]["datetime"],
          "total_spendng": user_df[number_events]["value"].sum()
        })


    lm = cp.LabelByInstance(instance_id="user_id",
                            labeling_function=total_spendng,
                            drop_null_labels=False,
                            verbose=False)

    label_times = lm.search(df, number_events=3)
    print(label_times)




if __name__ == '__main__':
    test_label_by_instance()
