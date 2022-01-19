import pandas as pd


def describe_label_times(label_times):
    """Prints out label info with transform settings that reproduce labels."""
    target_column = label_times.target_columns[0]
    is_discrete = label_times.is_discrete[target_column]

    if is_discrete:
        distribution = label_times[target_column].value_counts()
        distribution.sort_index(inplace=True)
        distribution.index = distribution.index.astype("str")
        distribution["Total:"] = distribution.sum()
    else:
        distribution = label_times[target_column].describe()

    print("Label Distribution\n" + "-" * 18, end="\n")
    print(distribution.to_string(), end="\n\n\n")

    metadata = label_times.settings
    target_column = metadata["label_times"]["target_columns"][0]
    target_type = metadata["label_times"]["target_types"][target_column]
    target_dataframe_name = metadata["label_times"]["target_dataframe_name"]

    settings = {
        "target_column": target_column,
        "target_dataframe_name": target_dataframe_name,
        "target_type": target_type,
    }

    settings.update(metadata["label_times"]["search_settings"])
    settings = pd.Series(settings)

    print("Settings\n" + "-" * 8, end="\n")
    settings.sort_index(inplace=True)
    print(settings.to_string(), end="\n\n\n")

    print("Transforms\n" + "-" * 10, end="\n")
    transforms = metadata["label_times"]["transforms"]
    for step, transform in enumerate(transforms):
        transform = pd.Series(transform)
        transform.sort_index(inplace=True)
        name = transform.pop("transform")
        transform = transform.add_prefix("  - ")
        transform = transform.add_suffix(":")
        transform = transform.to_string()
        header = "{}. {}\n".format(step + 1, name)
        print(header + transform, end="\n\n")

    if len(transforms) == 0:
        print("No transforms applied", end="\n\n")
