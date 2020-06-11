import pandas as pd


def describe_label_times(label_times):
    """Prints out label info with transform settings that reproduce labels."""
    if label_times.label_name is not None and label_times.is_discrete:
        print('Label Distribution\n' + '-' * 18, end='\n')
        distribution = label_times[label_times.label_name].value_counts()
        distribution.index = distribution.index.astype('str')
        distribution.sort_index(inplace=True)
        distribution['Total:'] = distribution.sum()
        print(distribution.to_string(), end='\n\n\n')

    settings = label_times.settings
    info = settings['label_times'].copy()
    info.update(info.pop('search_settings'))
    info = pd.Series(info)
    transforms = info.pop('transforms')

    print('Settings\n' + '-' * 8, end='\n')
    if info.isnull().all():
        print('No settings', end='\n\n\n')
    else:
        info.sort_index(inplace=True)
        print(info.to_string(), end='\n\n\n')

    print('Transforms\n' + '-' * 10, end='\n')
    for step, transform in enumerate(transforms):
        transform = pd.Series(transform)
        transform.sort_index(inplace=True)
        name = transform.pop('transform')
        transform = transform.add_prefix('  - ')
        transform = transform.add_suffix(':')
        transform = transform.to_string()
        header = '{}. {}\n'.format(step + 1, name)
        print(header + transform, end='\n\n')

    if len(transforms) == 0:
        print('No transforms applied', end='\n\n')
