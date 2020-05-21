import pandas as pd


def describe_label_times(label_times):
    """Prints out label info with transform settings that reproduce labels."""
    is_discrete = label_times.is_discrete[label_times.label_name]

    if is_discrete:
        print('Label Distribution\n' + '-' * 18, end='\n')
        distribution = label_times[label_times.label_name].value_counts()
        distribution.index = distribution.index.astype('str')
        distribution.sort_index(inplace=True)
        distribution['Total:'] = distribution.sum()
        print(distribution.to_string(), end='\n\n\n')

    metadata = label_times.settings
    transforms = metadata['transforms']
    label_name = metadata['label_name']
    label_type = metadata['target_types'].get(metadata['label_name'])
    target_entity = metadata['target_entity']
    settings = {'label_name': label_name, 'label_type': label_type, 'target_entity': target_entity}
    settings.update(metadata['search_settings'])
    settings = pd.Series(settings)

    print('Settings\n' + '-' * 8, end='\n')
    if settings.isnull().all():
        print('No settings', end='\n\n\n')
    else:
        settings.sort_index(inplace=True)
        print(settings.to_string(), end='\n\n\n')

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
