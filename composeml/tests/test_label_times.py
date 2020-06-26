from pytest import raises

from composeml.label_times import LabelTimes
from composeml.tests.utils import to_csv


def test_count_by_time_categorical(total_spent):
    given_answer = total_spent.bin(2, labels=range(2))
    given_answer = to_csv(given_answer.count_by_time)

    answer = [
        'time,0,1',
        '2019-01-01 08:00:00,0,1',
        '2019-01-01 08:30:00,0,2',
        '2019-01-01 09:00:00,0,3',
        '2019-01-01 09:30:00,0,4',
        '2019-01-01 10:00:00,0,5',
        '2019-01-01 10:30:00,1,5',
        '2019-01-01 11:00:00,2,5',
        '2019-01-01 11:30:00,3,5',
        '2019-01-01 12:00:00,4,5',
        '2019-01-01 12:30:00,5,5',
    ]

    assert given_answer == answer


def test_count_by_time_continuous(total_spent):
    given_answer = total_spent.count_by_time
    given_answer = to_csv(given_answer, header=True, index=True)

    answer = [
        'time,total_spent',
        '2019-01-01 08:00:00,1',
        '2019-01-01 08:30:00,2',
        '2019-01-01 09:00:00,3',
        '2019-01-01 09:30:00,4',
        '2019-01-01 10:00:00,5',
        '2019-01-01 10:30:00,6',
        '2019-01-01 11:00:00,7',
        '2019-01-01 11:30:00,8',
        '2019-01-01 12:00:00,9',
        '2019-01-01 12:30:00,10',
    ]

    assert given_answer == answer


def test_describe(capsys, total_spent):
    labels = ['A', 'B']
    total_spent.bin(2, labels=labels).describe()
    captured = capsys.readouterr()

    out = '\n'.join([
        'Label Distribution',
        '------------------',
        'A          5',
        'B          5',
        'Total:    10',
        '',
        '',
        'Settings',
        '--------',
        'num_examples_per_instance             -1',
        'target_column                total_spent',
        'target_entity                customer_id',
        'target_type                     discrete',
        '',
        '',
        'Transforms',
        '----------',
        '1. bin',
        '  - bins:              2',
        '  - labels:       [A, B]',
        '  - precision:         3',
        '  - quantiles:     False',
        '  - right:          True',
        '',
        '',
    ])

    assert captured.out == out


def test_describe_no_transforms(capsys):
    data = {'target': range(3)}
    LabelTimes(data).describe()
    captured = capsys.readouterr()
    out = '\n'.join([
        'Settings',
        '--------',
        'target_column        target',
        'target_entity          None',
        'target_type      continuous',
        '',
        '',
        'Transforms',
        '----------',
        'No transforms applied',
        '',
        '',
    ])

    assert captured.out == out


def test_distribution_categorical(total_spent):
    labels = range(2)
    given_answer = total_spent.bin(2, labels=labels).distribution
    given_answer = to_csv(given_answer)

    answer = [
        'total_spent,count',
        '0,5',
        '1,5',
    ]

    assert given_answer == answer


def test_distribution_continous(total_spent):
    assert total_spent.distribution is None


def test_target_type(total_spent):
    types = total_spent.target_types
    assert types['total_spent'] == 'continuous'
    total_spent = total_spent.threshold(5)
    types = total_spent.target_types
    assert types['total_spent'] == 'discrete'


def test_count(total_spent):
    given_answer = total_spent.count
    given_answer = to_csv(given_answer, index=True)

    answer = [
        'customer_id,count',
        '0,2',
        '1,3',
        '2,4',
        '3,1',
    ]

    assert given_answer == answer


def test_label_select_errors(total_spent):
    match = 'only one target exists'
    with raises(AssertionError, match=match):
        total_spent.select('a')

    lt = total_spent.copy()
    lt.target_columns.append('b')

    match = 'target name must be string'
    with raises(TypeError, match=match):
        total_spent.select(123)

    match = 'target "a" not found'
    with raises(AssertionError, match=match):
        lt.select('a')
