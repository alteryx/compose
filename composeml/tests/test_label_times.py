def test_count_by_time_categorical(total_spent):
    labels = range(2)
    given_answer = total_spent.bin(2, labels=labels).count_by_time
    given_answer = given_answer.to_csv(header=True).splitlines()

    answer = [
        'cutoff_time,0,1',
        '2019-01-01 08:00:00,0.0,1.0',
        '2019-01-01 08:30:00,0.0,2.0',
        '2019-01-01 09:00:00,0.0,3.0',
        '2019-01-01 09:30:00,0.0,4.0',
        '2019-01-01 10:00:00,0.0,5.0',
        '2019-01-01 10:30:00,1.0,5.0',
        '2019-01-01 11:00:00,2.0,5.0',
        '2019-01-01 11:30:00,3.0,5.0',
        '2019-01-01 12:00:00,4.0,5.0',
        '2019-01-01 12:30:00,5.0,5.0',
    ]

    assert given_answer == answer


def test_count_by_time_continuous(total_spent):
    given_answer = total_spent.count_by_time
    given_answer = given_answer.to_csv(header=True).splitlines()

    answer = [
        'cutoff_time,total_spent',
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


def test_describe(total_spent):
    assert total_spent.bin(2).describe() is None


def test_describe_no_settings(total_spent):
    total_spent = total_spent.copy()
    total_spent.settings.clear()
    assert total_spent.describe() is None


def test_distribution_categorical(total_spent):
    labels = range(2)
    given_answer = total_spent.bin(2, labels=labels).distribution
    given_answer = given_answer.to_csv(header=True).splitlines()

    answer = [
        'total_spent,count',
        '0,5',
        '1,5',
    ]

    assert given_answer == answer


def test_distribution_continous(total_spent):
    assert total_spent.distribution is None


def test_infer_type(total_spent):
    assert total_spent.infer_type() == 'continuous'

    total_spent = total_spent.threshold(5)
    total_spent.label_type = None
    assert total_spent.infer_type() == 'discrete'
