def can_be_type(type, string):
    """Return whether the string can be interpreted as a type.

    Args:
        type (type) : Type to apply on string.
        string (str) : String to check if can be type.

    Returns:
        bool : Whether string can be type.
    """
    try:
        type(string)
        return True

    except ValueError:
        return False


def format_number(n):
    if n == -1 or n == 'inf':
        n = float('inf')

    numeric = (int, float)
    assert isinstance(n, numeric), 'value must be numeric'
    return n


def is_finite_number(n):
    return n > 0 and abs(n) != float('inf')
