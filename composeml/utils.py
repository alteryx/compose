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
