"""

Gain evaluators (usually used with DCG-like metrics).

"""

import math


_LOG2 = math.log(2.0)


def _identity_gain(x):
    return x


def _exp2_gain(x):
    return math.exp(x * _LOG2) - 1.0


def get_gain_fn(name, **args):
    """Returns a gain callable corresponding to the provided gain name.

    Parameters
    ----------
    name : {'identity', 'exp2'}
        Name of the gain to return.

        - identity: ``lambda x : x``

        - exp2: ``lambda x : (2.0 ** x) - 1.0``

    Returns
    -------
    gain_fn : callable
        Callable that returns the gain of target values.

    """
    if name == 'identity':
        return _identity_gain
    elif name == 'exp2':
        return _exp2_gain
    raise ValueError(name + ' is not a valid gain type')
