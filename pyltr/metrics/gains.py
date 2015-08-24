"""

Gain evaluators (usually used with DCG-like metrics).

"""

import math


_LOG2 = math.log(2.0)


def identity_gain(x):
    return x


def exp2_gain(x):
    return math.exp(x * _LOG2) - 1.0


def get_gain_fn(name, **args):
    if name == 'identity':
        return identity_gain
    elif name == 'exp2':
        return exp2_gain
    raise ValueError(name + ' is not a valid gain type')
