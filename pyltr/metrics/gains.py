"""

Gain evaluators (usually used with DCG-like metrics).

"""

import numpy as np


def identity_gain(x):
    return x


def exp2_gain(x):
    return np.exp2(x) - 1.0


def get_gain_fn(name, **args):
    if name == 'identity':
        return identity_gain
    elif name == 'exp2':
        return exp2_gain
    raise ValueError(name + ' is not a valid gain type')
