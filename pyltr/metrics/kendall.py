"""

Kendall's rank correlation coefficient (tau).

TODO: better docs

"""

import numpy as np
from . import Metric
from sklearn.externals.six.moves import range


_EPS = np.finfo(np.float64).eps


class KendallTau(Metric):
    def __init__(self):
        super(KendallTau, self).__init__()

    def evaluate(self, qid, targets):
        n_targets = len(targets)
        if n_targets < 2:
            return 0.0

        concordant, discordant = 0, 0
        for i, t1 in enumerate(targets):
            for j in range(i + 1, n_targets):
                t2 = targets[j]
                if abs(t1 - t2) < _EPS:
                    continue
                rank_higher = i < j
                score_higher = t1 > t2
                if rank_higher == score_higher:
                    concordant += 1
                else:
                    discordant += 1
        return (concordant - discordant) / (n_targets * (n_targets - 1) / 2.0)

    # TODO: implement for efficiency
    # def calc_swap_deltas(self, qid, targets):
    #    n_targets = len(targets)
    #    deltas = np.zeros((n_targets, n_targets))
    #    return deltas
