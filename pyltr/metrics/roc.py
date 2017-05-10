"""

Area under the ROC curve.

TODO: better docs

"""

import numpy as np
from . import Metric
from sklearn.externals.six.moves import range


class AUCROC(Metric):
    def __init__(self, cutoff=0.5):
        super(AUCROC, self).__init__()
        self.cutoff = cutoff

    def evaluate(self, qid, targets):
        n_targets = len(targets)
        total_num_rel = 0
        for t in targets:
            if t >= self.cutoff:
                total_num_rel += 1

        if total_num_rel == 0 or total_num_rel == n_targets:
            return 0.0

        left_rel = 0
        cnt = 0
        for i, t in enumerate(targets):
            if t >= self.cutoff:
                left_rel += 1
                right_rel = total_num_rel - left_rel
                right_unrel = n_targets - i - right_rel - 1
                cnt += right_unrel

        return cnt / float(total_num_rel * (n_targets - total_num_rel))

    def calc_swap_deltas(self, qid, targets):
        n_targets = len(targets)
        deltas = np.zeros((n_targets, n_targets))
        rel = np.array(targets) >= self.cutoff
        total_num_rel = sum(rel)

        if total_num_rel == 0 or total_num_rel == n_targets:
            return deltas

        denom = total_num_rel * float(n_targets - total_num_rel)
        for i in range(n_targets):
            irel = rel[i]
            for j in range(i + 1, n_targets):
                jrel = rel[j]
                if not irel and jrel:
                    deltas[i, j] = (j - i) / denom
                elif irel and not jrel:
                    deltas[i, j] = (i - j) / denom

        return deltas
