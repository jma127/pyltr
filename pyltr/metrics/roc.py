"""

Area under the ROC curve.

TODO: better docs

"""

import numpy as np
from . import Metric
from overrides import overrides
from sklearn.externals.six.moves import range


class AUCROC(Metric):
    def __init__(self, cutoff=0.5):
        super(AUCROC, self).__init__()
        self.cutoff = cutoff

    @overrides
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
