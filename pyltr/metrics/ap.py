"""

(Mean) Average Precision

TODO: better docs

"""

import numpy as np
from . import Metric
from sklearn.externals.six.moves import range


class AP(Metric):
    def __init__(self, k=10, cutoff=0.5):
        super(AP, self).__init__()
        self.k = k
        self.cutoff = cutoff

    def evaluate(self, qid, targets):
        n_targets = len(targets)
        num_rel = 0
        total_prec = 0.0
        for i in range(n_targets):
            if targets[i] >= self.cutoff:
                num_rel += 1
                if i < self.k:
                    total_prec += num_rel / (i + 1.0)
        return (total_prec / num_rel) if num_rel > 0 else 0.0

    def calc_swap_deltas(self, qid, targets):
        n_targets = len(targets)
        deltas = np.zeros((n_targets, n_targets))
        total_num_rel = 0
        total_metric = 0.0
        for i in range(n_targets):
            if targets[i] >= self.cutoff:
                total_num_rel += 1
                if i < self.k:
                    total_metric += total_num_rel / (i + 1.0)
        metric = (total_metric / total_num_rel) if total_num_rel > 0 else 0.0

        num_rel_i = 0
        for i in range(min(n_targets, self.k)):
            if targets[i] >= self.cutoff:
                num_rel_i += 1
                num_rel_j = num_rel_i
                sub = num_rel_i / (i + 1.0)

                for j in range(i + 1, n_targets):
                    if targets[j] >= self.cutoff:
                        if j < self.k:
                            num_rel_j += 1
                            sub += 1 / (j + 1.0)
                    else:
                        add = (num_rel_j / (j + 1.0)) if j < self.k else 0.0
                        new_total_metric = total_metric + add - sub
                        new_num_rel = total_num_rel
                        new_metric = ((new_total_metric / new_num_rel)
                                      if new_num_rel > 0
                                      else 0.0)
                        deltas[i, j] = new_metric - metric

            else:
                num_rel_j = num_rel_i
                add = (num_rel_i + 1) / (i + 1.0)

                for j in range(i + 1, n_targets):
                    if targets[j] >= self.cutoff:
                        sub = (((num_rel_j + 1) / (j + 1.0))
                               if j < self.k
                               else 0.0)
                        new_total_metric = total_metric + add - sub
                        new_num_rel = total_num_rel
                        new_metric = ((new_total_metric / new_num_rel)
                                      if new_num_rel > 0
                                      else 0.0)
                        deltas[i, j] = new_metric - metric

                        if j < self.k:
                            num_rel_j += 1
                            add += 1 / (j + 1.0)

        return deltas

    def max_k(self):
        return self.k
