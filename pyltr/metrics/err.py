"""

Expected reciprocal rank.

TODO: better docs

"""

import numpy as np
from . import gains, Metric
from sklearn.externals.six import moves

range = moves.range
_EPS = np.finfo(np.float64).eps


class ERR(Metric):
    def __init__(self, highest_score, k=10, gain_type='exp2', add_num=0.0,
                 add_denom=1.0):
        super(ERR, self).__init__()
        self.highest_score = highest_score
        self.k = k
        self.gain_type = gain_type
        self.add_num = add_num
        self.add_denom = add_denom
        self._gain_fn = gains.get_gain_fn(gain_type)
        self._highest_gain = self._gain_fn(self.highest_score)

    def evaluate(self, qid, targets):
        residual = 1.0
        result = 0.0
        for i, t in enumerate(targets[:self.k]):
            assert t <= self.highest_score
            sprob = self._get_satisfied_prob(t)
            result += residual * sprob / (1.0 + i)
            residual *= (1.0 - sprob)
            if residual < _EPS:
                break
        return result

    def calc_swap_deltas(self, qid, targets):
        n_targets = len(targets)
        deltas = np.zeros((n_targets, n_targets))
        satisfied_probs = np.zeros(n_targets)
        prefix_sums = np.zeros(n_targets + 1)
        point_residuals = np.ones(n_targets + 1)

        for i, t in enumerate(targets):
            assert t <= self.highest_score
            sprob = self._get_satisfied_prob(t)
            satisfied_probs[i] = sprob
            prefix_sums[i + 1] = (
                prefix_sums[i] +
                ((point_residuals[i] * sprob / (1.0 + i))
                 if i < self.k else 0.0))
            point_residuals[i + 1] = point_residuals[i] * (1.0 - sprob)

        for i in range(min(n_targets, self.k)):
            for j in range(i + 1, n_targets):
                if satisfied_probs[i] == satisfied_probs[j]:
                    continue

                ratio = (1.0 - satisfied_probs[j]) / (1.0 - satisfied_probs[i])
                deltas[i, j] = (
                    # delta on i-th position
                    ((satisfied_probs[j] - satisfied_probs[i]) *
                     point_residuals[i] / (i + 1.0)) +
                    # delta on i+1 to j-1 positions
                    (prefix_sums[j] - prefix_sums[i + 1]) * (ratio - 1.0) +
                    # delta on j-th position
                    (((point_residuals[j] / (j + 1.0)) *
                      (satisfied_probs[i] * ratio - satisfied_probs[j]))
                     if j < self.k else 0.0))

        return deltas

    def max_k(self):
        return self.k

    def _get_satisfied_prob(self, t):
        return max(
            0.0,
            ((self._gain_fn(t) + self.add_num) /
             (self._highest_gain + self.add_denom)))
