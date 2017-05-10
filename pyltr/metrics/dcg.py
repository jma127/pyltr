"""

DCG and NDCG.

TODO: better docs

"""

import numpy as np
from . import gains, Metric
from sklearn.externals.six import moves

_EPS = np.finfo(np.float64).eps
range = moves.range

class DCG(Metric):
    def __init__(self, k=10, gain_type='exp2'):
        super(DCG, self).__init__()
        self.k = k
        self.gain_type = gain_type
        self._gain_fn = gains.get_gain_fn(gain_type)
        self._discounts = self._make_discounts(256)

    def evaluate(self, qid, targets):
        return sum(self._gain_fn(t) * self._get_discount(i)
                   for i, t in enumerate(targets) if i < self.k)

    def calc_swap_deltas(self, qid, targets, coeff=1.0):
        n_targets = len(targets)
        deltas = np.zeros((n_targets, n_targets))

        for i in range(min(n_targets, self.k)):
            for j in range(i + 1, n_targets):
                deltas[i, j] = coeff * \
                    (self._gain_fn(targets[i]) - self._gain_fn(targets[j])) * \
                    (self._get_discount(j) - self._get_discount(i))

        return deltas

    def max_k(self):
        return self.k

    def calc_random_ev(self, qid, targets):
        total_gains = sum(self._gain_fn(t) for t in targets)
        total_discounts = sum(self._get_discount(i)
                              for i in range(min(self.k, len(targets))))
        return total_gains * total_discounts / len(targets)

    @classmethod
    def _make_discounts(self, n):
        return np.array([1.0 / np.log2(i + 2.0) for i in range(n)])

    def _get_discount(self, i):
        if i >= self.k:
            return 0.0
        while i >= len(self._discounts):
            self._grow_discounts()
        return self._discounts[i]

    def _grow_discounts(self):
        self._discounts = self._make_discounts(len(self._discounts) * 2)


class NDCG(Metric):
    def __init__(self, k=10, gain_type='exp2'):
        super(NDCG, self).__init__()
        self.k = k
        self.gain_type = gain_type
        self._dcg = DCG(k=k, gain_type=gain_type)
        self._ideals = {}

    def evaluate(self, qid, targets):
        return (self._dcg.evaluate(qid, targets) /
                max(_EPS, self._get_ideal(qid, targets)))

    def calc_swap_deltas(self, qid, targets):
        ideal = self._get_ideal(qid, targets)
        if ideal < _EPS:
            return np.zeros((len(targets), len(targets)))
        return self._dcg.calc_swap_deltas(
            qid, targets, coeff=1.0 / ideal)

    def max_k(self):
        return self.k

    def calc_random_ev(self, qid, targets):
        return (self._dcg.calc_random_ev(qid, targets) /
                max(_EPS, self._get_ideal(qid, targets)))

    def _get_ideal(self, qid, targets):
        ideal = self._ideals.get(qid)
        if ideal is not None:
            return ideal
        sorted_targets = np.sort(targets)[::-1]
        ideal = self._dcg.evaluate(qid, sorted_targets)
        self._ideals[qid] = ideal
        return ideal
