import numpy as np
from sklearn.externals.six.moves import range
from ..util.group import check_qids, get_groups
from ..util.sort import get_sorted_y


class Metric(object):
    """Base LTR metric class.

    Subclasses must override evaluate() and cona optionally override various
    other methods.

    """
    def evaluate(self, qid, targets):
        """Evaluates the metric on a ranked list of targets.

        qid is guaranteed to be a hashable type s.t.
        sorted(targets1) == sorted(targets2) iff qid1 == qid2.

        """
        raise NotImplementedError()

    def calc_swap_deltas(self, qid, targets):
        """Returns an upper triangular matrix.

        Each (i, j) contains the change in the metric from swapping
        targets[i, j].

        Can be overridden for efficiency.

        """
        n_targets = len(targets)
        deltas = np.zeros((n_targets, n_targets))
        original = self.evaluate(qid, targets)
        max_k = self.max_k()
        if max_k is None or n_targets < max_k:
            max_k = n_targets

        for i in range(max_k):
            for j in range(i + 1, n_targets):
                tmp = targets[i]
                targets[i] = targets[j]
                targets[j] = tmp
                deltas[i, j] = self.evaluate(qid, targets) - original
                tmp = targets[i]
                targets[i] = targets[j]
                targets[j] = tmp

        return deltas

    def max_k(self):
        """Returns a value k for which:

        ``swap_delta()[i][j] == 0 for all i, j >= k``

        Returns None if no such value exists.

        """
        return None

    def evaluate_preds(self, qid, targets, preds):
        return self.evaluate(qid, get_sorted_y(targets, preds))

    def calc_random_ev(self, qid, targets):
        """Calculates the expectied value of the metric on randomized targets.

        The default implementation may be overriden with something smarter
        than repeated shuffles.

        """
        targets = np.copy(targets)
        scores = []
        for _ in range(100):
            np.random.shuffle(targets)
            scores.append(self.evaluate(qid, targets))
        return np.mean(scores)

    def calc_mean(self, qids, y, y_pred):
        """Calculates the mean of the metric among the provided predictions."""
        check_qids(qids)
        query_groups = get_groups(qids)
        return np.mean([self.evaluate_preds(qid, y[a:b], y_pred[a:b])
                        for qid, a, b in query_groups])

    def calc_mean_random(self, qids, y):
        """Calculates the EV of the mean of the metric with random ranking."""
        check_qids(qids)
        query_groups = get_groups(qids)
        return np.mean([self.calc_random_ev(qid, y[a:b])
                        for qid, a, b in query_groups])
