import numpy as np
from sklearn.externals.six.moves import range
from ..util.group import check_qids, get_groups
from ..util.sort import get_sorted_y


class Metric(object):
    """Base LTR metric class.

    Subclasses must override evaluate() and can optionally override various
    other methods.

    """
    def evaluate(self, qid, targets):
        """Evaluates the metric on a ranked list of targets.

        Parameters
        ----------
        qid : object
            Query id. Guaranteed to be a hashable type s.t.
            ``sorted(targets1) == sorted(targets2)`` iff ``qid1 == qid2``.
        targets : array_like of shape = [n_targets]
            List of targets for the query, in order of predicted score.

        Returns
        -------
        float
            Value of the metric on the provided list of targets.

        """
        raise NotImplementedError()

    def calc_swap_deltas(self, qid, targets):
        """Returns an upper triangular matrix.

        Each (i, j) contains the change in the metric from swapping
        targets[i, j].

        Parameters
        ----------
        qid : object
            See `evaluate`.
        targets : array_like of shape = [n_targets]
            See `evaluate`.

        Returns
        -------
        deltas = array_like of shape = [n_targets, n_targets]
            Upper triangular matrix, where ``deltas[i, j]`` is the change in
            the metric from swapping ``targets[i]`` with ``targets[j]``.

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
        """Returns a cutoff value for the metric.

        Returns
        -------
        k : int or None
            Value for which ``swap_delta()[i, j] == 0 for all i, j >= k``.
            None if no such value.

        """
        return None

    def evaluate_preds(self, qid, targets, preds):
        """Evaluates the metric on a ranked list of targets.

        Parameters
        ----------
        qid : object
            See `evaluate`.
        targets : array_like of shape = [n_targets]
            See `evaluate`.
        preds : array_like of shape = [n_targets]
            List of predicted scores corresponding to the targets. The
            `targets` array will be sorted by these predictions before
            evaluation.

        Returns
        -------
        float
            Value of the metric on the provided list of targets and
            predictions.

        """
        return self.evaluate(qid, get_sorted_y(targets, preds))

    def calc_random_ev(self, qid, targets):
        """Calculates the expectied value of the metric on randomized targets.

        This implementation just averages the metric over 100 shuffles.

        Parameters
        ----------
        qid : object
            See `evaluate`.
        targets : array_like of shape = [n_targets]
            See `evaluate`.

        Returns
        -------
        float
            Expected value of the metric from random ordering of targets.

        """
        targets = np.copy(targets)
        scores = []
        for _ in range(100):
            np.random.shuffle(targets)
            scores.append(self.evaluate(qid, targets))
        return np.mean(scores)

    def calc_mean(self, qids, targets, preds):
        """Calculates the mean of the metric among the provided predictions.

        Parameters
        ----------
        qids : array_like of shape = [n_targets]
            List of query ids. They must be grouped contiguously
            (i.e. ``pyltr.util.group.check_qids`` must pass).
        targets : array_like of shape = [n_targets]
            List of targets.
        preds : array_like of shape = [n_targets]
            List of predicted scores corresponding to the targets.

        Returns
        -------
        float
            Mean of the metric over provided query groups.

        """
        check_qids(qids)
        query_groups = get_groups(qids)
        return np.mean([self.evaluate_preds(qid, targets[a:b], preds[a:b])
                        for qid, a, b in query_groups])

    def calc_mean_random(self, qids, targets):
        """Calculates the EV of the mean of the metric with random ranking.

        Parameters
        ----------
        qids : array_like of shape = [n_targets]
            See `calc_mean`.
        targets : array_like of shape = [n_targets]
            See `calc_mean`.

        Returns
        -------
        float
            Expected value of the mean of the metric on random orderings of the
            provided query groups.

        """
        check_qids(qids)
        query_groups = get_groups(qids)
        return np.mean([self.calc_random_ev(qid, targets[a:b])
                        for qid, a, b in query_groups])
