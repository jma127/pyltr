"""

Training monitors for validation, early stopping, etc.

"""

import sklearn.utils
from . import AdditiveModel
from ..util.group import check_qids, get_groups
from ..util.sort import get_sorted_y


class ValidationMonitor(object):
    """Monitor for early stopping via validation set."""
    def __init__(self, X, y, qids, metric, stop_after=100,
                 trim_on_stop=True):
        self.X, self.y = sklearn.utils.check_X_y(
            X, y, dtype=sklearn.tree._tree.DTYPE)
        self.qids = qids
        self.metric = metric
        self.stop_after = stop_after
        self.trim_on_stop = trim_on_stop

        sklearn.utils.check_consistent_length(self.X, self.y, self.qids)
        check_qids(qids)

        self._query_groups = list(get_groups(self.qids))
        self._y_pred = None
        self._prev_iter = -1
        self._iter_scores = []
        self._best_score = None
        self._best_score_i = None

    def get_best_n(self):
        """Returns the n that maximizes score(ensemble[:n]).

        Should only be called after model.fit() has finished.

        """
        return self._best_score_i + 1

    def __call__(self, i, model, localvars):
        """Returns True if the model should stop early.

        Otherwise, returns a status string.

        """
        assert i == self._prev_iter + 1
        self._prev_iter = i

        if isinstance(model, AdditiveModel):
            if self._y_pred is None:
                self._y_pred = model.predict(self.X)
            else:
                self._y_pred += model.iter_y_delta(i, self.X)
            y_pred = self._y_pred
        else:
            y_pred = model.predict(self.X)

        score = 0.0
        for qid, a, b in self._query_groups:
            sorted_y = get_sorted_y(self.y[a:b], y_pred[a:b])
            score += self.metric.evaluate(qid, sorted_y)
        score /= len(self._query_groups)

        if self._best_score is None or score > self._best_score:
            self._best_score = score
            self._best_score_i = i

        since = i - self._best_score_i
        if self.trim_on_stop and \
                (since >= self.stop_after or i + 1 == model.n_estimators):
            model.trim(self._best_score_i + 1)
            return True

        return 'C:{:12.4f} B:{:12.4f} S:{:3d}'.format(
            score, self._best_score, since)
