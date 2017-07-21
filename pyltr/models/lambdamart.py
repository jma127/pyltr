"""

Implementation of LambdaMART.

Interface is very similar to sklearn's tree ensembles. In fact, the majority
of this code is just a port of GradientBoostingRegressor customized for LTR.
The most notable difference is that fit() now takes another `qids` parameter
containing query ids for all the samples.

"""

# Derivative of scikit-learn
#     https://github.com/scikit-learn/scikit-learn/
#     sklearn/ensemble/gradient_boosting.py
# License: BSD 3 clause

import numbers
import numpy as np
import scipy
import sklearn.ensemble
import sklearn.externals
import sklearn.utils
import sklearn.tree
import time
from . import AdditiveModel
from .. import metrics
from ..util.group import check_qids, get_groups
from ..util.sort import get_sorted_y_positions
from sklearn.externals.six.moves import range


class LambdaMART(AdditiveModel):
    """Tree-based learning to rank model.

    Parameters
    ----------

    metric : object
        The metric to be maximized by the model.
    learning_rate : float, optional (default=0.1)
        Shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.
    n_estimators : int, optional (default=100)
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.
    max_depth : int, optional (default=3)
        Maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.
        Ignored if ``max_leaf_nodes`` is not None.
    min_samples_split : int, optional (default=2)
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, optional (default=1)
        The minimum number of samples required to be at a leaf node.
    subsample : float, optional (default=1.0)
        The fraction of samples to be used for fitting the individual base
        learners. I have no idea why one would set this to something lower than
        one, and results will probably be strange if this is changed from the
        default.
    query_subsample : float, optional (default=1.0)
        The fraction of queries to be used for fitting the individual base
        learners.
    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:
          - If int, then consider `max_features` features at each split.
          - If float, then `max_features` is a percentage and
            `int(max_features * n_features)` features are considered at each
            split.
          - If "auto", then `max_features=sqrt(n_features)`.
          - If "sqrt", then `max_features=sqrt(n_features)`.
          - If "log2", then `max_features=log2(n_features)`.
          - If None, then `max_features=n_features`.
        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
        If not None then ``max_depth`` will be ignored.
    verbose : int, optional (default=0)
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.
    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    feature_importances_ : array, shape = [n_features]
        The feature importances (the higher, the more important the feature).
    oob_improvement_ : array, shape = [n_estimators]
        The improvement in loss (= deviance) on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.
    train_score_ : array, shape = [n_estimators]
        The i-th score ``train_score_[i]`` is the deviance (= loss) of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the deviance on the training data.
    estimators_ : ndarray of DecisionTreeRegressor, shape = [n_estimators, 1]
        The collection of fitted sub-estimators. ``loss_.K`` is 1 for binary
        classification, otherwise n_classes.
    estimators_fitted_ : int
        The number of sub-estimators actually fitted. This may be different
        from n_estimators in the case of early stoppage, trimming, etc.

    """
    def __init__(self, metric=None, learning_rate=0.1, n_estimators=100,
                 query_subsample=1.0, subsample=1.0, min_samples_split=2,
                 min_samples_leaf=1, max_depth=3, random_state=None,
                 max_features=None, verbose=0, max_leaf_nodes=None,
                 warm_start=True):
        super(LambdaMART, self).__init__()
        self.metric = metrics.dcg.NDCG() if metric is None else metric
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.query_subsample = query_subsample
        self.subsample = subsample
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state
        self.max_features = max_features
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start

    def fit(self, X, y, qids, monitor=None):
        """Fit lambdamart onto a dataset.

        Parameters
        ----------

        X : array_like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array_like, shape = [n_samples]
            Target values (integers in classification, real numbers in
            regression)
            For classification, labels must correspond to classes.
        qids : array_like, shape = [n_samples]
            Query ids for each sample. Samples must be grouped by query such
            that all queries with the same qid appear in one contiguous block.
        monitor : callable, optional
            The monitor is called after each iteration with the current
            iteration, a reference to the estimator and the local variables of
            ``_fit_stages`` as keyword arguments ``callable(i, self,
            locals())``. If the callable returns ``True`` the fitting procedure
            is stopped. The monitor can be used for various things such as
            computing held-out estimates, early stopping, model introspecting,
            and snapshoting.

        """

        if not self.warm_start:
            self._clear_state()

        X, y = sklearn.utils.check_X_y(X, y, dtype=sklearn.tree._tree.DTYPE)
        n_samples, self.n_features = X.shape

        sklearn.utils.check_consistent_length(X, y, qids)
        if y.dtype.kind == 'O':
            y = y.astype(np.float64)

        random_state = sklearn.utils.check_random_state(self.random_state)
        self._check_params()

        if not self._is_initialized():
            self._init_state()
            begin_at_stage = 0
            y_pred = np.zeros(y.shape[0])
        else:
            if self.n_estimators < self.estimators_.shape[0]:
                raise ValueError('n_estimators=%d must be larger or equal to '
                                 'estimators_.shape[0]=%d when '
                                 'warm_start==True'
                                 % (self.n_estimators,
                                    self.estimators_.shape[0]))
            begin_at_stage = self.estimators_.shape[0]
            self.estimators_fitted_ = begin_at_stage
            self.estimators_.resize((self.n_estimators, 1))
            self.train_score_.resize(self.n_estimators)
            if self.query_subsample < 1.0:
                self.oob_improvement_.resize(self.n_estimators)
            y_pred = self.predict(X)

        n_stages = self._fit_stages(X, y, qids, y_pred,
                                    random_state, begin_at_stage, monitor)

        if n_stages < self.estimators_.shape[0]:
            self.trim(n_stages)

        return self

    def predict(self, X):
        X = sklearn.utils.validation.check_array(
            X, dtype=sklearn.tree._tree.DTYPE, order='C')
        score = np.zeros((X.shape[0], 1))
        estimators = self.estimators_
        if self.estimators_fitted_ < len(estimators):
            estimators = estimators[:self.estimators_fitted_]
        sklearn.ensemble._gradient_boosting.predict_stages(
            estimators, X, self.learning_rate, score)

        return score.ravel()

    def iter_y_delta(self, i, X):
        assert i >= 0 and i < self.estimators_fitted_

        X = sklearn.utils.validation.check_array(
            X, dtype=sklearn.tree._tree.DTYPE, order='C')
        score = np.zeros((X.shape[0], 1))
        sklearn.ensemble._gradient_boosting.predict_stage(
            self.estimators_, i, X, self.learning_rate, score)

        return score.ravel()

    def trim(self, n):
        assert n <= self.estimators_fitted_

        self.estimators_fitted_ = n
        self.estimators_ = self.estimators_[:n]
        self.train_score_ = self.train_score_[:n]
        if hasattr(self, 'oob_improvement_'):
            self.oob_improvement_ = self.oob_improvement_[:n]

    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
        feature).

        Returns
        -------
        feature_importances_ : array, shape = [n_features]
            Array of summed variance reductions.

        """
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise NotFittedError("Estimator not fitted, call `fit` before"
                                 " `feature_importances_`.")

        total_sum = np.zeros((self.n_features, ), dtype=np.float64)
        for stage in self.estimators_:
            stage_sum = sum(tree.feature_importances_
                            for tree in stage) / len(stage)
            total_sum += stage_sum

        importances = total_sum / len(self.estimators_)
        return importances

    def _calc_lambdas_deltas(self, qid, y, y_pred):
        ns = y.shape[0]
        positions = get_sorted_y_positions(y, y_pred, check=False)
        actual = y[positions]

        swap_deltas = self.metric.calc_swap_deltas(qid, actual)
        max_k = self.metric.max_k()
        if max_k is None or ns < max_k:
            max_k = ns

        lambdas = np.zeros(ns)
        deltas = np.zeros(ns)

        for i in range(max_k):
            for j in range(i + 1, ns):
                if actual[i] == actual[j]:
                    continue

                delta_metric = swap_deltas[i, j]
                if delta_metric == 0.0:
                    continue

                a, b = positions[i], positions[j]
                # invariant: y_pred[a] >= y_pred[b]

                if actual[i] < actual[j]:
                    assert delta_metric > 0.0
                    logistic = scipy.special.expit(y_pred[a] - y_pred[b])
                    l = logistic * delta_metric
                    lambdas[a] -= l
                    lambdas[b] += l
                else:
                    assert delta_metric < 0.0
                    logistic = scipy.special.expit(y_pred[b] - y_pred[a])
                    l = logistic * -delta_metric
                    lambdas[a] += l
                    lambdas[b] -= l

                gradient = (1 - logistic) * l
                deltas[a] += gradient
                deltas[b] += gradient

        return lambdas, deltas

    def _update_terminal_regions(self, tree, X, y, lambdas, deltas, y_pred,
                                 sample_mask):
        terminal_regions = tree.apply(X)
        masked_terminal_regions = terminal_regions.copy()
        masked_terminal_regions[~sample_mask] = -1

        for leaf in np.where(tree.children_left ==
                             sklearn.tree._tree.TREE_LEAF)[0]:
            terminal_region = np.where(masked_terminal_regions == leaf)
            suml = np.sum(lambdas[terminal_region])
            sumd = np.sum(deltas[terminal_region])
            tree.value[leaf, 0, 0] = 0.0 if abs(sumd) < 1e-300 else (suml / sumd)

        y_pred += tree.value[terminal_regions, 0, 0] * self.learning_rate

    def _fit_stage(self, i, X, y, qids, y_pred, sample_weight, sample_mask,
                   query_groups, random_state):
        """Fit another tree to the boosting model."""
        assert sample_mask.dtype == np.bool

        n_samples = X.shape[0]

        all_lambdas = np.zeros(n_samples)
        all_deltas = np.zeros(n_samples)
        for qid, a, b, _ in query_groups:
            lambdas, deltas = self._calc_lambdas_deltas(qid, y[a:b],
                                                        y_pred[a:b])
            all_lambdas[a:b] = lambdas
            all_deltas[a:b] = deltas

        tree = sklearn.tree.DecisionTreeRegressor(
            criterion='friedman_mse',
            splitter='best',
            presort=True,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=0.0,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=random_state)

        if self.subsample < 1.0 or self.query_subsample < 1.0:
            sample_weight = sample_weight * sample_mask.astype(np.float64)

        tree.fit(X, all_lambdas, sample_weight=sample_weight,
                 check_input=False)

        self._update_terminal_regions(tree.tree_, X, y, all_lambdas,
                                      all_deltas, y_pred, sample_mask)
        self.estimators_[i, 0] = tree
        self.estimators_fitted_ = i + 1

        return y_pred

    def _fit_stages(self, X, y, qids, y_pred, random_state,
                    begin_at_stage=0, monitor=None):
        n_samples = X.shape[0]
        do_subsample = self.subsample < 1.0
        sample_weight = np.ones(n_samples, dtype=np.float64)

        n_queries = check_qids(qids)
        query_groups = np.array([(qid, a, b, np.arange(a, b))
                                 for qid, a, b in get_groups(qids)],
                                dtype=np.object)
        assert n_queries == len(query_groups)
        do_query_oob = self.query_subsample < 1.0
        query_mask = np.ones(n_queries, dtype=np.bool)
        query_idx = np.arange(n_queries)
        q_inbag = max(1, int(self.query_subsample * n_queries))

        if self.verbose:
            verbose_reporter = _VerboseReporter(self.verbose)
            verbose_reporter.init(self, begin_at_stage)

        for i in range(begin_at_stage, self.n_estimators):
            if do_query_oob:
                random_state.shuffle(query_idx)
                query_mask = np.zeros(n_queries, dtype=np.bool)
                query_mask[query_idx[:q_inbag]] = 1

            query_groups_to_use = query_groups[query_mask]
            sample_mask = np.zeros(n_samples, dtype=np.bool)
            for qid, a, b, sidx in query_groups_to_use:
                sidx_to_use = sidx
                if do_subsample:
                    query_samples_inbag = max(
                        1, int(self.subsample * (b - 1)))
                    random_state.shuffle(sidx)
                    sidx_to_use = sidx[:query_samples_inbag]
                sample_mask[sidx_to_use] = 1

            if do_query_oob:
                old_oob_total_score = 0.0
                for qid, a, b, _ in query_groups[~query_mask]:
                    old_oob_total_score += self.metric.evaluate_preds(
                        qid, y[a:b], y_pred[a:b])

            y_pred = self._fit_stage(i, X, y, qids, y_pred, sample_weight,
                                     sample_mask, query_groups_to_use,
                                     random_state)

            train_total_score, oob_total_score = 0.0, 0.0
            for qidx, (qid, a, b, _) in enumerate(query_groups):
                score = self.metric.evaluate_preds(
                    qid, y[a:b], y_pred[a:b])
                if query_mask[qidx]:
                    train_total_score += score
                else:
                    oob_total_score += score

            self.train_score_[i] = train_total_score / q_inbag
            if do_query_oob:
                if q_inbag < n_queries:
                    self.oob_improvement_[i] = \
                        ((oob_total_score - old_oob_total_score) /
                         (n_queries - q_inbag))

            early_stop = False
            monitor_output = None
            if monitor is not None:
                monitor_output = monitor(i, self, locals())
                if monitor_output is True:
                    early_stop = True

            if self.verbose > 0:
                verbose_reporter.update(i, self, monitor_output)

            if early_stop:
                break

        return i + 1

    def _is_initialized(self):
        return len(getattr(self, 'estimators_', [])) > 0

    def _init_state(self):
        self.estimators_ = np.empty((self.n_estimators, 1), dtype=np.object)
        self.estimators_fitted_ = 0
        self.train_score_ = np.zeros(self.n_estimators, dtype=np.float64)
        if self.query_subsample < 1.0:
            self.oob_improvement_ = np.zeros((self.n_estimators,),
                                             dtype=np.float64)

    def _clear_state(self):
        if hasattr(self, 'estimators_'):
            del self.estimators_
        if hasattr(self, 'esimators_trained'):
            del self.estimators_fitted_
        if hasattr(self, 'train_score_'):
            del self.train_score_
        if hasattr(self, 'oob_improvement_'):
            del self.oob_improvement_

    def _check_params(self):
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0 but "
                             "was %r" % self.n_estimators)

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than 0 but "
                             "was %r" % self.learning_rate)

        if not (0.0 < self.subsample <= 1.0):
            raise ValueError("subsample must be in (0,1] but "
                             "was %r" % self.subsample)

        if not (0.0 < self.query_subsample <= 1.0):
            raise ValueError("query_subsample must be in (0,1] but "
                             "was %r" % self.query_subsample)

        if isinstance(self.max_features, sklearn.externals.six.string_types):
            if self.max_features == "auto":
                max_features = self.n_features
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features)))
            else:
                raise ValueError("Invalid value for max_features: %r. "
                                 "Allowed string values are 'auto', 'sqrt' "
                                 "or 'log2'." % self.max_features)
        elif self.max_features is None:
            max_features = self.n_features
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if 0. < self.max_features <= 1.:
                max_features = max(int(self.max_features * self.n_features), 1)
            else:
                raise ValueError("max_features must be in (0,1]")

        self.max_features_ = max_features


class _VerboseReporter(object):
    """Reports verbose output to stdout.

    If ``verbose==1`` output is printed once in a while (when iteration mod
    verbose_mod is zero).; if larger than 1 then output is printed for
    each update.

    """
    def __init__(self, verbose):
        self.verbose = verbose

    def init(self, est, begin_at_stage=0):
        # header fields and line format str
        header_fields = ['Iter', 'Train score']
        verbose_fmt = ['{iter:>5d}', '{train_score:>12.4f}']
        # do oob?
        if est.query_subsample < 1:
            header_fields.append('OOB Improve')
            verbose_fmt.append('{oob_impr:>12.4f}')
        header_fields.append('Remaining')
        verbose_fmt.append('{remaining_time:>12s}')
        header_fields.append('Monitor Output')
        verbose_fmt.append('{monitor_output:>40s}')

        # print the header line
        print(('%5s ' + '%12s ' *
               (len(header_fields) - 2) + '%40s ') % tuple(header_fields))

        self.verbose_fmt = ' '.join(verbose_fmt)
        # plot verbose info each time i % verbose_mod == 0
        self.verbose_mod = 1
        self.start_time = time.time()
        self.begin_at_stage = begin_at_stage

    def update(self, j, est, monitor_output):
        """Update reporter with new iteration. """
        if monitor_output is True:
            print('Early termination at iteration ', j)
            return
        do_query_oob = est.query_subsample < 1
        # we need to take into account if we fit additional estimators.
        i = j - self.begin_at_stage  # iteration relative to the start iter
        if self.verbose > 1 or (i + 1) % self.verbose_mod == 0:
            oob_impr = est.oob_improvement_[j] if do_query_oob else 0
            remaining_time = ((est.n_estimators - (j + 1)) *
                              (time.time() - self.start_time) / float(i + 1))
            if remaining_time > 60:
                remaining_time = '{0:.2f}m'.format(remaining_time / 60.0)
            else:
                remaining_time = '{0:.2f}s'.format(remaining_time)
            if monitor_output is None:
                monitor_output = ''
            print(self.verbose_fmt.format(iter=j + 1,
                                          train_score=est.train_score_[j],
                                          oob_impr=oob_impr,
                                          remaining_time=remaining_time,
                                          monitor_output=monitor_output))
            if i + 1 >= 10:
                self.verbose_mod = 5
            if i + 1 >= 50:
                self.verbose_mod = 10
            if i + 1 >= 100:
                self.verbose_mod = 20
            if i + 1 >= 500:
                self.verbose_mod = 50
            if i + 1 >= 1000:
                self.verbose_mod = 100
