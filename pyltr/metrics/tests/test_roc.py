"""

Testing for AUC ROC metric.

"""

from . import helpers
import itertools
import numpy as np
import pyltr
import sklearn.metrics


class TestAUCROC(helpers.TestMetric):
    def get_metric(self):
        return pyltr.metrics.AUCROC()

    def get_queries_with_values(self):
        for i in range(0, 7):
            for tup in itertools.product(*([(0, 1)] * i)):
                if any(e != tup[0] for e in tup):
                    yield (np.array(tup),
                           sklearn.metrics.roc_auc_score(tup, range(i, 0, -1)))
                else:
                    yield np.array(tup), 0.0

    def get_queries(self):
        for i in range(0, 7):
            for tup in itertools.product(*([(0, 1)] * i)):
                yield np.array(tup)
