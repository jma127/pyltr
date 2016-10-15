"""

Testing for Kendall's Tau metric.

"""

from . import helpers
import itertools
import numpy as np
import pyltr


class TestKendallTau(helpers.TestMetric):
    def get_metric(self):
        return pyltr.metrics.KendallTau()

    def get_queries_with_values(self):
        yield [], 0.0
        yield [0], 0.0
        yield [1], 0.0
        yield [2], 0.0
        yield [0, 0, 0], 0.0
        yield [4, 2, 0], 1.0
        yield [1, 2.5, 4], -1.0
        yield [4, 4, 2], 2.0 / 3
        yield [2, 4, 1, 2], 1.0 / 6

    def get_queries(self):
        for i in range(0, 5):
            for tup in itertools.product(*([(0, 1, 2, 4)] * i)):
                yield np.array(tup)
