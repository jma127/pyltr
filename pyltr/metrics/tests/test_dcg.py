"""

Testing for (Normalized) DCG metric.

"""

from . import helpers
import itertools
import numpy as np
import pyltr


class TestDCG(helpers.TestMetric):
    def get_metric(self):
        return pyltr.metrics.DCG(k=3)

    def get_queries_with_values(self):
        yield [], 0.0
        yield [0], 0.0
        yield [1], 1.0
        yield [2], 3.0
        yield [2, 1, 0], 3.6309297535714578
        yield [0, 0, 0], 0.0
        yield [2, 5, 1], 23.058822360715183
        yield [2, 5, 1, 9], 23.058822360715183

    def get_queries(self):
        for i in range(0, 5):
            for tup in itertools.product(*([(0, 1, 2.5)] * i)):
                yield np.array(tup)


class TestNDCG(helpers.TestMetric):
    def get_metric(self):
        return pyltr.metrics.NDCG(k=3)

    def get_queries_with_values(self):
        yield [], 0.0
        yield [0], 0.0
        yield [1], 1.0
        yield [2], 1.0
        yield [2, 1, 0], 1.0
        yield [1, 2, 0], 0.7967075809905066
        yield [0, 0, 0], 0.0
        yield [2, 5, 1], 0.6905329824556825
        yield [2, 5, 1, 9], 0.04333885914794999
        yield [3, 2, 1, 1], 1.0

    def get_queries(self):
        for i in range(0, 5):
            for tup in itertools.product(*([(0, 1, 2.5)] * i)):
                yield np.array(tup)
