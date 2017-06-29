"""

Testing for (Mean) Average Precision metric.

"""

from . import helpers
import itertools
import numpy as np
import pyltr


class TestAP(helpers.TestMetric):
    def get_metric(self):
        return pyltr.metrics.AP(k=4)

    def get_queries_with_values(self):
        yield [], 0.0
        yield [0], 0.0
        yield [1], 1.0
        yield [1, 0], 1.0
        yield [0, 1], 0.5
        yield [1, 0, 1, 0], 5.0 / 6
        yield [0, 1, 1, 1], 23.0 / 36
        yield [1, 0, 1, 0, 1], 5.0 / 9
        yield [1, 0, 1, 0, 0], 5.0 / 6

    def get_queries(self):
        for i in range(0, 7):
            for tup in itertools.product(*([(0, 1)] * i)):
                yield np.array(tup)
