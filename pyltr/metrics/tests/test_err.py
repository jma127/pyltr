"""

Testing for Expected Reciprocal Rank metric.

"""

from . import helpers
import itertools
import numpy as np
import pyltr


class TestERR(helpers.TestMetric):
    def get_metric(self):
        return pyltr.metrics.ERR(4, k=3)

    def get_queries_with_values(self):
        yield [], 0.0
        yield [0], 0.0
        yield [1], 0.0625
        yield [2], 0.1875
        yield [3], 0.4375
        yield [4], 0.9375
        yield [0, 0, 0], 0.0
        yield [4, 2, 0], 0.943359375
        yield [1, 2.5, 4], 0.40663047881522385
        yield [2, 4, 1], 0.5694173177083334
        yield [2, 4, 1, 2], 0.5694173177083334

    def get_queries(self):
        for i in range(0, 5):
            for tup in itertools.product(*([(0, 1, 2, 4)] * i)):
                yield np.array(tup)

    def get_nulp(self):
        return 9999
