"""

Test helpers for metric classes.

Essentially, all you need to do is:

- Define a small set of queries along with expected metric values (for testing
  evaluate())
- Define a larger set of queries for testing other methods, using evaluate() as
  ground truth.

"""

import numpy as np
from numpy.testing import assert_approx_equal, assert_array_almost_equal_nulp


class TestMetric(object):
    def get_metric(self):
        """Metric class instance to test."""
        raise NotImplementedError()

    def get_queries_with_values(self):
        """Queries to test for evaluate()."""
        raise NotImplementedError()

    def get_queries(self):
        """Queries to test for methods except evaluate()."""
        raise NotImplementedError()

    def get_nulp(self):
        """NULP value for assertions."""
        return 99

    def test_evaluate(self):
        m = self.get_metric()
        for id, (q, v) in enumerate(self.get_queries_with_values()):
            # Test twice to ensure caching doesn't break things.
            assert_approx_equal(v, m.evaluate(id, q))
            assert_approx_equal(v, m.evaluate(id, q))

    def test_calc_swap_deltas(self):
        m = self.get_metric()
        for id, q in enumerate(self.get_queries()):
            n = len(q)
            q_cpy = np.copy(q)
            expected_deltas = np.zeros((n, n))
            deltas = m.calc_swap_deltas(id, q)
            orig = m.evaluate(id, q)

            for i in range(n):
                for j in range(n):
                    q_cpy[[i, j]] = q[[j, i]]
                    expected_deltas[i, j] = m.evaluate(id, q_cpy) - orig
                    q_cpy[[i, j]] = q[[i, j]]

            # Test that diagonal is 0.
            assert_array_almost_equal_nulp(
                0.0, np.diagonal(expected_deltas), nulp=self.get_nulp())

            # Test that the expected deltas are symmetric.
            assert_array_almost_equal_nulp(
                expected_deltas.transpose(), expected_deltas,
                nulp=self.get_nulp())

            # Test that deltas computed by the metric match the deltas from
            # evaluate().
            assert_array_almost_equal_nulp(
                np.triu(expected_deltas), deltas,
                nulp=self.get_nulp())

            # Test that max_k() works.
            if m.max_k() is not None:
                assert_array_almost_equal_nulp(
                    0.0, expected_deltas[m.max_k():, m.max_k():],
                    nulp=self.get_nulp())
