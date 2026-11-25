import unittest
import numpy as np

import sgd_seminar as seminar


class SgdSeminarTests(unittest.TestCase):
    def test_bb_reduces_quadratic(self):
        values, steps = seminar.barzilai_borwein_quadratic()
        self.assertLess(values[-1], values[0])
        self.assertGreater(len(steps), 0)

    def test_trust_region_stays_finite(self):
        trajectory = seminar.trust_region_experiment()
        self.assertTrue(np.all(np.isfinite(trajectory)))

    def test_bfgs_reduces_quadratic(self):
        values = seminar.bfgs_experiment()
        self.assertLess(values[-1], values[0])

    def test_lbfgs_converges_for_memory_values(self):
        results = seminar.lbfgs_experiment(d=40)
        for values, iterations, memory_bytes in results.values():
            self.assertTrue(np.all(np.isfinite(values)))
            self.assertGreater(iterations, 0)
            self.assertGreater(memory_bytes, 0)


if __name__ == "__main__":
    unittest.main()
