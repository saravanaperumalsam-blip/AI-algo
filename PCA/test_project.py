"""Tests for the PCA project.  Run:  python -m unittest -v"""
import unittest

import numpy as np

from train import load_data, load_model, reconstruct, train


class PCAProjectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = train(save=False)
        cls.bundle = load_model()

    def test_data(self):
        X, y = load_data()
        self.assertEqual(X.shape, (1797, 64))
        self.assertEqual(len(set(y)), 10)

    def test_components_are_sorted_by_importance(self):
        v = self.bundle["pca"].explained_variance_ratio_
        self.assertTrue(np.all(np.diff(v) <= 1e-12))
        self.assertAlmostEqual(v.sum(), 1.0, places=6)

    def test_far_fewer_than_64_components_keep_95_percent(self):
        self.assertLess(self.results["components_needed"]["95%"], 40)

    def test_reconstruction_error_shrinks_to_zero(self):
        err = list(self.results["reconstruction_mse"].values())
        self.assertTrue(all(a >= b for a, b in zip(err, err[1:])))
        X = self.bundle["test_X"]
        self.assertTrue(np.allclose(reconstruct(self.bundle["pca"], X, 64), X, atol=1e-6))

    def test_compressed_classifier_matches_full_pixels(self):
        acc = self.results["classifier_by_components"]
        self.assertGreater(self.results["final_classifier"]["test_accuracy"], 0.97)
        self.assertGreaterEqual(acc[20]["accuracy"], acc[64]["accuracy"] - 0.01)


if __name__ == "__main__":
    unittest.main()
