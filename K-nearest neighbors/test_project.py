"""Tests for the K-Nearest Neighbors project.  Run:  python -m unittest -v"""
import unittest

from train import CLASS_NAMES, load_data, load_model, predict_with_neighbours, train


class KNNProjectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = train(save=False)

    def test_data(self):
        X, y = load_data()
        self.assertEqual(X.shape, (178, 13))
        self.assertEqual(sorted(y.unique()), [0, 1, 2])

    def test_scaling_improves_accuracy_for_every_k(self):
        curve = self.results["k_curve"]
        for k, s, u in zip(curve["k"], curve["scaled"], curve["unscaled"]):
            self.assertGreater(s, u, f"k={k}")

    def test_scaled_model_is_much_better_on_the_test_set(self):
        test = self.results["test"]
        scaled = test["KNN scaled + tuned"]["accuracy"]
        unscaled = next(v["accuracy"] for n, v in test.items() if n.startswith("KNN unscaled"))
        self.assertGreater(scaled, 0.9)
        self.assertGreater(scaled - unscaled, 0.1)

    def test_training_wine_is_recognised_with_its_neighbours(self):
        bundle = load_model()
        X, y = bundle["train_X"], bundle["train_y"]
        pred, probs, neighbours = predict_with_neighbours(bundle, X.iloc[0].to_dict())
        self.assertEqual(pred, CLASS_NAMES[y.iloc[0]])
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=2)
        self.assertEqual(neighbours["Distance"].iloc[0], 0.0)  # its own copy is nearest
        self.assertTrue(neighbours["Distance"].is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
