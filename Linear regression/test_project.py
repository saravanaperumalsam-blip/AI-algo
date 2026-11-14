"""Tests for the Linear Regression project.  Run:  python -m unittest -v"""
import unittest

from train import FEATURES, load_data, predict, train


class LinearRegressionProjectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = train(save=False)

    def test_data_shape_and_no_missing_values(self):
        X, y = load_data()
        self.assertEqual(X.shape, (442, 10))
        self.assertEqual(list(X.columns), FEATURES)
        self.assertFalse(X.isnull().any().any())

    def test_models_beat_the_mean_baseline(self):
        comp = self.results["comparison"]
        baseline = comp["Baseline (predict the mean)"]["cv_rmse_mean"]
        for name, r in comp.items():
            if not name.startswith("Baseline"):
                self.assertLess(r["cv_rmse_mean"], baseline * 0.8, name)

    def test_best_model_explains_a_meaningful_share_of_variance(self):
        self.assertGreater(self.results["best_test_metrics"]["r2"], 0.35)

    def test_bmi_increases_predicted_progression(self):
        coef = self.results["linear_coefficients_per_std"]
        self.assertGreater(coef["bmi"], 0)

    def test_predict_returns_a_plausible_score(self):
        from train import load_model
        bundle = load_model()
        X, _ = load_data()
        patient = X.median().to_dict()
        score = predict(bundle, patient)
        self.assertTrue(25 <= score <= 346, score)


if __name__ == "__main__":
    unittest.main()
