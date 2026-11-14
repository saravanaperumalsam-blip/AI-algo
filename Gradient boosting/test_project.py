"""Tests for the Gradient Boosting project.  Run:  python -m unittest -v
(Training with hyperparameter search runs once and takes about 4 minutes.)"""
import unittest

from train import discount_curve, load_data, load_model, predict_profit, train

ORDER = {"Ship Mode": "Standard Class", "Segment": "Consumer", "Region": "West",
         "Category": "Furniture", "Sub-Category": "Chairs", "Sales": 500.0, "Quantity": 3,
         "Discount": 0.0}


class GradientBoostingProjectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = train(save=False)
        cls.comp = cls.results["comparison"]
        cls.bundle = load_model()

    def test_data_has_no_profit_leak(self):
        X, y = load_data()
        self.assertEqual(len(X), 9994)
        self.assertNotIn("Profit", X.columns)

    def test_boosting_beats_every_other_model_in_cross_validation(self):
        gb = self.comp["Gradient Boosting (tuned)"]["cv_r2"]
        for name, r in self.comp.items():
            if name != "Gradient Boosting (tuned)":
                self.assertGreater(gb, r["cv_r2"], name)

    def test_boosting_is_robust_on_the_test_set(self):
        gb = self.comp["Gradient Boosting (tuned)"]["test"]
        self.assertGreater(gb["r2"], 0.6)
        self.assertLess(gb["mae"], self.comp["Linear Regression"]["test"]["mae"])

    def test_discount_is_the_most_important_feature(self):
        self.assertEqual(next(iter(self.results["permutation_importance"])), "Discount")

    def test_bigger_discounts_reduce_predicted_profit(self):
        curve = discount_curve(self.bundle, ORDER)
        self.assertGreater(curve["Predicted profit"].iloc[0], curve["Predicted profit"].iloc[-1])
        self.assertGreater(predict_profit(self.bundle, ORDER),
                           predict_profit(self.bundle, {**ORDER, "Discount": 0.7}))


if __name__ == "__main__":
    unittest.main()
