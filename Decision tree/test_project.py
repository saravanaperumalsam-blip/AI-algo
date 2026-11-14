"""Tests for the Decision Tree project.  Run:  python -m unittest -v"""
import unittest

from train import FEATURES, load_data, load_model, predict_with_path, train


class DecisionTreeProjectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = train(save=False)
        cls.bundle = load_model()

    def test_data_has_expected_columns_and_no_profit_leak(self):
        X, y = load_data()
        self.assertEqual(len(X), 9994)
        self.assertEqual(list(X.columns), FEATURES)
        self.assertNotIn("Profit", X.columns)
        self.assertTrue(set(y.unique()) <= {0, 1})

    def test_unlimited_tree_overfits(self):
        full = self.results["models"]["Unlimited tree"]
        self.assertGreater(full["train"]["f1"], 0.98)
        self.assertLess(full["test"]["f1"], full["train"]["f1"] - 0.1)

    def test_tuned_tree_is_smaller_and_generalises_better(self):
        full = self.results["models"]["Unlimited tree"]
        tuned = self.results["models"]["Tuned tree"]
        self.assertLess(tuned["leaves"], full["leaves"])
        self.assertGreaterEqual(tuned["test"]["f1"], full["test"]["f1"])
        self.assertGreater(tuned["test"]["roc_auc"], 0.9)

    def test_discount_is_the_most_important_feature(self):
        top = next(iter(self.results["top_features"]))
        self.assertEqual(top, "Discount")

    def test_big_discount_predicts_loss_and_no_discount_predicts_profit(self):
        base = {"Ship Mode": "Standard Class", "Segment": "Consumer", "Region": "Central",
                "Category": "Office Supplies", "Sub-Category": "Binders",
                "Sales": 120.0, "Quantity": 3}
        p_high, path = predict_with_path(self.bundle, {**base, "Discount": 0.8})
        p_none, _ = predict_with_path(self.bundle, {**base, "Discount": 0.0})
        self.assertGreater(p_high, 0.5)
        self.assertLess(p_none, 0.5)
        self.assertTrue(path[0].startswith("Discount"))


if __name__ == "__main__":
    unittest.main()
