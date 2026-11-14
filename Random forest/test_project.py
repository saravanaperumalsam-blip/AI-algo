"""Tests for the Random Forest project.  Run:  python -m unittest -v
(Training runs once for all tests and takes about 2 minutes.)"""
import unittest

from train import load_data, load_model, predict_with_votes, train

HEALTHY = dict(male=0, age=40, currentSmoker=0, cigsPerDay=0, BPMeds=0, prevalentStroke=0,
               prevalentHyp=0, diabetes=0, totChol=190, sysBP=110, diaBP=70, BMI=22,
               heartRate=70, glucose=80)
AT_RISK = dict(male=1, age=68, currentSmoker=1, cigsPerDay=30, BPMeds=1, prevalentStroke=0,
               prevalentHyp=1, diabetes=1, totChol=300, sysBP=190, diaBP=105, BMI=32,
               heartRate=90, glucose=200)


class RandomForestProjectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = train(save=False)
        cls.comp = cls.results["comparison"]

    def test_data(self):
        X, y = load_data()
        self.assertEqual(X.shape, (4238, 14))
        self.assertAlmostEqual(y.mean(), 0.152, places=2)

    def test_forest_beats_a_single_tree(self):
        self.assertGreater(self.comp["Random Forest (tuned)"]["cv_auc_mean"],
                           self.comp["Single Decision Tree (tuned)"]["cv_auc_mean"])

    def test_default_forest_overfits_and_tuning_reduces_the_gap(self):
        default, tuned = self.comp["Random Forest (default)"], self.comp["Random Forest (tuned)"]
        self.assertGreater(default["train_auc"], 0.99)
        gap_default = default["train_auc"] - default["cv_auc_mean"]
        gap_tuned = tuned["train_auc"] - tuned["cv_auc_mean"]
        self.assertLess(gap_tuned, gap_default)

    def test_more_trees_do_not_hurt_oob_score(self):
        oob = self.results["oob_curve"]["oob_auc"]
        self.assertGreater(oob[-1], oob[0])

    def test_age_is_the_most_important_feature(self):
        self.assertEqual(next(iter(self.results["permutation_importance"])), "age")

    def test_risk_ordering_and_vote_count(self):
        bundle = load_model()
        low, low_votes, n = predict_with_votes(bundle, HEALTHY)
        high, high_votes, _ = predict_with_votes(bundle, AT_RISK)
        self.assertLess(low, high)
        self.assertLess(low_votes, high_votes)
        self.assertTrue(0 <= low_votes <= n)


if __name__ == "__main__":
    unittest.main()
