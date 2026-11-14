"""Tests for the Support Vector Machine project.  Run:  python -m unittest -v"""
import unittest

from train import diagnose, load_data, load_model, train


class SVMProjectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = train(save=False)

    def test_data(self):
        X, y = load_data()
        self.assertEqual(X.shape, (569, 30))
        self.assertEqual(int(y.sum()), 212)  # malignant cases

    def test_scaling_matters_for_rbf(self):
        k = self.results["kernel_cv_accuracy"]
        self.assertGreater(k["RBF"], k["RBF, unscaled"] + 0.03)

    def test_tuned_svm_is_highly_accurate_and_catches_cancer(self):
        rbf = self.results["test"]["SVM, RBF kernel (tuned)"]
        self.assertGreater(rbf["accuracy"], 0.95)
        self.assertGreater(rbf["recall_malignant"], 0.9)
        self.assertGreater(rbf["roc_auc"], 0.98)

    def test_only_a_minority_of_points_are_support_vectors(self):
        self.assertLess(self.results["support_vectors"]["share_of_training"], 0.3)

    def test_diagnose_matches_prediction_sign(self):
        bundle = load_model()
        X, y = bundle["test_X"], bundle["test_y"]
        label, d = diagnose(bundle, X.iloc[0].to_dict())
        self.assertEqual(label == "Malignant", d > 0)


if __name__ == "__main__":
    unittest.main()
