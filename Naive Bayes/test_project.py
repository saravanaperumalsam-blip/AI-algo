"""Tests for the Naive Bayes project.  Run:  python -m unittest -v"""
import unittest

from train import explain, load_data, load_model, train


class NaiveBayesProjectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = train(save=False)
        cls.bundle = load_model()

    def test_data_is_deduplicated_and_labelled(self):
        X, y = load_data()
        self.assertEqual(X.duplicated().sum(), 0)
        self.assertEqual(len(X), 5169)
        self.assertTrue(set(y.unique()) == {0, 1})

    def test_naive_bayes_is_accurate_and_precise(self):
        nb = next(v for k, v in self.results["test"].items() if k.startswith("Naive Bayes"))
        self.assertGreater(nb["f1"], 0.9)
        self.assertGreater(nb["precision"], 0.95)  # few legitimate messages blocked

    def test_naive_bayes_matches_logistic_regression(self):
        test = self.results["test"]
        nb = next(v for k, v in test.items() if k.startswith("Naive Bayes"))
        self.assertGreater(nb["f1"], test["Logistic Regression + TF-IDF"]["f1"] - 0.03)

    def test_obvious_messages(self):
        p_spam, words = explain(self.bundle, "Congratulations! You have won a prize, call now to claim")
        p_ham, _ = explain(self.bundle, "Are we still meeting for lunch tomorrow?")
        self.assertGreater(p_spam, 0.9)
        self.assertLess(p_ham, 0.1)
        self.assertGreater(len(words), 0)

    def test_prize_and_claim_are_spam_words(self):
        spam_words = self.results["top_spam_words"]
        self.assertIn("claim", spam_words)
        self.assertIn("prize", spam_words)


if __name__ == "__main__":
    unittest.main()
