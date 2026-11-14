"""Tests for the K-Means Clustering project.  Run:  python -m unittest -v"""
import unittest

import pandas as pd

from train import assign_segment, load_model, load_rfm, train


class KMeansProjectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = train(save=False)
        cls.bundle = load_model()

    def test_rfm_table(self):
        rfm = load_rfm()
        self.assertEqual(len(rfm), 793)
        self.assertTrue((rfm["recency_days"] >= 1).all())
        self.assertTrue((rfm["frequency"] >= 1).all())

    def test_inertia_decreases_as_k_grows(self):
        inertia = self.results["k_selection"]["inertia"]
        self.assertTrue(all(a > b for a, b in zip(inertia, inertia[1:])))

    def test_segments_cover_every_customer_and_differ(self):
        seg = pd.DataFrame(self.results["segments"])
        self.assertEqual(seg["customers"].sum(), 793)
        self.assertEqual(seg["segment"].nunique(), self.results["chosen_k"])
        self.assertGreater(seg["monetary"].max(), 2 * seg["monetary"].min())

    def test_champions_are_the_most_recent_buyers(self):
        seg = pd.DataFrame(self.results["segments"]).set_index("segment")
        self.assertEqual(seg["recency_days"].idxmin(), "Champions")

    def test_assign_obvious_customers(self):
        best, d = assign_segment(self.bundle, recency_days=10, frequency=12, monetary=8000)
        worst, _ = assign_segment(self.bundle, recency_days=900, frequency=1, monetary=50)
        self.assertEqual(best, "Champions")
        self.assertEqual(worst, "Low-value / lapsing")
        self.assertEqual(min(d, key=d.get), best)  # nearest centre is the assigned one


if __name__ == "__main__":
    unittest.main()
