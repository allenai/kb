
import unittest

import torch

from kb.common import F1Metric


class TestF1Metric(unittest.TestCase):
    def test_f1(self):

        f1 = F1Metric()

        predicted = [
            ['a', 'b', 'c'],
            ['d'],
            []
        ]
        gold = [
            ['b', 'd'],
            ['d', 'e'],
            ['f']
        ]

        f1(predicted, gold)
        predicted2 = [[6, 10]]
        gold2 = [[6, 15]]
        f1(predicted2, gold2)

        metrics = f1.get_metric()

        precision = 3 / 6
        recall = 3 / 7
        f1 = 2 * precision * recall / (precision + recall)

        expected_metrics = [precision, recall, f1]

        for m1, m2 in zip(metrics, expected_metrics):
            self.assertAlmostEqual(m1, m2)


if __name__ == '__main__':
    unittest.main()

