import unittest

import torch

from kb.metrics import MeanReciprocalRank, MicroF1


class TestMeanReciprocalRank(unittest.TestCase):
    def test_mrr(self):
        labels = torch.tensor([[1,2,3], [1,2,0]])
        predictions = torch.tensor([[
                [-4.00,  1.00,  0.00, -1.00],
                [-4.00,  1.00,  0.00, -1.00],
                [-4.00,  0.50,  0.00,  1.00]
            ], [
                [-4.00,  1.00,  0.00, -1.00],
                [-4.00,  1.00,  0.00, -1.00],
                [ 0.00,  0.00,  0.00,  0.00]
            ]])
        mask = torch.tensor([[1,1,1],[1,1,0]], dtype=torch.uint8)

        metric = MeanReciprocalRank()
        metric(predictions, labels, mask)

        expected = 0.8
        output = metric.get_metric(reset=True)
        self.assertAlmostEqual(expected, output)

        expected = 0.0
        output = metric.get_metric()
        self.assertAlmostEqual(expected, output)


class TestMicroF1(unittest.TestCase):
    def test_micro_f1(self):
        labels = torch.tensor([0, 1, 1, 1, 0, 0], dtype=torch.int32)
        predictions = torch.tensor([1, 1, 0, 1, 1, 0], dtype=torch.int32)

        metric = MicroF1(negative_label=0)
        metric(predictions, labels)

        precision, recall, f1 = metric.get_metric(reset=True)
        self.assertAlmostEqual(precision, 1/2)
        self.assertAlmostEqual(recall, 2/3)
        self.assertAlmostEqual(f1, 2 * 1/2 * 2/3 / (1/2 + 2/3))


        precision, recall, f1 = metric.get_metric(reset=True)
        self.assertAlmostEqual(precision, 0)
        self.assertAlmostEqual(recall, 0)
        self.assertAlmostEqual(f1, 0)
