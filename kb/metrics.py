''' Metric class for tracking correlations by saving predictions '''
import numpy as np
from overrides import overrides
from allennlp.training.metrics.metric import Metric
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from scipy.stats import pearsonr, spearmanr
import torch


@Metric.register("fastMatthews")
class FastMatthews(Metric):
    """Fast version of Matthews correlation.
    Computes confusion matrix on each batch, and computes MCC from this when
    get_metric() is called. Should match the numbers from the Correlation()
    class, but will be much faster and use less memory on large datasets.
    """

    def __init__(self, n_classes=2):
        assert n_classes >= 2
        self.n_classes = n_classes
        self.reset()
        self.corr_type = 'matthews'

    def __call__(self, predictions, labels):
        # Convert from Tensor if necessary
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        assert predictions.dtype in [np.int32, np.int64, int]
        assert labels.dtype in [np.int32, np.int64, int]

        C = confusion_matrix(labels.ravel(), predictions.ravel(),
                             labels=np.arange(self.n_classes, dtype=np.int32))
        assert C.shape == (self.n_classes, self.n_classes)
        self._C += C

    def mcc_from_confmat(self, C):
        # Code below from
        # https://github.com/scikit-learn/scikit-learn/blob/ed5e127b/sklearn/metrics/classification.py#L460
        t_sum = C.sum(axis=1, dtype=np.float64)
        p_sum = C.sum(axis=0, dtype=np.float64)
        n_correct = np.trace(C, dtype=np.float64)
        n_samples = p_sum.sum()
        cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
        cov_ypyp = n_samples ** 2 - np.dot(p_sum, p_sum)
        cov_ytyt = n_samples ** 2 - np.dot(t_sum, t_sum)
        mcc = cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)

        if np.isnan(mcc):
            return 0.
        else:
            return mcc

    def get_metric(self, reset=False):
        # Compute Matthews correlation from confusion matrix.
        # see https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
        correlation = self.mcc_from_confmat(self._C)
        if reset:
            self.reset()
        return correlation

    @overrides
    def reset(self):
        self._C = np.zeros((self.n_classes, self.n_classes),
                           dtype=np.int64)


@Metric.register("correlation")
class Correlation(Metric):
    """Aggregate predictions, then calculate specified correlation"""

    def __init__(self, corr_type):
        self._predictions = []
        self._labels = []
        if corr_type == 'pearson':
            corr_fn = pearsonr
        elif corr_type == 'spearman':
            corr_fn = spearmanr
        elif corr_type == 'matthews':
            corr_fn = matthews_corrcoef
        else:
            raise ValueError("Correlation type not supported")
        self._corr_fn = corr_fn
        self.corr_type = corr_type

    def _correlation(self, labels, predictions):
        corr = self._corr_fn(labels, predictions)
        if self.corr_type in ['pearson', 'spearman']:
            corr = corr[0]
        return corr

    def __call__(self, predictions, labels):
        """ Accumulate statistics for a set of predictions and labels.
        Values depend on correlation type; Could be binary or multivalued. This is handled by sklearn.
        Args:
            predictions: Tensor or np.array
            labels: Tensor or np.array of same shape as predictions
        """
        # Convert from Tensor if necessary
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        # Verify shape match
        assert predictions.shape == labels.shape, ("Predictions and labels must"
                                                   " have matching shape. Got:"
                                                   " preds=%s, labels=%s" % (
                                                           str(predictions.shape),
                                                           str(labels.shape)))
        if self.corr_type == 'matthews':
            assert predictions.dtype in [np.int32, np.int64, int]
            assert labels.dtype in [np.int32, np.int64, int]

        predictions = list(predictions.flatten())
        labels = list(labels.flatten())

        self._predictions += predictions
        self._labels += labels

    def get_metric(self, reset=False):
        correlation = self._correlation(self._labels, self._predictions)
        if reset:
            self.reset()
        return correlation

    @overrides
    def reset(self):
        self._predictions = []
        self._labels = []


@Metric.register('mrr')
class MeanReciprocalRank(Metric):
    def __init__(self):
        self._sum = 0.0
        self._n = 0.0

    def __call__(self, predictions, labels, mask):
        # Flatten
        labels = labels.view(-1)
        mask = mask.view(-1).float()
        predictions = predictions.view(labels.shape[0], -1)

        # MRR computation
        label_scores = predictions.gather(-1, labels.unsqueeze(-1))
        rank = predictions.ge(label_scores).sum(1).float()
        reciprocal_rank = 1 / rank
        self._sum += (reciprocal_rank * mask).sum().item()
        self._n += mask.sum().item()

    def get_metric(self, reset=False):
        mrr = self._sum / (self._n + 1e-13)
        if reset:
            self.reset()
        return mrr

    @overrides
    def reset(self):
        self._sum = 0.0
        self._n = 0.0


@Metric.register('microf1')
class MicroF1(Metric):
    def __init__(self, negative_label: int):
        """
        Micro-averaged F1 score

        Parameters
        ==========
        negative_label : ``int``
            Index of negative class label.
        """
        self._negative_label = negative_label
        self._tp = 0
        self._fp = 0
        self._fn = 0

    def __call__(self, predictions, labels, mask=None):
        if mask is None:
            mask = torch.ones_like(labels, dtype=torch.uint8)

        mask = mask.detach().cpu().numpy()

        gold_negative = labels.eq(self._negative_label).detach().cpu().numpy()
        pred_negative = predictions.eq(self._negative_label).detach().cpu().numpy()

        correct = (predictions == labels).detach().cpu().numpy()
        incorrect = (predictions != labels).detach().cpu().numpy()

        tp = correct & mask & ~gold_negative & ~pred_negative
        fp = incorrect & mask & ~pred_negative
        fn = incorrect & mask & ~gold_negative

        self._tp += tp.astype(np.int).sum()
        self._fp += fp.astype(np.int).sum()
        self._fn += fn.astype(np.int).sum()

    def get_metric(self, reset=False):
        precision = self._tp / (self._tp + self._fp + 1e-13)
        recall = self._tp / (self._tp + self._fn + 1e-13)
        f1 = 2 * precision * recall / (precision + recall + 1e-13)
        if reset:
            self.reset()
        return precision, recall, f1

    @overrides
    def reset(self):
        self._tp = 0
        self._fp = 0
        self._fn = 0
