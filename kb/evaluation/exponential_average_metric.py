from overrides import overrides

from allennlp.training.metrics.metric import Metric


@Metric.register("ema")
class ExponentialMovingAverage(Metric):
    """
    Keep an exponentially weighted moving average.
    alpha is the decay constant. Alpha = 1 means just keep the most recent value.
    alpha = 0.5 will have almost no contribution from 10 time steps ago.
    """
    def __init__(self, alpha:float = 0.5) -> None:
        self.alpha = alpha
        self.reset()

    @overrides
    def __call__(self, value):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """
        if self._ema is None:
            # first observation
            self._ema = value
        else:
            self._ema = self.alpha * value + (1.0 - self.alpha) * self._ema

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The average of all values that were passed to ``__call__``.
        """
        if self._ema is None:
            ret = 0.0
        else:
            ret = self._ema

        if reset:
            self.reset()

        return ret

    @overrides
    def reset(self):
        self._ema = None
