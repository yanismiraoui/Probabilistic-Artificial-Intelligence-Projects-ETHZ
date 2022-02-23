import abc
import warnings

import numpy as np
import torch


def ece(predicted_probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 30) -> float:
    """
    Computes the Expected Calibration Error (ECE).
    Many options are possible; in this implementation, we provide a simple version.

    Using a uniform binning scheme on the full range of probabilities, zero
    to one, we bin the probabilities of the predicted label only (ignoring
    all other probabilities). For the ith bin, we compute the avg predicted
    probability, p_i, and the bin's total accuracy, a_i.
    We then compute the ith calibration error of the bin, |p_i - a_i|.
    The final returned value is the weighted average of calibration errors of each bin.

    :param predicted_probabilities: Predicted probabilities, float array of shape (num_samples, num_classes)
    :param labels: True labels, int tensor of shape (num_samples,) with each entry in {0, ..., num_classes - 1}
    :param n_bins: Number of bins for histogram binning
    :return: ECE score as a float
    """
    num_samples, num_classes = predicted_probabilities.shape

    # Predictions are the classes with highest probability
    predictions = np.argmax(predicted_probabilities, axis=1)
    prediction_confidences = predicted_probabilities[range(num_samples), predictions]

    # Use uniform bins on the range of probabilities, i.e. closed interval [0.,1.]
    bin_upper_edges = np.histogram_bin_edges([], bins=n_bins, range=(0., 1.))
    bin_upper_edges = bin_upper_edges[1:]  # bin_upper_edges[0] = 0.

    probs_as_bin_num = np.digitize(prediction_confidences, bin_upper_edges)
    sums_per_bin = np.bincount(probs_as_bin_num, minlength=n_bins, weights=prediction_confidences)
    sums_per_bin = sums_per_bin.astype(np.float32)

    total_per_bin = np.bincount(probs_as_bin_num, minlength=n_bins) \
        + np.finfo(sums_per_bin.dtype).eps  # division by zero
    avg_prob_per_bin = sums_per_bin / total_per_bin

    onehot_labels = np.eye(num_classes)[labels]
    accuracies = onehot_labels[range(num_samples), predictions]  # accuracies[i] is 0 or 1
    accuracies_per_bin = np.bincount(probs_as_bin_num, weights=accuracies, minlength=n_bins) / total_per_bin

    prob_of_being_in_a_bin = total_per_bin / float(num_samples)

    ece_ret = np.abs(accuracies_per_bin - avg_prob_per_bin) * prob_of_being_in_a_bin
    ece_ret = np.sum(ece_ret)
    return float(ece_ret)


class ParameterDistribution(torch.nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract class that models a distribution over model parameters,
    usable for Bayes by backprop.
    You can implement this class using any distribution you want
    and try out different priors and variational posteriors.
    All torch.nn.Parameter that you add in the __init__ method of this class
    will automatically be registered and know to PyTorch.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        """
        Calculate the log-likelihood of the given values
        :param values: Values to calculate the log-likelihood on
        :return: Log-likelihood
        """
        pass

    @abc.abstractmethod
    def sample(self) -> torch.Tensor:
        """
        Sample from this distribution.
        Note that you only need to implement this method for variational posteriors, not priors.

        :return: Sample from this distribution. The sample shape depends on your semantics.
        """
        pass

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # DO NOT USE THIS METHOD
        # We only implement it since torch.nn.Module requires a forward method
        warnings.warn('ParameterDistribution should not be called! Use its explicit methods!')
        return self.log_likelihood(values)
