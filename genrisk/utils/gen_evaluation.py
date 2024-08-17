import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf


def hist2samp(target0, target1, label0, label1, nbins=20):
    """
    Plot histograms of the first differences of two time series data sets on the same axes.

    Parameters:
    target0 (pd.Series): Time series data for the first group.
    target1 (pd.Series): Time series data for the second group.
    label0 (str): Label for the first histogram, representing the first group.
    label1 (str): Label for the second histogram, representing the second as group.
    nbins (int): Number of bins to use in the histogram. Default is 20.

    Returns:
    None: This function plots histograms and does not return any value.
    """
    bins = np.linspace(-3, 3, nbins)
    width = 6 / nbins / 3
    (target0 - target0.shift(1)).hist(
        bins=bins + width, width=width / 1.2, density=True, label=label0, ax=plt.gca()
    )
    (target1 - target1.shift(1)).hist(
        bins=bins + 3 * width,
        width=width / 1.2,
        density=True,
        label=label1,
        ax=plt.gca(),
    )
    plt.legend()
    plt.grid(False)
    plt.show()


def pacf2samp(target0, target1, label0, label1, nlags=40):
    """
    Plot the partial autocorrelation functions (PACF) for two time series data sets.

    Parameters:
    target0 (pd.Series): Time series data for the first group.
    target1 (pd.Series): Time series data for the second group.
    label0 (str): Label for the first PACF plot.
    label1 (str): Label for the second PACF plot.
    nlags (int): Number of lags to include in the PACF plot. Default is 40.

    Returns:
    None: Plots the PACF for both time series data sets and does not return any value.
    """
    plt.figure(figsize=(6, 7))
    ax = plt.subplot(2, 1, 1)
    plot_pacf(target0, lags=nlags, title=f"PACF {label0}", ax=ax)
    ax = plt.subplot(2, 1, 2)
    plot_pacf(target1, lags=nlags, title=f"PACF {label1}", ax=ax)
    plt.show()


def pacf_error(target0, target1, nlags=10):
    """
    Calculate the mean squared error of the differences between the partial autocorrelation coefficients (PACF)
    of two time series data at specified lags.

    Parameters:
    target0 (pd.Series): Time series data for the first group.
    target1 (pd.Series): Time series data for the second group.
    nlags (int): Number of lags to include in the calculation. Default is 10.

    Returns:
    float: Mean squared error of the PACF differences up to the specified number of lags.
    """
    pacf0 = pacf(target0, nlags=nlags)[1:]
    pacf1 = pacf(target1, nlags=nlags)[1:]
    error = (pacf0 - pacf1) ** 2
    return error.mean()


def ks_test(target0, target1):
    """
    Perform a Kolmogorov-Smirnov test for the null hypothesis that two samples are drawn from the same distribution.

    Parameters:
    target0 (pd.Series): First sample of time series data.
    target1 (pd.Series): Second sample of time series data.

    Returns:
    float: p-value from the KS test indicating the probability of observing the data under the null hypothesis.
    """
    return ks_2samp(target0.to_numpy(), target1.to_numpy()).pvalue


def ks_lags_test(target0, target1):
    """
    Perform a Kolmogorov-Smirnov test on the first differences of two time series to test the null hypothesis that
    both series are drawn from the same distribution in their first differences.

    Parameters:
    target0 (pd.Series): First time series data.
    target1 (pd.Series): Second time series data.

    Returns:
    float: p-value from the KS test on the first differences, indicating the probability of observing the data under the null hypothesis.
    """
    return ks_2samp(
        (target0 - target0.shift(1)).values[1:], (target1 - target1.shift(1)).values[1:]
    ).pvalue
