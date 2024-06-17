import matplotlib.pyplot as plt
import numpy as np
import torch
from darts import TimeSeries


def plot_predictions(future_pred, future_target):
    """
    Plots the predicted and actual values on the same graph to visually compare their performance.

    This function takes in predicted and actual values, potentially as dictionaries or other iterable structures,
    converts them to lists if necessary, and plots them against their index. The plot includes labels and a legend
    to distinguish between real and predicted values.

    Parameters:
    future_pred (iterable): Predicted values. Can be a list, array, or a dictionary-like object that can be
                            converted to a list using `.values()`.
    future_target (iterable): Actual values corresponding to `future_pred`. The format can be similar to `future_pred`.

    Returns:
    None: This function does not return any values but displays a matplotlib plot showing the comparison between
          the predicted values and the actual values.

    Example usage:
        plot_predictions(model.predict(future_dates), actual_data)

    Notes:
    - The function assumes that both inputs are aligned and have the same length.
    - This visualization is useful for assessing the accuracy of predictions from forecasting models or simulations.
    """
    if not isinstance(future_pred, list):
        future_pred = list(future_pred.values())
    if not isinstance(future_pred, list):
        future_target = list(future_target.values())
    plt.figure(figsize=(10, 5))
    plt.plot(future_target, label="real_values")
    plt.plot(future_pred, label="predicted_values", linestyle="--")

    plt.legend()
    plt.title("Comparison of test and predict")
    plt.xlabel("Index")
    plt.ylabel("Values")

    plt.show()
