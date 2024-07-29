import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_fake_data(result_df, gen, col, n_samples=3):
    """
    Plots the original training data along with a number of sampled data sets.

    Parameters:
        result_df (DataFrame): The original DataFrame containing the training data.
        gen (Generator): The generator used to sample fake data.
        col (str): The column name to plot from the DataFrame.
        n_samples (int): The number of fake data samples to generate and plot.
    """
    plt.figure(figsize=(35, 6))
    plt.plot(result_df[col].values, label="train data", color="black")

    fake_data = gen.sample(result_df, n_samples=n_samples)
    for fake in fake_data:
        plt.plot(fake[col].values, alpha=0.5, label=f"fake data sample")

    plt.title("Comparison of Train Data and Sampled Fake Data")
    plt.legend()
    plt.show()


def plot_pair_wcr(
    ans_values,
    shift_model,
    test_ts,
    result_df,
    fake_data,
    col,
    error_func=lambda x, y: (x - y) ** 2,
):
    """
        Plot the weighted conditional risks (WCR) for test and fake data using a shift model.

        Parameters:
        ans (dict): Dictionary containing answers or true values.
        shift_model (model): Model that can fit data and calculate risk based on specified errors.
        test_ts (dict): Dictionary containing test time series data.
        result_df (DataFrame): DataFrame containing the results data.
        fake_data (list): List of DataFrames representing fake data.
    s    col (str): Column name in the fake data to compare against the ans values.
        error_func (function): Function to calculate errors between predictions and true values. Default is squared error.

        Returns:
        None: This function plots the WCR and does not return any value.
    """
    test_ts_values = np.array(test_ts.values)
    error0 = error_func(ans_values, test_ts_values).reshape(-1)

    risk0 = shift_model.fit(result_df, error0)

    errors = []
    for fake in fake_data:
        fake_errors = error_func(
            ans_values.reshape(-1), np.array(fake[col].values).reshape(-1)
        ).tolist()
        errors.extend(fake_errors)
    errors.extend(error0)

    fake_df = pd.concat(fake_data + [result_df], ignore_index=True)
    risk_fake = shift_model.fit(fake_df, np.array(errors))

    alpha_space = shift_model.alpha_space
    plt.plot(alpha_space, np.array(risk0)[:, 0], label="Test")
    plt.fill_between(
        alpha_space, np.array(risk0)[:, 1], np.array(risk0)[:, 2], alpha=0.1
    )
    plt.plot(alpha_space, np.array(risk_fake)[:, 0], label="Fake")
    plt.fill_between(
        alpha_space, np.array(risk_fake)[:, 1], np.array(risk_fake)[:, 2], alpha=0.1
    )

    plt.legend()
    plt.show()


def plot_test_wcr(
    ans_values, shift_model, test_ts, result_df, error_func=lambda x, y: (x - y) ** 2
):
    """
    Plot the weighted conditional risks (WCR) for test data using a shift model.

    Parameters:
    ans: np_array.
    shift_model (model): Model that can fit data and calculate risk based on specified errors.
    test_ts (dict): Dictionary containing test time series data.
    result_df (DataFrame): DataFrame containing the results data.
    error_func (function): Function to calculate errors between predictions and true values. Default is squared error.

    Returns:
    None: This function plots the WCR for test data and does not return any value.
    """
    test_ts_values = np.array(test_ts)
    error0 = error_func(ans_values, test_ts_values).reshape(-1)

    risk0 = shift_model.fit(result_df, error0)

    alpha_space = shift_model.alpha_space

    plt.plot(alpha_space, np.array(risk0)[:, 0], label="Test")
    plt.fill_between(
        alpha_space, np.array(risk0)[:, 1], np.array(risk0)[:, 2], alpha=0.1
    )
    plt.legend()
    plt.show()


def plot_fake_wcr(
    ans_values, shift_model, fake_data, col, error_func=lambda x, y: (x - y) ** 2
):
    """
    Plot the weighted conditional risks (WCR) for fake data using a shift model.

    Parameters:
    ans (dict): Dictionary containing answers or true values.
    shift_model (model): Model that can fit data and calculate risk based on specified errors.
    fake_data (list): List of DataDataframes representing fake data.
    col (str): Column name in the fake data to compare against the ans values.
    error_func (function): Function to calculate errors between predictions and true values. Default is squared error.

    Returns:
    None: This function plots the WCR for fake data and does not return any value.
    """

    errors = []
    for fake in fake_data:
        fake_errors = error_func(
            ans_values.reshape(-1), np.array(fake[col].values).reshape(-1)
        ).tolist()
        errors.extend(fake_errors)

    fake_df = pd.concat(fake_data, ignore_index=True)
    risk_fake = shift_model.fit(fake_df, np.array(errors))

    alpha_space = shift_model.alpha_space

    plt.plot(alpha_space, np.array(risk_fake)[:, 0], label="Fake", color="blue")
    plt.fill_between(
        alpha_space,
        np.array(risk_fake)[:, 1],
        np.array(risk_fake)[:, 2],
        color="blue",
        alpha=0.1,
    )
    plt.legend()
    plt.show()
