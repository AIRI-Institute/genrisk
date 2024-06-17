from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler


FreqType = Literal["weekofyear", "hoursofday", "dayofweek"]


class TimeSeriesProcessor:
    """
    Processes a time series DataFrame to prepare for time series forecasting, including encoding time-based features,
    scaling the data, and splitting into training and testing sets.

    Attributes:
    df (pd.DataFrame): The full dataset as a pandas DataFrame.
    test_size (float): Proportion of the dataset to set aside for testing.
    freq (str): The frequency of the time series data.
    freqs (List[FreqType]): List of temporal frequencies to encode as features.

    Methods:
    positional_encoding: Encodes specified frequencies in the datetime index as sine and cosine features.
    add_immutable: Adds encoded time features to a DataFrame and converts it to a TimeSeries object.
    process_dataset: Processes a DataFrame by encoding features, scaling, and converting to TimeSeries.
    prepare_data: Prepares and splits the data into training and testing sets, suitable for model training.
    """

    def __init__(
        self, df: pd.DataFrame, test_size: float, freq: str, freqs: List[FreqType]
    ):
        """
        Initializes the TimeSeriesProcessor with the dataset and configuration parameters.

        Parameters:
        df (pd.DataFrame): Input dataset as a pandas DataFrame.
        test_size (float): Fraction of the dataset to be used as the test set.
        freq (str): String representing the frequency of the time series data, e.g., 'D' for daily.
        freqs (List[FreqType]): List of string identifiers for the datetime components to be encoded.
        """
        self.df = df
        self.test_size = test_size
        self.freq = freq
        self.freqs = freqs
        self.scaler = None

    def positional_encoding(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Generates sine and cosine transformations of datetime indices for specified frequencies.

        Parameters:
        index (pd.DatetimeIndex): Datetime index from the DataFrame.

        Returns:
        pd.DataFrame: DataFrame containing the cosine and sine encoded features.
        """
        encoding = []
        for freq in self.freqs:
            if freq == "weekofyear":
                values = index.isocalendar().week
            elif freq == "hoursofday":
                values = index.hour
            elif freq == "dayofweek":
                values = index.dayofweek
            num_values = max(values) + 1
            steps = [x * 2.0 * np.pi / num_values for x in values]
            encoding.append(
                pd.DataFrame(
                    {f"{freq}_cos": np.cos(steps), f"{freq}_sin": np.sin(steps)},
                    index=index,
                )
            )

        return pd.concat(encoding, axis=1)

    def add_immutable(self, df: pd.DataFrame) -> TimeSeries:
        """
        Adds encoded time features to the provided DataFrame and returns it as a TimeSeries object.

        Parameters:
        df (pd.DataFrame): DataFrame whose index is used to generate time features.

        Returns:
        TimeSeries: Darts TimeSeries object with additional time features.
        """
        additional_columns = self.positional_encoding(df.index)
        return TimeSeries.from_dataframe(additional_columns, freq=self.freq)

    def process_dataset(
        self, df: pd.DataFrame
    ) -> Tuple[TimeSeries, TimeSeries, Scaler]:
        """
        Processes a DataFrame by encoding temporal features, applying scaling, and converting to TimeSeries.

        Parameters:
        df (pd.DataFrame): DataFrame to process.

        Returns:
        Tuple[TimeSeries, TimeSeries, Scaler]: Tuple containing the original and the feature-augmented TimeSeries and the scaler used.
        """
        df.index = pd.to_datetime(df.index)
        ts = TimeSeries.from_dataframe(df, freq=self.freq)

        if self.scaler is None:
            self.scaler = Scaler()
            ts = self.scaler.fit_transform(ts)
        else:
            ts = self.scaler.transform(ts)

        ts_with_features = self.add_immutable(df)

        return ts, ts_with_features

    def prepare_data(
        self, forecast_horizon: int
    ) -> Tuple[TimeSeries, TimeSeries, TimeSeries, TimeSeries, int, Scaler]:
        """
        Prepares data for training and testing, splits the dataset, and returns the necessary components for forecasting.

        Parameters:
        forecast_horizon (int): The number of periods ahead to forecast.

        Returns:
        Tuple[TimeSeries, TimeSeries, TimeSeries, TimeSeries, int, Scaler]: Tuple containing the training and testing time series,
        both raw and with covariates, the forecast horizon, and the scaler used.
        """
        split_idx = int(len(self.df) * (1 - self.test_size))

        df_train = self.df[:split_idx]
        df_test = self.df[split_idx:]

        train_ts, train_cov = self.process_dataset(df_train)
        test_ts, test_cov = self.process_dataset(df_test)

        return train_ts, train_cov, test_ts, test_cov, forecast_horizon, self.scaler
