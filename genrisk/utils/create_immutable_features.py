from typing import List
from typing import Literal

import numpy as np
import pandas as pd

FreqType = Literal["weekofyear", "hoursofday", "dayofweek"]


class AddImmutable:
    """
    A class to add immutable positional encodings to a pandas DataFrame with a datetime index.

    This class generates sine and cosine positional encodings based on specified frequency types
    (e.g., 'weekofyear', 'hoursofday', 'dayofweek') and appends these encodings as new columns
    to the provided DataFrame.
        df : pd.DataFrame
        The input DataFrame containing a datetime index to which positional encodings will be added.

    freqs : List[FreqType]
        A list of frequency types for which positional encodings will be generated. The supported
        frequency types are:
        - "weekofyear": Generates encodings based on the week of the year (1-53).
        - "hoursofday": Generates encodings based on the hour of the day (0-23).
        - "dayofweek": Generates encodings based on the day of the week (0-6, where Monday is 0).

    """
    def __init__(
            self, df: pd.DataFrame, freqs: List[FreqType]
    ):
        self.df = df
        self.freqs = freqs

    def _positional_encoding(self) -> pd.DataFrame:
        index = self.df.index
        encoding = []
        for freq in self.freqs:
            values = None
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

    def get(self):
        additional_columns = self._positional_encoding()

        df_concat = pd.concat([self.df, additional_columns], axis=1)
        df_concat.index.freq = self.df.index.freq

        return df_concat
