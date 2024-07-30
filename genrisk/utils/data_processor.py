from typing import List
from typing import Literal

import numpy as np
import pandas as pd

FreqType = Literal["weekofyear", "hoursofday", "dayofweek"]


class AddImmutable:
    def __init__(
        self, df: pd.DataFrame, freqs: List[FreqType]
    ):

        self.df = df
        self.freqs = freqs

    def positional_encoding(self) -> pd.DataFrame:
        index = self.df.index
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

    def get(self):
        additional_columns = self.positional_encoding()

        df_concat = pd.concat([self.df, additional_columns], axis=1)
        df_concat.index.freq = self.df.index.freq

        return df_concat
    
