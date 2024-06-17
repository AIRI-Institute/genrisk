from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler


FreqType = Literal["weekofyear", "hoursofday", "dayofweek"]


def positional_encoding(index: pd.DatetimeIndex, freqs: List[FreqType]) -> pd.DataFrame:
    encoding = []
    for freq in freqs:
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


def add_immutable(df: pd.DataFrame, freqs: List[FreqType], freq: str) -> TimeSeries:
    additional_columns = positional_encoding(df.index, freqs)
    return TimeSeries.from_dataframe(
        pd.concat([df, additional_columns], axis=1), freq=freq
    )


def process_dataset(
    df: pd.DataFrame, freq: str, freqs: List[FreqType], scaler: Optional[Scaler] = None
) -> Tuple[TimeSeries, TimeSeries, Scaler]:
    df.index = pd.to_datetime(df.index)
    ts = TimeSeries.from_dataframe(df, freq=freq)

    if scaler is None:
        scaler = Scaler()
        ts = scaler.fit_transform(ts)
    else:
        ts = scaler.transform(ts)

    ts_with_features = add_immutable(df, freqs, freq)

    return ts, ts_with_features, scaler


def prepare_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    freq: str,
    freqs: List[FreqType],
    forecast_horizon: int,
) -> Tuple[TimeSeries, TimeSeries, TimeSeries, TimeSeries, int, Scaler]:
    train_ts, train_cov, scaler = process_dataset(df_train, freq, freqs)
    test_ts, test_cov, _ = process_dataset(df_test, freq, freqs, scaler)

    return train_ts, train_cov, test_ts, test_cov, forecast_horizon, scaler
