import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.metrics import mae
from darts.metrics import mape
from darts.metrics import mse
from darts.metrics import r2_score
from darts.metrics import rmse
from darts.models import ARIMA
from darts.models import AutoARIMA
from darts.models import VARIMA
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from tqdm import tqdm
from utils.get_darts_data import get_data
from utils.get_darts_data import get_len_dim
from utils.get_data_article import get_data_article



def backtest(Y, X, model, input_size, horizon, scaler, dim, size_dim):
    abs_error = []
    abs_error_scale = []

    for i in range(len(Y) - input_size - horizon + 1):
        past_target = Y[i : i + input_size]
        future_target_original = Y[i + input_size : i + input_size + horizon]

        future_pred = model.predict(horizon, series=past_target, verbose=False)

        if scaler is not None:
            prepared_pred_values = np.zeros((len(future_pred), size_dim))
            prepared_pred_values[:, dim] = future_pred.values().reshape(-1)

            prepared_target_values = np.zeros((len(future_target_original), size_dim))
            prepared_target_values[:, dim] = future_target_original.values().reshape(-1)

            time_indices = future_pred.time_index
            prepared_pred_values = TimeSeries.from_times_and_values(
                time_indices, prepared_pred_values
            )

            time_indices = future_target_original.time_index
            prepared_target_values = TimeSeries.from_times_and_values(
                time_indices, prepared_target_values
            )

            future_pred_with_scaler_values = (
                scaler.inverse_transform(prepared_pred_values)
                .pd_dataframe()
                .iloc[:, dim]
            )
            future_target_with_scaler_values = (
                scaler.inverse_transform(prepared_target_values)
                .pd_dataframe()
                .iloc[:, dim]
            )

        future_pred = future_pred.values().reshape(-1)
        future_target_original = future_target_original.values().reshape(-1)

        if scaler is None:
            future_pred_with_scaler_values = future_pred
            future_target_with_scaler_values = future_target_original

        abs_error.append(
            torch.abs(
                torch.tensor(future_pred).view(-1)
                - torch.tensor(future_target_original).view(-1)
            )
        )
        abs_error_scale.append(
            torch.abs(
                torch.tensor(future_pred_with_scaler_values).view(-1)
                - torch.tensor(future_target_with_scaler_values).view(-1)
            )
        )

    return torch.stack(abs_error_scale), torch.stack(abs_error)



def train(name):
    (
        train,
        train_cov,
        test,
        test_cov,
        forcasting_horizon,
        scaler,
    ) = get_data_article()

    model = ARIMA(
        p=params[name]["p"],
        d=params[name]["d"],
        q=params[name]["q"],
        trend="n",
        seasonal_order=params[name]["seasonal_order"],
    )
    model.fit(train)

    model.save(f"arima/{name}/{name}_{dim}.pkl")

    def train(name, n_rnn_layers=1, n_epochs=20, training=False):
    input_chunk_length = input_size = 100 if name != "wiki" else 60
    if name != "article":
        train, train_cov, test, test_cov, forcasting_horizon, scaler = get_data(name)
    else:
        (
            train,
            train_cov,
            test,
            test_cov,
            forcasting_horizon,
            scaler,
        ) = get_data_article()

    if training:

        print("training...")
        model = RNNModel(
            input_chunk_length=input_chunk_length,
            model="LSTM",
            dropout=0.0,
            n_rnn_layers=n_rnn_layers,
            n_epochs=n_epochs,
            work_dir="lstm_new",
            save_checkpoints=False,
            pl_trainer_kwargs=pl_trainer_kwargs,
            random_state=42,
        )
        val_size = int(len(train) * 0.2)
        train_data, val_data = train[:-val_size], train[-val_size:]
        train_cov_data, val_cov_data = train_cov[:-val_size], train_cov[-val_size:]
        model.fit(
            train_data,
            future_covariates=train_cov_data,
            val_series=val_data,
            val_future_covariates=val_cov_data,
        )

        optimal_epochs = my_stopper.stopped_epoch + 1

        model = RNNModel(
            input_chunk_length=input_chunk_length,
            model="LSTM",
            dropout=0.0,
            n_rnn_layers=n_rnn_layers,
            n_epochs=optimal_epochs,
            work_dir="lstm_new",
            save_checkpoints=False,
            # pl_trainer_kwargs=pl_trainer_kwargs,
            random_state=42,
        )
        model.fit(train, future_covariates=train_cov)

        model.save(f"lstm_new/{name}.pkl")
 


def get_metric(name):
    dim_len = get_len_dim(name)
    input_size = 100 if name != "wiki" else 60

    if name != "article":
        data = get_data(name)
    else:
        (
            train,
            train_cov,
            test,
            test_cov,
            forecasting_horizon,
            scaler,
        ) = get_data_article()

    all_mae = 0
    all_mae_scaled = 0
    for dim in range(1):  # tqdm(range(dim_len)):
        train, train_cov, test, test_cov, forecasting_horizon, scaler = get_data(
            name, data, dim
        )

        print(scaler)

        try:
            model = ARIMA.load(f"arima/{name}/{name}_{dim}.pkl")
        except:
            retrain_model(name, dim)

        t_mae, t_mae_scaled = backtest(
            test, test_cov, model, input_size, forecasting_horizon, scaler, dim, dim_len
        )
        t_mae, t_mae_scaled = torch.mean(t_mae).item(), torch.mean(t_mae_scaled).item()
        all_mae += t_mae
        all_mae_scaled += t_mae_scaled

    print(all_mae, all_mae_scaled)
    test_metrics = {"mae": all_mae, "t_mae_scaled": all_mae_scaled}

    with open(f"metrics/arima/{name}_test_metrics.json", "w") as f:
        json.dump(test_metrics, f)


def get_pred(name, dim):
    dim_len = get_len_dim(name)
    input_size = 100 if name != "wiki" else 60

    if name != "article":
        train, train_cov, test, test_cov, forecasting_horizon, scaler = get_data(
            name, None, dim
        )
    else:
        (
            train,
            train_cov,
            test,
            test_cov,
            forecasting_horizon,
            scaler,
        ) = get_data_article()

    model = ARIMA.load(f"arima/{name}/{name}_{dim}.pkl")

    future_pred, future_target = backtest_plot(
        test[: input_size + forecasting_horizon],
        test_cov[: input_size + forecasting_horizon],
        model,
        input_size,
        forecasting_horizon,
        scaler,
        dim,
        dim_len,
    )

    plt.figure(figsize=(25, 4))
    plt.plot(future_target, label="Real values")
    plt.plot(future_pred, label="Predicted values")

    plt.legend()
    plt.title(name + f" dim = {dim}")
    plt.savefig(f"metrics/arima/{name}_png/{name}_{dim}.png")
    plt.show()


def backtest_plot(Y, X, model, input_size, horizon, scaler, dim, size_dim):
    past_target = Y[:input_size]
    future_target_original = Y[input_size : input_size + horizon]

    future_pred = model.predict(horizon, series=past_target, verbose=False)

    if scaler is not None:
        prepared_pred_values = np.zeros((len(future_pred), size_dim))
        prepared_pred_values[:, dim] = future_pred.values().reshape(-1)

        prepared_target_values = np.zeros((len(future_target_original), size_dim))
        prepared_target_values[:, dim] = future_target_original.values().reshape(-1)

        # Создаем TimeSeries объекты, если необходимо работать с ними дальше
        time_indices = future_pred.time_index
        prepared_pred_values = TimeSeries.from_times_and_values(
            time_indices, prepared_pred_values
        )

        time_indices = future_target_original.time_index
        prepared_target_values = TimeSeries.from_times_and_values(
            time_indices, prepared_target_values
        )

        future_pred_ans = (
            scaler.inverse_transform(prepared_pred_values).pd_dataframe().iloc[:, dim]
        )
        future_target_ans = (
            scaler.inverse_transform(prepared_target_values).pd_dataframe().iloc[:, dim]
        )

    if scaler is None:
        future_pred_ans = future_pred.values().reshape(-1)
        future_target_ans = future_target_original.values().reshape(-1)

    return np.array(future_pred_ans).reshape(-1), np.array(future_target_ans).reshape(
        -1
    )
