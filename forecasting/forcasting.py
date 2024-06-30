import os
from abc import ABC
from abc import abstractmethod

import numpy as np
import torch
from darts.models import ARIMA
from darts.models import AutoARIMA
from darts.models import RNNModel
from darts.models import TFTModel
from pytorch_lightning.callbacks import EarlyStopping
from typing import Tuple
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler


my_stopper = EarlyStopping(
    monitor="val_loss",
    patience=2,
    min_delta=0.0005,
    mode="min",
)


class BaseModelHandler(ABC):
    def __init__(self, dataset, 
                 test_percent, 
                 forecasting_horizont, 
                 target_columns,
                 conditional_columns):
        
        self.dataset = dataset
        self.test_percent = test_percent
        self.forecasting_horizont = forecasting_horizont
        self.target_columns = target_columns
        self.conditional_columns = conditional_columns
        self.scaler = None
        self._get_darts_format()

        self.model = None
        self.trained = False
        self.input_chunk_length = 100

        self.accelerator_type = "gpu" if torch.cuda.is_available() else "cpu"
        device_count = 1

        self.pl_trainer_kwargs = {
            "accelerator": self.accelerator_type,
            "devices": device_count,
            "callbacks": [my_stopper],
        }
    
    def _get_darts_format(self):
        split_idx = int(len(self.dataset) * (1 - self.test_percent))

        df_train = self.dataset[:split_idx]
        df_test = self.dataset[split_idx:]

        self.train_ts, self.train_cov = self.process_dataset(df_train)
        self.test_ts, self.test_cov = self.process_dataset(df_test)
    
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
        ts = TimeSeries.from_dataframe(df[self.target_columns], freq=df.index.freq)

        if self.scaler is None:
            self.scaler = Scaler()
            ts = self.scaler.fit_transform(ts)
        else:
            ts = self.scaler.transform(ts)

        ts_cov = TimeSeries.from_dataframe(df[self.conditional_columns], freq=df.index.freq)

        return ts, ts_cov
    
    def get_true_test(self):
        return self.test_ts.pd_dataframe()

    def ensure_directory(self, path):
        """Ensures that the directory exists."""
        os.makedirs(path, exist_ok=True)

    @abstractmethod
    def train_model(self, train, train_cov, params): #fit
        """Training method to be implemented by each model type handler."""
        pass

    def load_model(self):
        """Load model from file, to be possibly overridden by subclasses with specifics."""
        try:
            model_path = os.path.join(self.work_dir, f"{self.name}.pkl")
            self.model = self.model.load(model_path)
        except FileNotFoundError:
            raise Exception("Model not found. Please train the model first.")

    @abstractmethod
    def predict(self, series, horizon): #forecast
        """Predict method to be implemented by subclasses."""
        return

    def save_model(self, file_name, work_dir):
        """Save model to file, might need to be overridden depending on model specifics."""
        self.name = file_name
        self.work_dir = work_dir
        self.ensure_directory(self.work_dir)

        if self.model:
            model_path = os.path.join(self.work_dir, f"{self.name}.pkl")
            self.model.save(model_path)
        else:
            raise ValueError("No model to save.")

    def backtest(self):
        """
        Conducts a backtest by simulating the model's performance on unseen data using a sliding window approach.

        This method loads the model if not already trained, and uses historical data to predict future values
        step-by-step across the provided dataset. It also handles scaling and inverse scaling if a scaler is provided.

        Parameters:
        Y_train (TimeSeries or similar): The historical target series used for initial model training.
        X_train (TimeStartSeries or similar): The historical explanatory series used for initial model training.
        Y (TimeSeries or similar): The target series for which predictions are to be made.
        X (TimeSeries or similar): The explanatory series corresponding to Y for prediction purposes.
        horizon (int): The number of time steps to predict into the future.
        scaler (Scaler or None): A scaler object used for transforming the data before model training and inversely
                                 transforming predictions. If None, no scaling is performed.

        Returns:
        tuple: A tuple containing two numpy arrays:
               - future_preds: An array of predicted values reshaped into a single array for ease of analysis.
               - future_targets: An array of actual target values reshaped similarly for comparison.

        Raises:
        ValueError: If the model is not trained and cannot be loaded successfully.

        Notes:
        - It is assumed that the inputs are aligned and of the appropriate form for the model.
        - This method can be computationally intensive depending on the size of the input data and the complexity of the model.
        """
        if not self.trained:
            self.load_model()

        input_size = self.input_chunk_length
        horizon = self.forecasting_horizont

        Y = self.train_ts[-input_size:].append(self.test_ts)
        X = self.train_cov[-input_size:].append(self.test_cov)

        future_preds = []
        future_targets = []

        for i in range(0, len(Y) - input_size - horizon + 1, horizon):
            print(i)
            past_target = Y[i : i + input_size]
            future_target_original = Y[i + input_size : i + input_size + horizon]

            future_exog = X[i : i + input_size + horizon]
            future_pred = self.model.predict(
                horizon if horizon <= len(future_exog) else len(future_exog),
                series=past_target,
                future_covariates=future_exog,
                verbose=False,
            )

            if self.scaler is not None:
                future_pred_values = self.scaler.inverse_transform(future_pred).values()
                future_target_values = self.scaler.inverse_transform(
                    future_target_original
                ).values()

            future_pred = future_pred.values()
            future_target_original = future_target_original.values()

            if self.scaler is None:
                future_pred_values = future_pred
                future_target_values = future_target_original

            future_preds.append(future_pred_values)
            future_targets.append(future_target_values)

        return np.array(future_preds).reshape(-1), np.array(future_targets).reshape(-1)


class ARIMAModelHandler(BaseModelHandler):
    def __init__(self, dataset, test_percent, forecasting_horizont, target_columns, conditional_columns, params=None):
        super().__init__(dataset, test_percent, forecasting_horizont, target_columns, conditional_columns)
        if params is None:
            params = {
                "start_p": 0,
                "max_p": 2,
                "max_d": 1,
                "start_q": 0,
                "max_q": 2,
                "start_P": 0,
                "max_P": 2,
                "max_D": 1,
                "start_Q": 0,
                "max_Q": 2,
                "seasonal": True,
                "m": 7,
                "trace": True,
                "error_action": "ignore",  #'warn',
                "n_fits": 500,
                "stepwise": True,
                # 'scoring': 'mse',
            }

        self.params = params
        self.model = AutoARIMA(**params)

    def train_model(self):
        self.model.fit(self.train_ts, future_covariates=self.train_cov)
        self.trained = True

    def predict(self, all_test=True):
        if not self.trained:
            self.load_model()
        pred_len = len(self.test_cov) if all_test else self.forecasting_horizont
        return self.model.predict(pred_len, future_covariates=self.test_cov, verbose=False).pd_dataframe()
    
    def save_model(self, work_dir="forcasting_models", file_name="arima"):
        return super().save_model(file_name, work_dir)


class LSTMModelHandler(BaseModelHandler):
    def __init__(
        self,
        dataset, test_percent, forecasting_horizont, target_columns, conditional_columns,
        input_chunk_length=100,
        n_rnn_layers=1,
        n_epochs=20,
        dropout=0.0,
        
    ):
        super().__init__(dataset, test_percent, forecasting_horizont, target_columns, conditional_columns)
        self.input_chunk_length = input_chunk_length
        self.n_rnn_layers = n_rnn_layers
        self.n_epochs = n_epochs
        self.dropout = dropout

        self.model = RNNModel(
            input_chunk_length=self.input_chunk_length,
            model="LSTM",
            dropout=self.dropout,
            n_rnn_layers=self.n_rnn_layers,
            n_epochs=self.n_epochs,
            # work_dir=self.work_dir,
            save_checkpoints=False,
            random_state=42,
            pl_trainer_kwargs=self.pl_trainer_kwargs,
        )

    def train_model(self):
        print(f"Training on {self.accelerator_type.upper()}.")
        val_size = int(len(self.train_ts) * 0.1)
        train_data, val_data = self.train_ts[:-val_size], self.train_ts[-val_size:]
        train_cov_data, val_cov_data = self.train_cov[:-val_size], self.train_cov[-val_size:]

        self.model.fit(
            train_data,
            future_covariates=train_cov_data,
            val_series=val_data,
            val_future_covariates=val_cov_data,
        )
        self.trained = True

    def predict(self, all_test=True):
        if not self.trained:
            self.load_model()
        pred_len = len(self.test_cov) if all_test else self.forecasting_horizont
        return self.model.predict(
            pred_len,
            series=self.train_ts,
            future_covariates=self.train_cov[-self.input_chunk_length :].append(self.test_cov),
            verbose=False,
        ).pd_dataframe()
    
    def save_model(self, work_dir="forcasting_models", file_name="lstm"):
        return super().save_model(file_name, work_dir)
  

class TFTModelHandler(BaseModelHandler):
    def __init__(
        self,
        dataset, test_percent, forecasting_horizont, target_columns, conditional_columns,
        input_chunk_length=100,
        forcasting_horizon=24,
        n_epochs=20,
        dropout=0.0,
        work_dir="forcasting_models",
        name="tft",
    ):
        super().__init__(dataset, test_percent, forecasting_horizont, target_columns, conditional_columns)
        self.input_chunk_length = input_chunk_length
        self.forcasting_horizon = forcasting_horizon
        self.n_epochs = n_epochs
        self.dropout = dropout

        self.model = TFTModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.forcasting_horizon,
            hidden_size=16,
            lstm_layers=1,
            num_attention_heads=4,
            full_attention=False,
            hidden_continuous_size=8,
            dropout=0.0,
            n_epochs=self.n_epochs,
            # work_dir="tft_new",
            save_checkpoints=False,
            pl_trainer_kwargs=self.pl_trainer_kwargs,
            random_state=42,
            batch_size=8,
        )

    def train_model(self):
        print(f"Training on {self.accelerator_type.upper()}.")
        val_size = int(len(self.train_ts) * 0.1)
        train_data, val_data = self.train_ts[:-val_size], self.train_ts[-val_size:]
        train_cov_data, val_cov_data = self.train_cov[:-val_size], self.train_cov[-val_size:]

        self.model.fit(
            train_data,
            future_covariates=train_cov_data,
            val_series=val_data,
            val_future_covariates=val_cov_data,
        )
        self.trained = True

    def predict(self, all_test=True):
        if not self.trained:
            self.load_model()
        pred_len = len(self.test_cov) if all_test else self.forecasting_horizont
        return self.model.predict(
            pred_len,
            series=self.train_ts,
            future_covariates=self.train_cov[-self.input_chunk_length :].append(self.test_cov),
            verbose=False,
        ).pd_dataframe()
    
    def save_model(self, work_dir="forcasting_models", file_name="tft"):
        return super().save_model(file_name, work_dir)
