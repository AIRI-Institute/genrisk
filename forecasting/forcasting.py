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


my_stopper = EarlyStopping(
    monitor="val_loss",
    patience=2,
    min_delta=0.0005,
    mode="min",
)


class BaseModelHandler(ABC):
    def __init__(self, file_name, work_dir):
        self.name = file_name
        self.work_dir = work_dir
        self.ensure_directory(self.work_dir)
        self.model = None
        self.trained = False

        self.accelerator_type = "gpu" if torch.cuda.is_available() else "cpu"
        device_count = 1

        self.pl_trainer_kwargs = {
            "accelerator": self.accelerator_type,
            "devices": device_count,
            "callbacks": [my_stopper],
        }

    def ensure_directory(self, path):
        """Ensures that the directory exists."""
        os.makedirs(path, exist_ok=True)

    @abstractmethod
    def train_model(self, train, train_cov, params):
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
    def predict(self, series, horizon):
        """Predict method to be implemented by subclasses."""
        return

    def save_model(self):
        """Save model to file, might need to be overridden depending on model specifics."""
        if self.model:
            model_path = os.path.join(self.work_dir, f"{self.name}.pkl")
            self.model.save(model_path)
        else:
            raise ValueError("No model to save.")

    def backtest(self, Y_train, X_train, Y, X, horizon, scaler):
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
        Y = Y_train[-input_size:].append(Y)
        X = X_train[-input_size:].append(X)

        future_preds = []
        future_targets = []

        for i in range(0, len(Y) - input_size - horizon + 1, horizon):
            past_target = Y[i : i + input_size]
            future_target_original = Y[i + input_size : i + input_size + horizon]

            future_exog = X[i : i + input_size + horizon]
            future_pred = self.model.predict(
                horizon,
                series=past_target,
                future_covariates=future_exog,
                verbose=False,
            )

            if scaler is not None:
                future_pred_values = scaler.inverse_transform(future_pred).values()
                future_target_values = scaler.inverse_transform(
                    future_target_original
                ).values()

            future_pred = future_pred.values()
            future_target_original = future_target_original.values()

            if scaler is None:
                future_pred_values = future_pred
                future_target_values = future_target_original

            future_preds.append(future_pred_values)
            future_targets.append(future_target_values)

        return np.array(future_preds).reshape(-1), np.array(future_targets).reshape(-1)


class ARIMAModelHandler(BaseModelHandler):
    def __init__(self, params=None, work_dir="forcasting_models", name="arima"):
        super().__init__(name, work_dir)
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

    def train_model(self, train, train_cov):
        self.model.fit(train, future_covariates=train_cov)
        self.save_model()
        self.trained = True

    def predict(self, test_cov, horizon):
        if not self.trained:
            self.load_model()
        return self.model.predict(horizon, future_covariates=test_cov, verbose=False)


class LSTMModelHandler(BaseModelHandler):
    def __init__(
        self,
        input_chunk_length=100,
        n_rnn_layers=1,
        n_epochs=20,
        dropout=0.0,
        work_dir="forcasting_models",
        name="lstm",
    ):
        super().__init__(name, work_dir)
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
            work_dir=self.work_dir,
            save_checkpoints=False,
            random_state=42,
            pl_trainer_kwargs=self.pl_trainer_kwargs,
        )

    def train_model(self, train, train_cov):
        print(f"Training on {self.accelerator_type.upper()}.")
        val_size = int(len(train) * 0.1)
        train_data, val_data = train[:-val_size], train[-val_size:]
        train_cov_data, val_cov_data = train_cov[:-val_size], train_cov[-val_size:]

        self.model.fit(
            train_data,
            future_covariates=train_cov_data,
            val_series=val_data,
            val_future_covariates=val_cov_data,
        )
        self.save_model()
        self.trained = True

    def predict(self, train, train_cov, test_cov, horizon):
        if not self.trained:
            self.load_model()
        return self.model.predict(
            horizon,
            series=train,
            future_covariates=train_cov[-self.input_chunk_length :].append(test_cov),
            verbose=False,
        )


class TFTModelHandler(BaseModelHandler):
    def __init__(
        self,
        input_chunk_length=100,
        forcasting_horizon=24,
        n_epochs=20,
        dropout=0.0,
        work_dir="forcasting_models",
        name="tft",
    ):
        super().__init__(name, work_dir)
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
            work_dir="tft_new",
            save_checkpoints=False,
            pl_trainer_kwargs=self.pl_trainer_kwargs,
            random_state=42,
            batch_size=8,
        )

    def train_model(self, train, train_cov):
        print(f"Training on {self.accelerator_type.upper()}.")
        val_size = int(len(train) * 0.1)
        train_data, val_data = train[:-val_size], train[-val_size:]
        train_cov_data, val_cov_data = train_cov[:-val_size], train_cov[-val_size:]

        self.model.fit(
            train_data,
            future_covariates=train_cov_data,
            val_series=val_data,
            val_future_covariates=val_cov_data,
        )
        self.save_model()
        self.trained = True

    def predict(self, train, train_cov, test_cov, horizon):
        if not self.trained:
            self.load_model()
        return self.model.predict(
            horizon,
            series=train,
            future_covariates=train_cov[-self.input_chunk_length :].append(test_cov),
            verbose=False,
        )
