from darts.models import TFTModel
import logging
import matplotlib.pyplot as plt

from .base import BaseModelHandler

plt.style.use('default')
plt.rcParams['axes.grid'] = False


class TFTModelHandler(BaseModelHandler):
    def __init__(
            self,
            train, test, forecasting_horizont, target_columns, conditional_columns,
            input_chunk_length=100,
            forcasting_horizon=24,
            n_epochs=20,
            dropout=0.0,
    ):
        super().__init__(train, test, forecasting_horizont, target_columns, conditional_columns)
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
            save_checkpoints=False,
            pl_trainer_kwargs=self.pl_trainer_kwargs,
            random_state=42,
            batch_size=8,
        )

    def train_model(self):
        logging.info(f"Training on {self.accelerator_type.upper()}.")
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

    def forecast(self, all_test=True):
        if not self.trained:
            self.load_model()
        pred_len = len(self.test_cov) if all_test else self.forecasting_horizont
        future_pred = self.model.predict(
            pred_len,
            series=self.train_ts,
            future_covariates=self.train_cov[-self.input_chunk_length:].append(self.test_cov),
            verbose=False,
        )
        return self.scaler.inverse_transform(future_pred).pd_dataframe()

    def save_model(self, file_name="tft", work_dir="forcasting_models"):
        return super().save_model(file_name, work_dir)

    def load_model(self, file_name="tft", work_dir="forcasting_models"):
        return super().load_model(file_name, work_dir)
