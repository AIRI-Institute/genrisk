import numpy as np
import pandas as pd
import pytest

from genrisk.forecasting.arima import ARIMAModelHandler
from genrisk.forecasting.lstm import LSTMModelHandler
from genrisk.forecasting.tft import TFTModelHandler


class TestForecastingModels:
    @pytest.fixture(scope="class")
    def setup_data(self):
        np.random.seed(0)
        df = pd.DataFrame({
            'supply': np.random.randn(365),
            'weekofyear_cos': np.cos(np.linspace(0, 2 * np.pi, 365)),
            'weekofyear_sin': np.sin(np.linspace(0, 2 * np.pi, 365))
        })
        df.index = pd.date_range('2020-01-01', periods=365, freq='D')
        return df

    def _test_model(self, model_handler, setup_data):
        model_handler.train_model()

        # Test prediction on the test set
        test_ts_values = list(model_handler.get_true_test().values)
        ans_values = list(model_handler.predict().values)
        assert len(test_ts_values) == len(ans_values)

        # Test backtesting
        future_pred, future_target = model_handler.backtest()
        assert len(future_pred) == len(future_target)

    def test_arima(self, setup_data):
        arima = ARIMAModelHandler(
            dataset=setup_data,
            test_percent=0.1,
            forecasting_horizont=24,
            target_columns=['supply'],
            conditional_columns=['weekofyear_cos', 'weekofyear_sin'],
        )
        self._test_model(arima, setup_data)

    def test_lstm(self, setup_data):
        lstm = LSTMModelHandler(
            dataset=setup_data,
            test_percent=0.1,
            forecasting_horizont=24,
            target_columns=['supply'],
            conditional_columns=['weekofyear_cos', 'weekofyear_sin'],
        )
        self._test_model(lstm, setup_data)

    def test_tft(self, setup_data):
        tft = TFTModelHandler(
            dataset=setup_data,
            test_percent=0.1,
            forecasting_horizont=24,
            target_columns=['supply'],
            conditional_columns=['weekofyear_cos', 'weekofyear_sin'],
        )
        self._test_model(tft, setup_data)
