from darts.models import ARIMA
from pmdarima import auto_arima
import numpy as np
import matplotlib.pyplot as plt

from .base import BaseModelHandler

plt.style.use('default')
plt.rcParams['axes.grid'] = False


class ARIMAModelHandler(BaseModelHandler):
    def __init__(self, train, test, test_percent, forecasting_horizont, target_columns, conditional_columns,
                 params=None):
        super().__init__(train, test, test_percent, forecasting_horizont, target_columns, conditional_columns)
        if params is None:
            params = {
                "p": (0, 2),
                "d": (0, 1),
                "q": (0, 2),
                "P": (0, 2),
                "D": (0, 1),
                "Q": (0, 2),
                "m": 7,  # Seasonal period
                "seasonal": True,
            }

        self.params = params
        self.best_params = None
        self.best_score = float('inf')
        self.model = None
        self.val_size = int(len(self.train_ts) * 0.1)
        self.train_data, self.val_data = self.df_train[:-self.val_size][target_columns], self.df_train[-self.val_size:][
            target_columns]
        self.train_cov_data, self.val_cov_data = self.df_train[:-self.val_size][conditional_columns], \
        self.df_train[-self.val_size:][conditional_columns]

    def grid_search(self):
        self.model = auto_arima(
            self.train_data,
            seasonal=self.params['seasonal'],
            exogenous=self.train_cov_data,
            m=self.params['m'],
            start_p=self.params['p'][0], max_p=self.params['p'][1],
            start_d=self.params['d'][0], max_d=self.params['d'][1],
            start_q=self.params['q'][0], max_q=self.params['q'][1],
            start_P=self.params['P'][0], max_P=self.params['P'][1],
            start_D=self.params['D'][0], max_D=self.params['D'][1],
            start_Q=self.params['Q'][0], max_Q=self.params['Q'][1],
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
        )
        self.best_params = self.model.get_params()
        self.best_score = self.model.aic()
        return self.best_params, self.best_score

    @staticmethod
    def _evaluate(pred, true, k):
        residuals = pred - true
        rss = np.sum(residuals ** 2)
        n = len(true)
        aic = n * np.log(rss / n) + 2 * k
        return aic

    def train_model(self):
        param = None
        if self.model is None:
            param, score = self.grid_search()

        self.train_data, self.val_data = self.train_ts[:-self.val_size], self.train_ts[-self.val_size:]
        self.train_cov_data, self.val_cov_data = self.train_cov[:-self.val_size], self.train_cov[-self.val_size:]

        self.model = ARIMA(
            p=param['order'][0],
            d=param['order'][1],
            q=param['order'][2],
            seasonal_order=(
                param['seasonal_order'][0],
                param['seasonal_order'][1],
                param['seasonal_order'][2],
                param['seasonal_order'][3])
        )
        self.model.fit(self.train_data, future_covariates=self.train_cov_data)
        self.trained = True

    def forecast(self, all_test=True):
        if not self.trained:
            self.load_model()
        pred_len = len(self.test_cov) if all_test else self.forecasting_horizont
        future_pred = self.model.predict(pred_len, series=self.train_data,
                                         future_covariates=self.train_cov.append(self.test_cov), verbose=False)
        return self.scaler.inverse_transform(future_pred).pd_dataframe()

    def save_model(self, file_name="arima", work_dir="forcasting_models"):
        return super().save_model(file_name, work_dir)

    def load_model(self, file_name="arima", work_dir="forcasting_models"):
        self.model = ARIMA()
        return super().load_model(file_name, work_dir)
