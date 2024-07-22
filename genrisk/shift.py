import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.metrics import mae
from darts.metrics import mape
from darts.metrics import mse
from darts.metrics import r2_score
from darts.metrics import rmse
from quantile_forest import RandomForestQuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from tqdm.auto import tqdm


warnings.simplefilter("ignore", UserWarning)


class EmpiricalQuantile:
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y):
        self.quantile = np.quantile(y, q=self.alpha)

    def predict(self, X):
        return self.quantile


def ReLU(x):
    return x * (x > 0)


def worst_case_risk(mu, eta, alpha, h, error):
    # Eq. 5 in the paper.
    result = ReLU(mu - eta) / (1 - alpha) + eta + h * (error - mu) / (1 - alpha)
    return result[result > 0]


class ConditionalShift:
    def __init__(
        self,
        mutable_columns,
        immutable_columns,
        alpha_space,
        cv=10,
        expectation_model: object = None,
        quantile_model: object = None,
        verbose=False,
        mode="rf_quantiles",
    ):
        """
        Initialize the ConditionalShift model with specified configurations,
        handling mutable and immutable columns and determining risk assessments
        via expectation and quantile models based on a mode setting.

        Args:
            mutable_columns (list[str]): List of columns that can be modified or
                adjusted during model training.
            immutable_columns (list[str]): List of columns that are considered
                fixed and cannot be modified. This list can be empty.
            alpha_space (list[float]): Sequence of risk levels represented as
                fractions, defining the worst (1 - alpha) fraction of samples.
            cv (int): Number of splits in cross-validation for model validation.
            expectation_model (object): Scikit-learn compatible regression model
                used for estimating the expected error of the model based on both
                mutable and immutable variables. Defaults to GradientBoostingRegressor.
            quantile_model (object): Scikit-learn compatible regression model used
                for estimating the alpha-quantile error of the model based solely on
                immutable variables. The default is contextually set based on mode.
            verbose (bool): If True, display a progress bar during model fitting.
            mode (str): Mode of operation that affects the default setting of the
                quantile model. 'default' uses RandomForestQuantileRegressor, and
                'gbr_quantiles' uses GradientBoostingRegressor configured for quantile
                loss at each specified alpha in alpha_space.

        Raises:
            AssertionError: If mutable_columns is empty.

        Note:
            The `quantile_model` is dynamically set based on the presence of
            immutable columns and the specified mode. If immutable columns are empty,
            it defaults to EmpiricalQuantile models for each alpha.
        """
        assert mutable_columns, "List of mutable variables is empty."
        self.alpha_space = list(alpha_space)
        self.mutable_columns = mutable_columns
        self.immutable_columns = immutable_columns
        self.cv = cv
        self.verbose = verbose
        self.mode = mode
        self.expectation_model = expectation_model or GradientBoostingRegressor()
        if quantile_model is None:
            if len(immutable_columns):
                if mode == "rf_quantiles":
                    self.quantile_model = RandomForestQuantileRegressor()
                elif mode == "gbr_quantiles":
                    self.quantile_models = [
                        GradientBoostingRegressor(loss="quantile", alpha=alpha)
                        for alpha in self.alpha_space
                    ]
            else:
                self.quantile_model = [
                    EmpiricalQuantile(alpha=alpha) for alpha in self.alpha_space
                ]

    def fit(self, X, error):
        X_mi = X[self.mutable_columns + self.immutable_columns]
        X_i = X[self.immutable_columns] if self.immutable_columns else X_mi

        self.results = []

        self.mask = np.zeros((error.shape[0], len(self.alpha_space)), dtype=bool)
        self.eta = np.zeros((error.shape[0], len(self.alpha_space)))
        self.mu = np.zeros_like(error)

        for train, test in tqdm(
            KFold(n_splits=self.cv).split(X_mi), total=self.cv, disable=not self.verbose
        ):
            self.expectation_model.fit(X_mi.iloc[train], error[train])
            mu_train = self.expectation_model.predict(X_mi.iloc[train])
            self.mu[test] = self.expectation_model.predict(X_mi.iloc[test])

            if self.mode == "rf_quantiles":
                self._fit_default(X_i, mu_train, train, test)
            elif self.mode == "gbr_quantiles":
                self._fit_gbr_quantiles(X_i, mu_train, train, test)

        for i in range(len(self.alpha_space)):
            risk = worst_case_risk(
                self.mu, self.eta[:, i], self.alpha_space[i], self.mask[:, i], error
            )
            self.results.append(self._calc_risk(risk, error))

        return self.results

    def _fit_default(self, X_i, mu_train, train, test):
        self.quantile_model.fit(X_i.iloc[train], mu_train)
        self.eta[test] = self.quantile_model.predict(
            X_i.iloc[test], quantiles=self.alpha_space
        )

        for i in range(len(self.alpha_space)):
            self.mask[test, i] = self.mu[test] > self.eta[test, i]

    def _fit_gbr_quantiles(self, X_i, mu_train, train, test):
        for i in range(len(self.quantile_models)):
            quantile_model = self.quantile_models[i]
            quantile_model.fit(X_i.iloc[train], mu_train)
            self.eta[test, i] = quantile_model.predict(X_i.iloc[test])
            self.mask[test, i] = self.mu[test] > self.eta[test, i]

    def _calc_risk(self, risk, error):
        f_risk = risk.mean()
        ub_risk = f_risk + 1.96 * risk.std() / np.sqrt(error.shape[0])
        lb_risk = f_risk - 1.96 * risk.std() / np.sqrt(error.shape[0])
        return [f_risk, lb_risk, ub_risk]
