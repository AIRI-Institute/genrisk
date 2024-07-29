import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from quantile_forest import RandomForestQuantileRegressor
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.base')


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
    return ReLU(mu-eta) / (1-alpha) + eta + h * (error-mu) / (1-alpha)


class ConditionalShift:
    """
    Conditional shift model estimates the expected loss of a taraget model
    under the conditional shift in data. The model is based on Algorithm 1 in 
    Subbaswamy, Adarsh, Roy Adams, and Suchi Saria. "Evaluating model robustness 
    and stability to dataset shift." International Conference on Artificial 
    Intelligence and Statistics. PMLR, 2021.
    """
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
        Args:
            mutable_columns (list[str]): List of mutable columns.
            immutable_columns (list[str]): List of immutable columns. It can be empty.
            alpha (float): The risk coefficient is defined by the worst 
                (1 - alpha) fraction of samples.
            cv (int): The number of spits in cross-validation.
            expectation_model (object): The scikit-learn compatible model for 
                estimation the expected error of the model by both mutable and 
                immutable variables. It is GradientBoostingRegressor by default.
            quantile_model (object): The scikit-learn compatible model for 
                estimation the alpha-quantile error of the model by immutable 
                variables. It is GradientBoostingRegressor with the quantile 
                loss by default.
            verbose (bool): Show the progree bar.
        """
        assert len(mutable_columns), "List of mutable variables is empty."
        self.quantile_model = quantile_model
        self.expectation_model = expectation_model
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
        self.verbose = verbose

    def fit(self, X, error):
        """
        Fit the conditional shift model and calculate the worsct-case risk. The 
        result is stored in the following variables:

            - self.mask (numpy.ndarray): Binary mask for the worst-case subset.
            - self.risk (float): The expected error on the worst-case subset.
            - self.ub_risk (float): The upper bound of the error on the worst-case 
                subset with the 95% confidence.
            - self.lb_risk (float): The lower bound of the error on the worst-case 
                subset with the 95% confidence.
        
        Args:
            X (pd.DataFrame): Dataframe with both mutable and immutable columns.
            error (pd.Series): Series with errors of a target model.
        """
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
