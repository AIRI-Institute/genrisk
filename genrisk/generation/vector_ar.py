import pandas as pd
from statsmodels.tsa.api import VARMAX, SARIMAX
from genrisk.generation.base import BaseGenerator        

class VectorAR(BaseGenerator):
    """VectorAR generator."""
    def __init__(
            self, 
            target_columns: list[str], 
            conditional_columns: list[str], 
            k_ar: int=3, 
            maxiter: int=500
        ):
        """
        Args:
            target_columns (list[str]): A list of columns for generation.
            target_columns (list[str]): A list of columns for conditioning.
            k_ar (int): A number of lags to train the model.
            maxiter (int): A maxumum number of steps to train the model.
        """
        super().__init__(target_columns, conditional_columns)
        self.k_ar = k_ar
        self.maxiter = maxiter
        
    def fit(self, data: pd.DataFrame):
        """Fit the generator
        
        Args:
            data (pd.DataFrame): A dataframe with time series data.
        """
        if len(self.target_columns) > 1:
            model = VARMAX(
                endog=data[self.target_columns], 
                exog=data[self.conditional_columns] if self.conditional_columns else None, 
                order=(self.k_ar, 0),
                trend='c',
            )
        else:
            model = SARIMAX(
                endog=data[self.target_columns], 
                exog=data[self.conditional_columns] if self.conditional_columns else None, 
                order=(self.k_ar, 0, 0),
                trend='c',
            )
        self._results = model.fit(disp=False, maxiter=self.maxiter)
    
    def sample(self, data: pd.DataFrame, n_samples: int) -> list[pd.DataFrame]:
        """Sample time series data.

        Args:
            data (pd.DataFrame): A dataframe with columns for conditioning.
            n_samples (int): A number of samples to sample.
        
        Returns:
            list[pd.DataFrame]: A list with samples dataframes.
        """
        _fake = self._results.simulate(
            nsimulations=data.shape[0], 
            repetitions=n_samples,
        )
        _fake.index = data.index
        target_fakes = []
        for i in range(n_samples):
            loc_level1, _ = _fake.columns.get_loc_level(i, level=1)
            target_fake = _fake.iloc[:, loc_level1]
            target_fake.columns = self.target_columns
            target_fakes.append(target_fake)
        return super().postprocess_fake(data, target_fakes)
