import numpy as np
import pandas as pd
import pytest
from inspect import getmembers, isclass, isabstract
from genrisk import generation

models = [f[1] for f in getmembers(generation, isclass) if not isabstract(f[1])]

class TestOnSyntheticData:
    def setup_class(self):
        np.random.seed(0)
        X = np.random.randn(1000, 3)
        data = pd.DataFrame(X, columns=['X1', 'X2', 'X3'])
        data['y'] = (2*X[:, 1] > X[:, 0] + 0.5 * np.random.randn(1000)).astype('int')
        data.index = pd.date_range('2020-01-01', periods=1000, freq='D')
        self.data = data

    @pytest.mark.parametrize("model", models)
    def test_base(self, model):
        gen = model(target_columns=['X1', 'X2'], conditional_columns=['X3', 'y'])
        gen.fit(self.data)
        fakes = gen.sample(self.data, n_samples=10)

        assert len(fakes) == 10
        assert fakes[0].shape == self.data.shape
