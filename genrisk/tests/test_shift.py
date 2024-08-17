import pandas as pd
import numpy as np
from genrisk.shift import ConditionalShift


def test_conditional_shift():
    shift_model = ConditionalShift(
        mutable_columns=['X1'],
        immutable_columns=['X2'],
        alpha_space=list(np.linspace(0.05, 0.95, 10)),
    )
    X = pd.DataFrame(
        np.random.randn(1000, 2),
        columns=['X1', 'X2'],
    )
    error = pd.Series(np.random.randn(1000))
    output = shift_model.fit(X, error)
    assert len(shift_model.mask) == len(X)
    assert len(output) == len(list(np.linspace(0.05, 0.95, 10)))
    assert output[0]
    assert output[1]
    assert output[2]
