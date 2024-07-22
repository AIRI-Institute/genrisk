import pandas as pd
import numpy as np
from genrisk.shift import ConditionalShift

def test_conditional_shift():
    shift_model = ConditionalShift(mutable_columns=['X1'], immutable_columns=['X2'])
    X = pd.DataFrame(
        np.random.randn(1000, 2),
        columns=['X1', 'X2'],
    )
    error = pd.Series(np.random.randn(1000))
    shift_model.fit(X, error)
    assert len(shift_model.mask) == len(X)
    assert shift_model.risk
    assert shift_model.ub_risk
    assert shift_model.lb_risk
