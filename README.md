# GenRisk: Risk under Synthetic Shift in Time Series Data

GenRisk evaluates the stability of forecasting models to dataset shift using deep generative models and the Worst-Case Risk framework.

## Installation
Go into this this folder and run the following command:

```
pip install -e .
```

## Example of usage

Estimation the worst-case risk on validation data:

```python
from genrisk.shift import ConditionalShift

# X is a pandas dataframe with columns y-1, y-2, exog0, exog1
X_train, y_train = ...
X_val, y_val = ...

target_model = SomeForecastingModel(...)
target_model.fit(X_train, y_train)
y_pred = target_model.predict(X_val)
error_val = (y_pred - y_val)**2

# Looking for the dataset shift in y-1 and y-2
mutable_columns = ['y-1', 'y-2']
immutable_columns = ['exog0', 'exog1']

alpha = 0.9
shift_model = ConditionalShift(
    mutable_columns, 
    immutable_columns, 
    alpha=alpha,
)
shift_model.fit(X_val, error_val)

print(f'Worst-Case Risk for alpha {alpha} is {shift_model.risk:.4f})
```

Generation time-series data:
```python
from genrisk.generation import VectorAR
# data is a pandas DataFrame with timeseries X1 and X2
data = ...
gen = VectorAR(
    target_columns=['X2'], 
    conditional_columns=['X1'],
)
# fakes is a list of generated DataFrames
fakes = gen.sample(data.iloc[:100], n_samples=1)
```

Estimation the worst-case risk on fake data is the same as on validation data:

```python
shift_model.fit(X_fake, error_fake)
print(f'Worst-Case Risk for alpha {alpha} is {shift_model.risk:.4f})
```

More examples in `docs/examples`.

## Testing

Go into this folder and run
```
pytest tests
```

## Documentation

To buld the documentation, run 
```
mkdocs serve
```
and open http://127.0.0.1:8000/
