
# Linear Regression — Experiments

## Experiment 1: Linear Regression from Scratch

### Objective

Understand how Linear Regression learns weights and bias
using Gradient Descent implemented in Python.

### Dataset

- Type: Synthetic data generated using Python
- Number of samples: 100
- Input features: 1
- True relationship: y = 3x + 7 + noise
- Noise: Gaussian, mean 0, standard deviation 2
- Random seed: 42

### Training Configuration

- Initial weight: 0.0
- Initial bias: 0.0
- Learning rate: 0.005
- Epochs: 1000
- Loss function: Mean Squared Error (MSE)
- Optimisation: Full-batch Gradient Descent

### Results

| Measurement | Value |
|---|---:|
| True weight | 3.0000 |
| Learned weight | 3.1433 |
| True bias | 7.0000 |
| Learned bias | 6.5432 |
| R² (training data) | 0.9566 |

### Observations

1. The model learned a relationship close to the true equation.
2. The learned parameters differed from the true values
   because the dataset contained random noise.
3. Gradient Descent progressively reduced the training loss.
4. The learning rate controlled the size of parameter updates.

### Important Limitation

R² was calculated using the same dataset used for training.

Therefore, this experiment does not establish how well
the model generalises to unseen examples.

A future improvement is to introduce a train/test split.

### What I Learned

- How weights and bias influence predictions.
- How Mean Squared Error measures model mistakes.
- How gradients determine the direction of updates.
- How Gradient Descent improves parameters over time.
- Why evaluating on unseen data matters.



---

## Experiment 2: Linear Regression with Scikit-learn

### Objective

Learn how to train and evaluate Linear Regression
using Scikit-learn rather than implementing
Gradient Descent manually.

The main objective is to understand:
- How to prepare training and test datasets.
- How Scikit-learn learns model coefficients.
- How to make predictions on unseen data.
- How to evaluate performance using MSE and R².

### Dataset

- Type: Synthetic data generated using NumPy
- Number of samples: 100
- Number of input features: 1
- Input range: 0 to 10
- True relationship: y = 3x + 7 + noise
- Noise: Normal distribution, mean 0, standard deviation 2
- Random seed: 42

### Train/Test Split

| Dataset | Samples | Purpose |
|---|---:|---|
| Training | 80 | Learn model coefficients |
| Test | 20 | Evaluate unseen predictions |

The dataset was split using:

```python
train_test_split(
    X_sk,
    y_sk,
    test_size=0.2,
    random_state=42
)
```

### Model

Scikit-learn LinearRegression.

The model uses the equation:

$$
\hat{y} = wx + b
$$

Unlike my first implementation, Scikit-learn's
LinearRegression uses a least-squares solver
rather than my custom Gradient Descent loop.

### Training and Prediction

```python
lr = SklearnLR()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
```

- `SklearnLR()` creates the model object.
- `.fit()` learns the coefficient and intercept.
- `.predict()` generates predictions for 20 test inputs.

### Results

| Measurement | Value |
|---|---:|
| True coefficient | 3.0000 |
| Learned coefficient | 2.9197 |
| True intercept | 7.0000 |
| Learned intercept | 7.2858 |
| Test R² | 0.9686 |
| Test MSE | 2.6148 |

### Interpretation

**Coefficient**

The model learned a coefficient of approximately 2.92.

This means increasing X by one unit increases
the predicted Y by approximately 2.92 units.

The true coefficient used to create the data was 3.

**Intercept**

The model learned an intercept of approximately 7.29.

The true intercept was 7.

**R-squared**

The test R² was approximately 0.9686.

This indicates the model explains approximately
96.86% of the variation in the test targets
relative to the test-set mean baseline.

This does NOT mean 96.86% of predictions are correct.

**Mean Squared Error**

The test MSE was approximately 2.6148.

This is the average squared prediction error
across the 20 test examples.

Lower MSE indicates smaller squared errors.

### Important Observations

1. The learned coefficient and intercept were close
   to the original values used to generate the data.

2. The learned parameters were not identical
   because the data included random noise.

3. Scikit-learn handled the optimisation internally,
   so I did not need to implement gradients manually.

4. The model was evaluated on 20 examples
   that were not used to train it.

5. A high test R² suggests that Linear Regression
   models this synthetic relationship well.

### Comparison with Experiment 1

| Characteristic | From Scratch | Scikit-learn |
|---|---|---|
| Implementation | Custom Python | Scikit-learn |
| Training method | Gradient Descent | Least-squares solver |
| Samples generated | 100 | 100 |
| Training samples | 100 | 80 |
| Test samples | 0 | 20 |
| R² | 0.9566 (training) | 0.9686 (test) |

**Important:**

These results are not directly comparable
because the experiments used separately
generated datasets and different evaluation setups.

Experiment 1 evaluated training performance,
while Experiment 2 evaluated test performance.

Therefore, the higher R² in Experiment 2
does not prove Scikit-learn is a better model.

### What I Learned

- The difference between implementing an algorithm
  and using a machine-learning library.
- How `.fit()` learns model parameters.
- How `.predict()` generates unseen predictions.
- Why train/test separation matters.
- How to interpret coefficients and intercepts.
- How to interpret MSE and R².
- Why evaluation conditions must be consistent
  when comparing different models.

### Next Experiment

Compare ordinary Linear Regression
with Polynomial Regression and Ridge Regression.


---

## Experiment 3: Polynomial Regression (Degree 2)

### Objective

Investigate whether adding a squared input feature
improves Linear Regression predictions.

Ordinary Linear Regression models:

$$
\hat{y} = wx + b
$$

Polynomial Regression with degree 2 models:

$$
\hat{y} = w_1x + w_2x^2 + b
$$

The additional squared feature allows the model
to represent curved relationships.

### Dataset

This experiment uses the SAME synthetic dataset
and train/test split as Experiment 2.

- Total samples: 100
- Training samples: 80
- Test samples: 20
- True relationship: y = 3x + 7 + noise
- Random seed: 42

The original relationship is linear.

Therefore, adding a squared feature does not
necessarily improve predictive performance.

### Feature Engineering

```python
poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_poly_sk = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)
```

The transformer creates two features:

- x: original input
- x²: squared input

Example:

Original input: [4]

Transformed input: [4, 16]

Training data changes from shape (80, 1)
to shape (80, 2).

Test data changes from shape (20, 1)
to shape (20, 2).

The number of observations stays the same.

### Model Training

```python
lr_poly = SklearnLR()
lr_poly.fit(X_poly_sk, y_train)
```

The model learns:
- A coefficient for x
- A coefficient for x²
- An intercept

PolynomialFeatures creates the features.

LinearRegression learns the coefficients.

### Evaluation

```python
poly_predictions = lr_poly.predict(X_poly_test)

poly_r2 = r2_score(y_test, poly_predictions)

print(f"Polynomial R²: {poly_r2:.4f}")
```

### Results

| Measurement | Value |
|---|---|
| Polynomial degree | 2 |
| Number of transformed features | 2 |
| Training samples | 80 |
| Test samples | 20 |
| Test R² | To be recorded |

For comparison:

Ordinary Linear Regression test R²: **0.9686**

### Observations

1. Polynomial Regression can model curved relationships.
2. Adding x² increases model flexibility.
3. The true synthetic relationship is linear.
4. A more complex model is not automatically better.
5. Test performance must be measured before concluding
   whether the polynomial model performs better.

### What I Learned

- What polynomial feature engineering means.
- How x becomes [x, x²].
- Why `.fit_transform()` is used on training data.
- Why `.transform()` is used on test data.
- How LinearRegression can fit curved relationships.
- Why additional features may increase overfitting risk.

### Important Limitation

This is an initial educational comparison.

The same test set was used while exploring models.

In a more rigorous ML project, I should select
polynomial degree using validation data or
cross-validation and reserve the test set
for final evaluation.

Investigate whether greater model complexity
or regularisation improves test performance.


---

## Experiment 4: Ridge Regression (L2 Regularisation)

### Objective

Understand how Ridge Regression differs from
ordinary Linear Regression.

Investigate how L2 regularisation influences
the learned coefficient and model predictions.

### What Is Ridge Regression?

Ridge is a linear regression model that adds
a penalty for large coefficients.

Ordinary Linear Regression tries to minimise
squared prediction errors.

Ridge minimises squared prediction errors
plus a penalty based on squared coefficients.

A common Ridge objective is:

$$
\text{Cost} =
\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
+\alpha\sum_{j=1}^{p}w_j^2
$$

Here:

- alpha controls the regularisation strength.
- Larger alpha means stronger coefficient shrinkage.
- Excessive regularisation can cause underfitting.

### Dataset

This experiment uses the SAME synthetic dataset
and train/test split as Experiments 2 and 3.

- Total samples: 100
- Training samples: 80
- Test samples: 20
- Original relationship: y = 3x + 7 + noise
- Input features: 1

### Feature Standardisation

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

StandardScaler learns the mean and standard deviation
from the training inputs.

It then standardises each feature.

We use `.transform()` for test data so that
the same scaling rules are applied.

This prevents fitting preprocessing to test data.

### Training Ridge Regression

```python
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
```

`Ridge(alpha=1.0)` creates a Ridge model
with L2 regularisation strength set to 1.0.

`.fit()` trains the model using 80 scaled
training inputs and their corresponding targets.

Ridge learns:

- One coefficient
- One intercept

### Evaluation

```python
ridge_predictions = ridge.predict(X_test_scaled)

ridge_r2 = r2_score(y_test, ridge_predictions)

print(f"Ridge R²: {ridge_r2:.4f}")
print(f"Ridge coefficient: {ridge.coef_[0]:.4f}")
```

### Results

| Measurement | Value |
|---|---|
| Model | Ridge Regression |
| Regularisation | L2 |
| Alpha | 1.0 |
| Input features | 1 |
| Training samples | 80 |
| Test samples | 20 |
| Test R² | To be recorded |
| Ridge coefficient (scaled inputs) | To be recorded |

Ordinary Linear Regression test R²: **0.9686**

### Important: Interpreting the Ridge Coefficient

The Ridge model was trained using standardised X values.

Therefore, its coefficient represents the predicted
change in Y for a one-standard-deviation increase
in the original input X.

The ordinary Linear Regression coefficient
represents the predicted change in Y for a
one-unit increase in original X.

These coefficients use different input units
and must not be compared directly by magnitude.

To express the Ridge coefficient in original
X units, divide it by the training standard deviation.

### Observations

1. Ridge adds L2 regularisation to Linear Regression.
2. Regularisation discourages large coefficients.
3. Scaling is useful because Ridge penalises coefficients.
4. A smaller coefficient does not automatically mean
   better predictions.
5. Ridge must be evaluated on unseen data.
6. Ridge is not guaranteed to outperform ordinary
   Linear Regression on every dataset.

### Important Limitation

This experiment uses a single input feature
and a synthetic relationship that is already linear.

Ridge may offer little benefit in this situation.

Regularisation can become more useful when models
have many features, correlated inputs, or a greater
risk of overfitting.

Also, this experiment uses a fixed alpha of 1.0.

A more rigorous experiment would compare several
alpha values using validation data or cross-validation,
then evaluate the selected model on a final test set.

### What I Learned

- What L2 regularisation means.
- How Ridge differs from ordinary Linear Regression.
- What alpha controls.
- Why feature scaling matters.
- Why coefficients must be interpreted in their units.
- Why regularisation does not guarantee better performance.
- Why validation data is useful for tuning alpha.


---

## Experiment 5: Final Model Comparison and Learning

### Results Summary

All three Scikit-learn models used the same synthetic
dataset, training examples and test examples.

| Model | Test R² | Test MSE |
|---|---:|---:|
| Linear Regression | 0.9686 | 2.6148 |
| Polynomial (degree 2) | 0.9695 | 2.5434 |
| Ridge (alpha=1.0) | 0.9689 | 2.5914 |

### Interpretation

Polynomial Regression achieved the highest test R²
and lowest test MSE in this particular experiment.

However, the differences between the three models
were small.

The underlying data was generated from a linear
relationship with random noise.

Therefore, additional model complexity is not
necessarily justified by these results.

### Ridge Coefficient Interpretation

Ridge coefficient on scaled X: 8.3983

Ridge coefficient in original X units: 2.8837

Original Linear Regression coefficient: 2.9197

These original-unit coefficients are reasonably close.

### Important Experiment Limitations

- Only 100 synthetic examples were generated.
- Only 20 examples were used for testing.
- The models were compared using one test split.
- Polynomial degree and Ridge alpha were not tuned
  using separate validation data.
- These results do not establish which model would
  perform best on other datasets.

### What I Learned

1. A more complex model does not always generalise better.
2. Regularisation changes how coefficients are learned.
3. Feature scaling changes coefficient interpretation.
4. Models must be compared on consistent datasets.
5. A high R² does not guarantee real-world success.
6. Validation and cross-validation help select models
   without repeatedly using the final test set.

### Next Improvements

- Add residual plots.
- Compare models using cross-validation.
- Experiment with multiple Ridge alpha values.
- Introduce a validation strategy.
- Use Scikit-learn Pipelines to prevent data leakage.
