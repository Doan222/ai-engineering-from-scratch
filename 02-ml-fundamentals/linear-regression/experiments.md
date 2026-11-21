
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

Investigate whether greater model complexity
or regularisation improves test performance.


