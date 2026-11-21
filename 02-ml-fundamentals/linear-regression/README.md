# Linear Regression — From Theory to Python

**Learning journey:** Applied AI Engineering  
**Topic:** Supervised learning · Regression  
**Level:** Beginner → Practical ML engineering  
**Status:** Actively learning and improving

> This is my learning notebook: explanations in plain English, mathematical intuition, Python implementations and practical engineering lessons. Examples use **synthetic or illustrative data**, not confidential business data.

## Table of Contents

1. [What is Linear Regression?](#1-what-is-linear-regression)
2. [How does the model learn? Loss and MSE](#2-how-does-the-model-learn-loss-and-mse)
3. [Gradient Descent](#3-gradient-descent)
4. [Learning rate, epochs and batches](#4-learning-rate-epochs-and-batches)
5. [Ordinary Least Squares vs Gradient Descent](#5-ordinary-least-squares-vs-gradient-descent)
6. [Simple and Multiple Linear Regression](#6-simple-and-multiple-linear-regression)
7. [Feature standardisation](#7-feature-standardisation)
8. [Polynomial Regression](#8-polynomial-regression)
9. [Residuals and residual plots](#9-residuals-and-residual-plots)
10. [Evaluation metrics](#10-evaluation-metrics)
11. [Underfitting, overfitting and generalisation](#11-underfitting-overfitting-and-generalisation)
12. [Regularisation and Ridge Regression](#12-regularisation-and-ridge-regression)
13. [Ridge vs Lasso vs Elastic Net](#13-ridge-vs-lasso-vs-elastic-net)
14. [Assumptions and limitations](#14-assumptions-and-limitations)
15. [Training, validation and test data](#15-training-validation-and-test-data)
16. [Data leakage](#16-data-leakage)
17. [Feature engineering](#17-feature-engineering)
18. [Business use case: home-charger installation costs](#18-business-use-case-home-charger-installation-costs)
19. [My complete ML workflow](#19-my-complete-ml-workflow)
20. [Python and Scikit-learn reference](#20-python-and-scikit-learn-reference)
21. [My experiments](#21-my-experiments)
22. [Next learning goals](#22-next-learning-goals)

---

## 1. What is Linear Regression?

**Linear Regression** is a **supervised learning** method that predicts a *continuous numerical value* from input features. In supervised learning, we train using examples with known answers.

Examples:
- Estimate house prices from property characteristics.
- Predict vehicle energy consumption from journey information.
- Estimate home-charger installation costs from survey information.
- Model vehicle depreciation from vehicle age.

### The basic equation

$$
\hat{y}=wx+b
$$

| Symbol | Meaning |
|---|---|
| $x$ | Input feature: information supplied to the model |
| $y$ | Actual target: the answer in the dataset |
| $\hat{y}$ | Predicted target: the model's estimate |
| $w$ | Weight/coefficient: how prediction changes when $x$ changes by one unit |
| $b$ | Bias/intercept: predicted value when $x=0$ |

**Think of weight and bias as adjustable settings.** Weight controls the line's slope; bias moves the whole line up or down. Mathematical *bias* here does not mean unfairness or discrimination.

Example: if $\hat y=3x+7$ and $x=5$, the prediction is $3(5)+7=22$.

**Important:** The model learns $w$ and $b$ from training examples; we normally do not specify them ourselves. An intercept of 7 describes the model prediction at $x=0$, but that might not be meaningful if zero is outside the observed range.

## 2. How does the model learn? Loss and MSE

The model needs to measure how wrong its predictions are. A **loss function** converts prediction mistakes into a number we can minimise.

**Mean Squared Error (MSE)** is a common regression loss:

$$
\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
$$

Read this as:
1. Predict a value for every training example.
2. Subtract its prediction from its actual value.
3. Square the difference, so negative and positive errors do not cancel.
4. Add the squared errors and divide by the number of examples.

For example, an error of 5 contributes $25$ to the sum, while an error of 2 contributes $4$: **large mistakes are penalised more heavily**.

**Goal:** Find weights and intercepts that minimise squared training errors. A low *training* loss alone does not prove the model will perform well on new data.

## 3. Gradient Descent

**Gradient Descent** is an *optimisation algorithm*: a method of adjusting parameters to reduce loss. It is **one way**, but not the only way, to train Linear Regression.

Training loop:
1. Start with initial weight and bias values.
2. Predict using the current parameters.
3. Measure the loss.
4. Calculate the gradients, which describe how loss changes as parameters change.
5. Move the parameters in the direction opposite the gradients.
6. Repeat.

For one weight, the update rule is:

$$
w_{\mathrm{new}}=w_{\mathrm{old}}-\alpha\frac{\partial L}{\partial w}
$$

- $L$ is the loss.
- $\frac{\partial L}{\partial w}$ is the gradient of loss with respect to the weight.
- $\alpha$ is the **learning rate**, which controls update size.

The intercept is updated similarly. Gradients are calculated from data; they are not random guesses.

## 4. Learning rate, epochs and batches

The **learning rate** determines the size of each Gradient Descent adjustment.

- **Too small:** progress can be very slow.
- **Too large:** updates can overshoot and training may fail to converge.
- **Appropriate:** training can move steadily towards a lower loss.

| Term | Plain-English meaning |
|---|---|
| **Epoch** | One complete pass through the training examples |
| **Batch** | A group of examples used to compute an update |
| **Iteration** | One parameter-update step |
| **Convergence** | Further updates produce little meaningful objective improvement |

These ideas matter to our **custom Gradient Descent implementation**. Scikit-learn's standard `LinearRegression()` does not expose a learning rate or number of epochs because it uses a least-squares solver.

## 5. Ordinary Least Squares vs Gradient Descent

A **model** is not the same thing as the **method used to train it**.

**Ordinary Least Squares (OLS)** chooses coefficients to minimise the sum of squared residuals:

$$
\sum_{i=1}^{n}(y_i-\hat y_i)^2
$$

| Approach | What happens? |
|---|---|
| **Gradient Descent** | Repeatedly adjust parameters to reduce loss |
| **Least-squares solver** | Use numerical linear algebra to solve the least-squares problem |

Scikit-learn's `LinearRegression()` uses a least-squares solver, **not** our hand-written Gradient Descent loop. Both approaches aim to fit a linear model.

## 6. Simple and Multiple Linear Regression

**Simple Linear Regression** has one input feature:

$$\hat y=wx+b$$

**Multiple Linear Regression** has several input features:

$$\hat y=w_1x_1+w_2x_2+w_3x_3+b$$

For home-charger installation-cost estimation, features might include cable length, meter-to-charger distance and a numerical measure of installation complexity.

Each feature has its **own weight**. A coefficient shows the change in prediction associated with a one-unit change in that feature **while holding the other included features fixed**. It does not by itself prove causation.

## 7. Feature standardisation

Numerical features may have very different ranges: house size could be 3,000 while bedrooms could be 3. Some optimisation and regularisation methods are sensitive to these differences.

**StandardScaler** transforms each numeric feature using training statistics:

$$z=\frac{x-\mu}{\sigma}$$

- $x$: original value
- $\mu$: training feature mean
- $\sigma$: training feature standard deviation

For mean 50, standard deviation 10 and input 70, the scaled value is $(70-50)/10=2$; it is two standard deviations above the training mean.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()  # No training statistics learned yet
X_train_scaled = scaler.fit_transform(X_train)  # Learn mean/std, then scale training inputs
X_test_scaled = scaler.transform(X_test)  # Reuse training mean/std on test inputs
```

**Why not `fit_transform(X_test)`?** It would calculate **new** statistics from test inputs and potentially produce a different representation from the one the model learned. Fit preprocessing on training data; use the **same fitted transformer** on validation, test and future production inputs.

**Important nuance:** Ordinary unregularised Linear Regression often works without feature scaling. Scaling is particularly important for gradient-based training and regularised models such as Ridge and Lasso. Constant features require special handling; Scikit-learn's scaler handles zero variance without dividing by zero.

## 8. Polynomial Regression

What if the true relationship is **curved**? For instance, a car may lose value rapidly at first, then more slowly.

A degree-2 polynomial model uses both $x$ and $x^2$:

$$\hat y=w_1x+w_2x^2+b$$

If $x=4$, the transformed features are $[4,16]$.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree=2, include_bias=False)  # Create x and x² columns
X_poly_train = poly.fit_transform(X_train)  # Transform training inputs
X_poly_test = poly.transform(X_test)  # Same transformation for test inputs

lr_poly = LinearRegression()  # New regression model
lr_poly.fit(X_poly_train, y_train)  # Learn coefficients for x, x² and intercept
```

`PolynomialFeatures` **creates columns**; `LinearRegression` **learns their coefficients**. The model can represent a curve but remains *linear in its coefficients*.

Increasing polynomial degree increases flexibility, **not guaranteed accuracy**. Choose degree using validation data rather than final test results.

## 9. Residuals and residual plots

A **residual** is an observed target minus its fitted prediction:

$$e_i=y_i-\hat y_i$$

| Actual installation cost | Prediction | Residual | Meaning |
|---:|---:|---:|---|
| £1,000 | £900 | +£100 | Underestimated |
| £1,500 | £1,600 | −£100 | Overestimated |
| £2,000 | £2,000 | £0 | Exact |

```python
residuals = y_test - y_pred  # One test residual for each test prediction
```

A **residual plot** places residuals against predicted values or input values. Ideally, it has no obvious systematic pattern:

- **Curved pattern:** possibly a missing nonlinear relationship.
- **Wider spread at high predictions:** possibly changing error variance.
- **Several very large residuals:** investigate potential outliers, data issues or missing predictors.

Metrics summarise overall errors; residual plots reveal **how those errors behave**.

## 10. Evaluation metrics

Evaluate predictions against actual values on **unseen data**.

### Mean Absolute Error (MAE)

$$\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|$$

Average *absolute* error. Example: an MAE of £150 means average absolute error is £150 across evaluation examples.

### Mean Squared Error (MSE)

$$\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2$$

Average squared error. Large mistakes contribute especially heavily; units are squared (e.g., £²).

### Root Mean Squared Error (RMSE)

$$\mathrm{RMSE}=\sqrt{\mathrm{MSE}}$$

Square root of MSE, bringing the number back to original target units (e.g., pounds).

### R-squared ($R^2$)

$$R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}$$

Compares the model's squared errors against a baseline that always predicts the **evaluation-set target mean**, $\bar y$.

| R² value | Interpretation |
|---|---|
| **1.0** | Perfect predictions |
| **0.8** | Explains 80% of target variation relative to that mean baseline |
| **0.0** | Same squared error as the mean baseline |
| **Negative** | Worse than the mean baseline |

**R² = 0.8 does NOT mean 80% of predictions are correct.** R² is not classification accuracy, and high R² does not guarantee good production performance.

| Metric | Main question | Better | Units |
|---|---|---|---|
| MAE | Average absolute size of mistakes? | Lower | Target units |
| MSE | Average squared mistakes? | Lower | Squared units |
| RMSE | Error scale with extra sensitivity to large mistakes? | Lower | Target units |
| R² | Improvement over mean baseline in explained variation? | Higher | No units |

## 11. Underfitting, overfitting and generalisation

**Underfitting:** The model is too simple to capture important patterns. Training and validation errors may both be high.

**Overfitting:** The model fits training-specific noise or detail that does not carry over to new examples. Training performance may be strong but validation performance poor.

**Generalisation:** The ability to make useful predictions on previously unseen examples.

**The goal is not the smallest training error; it is reliable performance on new data.** A higher-degree polynomial may fit the training data better while doing worse on validation data.

## 12. Regularisation and Ridge Regression

**Regularisation** discourages needlessly complex solutions. **Ridge Regression** adds an **L2 penalty** for large coefficients.

Scikit-learn Ridge's objective is equivalent to:

$$\sum_{i=1}^{n}(y_i-\hat y_i)^2+\alpha\sum_{j=1}^{p}w_j^2$$

- First part: squared prediction mistakes.
- Second part: sum of squared **feature coefficients**.
- $\alpha$: strength of regularisation (a **hyperparameter**, not a learned weight).
- The intercept is generally **not penalised**.

| Alpha | Effect |
|---|---|
| 0 | No L2 penalty |
| Small positive | Less shrinkage |
| Large positive | Stronger shrinkage; too much may underfit |

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)  # Choose regularisation strength
ridge.fit(X_train_scaled, y_train)  # Learn coefficients from scaled training inputs
ridge_predictions = ridge.predict(X_test_scaled)  # Predict with the same scaling
```

**Why scale first?** Ridge penalises coefficient sizes, and feature measurement units affect coefficient sizes. Standardisation makes the penalty more comparable across features.

**Be careful interpreting `ridge.coef_`:** When X is standardised, each coefficient describes a **one-standard-deviation increase** in X, not a one-original-unit increase. It is not directly comparable to the unscaled ordinary Linear Regression coefficient.

**Do not confuse:** Ridge `alpha` controls *regularisation strength*. Gradient Descent's *learning rate* controls parameter-update size.

## 13. Ridge vs Lasso vs Elastic Net

**Yes, these are all linear regression models.** Their prediction formulas are linear in the coefficients, but their training objectives differ.

| Model | Training objective, in plain English | Main characteristic |
|---|---|---|
| Ordinary Linear Regression | Minimise squared errors | No coefficient penalty |
| **Ridge** | Squared errors + L2 penalty | Shrinks coefficients towards zero |
| **Lasso** | Squared errors + L1 penalty | Can set some coefficients *exactly* to zero |
| **Elastic Net** | Squared errors + L1 and L2 penalties | Combines shrinkage and feature selection |

**Lasso:** Can help when many features are available and some may not be needed. **Elastic Net:** Combines L1 and L2 behaviour. Neither method is automatically better; compare suitable candidates using validation data.

## 14. Assumptions and limitations

Understand these assumptions especially when interpreting coefficients or statistical confidence intervals:

- **Linearity:** The expected target is modelled as a linear combination of included features and coefficients. Polynomial *features* can represent some curves.
- **Independent errors:** Dependence between observations (e.g., time series, repeated measures from one property) needs special care.
- **Constant error variance (homoscedasticity):** Classical inference commonly assumes error variance stays approximately constant across fitted values.
- **Multicollinearity:** Highly related predictors can make separate coefficient estimates unstable. Example: vehicle age and model year.
- **Normal errors:** Mainly important for some classical statistical inference procedures, especially in small samples; **not a strict requirement just to fit or predict**.

Violating an assumption does not automatically make every prediction useless. Investigate how it affects the specific decision, coefficient interpretation or uncertainty estimate.

Other limitations: Linear Regression can be sensitive to outliers and may extrapolate poorly far outside the training feature range.

## 15. Training, validation and test data

We separate data because **training a model** and **checking whether it generalises** are different jobs.

| Split | What it is for |
|---|---|
| **Training** | Learn coefficients and fitted preprocessing statistics |
| **Validation** | Compare candidate models and choose settings |
| **Test** | Final evaluation of the already selected approach |

Examples of decisions made using validation data:

- **Model selection:** Linear Regression, Ridge, Lasso or Polynomial Regression?
- **Hyperparameter tuning:** Ridge `alpha=0.1` or `1.0`? Polynomial `degree=2` or `3`?

We do **not** have to try every possible model. Start with a simple baseline and compare sensible alternatives.

**Illustrative example (not my measured results):**

| Candidate | Validation MAE |
|---|---:|
| Linear Regression | £150 |
| Ridge, alpha = 1 | £120 |
| Polynomial, degree = 2 | £110 |

Here, degree-2 Polynomial Regression has the lowest *validation* MAE. We might select it, then evaluate it on the untouched test set.

**Important:** Repeatedly looking at test results to choose a model turns the test set into another validation set and makes the final estimate optimistic. For limited data, **cross-validation** can help choose models more reliably.

After model selection, it is common to refit the chosen method using available training **and validation** data, while keeping the final test set untouched.

## 16. Data leakage

**Data leakage** happens when information that would not genuinely be available at the time of prediction affects training or model selection.

Examples:
- Standardising with statistics computed across the whole dataset before splitting.
- Using test results to choose features or hyperparameters.
- Using an installation's final cost-related outcome as an input to predict its pre-installation cost.
- Placing highly related records (e.g., repeat surveys for the same property) across training and test splits when that makes evaluation unrealistic.

**Safe principle:** Split appropriately first; fit preprocessing only on the relevant training partition; use the fitted transformation on validation, test and production inputs.

A Scikit-learn **Pipeline** can combine preprocessing and modelling consistently, including during cross-validation. Preventing leakage is part of good ML engineering, not just a coding style preference.

## 17. Feature engineering

**Feature engineering** means creating, selecting or improving inputs so the model can represent useful patterns.

Possible installation-cost features:
- Cable length.
- Distance from meter to charger location.
- Property type (categorical; requires encoding for Linear Regression).
- Survey or installation complexity.
- Polynomial terms or interactions where justified.

A useful feature should be **relevant**, **reliable**, and **available when the prediction must be made**. More features do not automatically improve the model.

## 18. Business use case: home-charger installation costs

### Business question

Can we estimate installation costs earlier in the customer journey to support quotations and decision-making?

### Example features and target

| Role | Candidate data |
|---|---|
| Inputs (X) | Cable length, meter distance, property characteristics, installation complexity |
| Target (y) | Actual final installation cost (£) |

### Candidate approach

1. Understand the business objective and when the estimate is needed.
2. Gather historical data with suitable permissions, or use synthetic data for demonstrations.
3. Check completeness, units, outliers and which features would be known before installation.
4. Split data in a way that reflects real future use.
5. Fit a baseline Linear Regression model.
6. Compare appropriate alternative models and hyperparameters using validation or cross-validation.
7. Evaluate once using the final held-out test set.
8. Check whether errors are acceptable in pounds and across installation types.

**Business value is not the same as a good R².** Evaluate whether errors are acceptable, whether the system could reduce manual estimation, and what happens when a quotation is wrong.

## 19. My complete ML workflow

```text
Business question
      ↓
Define target (y) and available input features (X)
      ↓
Inspect and clean the data
      ↓
Split into training / validation / test
      ↓
Fit preprocessing on training data only
      ↓
Train a simple baseline model
      ↓
Compare candidates and settings on validation data
      ↓
Inspect residuals, MAE, RMSE and R²
      ↓
Select and refit the chosen method (without test data)
      ↓
Evaluate on untouched test data
      ↓
Assess business value, risks and monitoring needs
```

For small datasets, cross-validation often replaces a single fixed validation split. Preprocessing must be fitted **inside each training fold** to prevent leakage.

## 20. Python and Scikit-learn reference

| Syntax | Beginner explanation |
|---|---|
| `import` | Load functionality from a library |
| `as` | Give the import a shorter alternative name |
| `=` | Assign the right-hand value to the left-hand variable |
| `()` | Call a function/method or create an object |
| `.` | Access an object's attribute or method |
| `[0]` | Take the first element (Python starts counting at zero) |
| `.fit(X, y)` | Learn model parameters from input/target examples |
| `.predict(X)` | Use a trained model to produce predictions |
| `.transform(X)` | Apply an already fitted data transformation |
| `.fit_transform(X)` | Fit a transformer then transform X |
| `.coef_` | Coefficients learned during fitting |
| `.intercept_` | Intercept learned during fitting |
| `.shape` | Dimensions of a NumPy array |
| `random_state=42` | Fix randomness for reproducible data splitting |
| `:.4f` | Display four digits after the decimal point |

**Parameters vs hyperparameters:** Model parameters (e.g., weights) are learned by `.fit()`; hyperparameters (e.g., Ridge `alpha`, polynomial `degree`) are selected by us, preferably using validation.

### The script I practised

1. Generate **100 synthetic examples** following $y=3x+7+\text{noise}$.
2. Split into **80 training** and **20 test** examples.
3. Train an ordinary Scikit-learn Linear Regression model.
4. Make test predictions and calculate R² and MSE.
5. Create polynomial features ($x$, $x^2$) and fit a separate model.
6. Standardise inputs, then train Ridge Regression.

See the accompanying Python files in this folder as they are added.

## 21. My experiments

These are results from my earlier exercises with **synthetic data**. They are not business performance claims.

### Scikit-learn Linear Regression

| Measure | Result |
|---|---:|
| True data-generating coefficient | 3.0 |
| True data-generating intercept | 7.0 |
| Fitted coefficient | ≈ 2.9197 |
| Fitted intercept | ≈ 7.2858 |
| Test R² | ≈ 0.9686 |
| Test MSE | ≈ 2.6148 |

Because the data includes noise and the model learns from a sample, fitted coefficients are not necessarily exactly equal to the values used to generate the data.

### Custom Gradient Descent exercise

| Measure | Result |
|---|---:|
| Fitted weight | ≈ 3.1433 |
| Fitted intercept | ≈ 6.5432 |
| R² reported in that exercise | ≈ 0.9566 |

These exercises used separate generated data/workflows; **do not interpret their R² numbers as a controlled head-to-head comparison** unless both models are evaluated on the same test examples.

### My key observations

- Learning weights is different from manually setting the true generating equation.
- A model can have nonzero prediction error even when the data is generated from a linear relationship, because we added random noise.
- Extra polynomial features do not automatically improve results.
- Ridge changes training by penalising large coefficients.
- Feature scaling changes what a coefficient means numerically.
- Evaluation should represent unseen examples and the business context.


---

### Repository files (planned)

- `README.md` — this theory and learning document.
- `linear_regression_scratch.py` — my implementation using Gradient Descent.
- `linear_regression_sklearn.py` — Scikit-learn Linear, Polynomial and Ridge exercises.
- `experiments.md` — reproducible experiments and notes.


