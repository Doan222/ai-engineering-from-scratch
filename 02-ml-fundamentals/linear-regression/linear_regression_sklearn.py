
# ============================================================
# SCIKIT-LEARN: LINEAR REGRESSION, POLYNOMIAL REGRESSION & RIDGE
# ============================================================
#
# WHAT ARE WE TRYING TO LEARN?
#
# 1. Create a synthetic dataset (artificial example data).
# 2. Train a Linear Regression model.
# 3. Evaluate how accurately it predicts unseen data.
# 4. Try Polynomial Regression to allow curved relationships.
# 5. Standardise the numerical input values.
# 6. Train Ridge Regression to introduce regularisation.
#
# KEY ML IDEA:
# X = input features (the information used to predict).
# y = target values (the answers we want to predict).
#
# fit() = train a model or fit a data transformer.
# predict() = use a trained model to generate predictions.
# transform() = change how input data is represented.
#
# ============================================================
# PART 1 — IMPORT LIBRARIES
# ============================================================

# Import LinearRegression from Scikit-learn.
# "as SklearnLR" gives the class a shorter, alternative name.
# We use this name so it is distinguishable from our
# previously created custom LinearRegression class.
from sklearn.linear_model import LinearRegression as SklearnLR

# Ridge is Linear Regression with L2 regularisation.
# Regularisation discourages unnecessarily large coefficients.
# This can help reduce overfitting.
from sklearn.linear_model import Ridge

# PolynomialFeatures creates additional feature columns,
# such as x², so Linear Regression can fit curved patterns.
#
# StandardScaler standardises numerical features using:
# (value - training_mean) / training_standard_deviation
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Allows us to divide a dataset into:
# Training data -> used to train the model.
# Test data -> used to evaluate the trained model.
from sklearn.model_selection import train_test_split

# Import two evaluation metrics:
# MSE: average squared prediction error (lower is better).
# R²: how much variation the model explains relative
#     to predicting the mean of the actual target values.
from sklearn.metrics import mean_squared_error, r2_score

# NumPy provides numerical arrays and mathematical operations.
# "np" is a conventional short name (alias) for NumPy.
import numpy as np


# ============================================================
# PART 2 — GENERATE SYNTHETIC DATA
# ============================================================

# Set the random seed to 42.
# This makes NumPy's random results reproducible.
# Running the script again gives the same generated data.
# The number 42 is arbitrary.
np.random.seed(42)

# Generate 100 random X values between 0 and 10.
#
# np.random.uniform() creates uniformly distributed numbers.
# (0, 10, ...) specifies the lower and upper bounds.
#
# (100, 1) describes the array's shape:
# 100 rows = 100 examples.
# 1 column = 1 input feature.
#
# X_sk is a NumPy array with shape (100, 1).
X_sk = np.random.uniform(0, 10, (100, 1))

# Create the target values using:
# y = 3x + 7 + random_noise
#
# 3.0 = true coefficient (slope).
# 7.0 = true intercept.
#
# X_sk.squeeze() changes the shape from (100, 1) to (100,).
# It removes the extra one-column dimension without
# changing the actual numbers in X_sk.
#
# np.random.normal(0, 2.0, 100) creates 100 noise values:
# 0 = average noise.
# 2.0 = standard deviation of noise.
# 100 = number of noise values.
#
# We add noise to make the dataset more realistic.
# Without noise, all examples would lie exactly on a line.
#
# y_sk contains 100 target values, shape (100,).
y_sk = (
    3.0 * X_sk.squeeze()
    + 7.0
    + np.random.normal(0, 2.0, 100)
)


# ============================================================
# PART 3 — SPLIT TRAINING AND TEST DATA
# ============================================================

# Divide our 100 examples into training and testing sets.
#
# test_size=0.2 means 20% goes to testing.
# Therefore:
# 80 examples -> training.
# 20 examples -> testing.
#
# random_state=42 makes the split reproducible.
# This controls the split's randomness separately from
# np.random.seed(42), which controlled data generation.
#
# The function shuffles and splits X and y consistently.
# Corresponding input and target values stay together.
#
# The results are assigned to four variables in order:
#
# X_train -> 80 training inputs, shape (80, 1).
# X_test  -> 20 test inputs, shape (20, 1).
# y_train -> 80 actual training targets, shape (80,).
# y_test  -> 20 actual test targets, shape (20,).
#
# IMPORTANT:
# We train using training data, not test data.
# Test data represents examples not seen during training.
X_train, X_test, y_train, y_test = train_test_split(
    X_sk,
    y_sk,
    test_size=0.2,
    random_state=42
)


# ============================================================
# PART 4 — TRAIN ORDINARY LINEAR REGRESSION
# ============================================================

# Create a new Scikit-learn Linear Regression object.
#
# SklearnLR() creates the model.
# lr is the variable that stores the model object.
#
# At this point, the model has NOT learned any coefficients.
lr = SklearnLR()

# Train the model.
#
# fit() learns the relationship between X_train and y_train.
#
# The model tries to find the best coefficient w
# and intercept b for the equation:
#
# predicted_y = w * x + b
#
# It chooses values that minimise squared prediction errors
# on the training data.
#
# Unlike our earlier custom gradient descent implementation,
# Scikit-learn's LinearRegression uses a least-squares solver.
#
# After fitting, the learned parameters are stored in lr.
lr.fit(X_train, y_train)

# Make predictions using the 20 test inputs.
#
# predict() uses the coefficient and intercept learned
# during training.
#
# It does NOT train the model again.
#
# Input: X_test -> shape (20, 1).
# Output: y_pred -> shape (20,).
#
# y_pred contains 20 predicted target values.
y_pred = lr.predict(X_test)


# ============================================================
# PART 5 — EVALUATE LINEAR REGRESSION
# ============================================================

# Print a heading in the console.
print("=== Scikit-learn Linear Regression ===")

# Print the coefficient learned by the model.
#
# lr.coef_ contains the fitted coefficients as an array.
# The final underscore (_) is a Scikit-learn naming convention
# indicating an attribute learned during fitting.
#
# [0] selects the first coefficient from the array.
# Python indexing starts at 0.
#
# :.4f displays a number with four decimal places.
# f"..." is an f-string: expressions inside {}
# are evaluated and inserted into the text.
#
# Because we have one feature, we have one coefficient.
# Our synthetic data originally used a true coefficient of 3.
print(f"Coefficient (w): {lr.coef_[0]:.4f}")

# Print the learned intercept (b).
#
# The intercept is the predicted value when x = 0.
# Our synthetic data originally used a true intercept of 7.
#
# The learned result may differ because we added noise
# and trained using a sample of the data.
print(f"Intercept (b): {lr.intercept_:.4f}")

# Evaluate the model using R-squared (R²).
#
# y_test = actual answers for the 20 test examples.
# y_pred = model predictions for those same examples.
#
# R² measures how well the predictions explain variation
# in the actual target values compared with a mean baseline.
#
# R² = 1 means perfect predictions.
# R² = 0 means no better than predicting the test mean.
# R² can also be negative.
#
# R² is NOT classification accuracy.
print(f"R-squared (test): {r2_score(y_test, y_pred):.4f}")

# Calculate Mean Squared Error (MSE).
#
# For each test example:
# 1. Calculate prediction - actual value.
# 2. Square that difference.
# 3. Average the squared differences.
#
# MSE = average((predicted - actual)²)
#
# Lower MSE means smaller squared prediction errors.
# MSE = 0 means perfect predictions.
#
# MSE is measured in squared units of the target variable.
print(f"MSE (test): {mean_squared_error(y_test, y_pred):.4f}")


# ============================================================
# PART 6 — POLYNOMIAL REGRESSION
# ============================================================

# WHY TRY POLYNOMIAL REGRESSION?
#
# Ordinary Linear Regression using one feature models:
#
# y = w*x + b
#
# This is a straight line.
#
# Sometimes relationships are curved.
# PolynomialFeatures can create extra features such as x²,
# allowing Linear Regression to represent a curve.

# Create a PolynomialFeatures transformer.
#
# degree=2 means create polynomial terms up to power 2.
#
# With one original feature x, the output features are:
# x and x².
#
# Example:
# Original x = 4
# Transformed features = [4, 16]
#
# include_bias=False excludes an additional constant
# column of ones.
#
# LinearRegression already learns an intercept by default,
# so we don't need that extra constant column.
#
# IMPORTANT:
# This creates the transformer but transforms no data yet.
poly = PolynomialFeatures(degree=2, include_bias=False)

# FIT + TRANSFORM THE TRAINING DATA.
#
# fit:
# Determine which polynomial terms to create based
# on the configuration and the input feature structure.
#
# transform:
# Generate those polynomial feature columns.
#
# Example:
# Original input [3] -> transformed input [3, 9].
#
# Before: X_train has shape (80, 1).
# After: X_poly_sk has shape (80, 2).
#
# We still have 80 examples.
# We now have 2 input features instead of 1.
#
# PolynomialFeatures does not learn relationships with y.
# It only changes how X is represented.
X_poly_sk = poly.fit_transform(X_train)

# TRANSFORM THE TEST DATA.
#
# Apply the same polynomial feature transformation
# to the 20 test examples.
#
# We do NOT fit the transformer again.
# We reuse the already fitted poly object.
#
# Before: X_test has shape (20, 1).
# After: X_poly_test has shape (20, 2).
#
# ML BEST PRACTICE:
# Fit preprocessing on training data.
# Reuse fitted preprocessing on test data.
#
# For PolynomialFeatures specifically, refitting on test
# inputs with the same feature structure would normally
# create identical columns. However, reusing the fitted
# transformer is the correct general workflow.
X_poly_test = poly.transform(X_test)

# Create a NEW Linear Regression model.
#
# This is separate from the original lr model.
# We want to compare straight-line and polynomial models.
#
# It has not been trained yet.
lr_poly = SklearnLR()

# Train the new model using the polynomial training features.
#
# X_poly_sk contains two columns: x and x².
# y_train still contains the original 80 target values.
#
# The model now learns an equation of the form:
#
# predicted_y = w1*x + w2*x² + b
#
# w1 = coefficient for x.
# w2 = coefficient for x².
# b  = intercept.
#
# The model can represent curved relationships because
# the second feature is x².
#
# We are still using LinearRegression!
# The difference is the features supplied to the model.
lr_poly.fit(X_poly_sk, y_train)

# Evaluate the polynomial model on test data.
#
# First:
# lr_poly.predict(X_poly_test) generates 20 predictions.
#
# Then:
# r2_score(y_test, ...) compares these predictions
# with the 20 actual test target values.
#
# Finally:
# print() displays the result.
#
# \n inserts a newline before the printed text.
# :.4f formats R² to four decimal places.
#
# Compare this R² with the original Linear Regression R².
# A more complex model is not automatically better.
#
# Our synthetic target is linear (3x + 7 + noise), so
# adding x² may not improve performance on unseen data.
print(
    f"\nPolynomial degree 2 R-squared: "
    f"{r2_score(y_test, lr_poly.predict(X_poly_test)):.4f}"
)


# ============================================================
# PART 7 — FEATURE SCALING WITH STANDARDSCALER
# ============================================================

# WHY SCALE FEATURES?
#
# Some ML algorithms are sensitive to feature magnitudes.
#
# Example:
# House size = 250 square metres.
# House age  = 10 years.
#
# Standardisation expresses values relative to each
# feature's training mean and standard deviation.
#
# Formula:
# scaled_x = (x - mean) / standard_deviation
#
# The resulting non-constant training feature typically
# has mean 0 and standard deviation 1.
#
# Ridge benefits from consistent feature scaling because
# regularisation penalises coefficients.

# Create a StandardScaler object.
#
# At this point, it does not know the training mean
# or standard deviation.
# No input data has been transformed yet.
scaler = StandardScaler()

# FIT + TRANSFORM TRAINING DATA.
#
# FIT:
# Learn the mean and standard deviation of X_train.
#
# TRANSFORM:
# Scale each training value using those statistics.
#
# Example:
# Training mean = 50.
# Training standard deviation = 10.
# Original value = 70.
#
# Scaled value = (70 - 50) / 10 = 2.
#
# This means the value is two standard deviations
# above the training mean.
#
# The fitted scaler remembers its training statistics:
# scaler.mean_  -> learned mean.
# scaler.scale_ -> scaling factor.
#
# Input shape: (80, 1).
# Output shape: (80, 1).
#
# Unlike PolynomialFeatures, scaling does not add columns.
# It changes the numerical values of existing columns.
X_train_scaled = scaler.fit_transform(X_train)

# TRANSFORM TEST DATA ONLY.
#
# This is one of the most important ML engineering practices.
#
# Use the SAME training mean and standard deviation
# to standardise the 20 test inputs.
#
# DO NOT use scaler.fit_transform(X_test).
#
# WHY NOT?
#
# Calling fit_transform() on test data would calculate
# a new mean and standard deviation from X_test.
#
# Then training and testing inputs would use
# different scaling rules.
#
# The model was trained using one scale, but would
# receive test values expressed on another scale.
#
# Fitting preprocessing on test data is also a form
# of test-data leakage because test-set information
# influences how that test set is processed.
#
# Correct workflow:
# TRAIN -> fit_transform().
# TEST  -> transform().
#
# The same fitted scaler should later be used
# for real-world production predictions.
#
# Input shape: (20, 1).
# Output shape: (20, 1).
X_test_scaled = scaler.transform(X_test)


# ============================================================
# PART 8 — RIDGE REGRESSION
# ============================================================

# WHY RIDGE REGRESSION?
#
# Ordinary Linear Regression minimises prediction errors.
#
# Ridge Regression also discourages large coefficients.
# This is known as L2 regularisation.
#
# Regularisation can help reduce overfitting.
#
# alpha controls the strength of the coefficient penalty.
# alpha = 0 means no regularisation penalty.
# Larger alpha means stronger regularisation.
#
# Too much regularisation may cause underfitting.
#
# Ridge does not automatically create polynomial features.
# In this section, we use the original X feature,
# standardised by StandardScaler.

# Create a Ridge Regression model.
#
# alpha=1.0 sets the regularisation strength.
#
# We have only created the model.
# It has not learned any coefficients yet.
ridge = Ridge(alpha=1.0)

# TRAIN THE RIDGE MODEL.
#
# Input: X_train_scaled -> 80 scaled training examples.
# Target: y_train -> 80 original target values.
#
# Ridge learns coefficients that balance:
# 1. Keeping predictions close to actual Y values.
# 2. Discouraging unnecessarily large coefficients.
#
# The model uses the scaled X feature.
# Our y_train values have not been standardised.
#
# Because we have one input feature, Ridge learns:
# One coefficient and one intercept.
#
# Learned attributes:
# ridge.coef_      -> coefficient array.
# ridge.intercept_ -> intercept.
#
# IMPORTANT:
# Training happens here, not when Ridge() was created.
ridge.fit(X_train_scaled, y_train)

# EVALUATE RIDGE REGRESSION.
#
# ridge.predict(X_test_scaled):
# Make predictions for the 20 scaled test inputs.
#
# We use X_test_scaled rather than X_test because
# Ridge was trained on standardised inputs.
#
# r2_score(y_test, predictions):
# Compare predictions with actual Y values.
#
# The result lets us compare Ridge with the other models.
print(
    f"Ridge R-squared: "
    f"{r2_score(y_test, ridge.predict(X_test_scaled)):.4f}"
)

# DISPLAY THE RIDGE COEFFICIENT.
#
# ridge.coef_ is an array of fitted coefficients.
# [0] selects the first coefficient.
# :.4f prints it to four decimal places.
#
# IMPORTANT:
# This coefficient describes the relationship with
# STANDARDISED X, not the original unscaled X.
#
# Therefore, it cannot be compared directly with the
# ordinary Linear Regression coefficient unless
# we account for the difference in feature scaling.
print(f"Ridge coefficient: {ridge.coef_[0]:.4f}")
