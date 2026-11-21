
# ============================================================
# LINEAR REGRESSION FROM SCRATCH — STUDY VERSION
# ============================================================
# GOAL:
# Build a Linear Regression model without using scikit-learn.
#
# Our model learns the equation:
#     y = w * x + b
#
# w = weight (slope): how much y changes when x increases by 1
# b = bias (intercept): predicted y when x = 0
#
# LEARNING PROCESS:
# 1. Make predictions
# 2. Calculate how wrong the predictions are (cost)
# 3. Calculate gradients (direction of error reduction)
# 4. Update weight and bias
# 5. Repeat until the model improves
# ============================================================


# ------------------------------------------------------------
# PART 1: GENERATE TRAINING DATA
# ------------------------------------------------------------

import random  # Python's built-in module for generating random numbers

# IMPORTANT:
# A seed makes our random numbers reproducible.
# Running the script again produces the same training data.
random.seed(42)

# These are the TRUE parameters used to generate our data.
# The model does NOT receive these values during training.
TRUE_W = 3.0
TRUE_B = 7.0

# Number of training examples
N_SAMPLES = 100

# Generate 100 random input values between 0 and 10.
#
# random.uniform(0, 10) generates one random decimal number.
# range(N_SAMPLES) repeats the operation 100 times.
# _ means we don't need the loop variable.
#
# X is a list containing 100 input values.
X = [random.uniform(0, 10) for _ in range(N_SAMPLES)]

# Generate the correct target values (y).
#
# TRUE_W * x + TRUE_B is the original linear relationship.
# random.gauss(0, 2.0) adds normally distributed random noise:
#   0   = mean of the noise
#   2.0 = standard deviation of the noise
#
# Noise makes the data more realistic.
# Without noise, all points would lie exactly on one line.
#
# Each x in X produces one corresponding y value.
y = [
    TRUE_W * x + TRUE_B + random.gauss(0, 2.0)
    for x in X
]


# ------------------------------------------------------------
# PART 2: DEFINE OUR LINEAR REGRESSION MODEL
# ------------------------------------------------------------

# A class is a blueprint for creating model objects.
# Defining a class does NOT automatically train a model.
class LinearRegression:

    # --------------------------------------------------------
    # METHOD 1: INITIALISE THE MODEL
    # --------------------------------------------------------

    def __init__(self, learning_rate=0.01):
        # __init__ runs automatically when we create an object.
        #
        # self refers to the particular model object.
        # self.w and self.b belong to that object.

        self.w = 0.0  # Initial weight
        self.b = 0.0  # Initial bias

        # Learning rate controls the size of parameter updates.
        # Too large: training may overshoot or diverge.
        # Too small: training may be very slow.
        self.lr = learning_rate

        # Store the cost after every training epoch.
        # Useful for checking whether training is improving.
        self.cost_history = []

    # --------------------------------------------------------
    # METHOD 2: MAKE PREDICTIONS
    # --------------------------------------------------------

    def predict(self, X):
        # Apply y = w*x + b to every x in X.
        #
        # The list comprehension returns a list of predictions.
        # It uses the model's CURRENT weight and bias.
        return [self.w * x + self.b for x in X]

    # --------------------------------------------------------
    # METHOD 3: CALCULATE COST (MEAN SQUARED ERROR)
    # --------------------------------------------------------

    def compute_cost(self, X, y):
        # Calculate predictions using the current model.
        predictions = self.predict(X)

        # Number of actual target values
        n = len(y)

        # MSE = average of squared prediction errors.
        #
        # zip(predictions, y) pairs each prediction
        # with its corresponding actual value.
        #
        # pred - actual = prediction error
        # ** 2 = square the error
        # sum(...) = add all squared errors
        # / n = calculate their average
        #
        # Lower MSE generally means a better fit to this data.
        cost = sum(
            (pred - actual) ** 2
            for pred, actual in zip(predictions, y)
        ) / n

        return cost

    # --------------------------------------------------------
    # METHOD 4: CALCULATE GRADIENTS
    # --------------------------------------------------------

    def compute_gradients(self, X, y):
        predictions = self.predict(X)
        n = len(y)

        # dw = derivative of MSE with respect to weight (w).
        #
        # It tells us how cost changes when we change w.
        #
        # The x is included because changing w affects
        # predictions in proportion to each input value.
        dw = (2 / n) * sum(
            (pred - actual) * x
            for pred, actual, x in zip(predictions, y, X)
        )

        # db = derivative of MSE with respect to bias (b).
        #
        # Bias adds the same amount to every prediction,
        # so the gradient does not need to multiply by x.
        db = (2 / n) * sum(
            pred - actual
            for pred, actual in zip(predictions, y)
        )

        # These are LOCAL variables.
        # They are returned to whichever method called this one.
        return dw, db

    # --------------------------------------------------------
    # METHOD 5: TRAIN THE MODEL USING GRADIENT DESCENT
    # --------------------------------------------------------

    def fit(self, X, y, epochs=1000, print_every=200):
        # An epoch here means one complete gradient calculation
        # and parameter update using the full training dataset.
        #
        # range(epochs) produces 0, 1, 2, ..., 999
        # when epochs = 1000.
        for epoch in range(epochs):

            # Step 1: Calculate gradients with CURRENT parameters.
            dw, db = self.compute_gradients(X, y)

            # Step 2: Update the model parameters.
            #
            # -= means subtract and assign the new value.
            #
            # Gradient descent moves AGAINST the gradient.
            # A negative gradient therefore increases a parameter.
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # Step 3: Calculate cost AFTER updating parameters.
            cost = self.compute_cost(X, y)

            # Step 4: Save cost for later analysis.
            self.cost_history.append(cost)

            # Step 5: Print progress at selected epochs.
            #
            # % is the modulo operator (remainder).
            # epoch % 200 == 0 is true at 0, 200, 400, etc.
            #
            # :.4f formats a number to 4 decimal places.
            # :4d formats an integer in a field of width 4.
            if epoch % print_every == 0:
                print(
                    f"  Epoch {epoch:4d} | "
                    f"Cost: {cost:.4f} | "
                    f"w: {self.w:.4f} | "
                    f"b: {self.b:.4f}"
                )

        # Return the trained model object itself.
        return self

    # --------------------------------------------------------
    # METHOD 6: EVALUATE THE MODEL USING R-SQUARED
    # --------------------------------------------------------

    def r_squared(self, X, y):
        predictions = self.predict(X)

        # Calculate the average of actual target values.
        y_mean = sum(y) / len(y)

        # SS_res = total squared errors from OUR model.
        ss_res = sum(
            (actual - pred) ** 2
            for actual, pred in zip(y, predictions)
        )

        # SS_tot = total squared errors if we always
        # predicted the average target value.
        ss_tot = sum(
            (actual - y_mean) ** 2
            for actual in y
        )

        # R² = 1 - SS_res / SS_tot
        #
        # R² = 1: perfect predictions
        # R² = 0: same squared error as predicting the mean
        # R² < 0: worse than predicting the mean
        #
        # NOTE: If all y values are identical, ss_tot = 0.
        # This simple implementation does not handle that case.
        return 1 - (ss_res / ss_tot)


# ------------------------------------------------------------
# PART 3: CREATE, TRAIN AND EVALUATE THE MODEL
# ------------------------------------------------------------

# This only displays a heading.
print("=== Training Linear Regression (Gradient Descent) ===")

# Create an instance of the LinearRegression class.
#
# Python automatically calls __init__ here.
# Other methods do NOT run automatically.
#
# Override the default learning rate with 0.005.
model = LinearRegression(learning_rate=0.005)

# Start training!
#
# fit() calls compute_gradients() and compute_cost().
# During training, the model updates its own w and b.
model.fit(X, y, epochs=1000, print_every=200)

# Display the equation learned by the model.
#
# \n starts a new line.
# f"..." is an f-string: it inserts variable values.
# :.4f displays a number with 4 decimal places.
print(f"\nLearned: y = {model.w:.4f}x + {model.b:.4f}")

# Display the original equation used to generate the data.
print(f"True:    y = {TRUE_W}x + {TRUE_B}")

# Calculate and display R-squared.
#
# IMPORTANT:
# This evaluates the model on its TRAINING data.
# For a real ML project, use separate test data to check
# how well the model generalises to unseen examples.
print(f"R-squared: {model.r_squared(X, y):.4f}")
