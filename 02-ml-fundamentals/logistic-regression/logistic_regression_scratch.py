
# ============================================================
# LOGISTIC REGRESSION FROM SCRATCH — STUDY VERSION
# ============================================================
# GOAL:
# Build a binary classification model without scikit-learn.
#
# Logistic Regression is a SUPERVISED LEARNING algorithm.
#
# Supervised Learning means:
# We train a model using examples where we already know
# the correct answers (labels).
#
# Unlike Linear Regression, which predicts continuous values,
# Logistic Regression predicts probabilities between 0 and 1.
#
# EXAMPLE:
# Predict whether a home charger installation is complex.
#
# 0 = Standard installation
# 1 = Complex installation
#
# Our model learns the equation:
#
#     z = w1*x1 + w2*x2 + b
#
# Then converts z into a probability:
#
#     p = 1 / (1 + e^(-z))
#
# This second equation is called the SIGMOID FUNCTION.
#
# VARIABLES:
# x1, x2 = input features (information about each example)
# w1, w2 = weights learned by the model
# b      = bias learned by the model
# z      = weighted sum before applying sigmoid
# p      = predicted probability of class 1
#
# EXAMPLE:
# If the model predicts p = 0.85:
#
# It estimates an 85% probability of class 1.
#
# Using a classification threshold of 0.5:
#
# p >= 0.5 --> Predict class 1
# p < 0.5  --> Predict class 0
#
# LEARNING PROCESS:
# 1. Make probability predictions.
# 2. Calculate how wrong predictions are (loss).
# 3. Calculate gradients (direction of loss change).
# 4. Update weights and bias using Gradient Descent.
# 5. Repeat until training is complete.
# 6. Evaluate predictions using classification accuracy.
#
# IMPORTANT:
# Logistic Regression is a linear classification model
# even though its sigmoid function is nonlinear.
#
# It creates a linear decision boundary in feature space.
#
# This implementation uses:
# - Binary Cross-Entropy loss
# - Batch Gradient Descent
# - Two input features
# - Binary labels (0 and 1)
#
# ============================================================


# ------------------------------------------------------------
# PART 1: IMPORT REQUIRED MODULES
# ------------------------------------------------------------

# Python's built-in math module contains mathematical functions.
#
# We need:
#
# math.exp(x):
# Calculates e raised to the power x.
#
# e is a mathematical constant approximately equal to 2.71828.
#
# Example:
# math.exp(0) = 1
#
# math.log(x):
# Calculates the natural logarithm of x.
#
# Example:
# math.log(1) = 0
#
# We use:
# math.exp() in the sigmoid function.
# math.log() in Binary Cross-Entropy loss.

import math


# ------------------------------------------------------------
# PART 2: DEFINE THE SIGMOID FUNCTION
# ------------------------------------------------------------

# The sigmoid function converts any real number into
# a value between 0 and 1.
#
# Formula:
#
#                  1
# sigmoid(z) = -----------
#              1 + e^(-z)
#
# EXAMPLES:
#
# z = 0   --> sigmoid = 0.5000
# z = 2   --> sigmoid = 0.8808
# z = -2  --> sigmoid = 0.1192
#
# Larger positive z --> probability approaches 1.
# Larger negative z --> probability approaches 0.
#
# WHY DO WE NEED SIGMOID?
#
# The weighted sum:
#
# z = w1*x1 + w2*x2 + b
#
# can produce any number, including negative numbers.
#
# But classification probabilities must be between 0 and 1.
#
# Sigmoid transforms z into a valid probability.


def sigmoid(z):

    # def creates a function.
    #
    # sigmoid is the function name.
    #
    # (z) means the function receives one input called z.
    #
    # The colon (:) begins the function body.
    #
    # IMPORTANT:
    # The negative-z branch below avoids numerical overflow
    # when z is a very large negative number.

    if z < 0:

        # if checks whether a condition is true.
        #
        # z < 0 means:
        # Is z less than zero?
        #
        # If yes, use a mathematically equivalent version
        # of the sigmoid formula:
        #
        #               e^z
        # sigmoid(z) = -------
        #              1 + e^z
        #
        # This avoids calculating e^(-z), which can become
        # extremely large when z is very negative.

        return math.exp(z) / (1 + math.exp(z))

    # If z is zero or positive, use the standard formula.
    #
    # math.exp(-z) calculates e raised to -z.
    #
    # 1 + math.exp(-z) calculates the denominator.
    #
    # return sends the result back to whoever called sigmoid().

    return 1 / (1 + math.exp(-z))


# ------------------------------------------------------------
# PART 3: DEFINE OUR LOGISTIC REGRESSION MODEL
# ------------------------------------------------------------

# A class is a blueprint for creating objects.
#
# Think of a class as a template.
#
# A blueprint describes what a model contains
# and what operations the model can perform.
#
# Our LogisticRegression class contains:
#
# ATTRIBUTES:
# - weights
# - bias
# - learning rate
# - loss history
#
# METHODS:
# 1. __init__()      --> Initialise the model
# 2. predict_proba() --> Predict a probability
# 3. predict()       --> Predict class 0 or 1
# 4. compute_loss()  --> Calculate model loss
# 5. fit()           --> Train the model
# 6. accuracy()      --> Evaluate classification accuracy
#
# IMPORTANT:
# Defining a class does NOT train a model.
#
# Training starts only when we call model.fit().


class LogisticRegression:

    # --------------------------------------------------------
    # METHOD 1: INITIALISE THE MODEL
    # --------------------------------------------------------

    def __init__(self, n_features, learning_rate=0.01):

        # __init__ is a special Python method.
        #
        # It runs automatically when we create an object.
        #
        # self refers to the particular object being created.
        #
        # Example:
        #
        # model = LogisticRegression(n_features=2)
        #
        # Python automatically calls __init__.
        #
        # PARAMETERS:
        #
        # n_features:
        # Number of input features for each example.
        #
        # Example:
        # [10, 4] contains two features.
        #
        # Therefore:
        # n_features = 2
        #
        # learning_rate=0.01:
        # Default learning rate.
        #
        # We can override it when creating the model.

        # Initialise one weight for every feature.
        #
        # [0.0] creates a list containing one zero.
        #
        # * n_features repeats that zero.
        #
        # Example:
        #
        # n_features = 2
        #
        # [0.0] * 2 --> [0.0, 0.0]
        #
        # self.weights now contains:
        #
        # w1 = 0.0
        # w2 = 0.0
        #
        # These weights will change during training.

        self.weights = [0.0] * n_features

        # Bias is an additional adjustable parameter.
        #
        # Formula:
        #
        # z = w1*x1 + w2*x2 + b
        #
        # Bias shifts the decision boundary.
        #
        # Initially, bias = 0.

        self.bias = 0.0

        # Store the learning rate.
        #
        # self.lr belongs to this model object.
        #
        # Learning rate controls the size of parameter updates.
        #
        # Too large:
        # Training may overshoot or become unstable.
        #
        # Too small:
        # Training may improve very slowly.

        self.lr = learning_rate

        # Store loss values during training.
        #
        # [] creates an empty list.
        #
        # After training, an example might look like:
        #
        # [0.6931, 0.6540, 0.6210, ...]
        #
        # This helps us analyse whether the model improves.

        self.loss_history = []


    # --------------------------------------------------------
    # METHOD 2: PREDICT A PROBABILITY
    # --------------------------------------------------------

    def predict_proba(self, x):

        # predict_proba means predict probability.
        #
        # INPUT:
        #
        # x represents ONE example.
        #
        # Example:
        #
        # x = [10, 4]
        #
        # It contains two feature values.
        #
        # CURRENT MODEL PARAMETERS:
        #
        # self.weights = [w1, w2]
        # self.bias = b
        #
        # STEP 1:
        # Calculate the weighted sum:
        #
        # z = w1*x1 + w2*x2 + b
        #
        # We use zip() to pair weights with feature values.
        #
        # Example:
        #
        # weights = [0.2, 0.5]
        # x = [10, 4]
        #
        # zip(weights, x) creates pairs:
        #
        # (0.2, 10)
        # (0.5, 4)
        #
        # The generator expression:
        #
        # w * xi for w, xi in zip(self.weights, x)
        #
        # calculates:
        #
        # 0.2 * 10 = 2
        # 0.5 * 4  = 2
        #
        # sum(...) adds these results:
        #
        # 2 + 2 = 4
        #
        # Then add bias.
        #
        # If bias = -1:
        #
        # z = 4 + (-1) = 3
        #
        # Python syntax:
        #
        # w = current weight
        # xi = corresponding feature value
        # for = repeat for each pair
        # in = take values from
        # zip() = pair elements together
        # sum() = add the generated numbers
        #
        # NOTE:
        # zip() stops at the shorter input sequence.
        # This simple model assumes the feature count matches
        # the number of weights.

        z = sum(
            w * xi
            for w, xi in zip(self.weights, x)
        ) + self.bias

        # STEP 2:
        # Convert z into a probability.
        #
        # Example:
        #
        # z = 3
        #
        # sigmoid(3) is approximately 0.9526.
        #
        # The model predicts approximately:
        #
        # 95.26% probability of class 1.
        #
        # IMPORTANT:
        # This is a model-estimated probability.
        # It is not automatically a well-calibrated probability.

        return sigmoid(z)


    # --------------------------------------------------------
    # METHOD 3: PREDICT CLASS 0 OR CLASS 1
    # --------------------------------------------------------

    def predict(self, x, threshold=0.5):

        # predict() returns the final predicted class.
        #
        # Unlike predict_proba(), it returns 0 or 1.
        #
        # threshold=0.5:
        # Default classification threshold.
        #
        # Example:
        #
        # Probability = 0.85
        # Threshold = 0.50
        #
        # 0.85 >= 0.50 --> True
        #
        # Therefore:
        # Predicted class = 1
        #
        # Another example:
        #
        # Probability = 0.30
        #
        # 0.30 >= 0.50 --> False
        #
        # Therefore:
        # Predicted class = 0
        #
        # PYTHON SYNTAX:
        #
        # >= means greater than or equal to.
        #
        # if ... else ... is a conditional expression.
        #
        # Read:
        # "Return 1 if the probability meets the threshold;
        # otherwise return 0."
        #
        # predict_proba(x) is called first.
        #
        # The calculated probability is then compared
        # against the threshold.

        return 1 if self.predict_proba(x) >= threshold else 0


    # --------------------------------------------------------
    # METHOD 4: CALCULATE LOSS (BINARY CROSS-ENTROPY)
    # --------------------------------------------------------

    def compute_loss(self, X, y):

        # In Machine Learning, loss measures how wrong
        # the model's predictions are.
        #
        # Linear Regression commonly uses:
        #
        # Mean Squared Error (MSE).
        #
        # Logistic Regression commonly uses:
        #
        # Binary Cross-Entropy (BCE), also called Log Loss.
        #
        # FORMULA:
        #
        # Loss = -(1/n) * SUM[
        #     y*log(p) + (1-y)*log(1-p)
        # ]
        #
        # WHERE:
        #
        # n = number of examples
        # y = actual label (0 or 1)
        # p = predicted probability of class 1
        # log = natural logarithm
        #
        # WHY USE BINARY CROSS-ENTROPY?
        #
        # It penalises confident incorrect predictions heavily.
        #
        # EXAMPLE:
        #
        # Actual class = 1
        #
        # Prediction A: p = 0.9
        # Prediction B: p = 0.1
        #
        # Prediction A is much better.
        #
        # Loss A = -log(0.9) = approximately 0.105
        #
        # Loss B = -log(0.1) = approximately 2.303
        #
        # Lower loss is better.
        #
        # IMPORTANT:
        # Loss measures probability prediction quality.
        # Accuracy measures correct class predictions.
        # They are not the same metric.

        # Calculate the number of actual labels.
        #
        # len(y) returns the length of the list.
        #
        # Example:
        #
        # y = [0, 1, 0, 1]
        #
        # len(y) = 4

        n = len(y)

        # Create an accumulator for total loss.
        #
        # An accumulator stores a running total.
        #
        # Initially:
        # total = 0.0

        total = 0.0

        # Loop through every training example.
        #
        # range(n) produces:
        #
        # 0, 1, 2, ..., n-1
        #
        # i represents the current example's index.

        for i in range(n):

            # Predict the probability for example i.
            #
            # X[i] selects one row from X.
            #
            # Example:
            #
            # X[0] = [10, 4]
            #
            # predict_proba(X[0]) returns its probability.

            p = self.predict_proba(X[i])

            # Protect against log(0).
            #
            # math.log(0) is undefined.
            #
            # Predicted probabilities can become extremely
            # close to zero or one.
            #
            # We therefore restrict p to:
            #
            # [0.000000000000001, 0.999999999999999]
            #
            # 1e-15 means:
            #
            # 1 * 10^(-15)
            #
            # min(1 - 1e-15, p):
            # Choose whichever is smaller.
            #
            # max(1e-15, ...):
            # Choose whichever is larger.
            #
            # This is called probability clipping.
            #
            # It prevents undefined logarithms.
            #
            # NOTE:
            # Clipping is applied when calculating loss.
            # It does not change the model's saved weights.

            p = max(1e-15, min(1 - 1e-15, p))

            # Calculate the contribution of this example.
            #
            # Formula:
            #
            # y*log(p) + (1-y)*log(1-p)
            #
            # CASE 1:
            #
            # If actual label y = 1:
            #
            # 1*log(p) + 0*log(1-p)
            #
            # Result:
            #
            # log(p)
            #
            # CASE 2:
            #
            # If actual label y = 0:
            #
            # 0*log(p) + 1*log(1-p)
            #
            # Result:
            #
            # log(1-p)
            #
            # += means:
            #
            # Add this value to the existing total.
            #
            # Equivalent to:
            #
            # total = total + new_value
            #
            # This expression accumulates log-likelihood
            # terms, which are zero or negative.

            total += (
                y[i] * math.log(p)
                + (1 - y[i]) * math.log(1 - p)
            )

        # Convert the accumulated result into average BCE.
        #
        # -total:
        # Changes the negative log-likelihood into loss.
        #
        # / n:
        # Calculates the average across all examples.
        #
        # The final value is a single number.
        #
        # Example:
        #
        # Loss = 0.42
        #
        # The goal of training is to reduce this loss.

        return -total / n


    # --------------------------------------------------------
    # METHOD 5: TRAIN USING BATCH GRADIENT DESCENT
    # --------------------------------------------------------

    def fit(self, X, y, epochs=1000, print_every=200):

        # In Machine Learning, fit means train.
        #
        # The model receives:
        #
        # X = input features
        # y = correct labels
        #
        # epochs=1000:
        # Maximum number of complete training passes.
        #
        # print_every=200:
        # Display progress every 200 epochs.
        #
        # We use BATCH GRADIENT DESCENT.
        #
        # This means:
        #
        # 1. Calculate gradients using ALL training examples.
        # 2. Average their contributions.
        # 3. Update weights and bias once per epoch.
        #
        # IMPORTANT:
        # Training adjusts parameters to reduce BCE loss.
        #
        # It does not directly maximise accuracy.

        # Number of training examples.
        #
        # Example:
        #
        # y = [0, 1, 0, 1]
        #
        # n = 4

        n = len(y)

        # Calculate the number of input features.
        #
        # X[0] selects the first example.
        #
        # Example:
        #
        # X[0] = [10, 4]
        #
        # len(X[0]) = 2
        #
        # Therefore:
        #
        # n_features = 2

        n_features = len(X[0])

        # Begin the training loop.
        #
        # range(epochs) produces:
        #
        # 0, 1, 2, ..., 999
        #
        # when epochs = 1000.
        #
        # One epoch represents:
        #
        # - Processing all training examples
        # - Calculating average gradients
        # - Updating model parameters
        #
        # The parameters are updated once per epoch.

        for epoch in range(epochs):

            # ------------------------------------------------
            # STEP 1: INITIALISE GRADIENT ACCUMULATORS
            # ------------------------------------------------

            # dw stores gradients for the weights.
            #
            # We have two features, so two weight gradients.
            #
            # Example:
            #
            # dw = [0.0, 0.0]
            #
            # dw[0] corresponds to weight w1.
            # dw[1] corresponds to weight w2.
            #
            # These values are reset at every epoch.

            dw = [0.0] * n_features

            # db stores the bias gradient.
            #
            # Initially:
            #
            # db = 0.0
            #
            # It will accumulate the errors from all examples.

            db = 0.0

            # ------------------------------------------------
            # STEP 2: PROCESS EVERY TRAINING EXAMPLE
            # ------------------------------------------------

            for i in range(n):

                # Predict probability using CURRENT parameters.
                #
                # Example:
                #
                # X[i] = [10, 4]
                #
                # p = 0.65
                #
                # This is the model's estimated probability
                # of class 1.

                p = self.predict_proba(X[i])

                # Calculate the prediction error.
                #
                # Formula:
                #
                # error = p - y
                #
                # Example 1:
                #
                # p = 0.65
                # y = 1
                #
                # error = -0.35
                #
                # The probability is below the target.
                #
                # Example 2:
                #
                # p = 0.65
                # y = 0
                #
                # error = 0.65
                #
                # The probability is above the target.
                #
                # IMPORTANT MATHEMATICAL POINT:
                #
                # For sigmoid + Binary Cross-Entropy,
                # the derivative of loss with respect to z
                # simplifies to:
                #
                # dLoss/dz = p - y
                #
                # This simplifies gradient calculation.

                error = p - y

                # --------------------------------------------
                # STEP 3: CALCULATE WEIGHT GRADIENTS
                # --------------------------------------------

                # Loop through every feature.
                #
                # Example:
                #
                # n_features = 2
                #
                # range(2) produces:
                #
                # j = 0
                # j = 1
                #
                # Each feature has a corresponding weight.

                for j in range(n_features):

                    # Add this example's contribution
                    # to the gradient for weight j.
                    #
                    # Formula:
                    #
                    # dLoss/dwj = (p-y) * xj
                    #
                    # For the complete dataset:
                    #
                    # dwj = (1/n) * SUM[(p-y)*xj]
                    #
                    # EXAMPLE:
                    #
                    # error = -0.35
                    #
                    # X[i] = [10, 4]
                    #
                    # For weight 1:
                    #
                    # -0.35 * 10 = -3.5
                    #
                    # For weight 2:
                    #
                    # -0.35 * 4 = -1.4
                    #
                    # SAME ERROR, DIFFERENT CONTRIBUTIONS.
                    #
                    # Why?
                    #
                    # Because each weight is multiplied
                    # by a different feature value.
                    #
                    # Python indexing:
                    #
                    # X[i][j]
                    #
                    # First [i] selects the example.
                    # Second [j] selects its feature.
                    #
                    # Example:
                    #
                    # X[0] = [10, 4]
                    #
                    # X[0][0] = 10
                    # X[0][1] = 4
                    #
                    # += accumulates gradient contributions.
                    #
                    # We do NOT update weights here.
                    #
                    # We first collect contributions
                    # from all training examples.

                    dw[j] += error * X[i][j]

                # --------------------------------------------
                # STEP 4: CALCULATE BIAS GRADIENT
                # --------------------------------------------

                # Bias is added directly to z:
                #
                # z = w1*x1 + w2*x2 + b
                #
                # The derivative of z with respect to b is 1.
                #
                # Therefore:
                #
                # dLoss/db = p - y
                #
                # Unlike the weight gradients,
                # there is no multiplication by a feature.
                #
                # Accumulate this example's error.

                db += error

            # ------------------------------------------------
            # STEP 5: UPDATE THE WEIGHTS
            # ------------------------------------------------

            # At this stage:
            #
            # dw contains accumulated weight gradients.
            # db contains the accumulated bias gradient.
            #
            # We now average gradients by dividing by n.
            #
            # GRADIENT DESCENT FORMULA:
            #
            # new_weight = old_weight - learning_rate*gradient
            #
            # WHY SUBTRACT?
            #
            # The gradient points in the direction
            # of increasing loss.
            #
            # We move in the opposite direction
            # to reduce loss.
            #
            # A positive gradient decreases a weight.
            # A negative gradient increases a weight.

            for j in range(n_features):

                # self.weights[j]:
                # Current value of weight j.
                #
                # self.lr:
                # Learning rate.
                #
                # dw[j] / n:
                # Average gradient for weight j.
                #
                # -= means:
                #
                # Subtract the calculated adjustment
                # and store the updated value.
                #
                # EXAMPLE:
                #
                # Old weight = 0.5
                # Learning rate = 0.1
                # Average gradient = 0.2
                #
                # New weight:
                #
                # 0.5 - (0.1 * 0.2)
                #
                # = 0.48

                self.weights[j] -= self.lr * (dw[j] / n)

            # ------------------------------------------------
            # STEP 6: UPDATE THE BIAS
            # ------------------------------------------------

            # Apply the same Gradient Descent principle.
            #
            # new_bias = old_bias - learning_rate*gradient
            #
            # db / n:
            # Average bias gradient.
            #
            # Example:
            #
            # Old bias = 0.2
            # Learning rate = 0.1
            # Average gradient = -0.3
            #
            # New bias:
            #
            # 0.2 - (0.1 * -0.3)
            #
            # = 0.23
            #
            # Because the gradient was negative,
            # subtracting it increased the bias.

            self.bias -= self.lr * (db / n)

            # ------------------------------------------------
            # STEP 7: CALCULATE LOSS AFTER THE UPDATE
            # ------------------------------------------------

            # The model now has UPDATED weights and bias.
            #
            # Calculate BCE loss using the new parameters.
            #
            # compute_loss() internally calls predict_proba().
            #
            # We calculate this after the update to measure
            # the model's new training performance.

            loss = self.compute_loss(X, y)

            # ------------------------------------------------
            # STEP 8: SAVE LOSS HISTORY
            # ------------------------------------------------

            # append() adds an item to the end of a list.
            #
            # Example:
            #
            # self.loss_history initially:
            #
            # []
            #
            # After epoch 0:
            #
            # [0.69]
            #
            # After epoch 1:
            #
            # [0.69, 0.66]
            #
            # The actual values depend on the training data.
            #
            # This allows us to plot a learning curve later.

            self.loss_history.append(loss)

            # ------------------------------------------------
            # STEP 9: PRINT TRAINING PROGRESS
            # ------------------------------------------------

            # % is the modulo operator.
            #
            # It calculates the remainder after division.
            #
            # Example:
            #
            # 200 % 200 = 0
            # 201 % 200 = 1
            #
            # Therefore:
            #
            # epoch % print_every == 0
            #
            # is true at:
            #
            # 0, 200, 400, 600, 800
            #
            # when print_every = 200.
            #
            # == checks equality.
            #
            # This avoids printing every epoch.

            if epoch % print_every == 0:

                # f"..." is an f-string.
                #
                # It inserts Python values into text.
                #
                # {epoch:4d}:
                # Display the integer in a field of width 4.
                #
                # {loss:.4f}:
                # Display loss with 4 decimal places.
                #
                # {self.weights[0]:.3f}:
                # Display weight 1 with 3 decimal places.
                #
                # {self.weights[1]:.3f}:
                # Display weight 2 with 3 decimal places.
                #
                # This display is specifically designed
                # for our TWO-FEATURE example.
                #
                # A model with one feature would need
                # a different printing statement.

                print(
                    f"  Epoch {epoch:4d} | "
                    f"Loss: {loss:.4f} | "
                    f"w: [{self.weights[0]:.3f}, "
                    f"{self.weights[1]:.3f}] | "
                    f"b: {self.bias:.3f}"
                )

        # Return the trained model object itself.
        #
        # self refers to the current model.
        #
        # This allows:
        #
        # model.fit(X_train, y_train)
        #
        # or:
        #
        # model = LogisticRegression(2).fit(X_train, y_train)

        return self


    # --------------------------------------------------------
    # METHOD 6: EVALUATE CLASSIFICATION ACCURACY
    # --------------------------------------------------------

    def accuracy(self, X, y):

        # Accuracy measures the percentage of examples
        # classified correctly.
        #
        # FORMULA:
        #
        # Accuracy = Correct Predictions / Total Predictions
        #
        # EXAMPLE:
        #
        # Actual labels:
        # [0, 1, 0, 1]
        #
        # Predicted labels:
        # [0, 1, 1, 1]
        #
        # Correct predictions = 3
        # Total examples = 4
        #
        # Accuracy = 3/4 = 0.75
        #
        # This means 75% accuracy.

        # Calculate the number of correct predictions.
        #
        # range(len(y)):
        # Iterate through every actual label.
        #
        # self.predict(X[i]):
        # Predict class 0 or 1 for example i.
        #
        # == y[i]:
        # Check whether the prediction matches reality.
        #
        # if:
        # Include only matching predictions.
        #
        # 1 for i in ...:
        # Generate one number 1 for each correct prediction.
        #
        # sum(...):
        # Count the correct predictions by adding the ones.
        #
        # IMPORTANT:
        # This uses the default threshold of 0.5.

        correct = sum(
            1
            for i in range(len(y))
            if self.predict(X[i]) == y[i]
        )

        # Divide the number of correct predictions
        # by the total number of actual labels.
        #
        # The result is between 0 and 1.
        #
        # 1.0 = 100% accuracy
        # 0.5 = 50% accuracy
        # 0.0 = 0% accuracy
        #
        # NOTE:
        # Accuracy alone can be misleading when classes
        # are imbalanced.
        #
        # Other important classification metrics include:
        #
        # - Precision
        # - Recall
        # - F1-score
        # - Confusion Matrix
        #
        # We will study these in later experiments.

        return correct / len(y)


# ------------------------------------------------------------
# PART 4: CREATE OUR DATASET
# ------------------------------------------------------------

# We will create a small dataset manually.
#
# Unlike our previous Linear Regression example,
# we are NOT generating random data.
#
# Therefore:
# We do not need random.seed().
#
# Each example contains TWO input features.
#
# EXAMPLE BUSINESS SCENARIO:
#
# Imagine predicting whether a home charger installation
# will be classified as complex.
#
# Feature 1 = cable length in metres
# Feature 2 = distance from meter to installation point
#
# Target:
#
# 0 = Standard installation
# 1 = Complex installation
#
# IMPORTANT:
# This is a hypothetical example.
#
# These values and labels are manually invented
# for learning Python and Logistic Regression.
#
# They are NOT real JLR installation records.
#
# The labels are illustrative, not based on
# validated electrical installation rules.

# X contains all input examples.
#
# The outer [] is a list containing multiple examples.
#
# Each inner [] is a list containing feature values.
#
# Example:
#
# X[0] = [10, 4]
#
# This example has:
#
# Feature 1 = 10
# Feature 2 = 4
#
# X[1] = [20, 8]
#
# Feature 1 = 20
# Feature 2 = 8

X = [
    [10, 4],
    [20, 8],
    [15, 6],
    [30, 12],
    [25, 9]
]

# Calculate the number of examples.
#
# len(X) counts the number of rows.
#
# Here:
#
# N = 5
#
# IMPORTANT:
# N represents the number of examples.
#
# It does NOT represent the number of features.

N = len(X)

# y contains the correct class labels.
#
# Each value in y corresponds to one row in X.
#
# Example:
#
# X[0] = [10, 4]  --> y[0] = 0
# X[1] = [20, 8]  --> y[1] = 1
# X[2] = [15, 6]  --> y[2] = 0
# X[3] = [30, 12] --> y[3] = 1
# X[4] = [25, 9]  --> y[4] = 1
#
# These are the correct answers during training.
#
# The model does not initially know
# the relationship between X and y.
#
# It must learn useful weights and bias.

y = [0, 1, 0, 1, 1]


# ------------------------------------------------------------
# PART 5: SPLIT DATA INTO TRAINING AND TEST SETS
# ------------------------------------------------------------

# In Machine Learning, we normally separate data into:
#
# TRAINING DATA:
# Used to learn model parameters.
#
# TEST DATA:
# Used to evaluate predictions on unseen examples.
#
# Our target split:
#
# 80% training
# 20% testing
#
# IMPORTANT:
# This is an extremely small example.
#
# With only 5 records:
#
# Training = 4 records
# Testing = 1 record
#
# A single test example cannot provide a reliable
# estimate of performance on future real-world data.

# Calculate the split position.
#
# 0.8 means 80%.
#
# N = 5
#
# 0.8 * 5 = 4.0
#
# int() converts the result into an integer.
#
# Therefore:
#
# split = 4

split = int(0.8 * N)

# Split input features into training and test datasets.
#
# Python slicing syntax:
#
# X[:split]
#
# Means:
# Start from the beginning and stop BEFORE split.
#
# If split = 4:
#
# X[:4] selects positions:
#
# 0, 1, 2, 3
#
# X[split:]
#
# Means:
# Start at position split and continue to the end.
#
# Therefore:
#
# X_train contains the first 4 examples.
# X_test contains the last example.
#
# Python allows assigning two variables at once:
#
# X_train, X_test = ...
#
# This is called unpacking.

X_train, X_test = X[:split], X[split:]

# Apply the SAME split to the target labels.
#
# This is extremely important.
#
# Every input example must remain paired
# with its correct label.
#
# TRAINING:
#
# X_train:
# [[10, 4], [20, 8], [15, 6], [30, 12]]
#
# y_train:
# [0, 1, 0, 1]
#
# TEST:
#
# X_test:
# [[25, 9]]
#
# y_test:
# [1]

y_train, y_test = y[:split], y[split:]

# IMPORTANT LIMITATION:
#
# This is a sequential split, not a random split.
#
# It takes the first 80% for training
# and the remaining 20% for testing.
#
# Real datasets should be split using an appropriate
# strategy for how the data was collected.
#
# For ordinary independent classification records,
# a shuffled, stratified train/test split is often useful.
#
# For time-dependent data, a chronological split
# may be more appropriate.
#
# Our simple example keeps slicing so we can
# understand Python without using external libraries.


# ------------------------------------------------------------
# PART 6: CREATE OUR LOGISTIC REGRESSION MODEL
# ------------------------------------------------------------

# Display a heading.
#
# \n means a new line.
#
# print() displays text in the console.

print("\n=== Training Logistic Regression ===")

# Create an instance of the LogisticRegression class.
#
# model is a variable containing our model object.
#
# LogisticRegression(...) calls the class constructor.
#
# Python automatically executes __init__().
#
# PARAMETERS:
#
# n_features=2:
# Each example contains two input features.
#
# learning_rate=0.1:
# Override the default learning rate of 0.01.
#
# INITIAL STATE:
#
# weights = [0.0, 0.0]
# bias = 0.0
# learning rate = 0.1
# loss history = []
#
# IMPORTANT:
# Creating the object does NOT train it.
#
# At this stage, the parameters are still zero.
#
# With zero weights and zero bias:
#
# z = 0
# sigmoid(0) = 0.5
#
# Every input initially receives probability 0.5.
#
# The model needs training to learn different predictions.

model = LogisticRegression(
    n_features=2,
    learning_rate=0.1
)


# ------------------------------------------------------------
# PART 7: TRAIN THE MODEL
# ------------------------------------------------------------

# Call the fit() method.
#
# The dot (.) means:
# Access a method belonging to an object.
#
# model.fit(...) means:
# Run fit() on this model.
#
# INPUTS:
#
# X_train:
# Training input features.
#
# y_train:
# Correct training labels.
#
# epochs=1000:
# Perform 1000 parameter updates.
#
# print_every=200:
# Display progress every 200 epochs.
#
# DURING TRAINING:
#
# 1. Compute probabilities.
# 2. Calculate errors.
# 3. Accumulate gradients.
# 4. Average gradients.
# 5. Update weights and bias.
# 6. Calculate BCE loss.
# 7. Store loss history.
# 8. Repeat.
#
# IMPORTANT:
# Only TRAINING data is used in fit().
#
# Test data is kept separate.
#
# NOTE:
# With unscaled features and a learning rate of 0.1,
# this model may take large parameter steps.
#
# Loss is not guaranteed to decrease smoothly.
#
# We keep your original parameters for this experiment,
# but feature scaling and smaller learning rates
# are important topics to investigate.

model.fit(
    X_train,
    y_train,
    epochs=1000,
    print_every=200
)


# ------------------------------------------------------------
# PART 8: EVALUATE TRAINING ACCURACY
# ------------------------------------------------------------

# Training accuracy measures performance on examples
# that the model has already seen during training.
#
# model.accuracy(X_train, y_train):
#
# 1. Predict classes for training examples.
# 2. Compare predictions with actual labels.
# 3. Count correct predictions.
# 4. Divide by total training examples.
#
# :.4f displays 4 decimal places.
#
# Example:
#
# 0.7500 means 75% accuracy.
#
# IMPORTANT:
# High training accuracy alone does not guarantee
# good performance on unseen examples.
#
# The model might overfit training data.

print(
    f"\nTrain accuracy: "
    f"{model.accuracy(X_train, y_train):.4f}"
)


# ------------------------------------------------------------
# PART 9: EVALUATE TEST ACCURACY
# ------------------------------------------------------------

# Test accuracy evaluates examples not used
# during Gradient Descent training.
#
# model.accuracy(X_test, y_test):
#
# Uses the trained weights and bias.
#
# It does NOT update model parameters.
#
# This is evaluation, not training.
#
# IMPORTANT:
# Our test set contains just ONE example.
#
# Therefore:
#
# If the model predicts correctly:
# Test accuracy = 1.0000
#
# If the model predicts incorrectly:
# Test accuracy = 0.0000
#
# Neither result alone demonstrates
# that the model will generalise well.
#
# A larger representative test dataset
# is needed for reliable evaluation.

print(
    f"Test accuracy:  "
    f"{model.accuracy(X_test, y_test):.4f}"
)


# ------------------------------------------------------------
# PART 10: DISPLAY LEARNED WEIGHTS
# ------------------------------------------------------------

# After training, the model has learned two weights.
#
# self.weights is a list.
#
# model.weights[0]:
# First learned weight.
#
# model.weights[1]:
# Second learned weight.
#
# A weight describes how its associated feature
# contributes to the weighted sum z.
#
# Positive weight:
# Increasing that feature increases z and therefore
# increases the predicted probability of class 1,
# when other features remain constant.
#
# Negative weight:
# Increasing that feature decreases z and therefore
# decreases the predicted probability of class 1,
# when other features remain constant.
#
# IMPORTANT:
# Raw weight magnitudes should not be compared
# without considering feature scales and correlations.
#
# A larger weight does not automatically mean
# a feature is more important.

print(
    f"Weights: "
    f"[{model.weights[0]:.4f}, "
    f"{model.weights[1]:.4f}]"
)


# ------------------------------------------------------------
# PART 11: DISPLAY LEARNED BIAS
# ------------------------------------------------------------

# Bias is the constant added to the weighted sum.
#
# Formula:
#
# z = w1*x1 + w2*x2 + b
#
# It influences the decision boundary.
#
# When all features are zero:
#
# z = b
#
# Therefore:
#
# sigmoid(b)
#
# is the predicted probability of class 1
# when all feature values equal zero.
#
# IMPORTANT:
# This interpretation is mathematical.
# Zero feature values may not describe
# a meaningful real-world installation.

print(f"Bias: {model.bias:.4f}")


# ------------------------------------------------------------
# PART 12: INTERPRET THE MODEL
# ------------------------------------------------------------

# Our model has now completed training.
#
# The learned equation is:
#
# z = w1*x1 + w2*x2 + b
#
# Probability = sigmoid(z)
#
# Classification:
#
# If probability >= 0.5:
#     Predict class 1
#
# Otherwise:
#     Predict class 0
#
# IMPORTANT MATHEMATICAL INSIGHT:
#
# Sigmoid(0) = 0.5
#
# Therefore, with the default threshold of 0.5,
# the decision boundary occurs when:
#
# z = 0
#
# For two features:
#
# w1*x1 + w2*x2 + b = 0
#
# This is a straight line in two-dimensional
# feature space.
#
# Even though sigmoid is nonlinear,
# standard Logistic Regression uses a linear
# decision boundary in the original features.


# ------------------------------------------------------------
# PART 13: KEY DIFFERENCES FROM LINEAR REGRESSION
# ------------------------------------------------------------

# LINEAR REGRESSION:
#
# Task:
# Predict a continuous numerical value.
#
# Examples:
# - Installation cost
# - Energy consumption
# - Installation duration
#
# Equation:
#
# y = w*x + b
#
# Output:
# Any real number.
#
# Common loss:
# Mean Squared Error (MSE).
#
# Common evaluation:
# R-squared, MAE, RMSE.
#
#
# LOGISTIC REGRESSION:
#
# Task:
# Predict the probability of a binary class.
#
# Examples:
# - Standard vs complex installation
# - Customer churn vs retention
# - Fraud vs legitimate transaction
#
# Equation:
#
# z = w1*x1 + w2*x2 + b
#
# p = sigmoid(z)
#
# Output:
# Probability between 0 and 1.
#
# Common loss:
# Binary Cross-Entropy (BCE).
#
# Common evaluation:
# Accuracy, Precision, Recall, F1-score, ROC-AUC.
#
#
# BOTH ALGORITHMS:
#
# - Are supervised learning algorithms.
# - Learn weights and bias.
# - Can be trained with Gradient Descent.
# - Use input features to make predictions.
# - Can overfit or underfit.
# - Need evaluation on unseen data.


# ------------------------------------------------------------
# PART 14: IMPORTANT EXPERIMENT LIMITATIONS
# ------------------------------------------------------------

# This project is designed for LEARNING.
#
# LIMITATION 1: VERY SMALL DATASET
#
# We have only 5 examples.
#
# Four are used for training.
# One is used for testing.
#
# This is insufficient to evaluate real-world performance.
#
#
# LIMITATION 2: NO FEATURE SCALING
#
# We use raw numerical feature values.
#
# Features with larger scales can create
# larger gradient contributions.
#
# This can make Gradient Descent less stable.
#
# Future experiments should investigate standardisation.
#
#
# LIMITATION 3: FIXED LEARNING RATE
#
# We use learning_rate=0.1.
#
# This may be too high for the current feature scales.
#
# Future experiments should compare:
#
# 0.1
# 0.01
# 0.001
#
# Compare the resulting loss curves and accuracy.
#
#
# LIMITATION 4: NO SHUFFLING
#
# We split data in its existing order.
#
# This may introduce bias if the data is ordered.
#
# A larger, representative dataset and an appropriate
# split strategy are needed for real applications.
#
#
# LIMITATION 5: ACCURACY ONLY
#
# Accuracy does not explain which types of
# classification errors the model makes.
#
# Future experiments should include:
#
# - Confusion Matrix
# - Precision
# - Recall
# - F1-score
#
#
# LIMITATION 6: NO REGULARISATION
#
# Our model does not penalise very large weights.
#
# Regularisation can help control model complexity.
#
# Future experiments can investigate L1 and L2.
#
#
# LIMITATION 7: NO EARLY STOPPING
#
# Training always runs for the requested epochs.
#
# We do not stop automatically if improvement becomes
# very small or validation performance deteriorates.
#
#
# LIMITATION 8: SYNTHETIC EXAMPLE LABELS
#
# Our labels were invented for teaching.
#
# This model must not be used for actual installation
# suitability or electrical safety decisions.
#
# Real-world applications require validated data,
# appropriate engineering rules and human oversight.
#
#
# FINAL SUMMARY:
#
# We have successfully implemented the core mechanics
# of binary Logistic Regression without scikit-learn.
#
# We now understand:
#
# 1. Sigmoid function
# 2. Weights and bias
# 3. Probability prediction
# 4. Classification threshold
# 5. Binary Cross-Entropy loss
# 6. Gradient calculation
# 7. Batch Gradient Descent
# 8. Training and test datasets
# 9. Classification accuracy
# 10. Model limitations
#
# ============================================================
# END OF LOGISTIC REGRESSION FROM SCRATCH — STUDY VERSION
# ============================================================
