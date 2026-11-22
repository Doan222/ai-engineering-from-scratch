# Logistic Regression — From Scratch to Scikit-learn

## Project Overview

This project explores **Logistic Regression**, a supervised machine learning algorithm primarily used for classification.

The objective is to understand how Logistic Regression works, implement its core components from scratch using Python, and compare the implementation with Scikit-learn.

This project is part of my **AI Engineering from Scratch** learning portfolio.

### Learning Objectives

- Understand supervised learning and binary classification.
- Learn how sigmoid converts linear scores into probabilities.
- Understand Binary Cross-Entropy and Gradient Descent.
- Implement a reusable Logistic Regression model in Python.
- Explore decision boundaries and classification thresholds.
- Understand multi-class classification using Softmax.
- Evaluate models using classification metrics.
- Compare from-scratch and Scikit-learn implementations.

### Business Context

A potential application is predicting whether a home-charger installation requires additional electrical work.

For example:

- **Class 0:** No additional work required.
- **Class 1:** Additional work required.

The project uses synthetic data for learning purposes. It does not represent real customer records or a validated installation assessment model.

---

## 1. What Is Logistic Regression?

**Logistic Regression is a supervised machine learning algorithm used for classification.**

Despite its name, it estimates class probabilities rather than continuous numerical targets.

### Important Terminology

- **Supervised Learning:** The model learns from examples containing input features and known outcomes.
- **Classification:** Predicting which predefined category an observation belongs to.
- **Binary Classification:** Predicting one of two classes, usually represented as 0 and 1.
- **Probability:** A value between 0 and 1 representing an estimated likelihood.

### Linear Regression vs Logistic Regression

| Component | Linear Regression | Logistic Regression |
|---|---|---|
| Problem | Regression | Classification |
| Output | Continuous numerical value | Class probability |
| Example | Installation cost: £850 | Probability of extra work: 85% |
| Output range | Unbounded | Between 0 and 1 |
| Common loss | Mean Squared Error | Binary Cross-Entropy |
| Evaluation | MAE, MSE, R² | Accuracy, precision, recall, F1, log loss |

A linear equation can produce any real number, but a probability must be between 0 and 1.

Logistic Regression addresses this by applying the **sigmoid function** to a linear score.

---

## 2. How Logistic Regression Works

The prediction process follows four stages:

**Input Features → Linear Score → Sigmoid Probability → Classification Decision**

1. **Input Features:** Numerical characteristics used to make predictions.
2. **Linear Score:** The model combines features using learned weights and bias.
3. **Sigmoid:** Converts the numerical score into a probability.
4. **Classification Threshold:** Converts the probability into a predicted class.

### Important Terminology

| Term | Meaning |
|---|---|
| Feature (X) | An input variable used for prediction |
| Label (y) | The correct outcome in the dataset |
| Weight (w) | A learned parameter controlling a feature's influence |
| Bias (b) | A learned parameter shifting the linear score |
| Logit (z) | The raw linear score |
| Probability (p) | Estimated probability of class 1 |
| Threshold | Cutoff used to determine the predicted class |

### Training vs Inference

**Training:** The model learns weights and bias from labelled data by minimising a loss function.

**Inference:** The trained model uses its learned parameters to make predictions on new observations.

During normal inference, the model parameters remain unchanged.

---

## 3. The Sigmoid Function

### What Is Sigmoid?

Sigmoid is a mathematical function that converts a numerical score into a value strictly between 0 and 1.

Binary Logistic Regression uses sigmoid to estimate the probability of an observation belonging to class 1.

### Essential Formula

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Where:

- `z` represents the linear score.
- `e` is Euler's number, approximately 2.718.
- `σ(z)` represents the resulting probability.

### Example

If `z = 2`, sigmoid produces approximately `0.8808`.

This represents an estimated **88.08% probability of class 1**.

With a classification threshold of 0.5, the predicted class is 1.

### Python Implementation

```python
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

print(sigmoid(2))  # Approximately 0.8808
```

For numerical stability, practical implementations must handle extreme input values that could cause exponential overflow.

**Key Takeaway:** Sigmoid converts a linear score into a probability. It does not learn parameters itself.

---

## 4. Classification Thresholds

Logistic Regression predicts a probability, but the final classification depends on a decision threshold.

The most common starting threshold is `0.5`.

```python
prediction = 1 if probability >= 0.5 else 0
```

- Probability ≥ 0.5 → Class 1
- Probability below 0.5 → Class 0

### Business Example

Suppose a model predicts a 54.98% probability that an installation requires additional electrical work.

At a threshold of 0.5, the model predicts Class 1.

However, this does not necessarily mean the business should automatically arrange additional work. The prediction might instead trigger an additional review.

### Why Threshold Selection Matters

**Lower threshold:**

- Flags more potentially positive cases.
- Generally increases recall.
- May increase false positives and unnecessary reviews.

**Higher threshold:**

- Flags fewer cases.
- May reduce false positives.
- Can increase false negatives and missed cases.

The most suitable threshold depends on the business consequences of incorrect predictions.

**Key Takeaway:** Changing the classification threshold changes the predicted classes without retraining the model or changing its underlying probabilities.

---

## 5. Binary Cross-Entropy (Log Loss)

### What Is a Loss Function?

A loss function measures how wrong a model's predictions are.

In Linear Regression, Mean Squared Error (MSE) is commonly used.

For binary Logistic Regression, **Binary Cross-Entropy (BCE)**, also called Log Loss, is commonly used.

BCE evaluates the quality of predicted probabilities rather than only whether the final classification is correct.

### Example

Suppose the actual label is Class 1.

| Predicted probability | BCE loss | Interpretation |
|---|---|---|
| 0.99 | 0.010 | Very good prediction |
| 0.80 | 0.223 | Good prediction |
| 0.50 | 0.693 | Uncertain prediction |
| 0.10 | 2.303 | Confidently wrong prediction |

BCE strongly penalises a model when it assigns very low probability to the correct class.

### Essential Formula

$$
L=-[y\log(p)+(1-y)\log(1-p)]
$$

Where:

| Symbol | Meaning |
|---|---|
| L | Loss for one observation |
| y | Actual label, either 0 or 1 |
| p | Predicted probability of class 1 |
| log | Natural logarithm |
| 1 − p | Predicted probability of class 0 |

For an entire dataset, the average loss is calculated across observations.

In practical implementations, numerical stability is important because the logarithm of zero is undefined.

**Key Takeaway:** Lower BCE indicates that the model assigns more probability to the correct labels on the evaluated dataset.

---

## 6. Gradient Descent — How Logistic Regression Learns

Logistic Regression and Linear Regression use the same general parameter-optimisation principle.

**Gradient Descent** is an optimisation algorithm that adjusts model parameters to reduce loss.

### Training Process

1. **Make Predictions:** Calculate probabilities using current weights and bias.
2. **Calculate Loss:** Measure prediction errors using Binary Cross-Entropy.
3. **Calculate Gradients:** Determine how each parameter affects loss.
4. **Update Parameters:** Adjust weights and bias in the opposite direction of their gradients.
5. **Repeat:** Continue over multiple epochs until a stopping condition is reached.

### Important Terminology

| Term | Meaning |
|---|---|
| Gradient | Measures how changing a parameter affects loss |
| Gradient Descent | Algorithm for updating parameters to minimise loss |
| Learning Rate | Controls the size of parameter updates |
| Epoch | One complete pass through the training dataset |
| Iteration | One parameter update |
| Convergence | When optimisation approaches a stable solution |

### Parameter Updates

```python
self.w -= self.lr * dw
self.b -= self.lr * db
```

Where:

- `self.w` and `self.b` are the model parameters.
- `self.lr` is the learning rate.
- `dw` and `db` are gradients.

For a multi-feature implementation, each weight is updated using its corresponding gradient.

### Parameters vs Hyperparameters

**Parameters:** Values learned by the model, such as weights and bias.

**Hyperparameters:** Values chosen by the engineer, such as learning rate and number of epochs.

### Engineering Observations

- Gradient Descent uses calculated gradients rather than random parameter changes.
- A very small learning rate can lead to slow training.
- A very large learning rate can cause unstable optimisation.
- Training usually requires multiple parameter updates.
- Feature scaling can improve optimisation.
- Monitoring training loss helps identify convergence problems.

---

## 7. The Decision Boundary

A **decision boundary** separates regions where a classifier predicts different classes.

A **decision region** is an area where the model produces the same predicted class.

For standard binary Logistic Regression with two input features and a threshold of 0.5, the decision boundary is a straight line.

### How It Works

- One side of the boundary predicts Class 0.
- The other side predicts Class 1.
- At the boundary, the estimated probability is 0.5.
- Observations close to the boundary have probabilities close to 0.5.

The learned weights and bias determine the boundary's position and orientation.

### Engineering Limitations

Standard Logistic Regression produces a linear decision boundary in its supplied feature space.

If the relationships between features and labels are complex or nonlinear, a straight line may not separate the classes effectively.

Possible approaches include:

- Adding polynomial or interaction features.
- Testing nonlinear classifiers.
- Comparing models using validation performance.

**Key Takeaway:** A decision boundary separates predicted classes but does not guarantee perfect classification accuracy.

---

## 8. Multi-Class Classification with Softmax

Binary classification predicts one of two classes.

**Multi-class classification** predicts one of three or more mutually exclusive categories.

### Example

- Class 0: Standard installation.
- Class 1: Additional electrical work.
- Class 2: Detailed site assessment.

### Classification Types

| Type | Meaning |
|---|---|
| Binary | One of two possible classes |
| Multi-class | One of three or more mutually exclusive classes |
| Multi-label | Multiple labels may apply to the same observation |

For example, if a property can require both electrical work and a site assessment, predicting both independently becomes a multi-label problem.

### What Is Softmax?

Softmax converts multiple raw class scores, called logits, into a probability distribution.

The probabilities across the classes add up to 1.

### Example

| Class | Raw score | Probability |
|---|---|---|
| Standard installation | 2.0 | 65.9% |
| Additional electrical work | 1.0 | 24.2% |
| Detailed site assessment | 0.1 | 9.9% |
| **Total** | | **100%** |

The predicted class is the one with the highest probability.

### Argmax

Argmax returns the index of the highest value.

```python
probabilities = [0.659, 0.242, 0.099]

predicted_class = probabilities.index(max(probabilities))

print(predicted_class)  # 0
```

### Softmax vs Sigmoid

- **Sigmoid:** Converts a score into a binary class probability.
- **Softmax:** Converts multiple scores into probabilities that sum to 1.

In multi-label classification, independent sigmoid outputs can be used because more than one label may be positive.

---

## 9. One-Hot Encoding and Categorical Cross-Entropy

### One-Hot Encoding

One-hot encoding represents a class using a list of zeros with one position containing a 1.

| Actual class | One-hot representation |
|---|---|
| Class 0 | `[1, 0, 0]` |
| Class 1 | `[0, 1, 0]` |
| Class 2 | `[0, 0, 1]` |

These representations can be used as targets when training a multi-class model.

### Categorical Cross-Entropy (CCE)

CCE is commonly used to measure probability prediction errors in multi-class classification.

It penalises models that assign low probability to the correct category.

For one observation with a one-hot target:

`Loss = -log(probability assigned to the correct class)`

**Key Takeaway:** BCE is commonly used for binary classification, while CCE is commonly used with Softmax for multi-class classification.

---

## 10. Model Evaluation — How Do We Know If It Works?

Training a model is only part of ML engineering.

We must also evaluate whether it performs well on data it has not seen during training.

### Training, Validation and Test Data

- **Training Set:** Used to learn model parameters.
- **Validation Set:** Used to select models, tune hyperparameters and choose thresholds.
- **Test Set:** Used for final evaluation on previously unseen observations.

A model can achieve low training loss but perform poorly on new data, which may indicate overfitting.

### Classification Metrics

| Metric | What It Measures |
|---|---|
| Accuracy | Proportion of all predictions that are correct |
| Precision | Proportion of predicted positives that are actually positive |
| Recall | Proportion of actual positives correctly identified |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Ability of model scores to rank positive cases above negative cases |
| Log Loss | Quality of predicted probabilities |

### Confusion Matrix

A confusion matrix breaks classification results into four outcomes.

| | Actual Positive | Actual Negative |
|---|---|---|
| Predicted Positive | True Positive (TP) | False Positive (FP) |
| Predicted Negative | False Negative (FN) | True Negative (TN) |

### Business Interpretation

- **False Positive:** May trigger an unnecessary installation review.
- **False Negative:** May fail to identify an installation requiring additional work.

The business must determine which type of error is more costly.

### Class Imbalance

Class imbalance occurs when some classes appear much more frequently than others.

For example, if only 10% of installations require additional work, a model predicting "No additional work" for every installation would achieve 90% accuracy while missing every positive case.

Accuracy alone can therefore be misleading.

Precision, recall, F1-score, PR-AUC and the confusion matrix may provide more useful information.

---

## 11. Practical ML Engineering Considerations

When developing a Logistic Regression model:

1. **Understand the business problem:** Define the target classes and consequences of incorrect predictions.
2. **Prepare the dataset:** Check missing values, feature types, distributions and data quality.
3. **Split the dataset:** Separate training and evaluation data to avoid leakage.
4. **Scale features when appropriate:** Particularly when using gradient-based optimisation or regularisation.
5. **Train a baseline:** Start with a simple Logistic Regression model.
6. **Monitor training:** Track BCE loss and convergence.
7. **Evaluate appropriately:** Choose metrics relevant to the business objective.
8. **Tune thresholds:** Balance false positives and false negatives.
9. **Compare experiments:** Test model configurations systematically.
10. **Document findings:** Record parameters, metrics, limitations and engineering decisions.

### Important Limitations

Logistic Regression is a useful baseline because it is relatively simple, efficient and interpretable.

However:

- It models a linear relationship between features and log-odds.
- Complex nonlinear relationships may require feature engineering or another model.
- Extreme values and poorly prepared data can affect model behaviour.
- Regularisation may help control model complexity.
- Probability calibration should be assessed when probabilities inform important decisions.

---

## 12. Key Takeaways

- **Logistic Regression:** Supervised classification algorithm that estimates class probabilities.
- **Sigmoid:** Converts a linear score into a binary class probability.
- **Binary Cross-Entropy:** Measures prediction quality for binary probabilistic classification.
- **Gradient Descent:** Updates weights and bias to minimise loss.
- **Decision Boundary:** Separates regions where different classes are predicted.
- **Softmax:** Produces probabilities for mutually exclusive multi-class classification.
- **Categorical Cross-Entropy:** Measures multi-class probability prediction loss.
- **Evaluation Metrics:** Help determine whether a model performs well and meets business objectives.

### Final ML Engineering Principle

A successful classifier is not simply one that achieves high accuracy.

It should perform reliably on unseen data, minimise business-relevant errors, and be evaluated, documented and monitored appropriately.

---

## Implementation and Experiments

### Planned Repository Files

```text
logistic-regression/
├── README.md
├── experiments.md
├── logistic_regression_scratch.py
├── logistic_regression_sklearn.py
└── requirements.txt
```

### Experiment Status

| Experiment | Status |
|---|---|
| Logistic Regression from scratch | Pending complete implementation and evaluation |
| Scikit-learn Logistic Regression | Pending |
| Learning rate and epoch comparison | Pending documented results |
| Classification threshold comparison | Pending |
| Precision, recall, F1 and confusion matrix | Pending |
| ROC-AUC evaluation | Pending |
| From-scratch vs Scikit-learn comparison | Pending |

Results will be added after running and evaluating the implementations.

## References

- [AI Engineering from Scratch — Logistic Regression](https://aiengineeringfromscratch.com/learn/ml-fundamentals/logistic-regression)
- [Scikit-learn — LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
