Logistic Regression — ML Engineering Study Notes
1. What Is Logistic Regression?
Logistic Regression is a supervised machine learning algorithm primarily used for classification. Despite its name, it predicts class probabilities rather than continuous numerical values.
For example, it can predict the probability that a charger installation will require additional electrical work.
Supervised learning: The model learns from historical examples containing input features and known outcomes (labels).
Classification: Predicting which predefined category an observation belongs to.
Binary classification: Predicting one of two classes, usually represented as 0 and 1.
Probability: A value between 0 and 1 expressing how likely the model considers an outcome.
Linear Regression vs Logistic Regression

Linear Regression predicts numerical quantities, whereas Logistic Regression estimates class probabilities.
A linear equation can produce any real number, including negative values or numbers greater than 1. Probabilities must remain between 0 and 1.
Logistic Regression solves this by combining a linear model with the sigmoid function.

2. How Logistic Regression Works
The prediction process follows four stages:
Input Features → Linear Score → Sigmoid Probability → Classification Decision
Input features: Numerical characteristics used by the model, such as cable length and meter distance.
Linear score: The model combines features using learned weights and bias.
Sigmoid: Converts the linear score into a probability.
Classification threshold: Converts the probability into a predicted class.
Important terminology

Weights and bias are model parameters: the model learns their values during training.
Training vs Inference
These are two different stages of an ML workflow.
Training: The model learns appropriate weights and bias from labelled data by minimising a loss function.
Inference: The trained model uses its learned parameters to generate predictions for new observations.
During normal inference, the parameters remain unchanged.

3. The Sigmoid Function
What is it?
Sigmoid is a mathematical function that converts a numerical score into a value strictly between 0 and 1.
Why is it used?
Binary Logistic Regression needs to estimate the probability of an observation belonging to class 1.
Essential formula
σ(z) = 1 / (1 + e⁻ᶻ)
Where:
z is the linear score produced by the model.
e is Euler's number, approximately 2.718.
σ(z) is the resulting probability.
Example
If the linear score is z = 2, sigmoid produces approximately 0.8808.
This means the model estimates an 88.08% probability of class 1.
With a threshold of 0.5, the predicted class is 1.
Python implementation
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

print(sigmoid(2))  # 0.8808
For production-quality code, numerical stability must also be considered, because very large exponential calculations can overflow.
Key takeaway: Sigmoid transforms the model's raw score into a probability. It does not learn parameters itself.

4. Classification Thresholds
Logistic Regression generates a probability, but the final classification depends on a decision threshold.
The most common starting threshold is 0.5.
Probability ≥ 0.5 → Class 1
Probability < 0.5 → Class 0
Business example
A model predicts a 54.98% probability that an installation requires additional electrical work.
At a threshold of 0.5, the model predicts Class 1.
However, should the business automatically arrange additional work?
Not necessarily. The prediction might instead trigger an additional review or survey.
Why threshold selection matters
Lower threshold:
Flags more potentially positive cases.
Typically increases recall.
May increase false positives and unnecessary reviews.
Higher threshold:
Flags fewer cases.
Often reduces false positives.
May increase false negatives and missed cases.
The best threshold depends on the business consequences of incorrect decisions.
Changing the threshold changes class predictions without retraining the model or changing its underlying predicted probabilities.

5. Binary Cross-Entropy (Log Loss)
What is a loss function?
A loss function measures how wrong a model's predictions are.
In Linear Regression, we commonly use Mean Squared Error (MSE).
For binary Logistic Regression, we commonly use Binary Cross-Entropy (BCE), also called Log Loss.
BCE evaluates the quality of predicted probabilities rather than only whether the final class prediction is correct.
How it works
Suppose the actual label is Class 1.

BCE strongly penalises a model when it assigns very low probability to the correct class.
Essential formula
L = −[y log(p) + (1 − y) log(1 − p)]

For an entire dataset, we calculate the average loss across observations.
Practical implementations must prevent undefined logarithm calculations when probabilities are exactly 0 or 1, for example by using clipping or numerically stable loss functions.
Lower BCE indicates that the model assigns more probability to the correct labels on the evaluated dataset.

6. Gradient Descent — How Logistic Regression Learns
Logistic Regression and Linear Regression use the same general parameter-optimisation principle.
Gradient Descent is an optimisation algorithm used to adjust model parameters to reduce loss.
Training cycle
Make predictions: Use the current weights and bias to calculate probabilities.
Calculate loss: Use Binary Cross-Entropy to measure prediction errors.
Calculate gradients: Determine how each parameter affects the loss.
Update parameters: Adjust weights and bias in the direction that reduces loss locally.
Repeat: Continue over multiple epochs until a stopping condition is reached.
Important terminology

Example parameter update
self.w -= self.lr * dw
self.b -= self.lr * db
Where:
self.w and self.b are model parameters.
self.lr is the learning rate.
dw and db are gradients.
Important engineering observations
Gradient Descent uses calculated gradients, not random parameter changes.
A learning rate that is too small can make training slow.
A learning rate that is too large can cause unstable training or divergence.
Training normally requires multiple updates.
Feature scaling can improve gradient-based optimisation.
Training loss should be monitored to identify convergence or unstable behaviour.
Parameters vs Hyperparameters
Parameters are learned by the model, such as weights and bias.
Hyperparameters are chosen or tuned by the engineer, such as learning rate and number of epochs.

7. The Decision Boundary
A decision boundary separates regions where a classifier predicts different classes.
For standard binary Logistic Regression with two input features and a 0.5 threshold, the boundary is a straight line.
A decision region is an area of the feature space where the model predicts the same class.
How it works
One side of the boundary predicts Class 0.
The other side predicts Class 1.
At the default boundary, the model estimates a 50% probability.
Observations near the boundary have probabilities close to 0.5.
The learned weights and bias determine the position and orientation of the boundary.
Engineering limitations
Standard Logistic Regression creates a linear decision boundary in the supplied feature space.
If the relationship between the input features and target is highly nonlinear, a straight line may not separate the classes effectively.
Possible approaches include:
Adding polynomial or interaction features.
Trying nonlinear classifiers.
Comparing models using validation performance.
A decision boundary is the model's classification rule. It does not guarantee perfect separation or represent a physical boundary in the real world.

8. Multi-Class Classification with Softmax
Binary Logistic Regression predicts between two classes.
Multi-class classification involves predicting one of three or more mutually exclusive categories.
For example:
Class 0: Standard installation
Class 1: Additional electrical work
Class 2: Detailed site assessment
Binary vs Multi-Class vs Multi-Label

A property might require both additional electrical work and a site assessment. If the model must predict both simultaneously, that is a multi-label problem rather than mutually exclusive multi-class classification.
What is Softmax?
Softmax converts multiple raw class scores (logits) into a probability distribution.
Each class receives a probability, and the probabilities add up to 1.
Example

The predicted class is the one with the highest probability.
Argmax
Argmax identifies the index of the largest value.
For example:
probabilities = [0.659, 0.242, 0.099]

predicted_class = probabilities.index(max(probabilities))

print(predicted_class)  # 0
The model predicts Class 0 because it has the highest probability.
Softmax vs Sigmoid
Sigmoid: Converts one score into a binary class probability.
Softmax: Converts multiple class scores into probabilities that sum to 1.
In multi-label classification, independent sigmoid outputs may be used instead of Softmax because more than one label can be positive.

9. One-Hot Encoding and Categorical Cross-Entropy
One-hot encoding
One-hot encoding represents a class label using a list of zeros with a single 1 indicating the correct class.

One-hot encoded targets can be used when training a multi-class classifier.
Categorical Cross-Entropy (CCE)
CCE is commonly used to measure the quality of predicted probabilities in multi-class classification.
It penalises models that assign low probability to the correct class.
For a single observation with a one-hot target, the loss is:
Loss = -log(probability assigned to the correct class)
BCE is commonly used for binary classification, while CCE is commonly used with Softmax for mutually exclusive multi-class classification.

10. Model Evaluation — How Do We Know If It Works?
Training a model is only part of ML engineering. We also need to evaluate how it performs on data it has not seen during training.
Train / Validation / Test
Training set: Used to learn model parameters.
Validation set: Used to tune hyperparameters, thresholds and model choices.
Test set: Used for a final evaluation on previously unseen examples.
A model can achieve low training loss but perform poorly on unseen data. This may indicate overfitting.
Classification metrics

Confusion Matrix
A confusion matrix breaks predictions into four outcomes.

Business interpretation:
A false positive may trigger an unnecessary installation review.
A false negative may fail to identify a property requiring additional work.
The business must determine which error is more costly before selecting a classification threshold.
Class Imbalance
Class imbalance happens when one category appears much more frequently than another.
For example, if only 10% of installations require additional work, a model predicting "No additional work" for every property would achieve 90% accuracy but identify none of the actual positive cases.
In imbalanced classification problems, accuracy alone can be misleading.
Precision, recall, F1-score, PR-AUC and the confusion matrix can provide a more useful evaluation.

11. Practical ML Engineering — What Should I Remember?
When building Logistic Regression models:
Understand the business problem: Identify the target classes and cost of incorrect decisions.
Prepare the data: Check missing values, feature types, distributions and data quality.
Split the dataset: Separate training and evaluation data to avoid leakage.
Scale features when appropriate: Particularly when using gradient-based optimisation or regularisation.
Train a baseline model: Start with a simple Logistic Regression implementation.
Monitor training: Track BCE loss and check convergence.
Evaluate properly: Use metrics that reflect the business objective.
Tune the threshold: Consider false-positive and false-negative trade-offs.
Compare experiments: Test changes systematically, including regularisation and learning rates.
Document results: Record data preparation, model configuration, metrics, limitations and decisions.
Important limitations
Logistic Regression is a strong baseline because it is relatively simple, computationally efficient and interpretable.
However:
It models linear relationships in the log-odds of the target.
It may struggle with complex nonlinear patterns without feature engineering.
It can be affected by extreme feature values and poorly prepared data.
Regularisation may be needed to control model complexity.
Predicted probabilities should be assessed for calibration when they are used in important decisions.

12. Key Takeaways
Logistic Regression is a supervised classification method that estimates class probabilities.
Sigmoid converts a linear score into a binary class probability.
Binary Cross-Entropy measures how well predicted probabilities match actual binary labels.
Gradient Descent adjusts weights and bias to minimise training loss.
Decision Boundary separates the regions where the classifier predicts different classes.
Softmax extends the probability-based approach to mutually exclusive multi-class classification.
Categorical Cross-Entropy measures multi-class probability prediction error.
Evaluation metrics help determine whether the model performs well and whether its decisions are useful for the business.
Final ML Engineering Principle
A successful classification model is not simply one that achieves high accuracy. It must make reliable predictions on unseen data, minimise the errors that matter to the business, and be evaluated, documented and monitored appropriately.

