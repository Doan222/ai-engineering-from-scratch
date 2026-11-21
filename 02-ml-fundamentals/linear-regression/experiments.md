
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
