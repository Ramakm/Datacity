export interface Question {
  id: string;
  title: string;
  category: "math" | "stats" | "ml";
  difficulty: "easy" | "medium" | "hard";
  description: string;
  hints: string[];
  starterCode: string;
  testCases: string;
  solution: string;
}

export const questions: Question[] = [
  // MATH Questions
  {
    id: "math-1",
    title: "Matrix Multiplication",
    category: "math",
    difficulty: "easy",
    description: `Implement a function that multiplies two matrices A and B.

Given two 2D arrays (matrices), return their product.

Rules:
- Matrix A has dimensions (m x n)
- Matrix B has dimensions (n x p)
- Result will have dimensions (m x p)
- If matrices cannot be multiplied, return None

Example:
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
Result = [[19, 22], [43, 50]]`,
    hints: [
      "Result[i][j] = sum of A[i][k] * B[k][j] for all k",
      "Check if A's columns equal B's rows",
      "Use nested loops: outer for rows of A, middle for cols of B, inner for the dot product"
    ],
    starterCode: `def matrix_multiply(A, B):
    """
    Multiply two matrices A and B.

    Args:
        A: List[List[float]] - First matrix
        B: List[List[float]] - Second matrix

    Returns:
        List[List[float]] or None if multiplication not possible
    """
    # Your code here
    pass

# Test
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
print(matrix_multiply(A, B))  # Expected: [[19, 22], [43, 50]]`,
    testCases: `# Test Cases
assert matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]]
assert matrix_multiply([[1, 2, 3]], [[1], [2], [3]]) == [[14]]
assert matrix_multiply([[1, 2]], [[1, 2, 3]]) == None  # Incompatible dimensions
print("All tests passed!")`,
    solution: `def matrix_multiply(A, B):
    if len(A[0]) != len(B):
        return None

    m, n, p = len(A), len(A[0]), len(B[0])
    result = [[0] * p for _ in range(m)]

    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]

    return result`
  },
  {
    id: "math-2",
    title: "Gradient Descent Step",
    category: "math",
    difficulty: "medium",
    description: `Implement a single step of gradient descent for a simple quadratic function.

Given f(x) = x^2, find the gradient and update x using:
x_new = x - learning_rate * gradient

The gradient of x^2 is 2x.

Implement the function that takes current x and learning rate, returns new x.

Example:
x = 4, learning_rate = 0.1
gradient = 2 * 4 = 8
x_new = 4 - 0.1 * 8 = 3.2`,
    hints: [
      "The gradient of f(x) = x^2 is f'(x) = 2x",
      "Apply the update rule: x_new = x - lr * gradient",
      "Make sure to use the correct sign (subtract gradient)"
    ],
    starterCode: `def gradient_descent_step(x, learning_rate):
    """
    Perform one step of gradient descent for f(x) = x^2.

    Args:
        x: float - Current value
        learning_rate: float - Learning rate

    Returns:
        float - New value after gradient step
    """
    # Your code here
    pass

# Test
print(gradient_descent_step(4, 0.1))  # Expected: 3.2
print(gradient_descent_step(10, 0.05))  # Expected: 9.0`,
    testCases: `# Test Cases
assert abs(gradient_descent_step(4, 0.1) - 3.2) < 1e-6
assert abs(gradient_descent_step(10, 0.05) - 9.0) < 1e-6
assert abs(gradient_descent_step(-3, 0.1) - (-2.4)) < 1e-6
assert abs(gradient_descent_step(0, 0.1) - 0) < 1e-6
print("All tests passed!")`,
    solution: `def gradient_descent_step(x, learning_rate):
    gradient = 2 * x  # Derivative of x^2
    x_new = x - learning_rate * gradient
    return x_new`
  },
  {
    id: "math-3",
    title: "Softmax Function",
    category: "math",
    difficulty: "medium",
    description: `Implement the softmax function.

Softmax converts a vector of real numbers into a probability distribution.

Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j

For numerical stability, subtract max(x) from all elements first:
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

Example:
x = [1.0, 2.0, 3.0]
Result ≈ [0.09, 0.24, 0.67]`,
    hints: [
      "First subtract max(x) for numerical stability",
      "Compute exp() for each element",
      "Divide each by the sum of all exp values",
      "Result should sum to 1.0"
    ],
    starterCode: `import math

def softmax(x):
    """
    Compute softmax of a list of numbers.

    Args:
        x: List[float] - Input vector

    Returns:
        List[float] - Probability distribution
    """
    # Your code here
    pass

# Test
result = softmax([1.0, 2.0, 3.0])
print(result)  # Should sum to ~1.0
print(sum(result))`,
    testCases: `# Test Cases
import math

result = softmax([1.0, 2.0, 3.0])
assert abs(sum(result) - 1.0) < 1e-6, "Should sum to 1"
assert result[2] > result[1] > result[0], "Larger inputs should have larger probabilities"

result2 = softmax([0, 0, 0])
assert all(abs(r - 1/3) < 1e-6 for r in result2), "Equal inputs should give equal probabilities"

print("All tests passed!")`,
    solution: `import math

def softmax(x):
    max_x = max(x)
    exp_x = [math.exp(xi - max_x) for xi in x]
    sum_exp = sum(exp_x)
    return [e / sum_exp for e in exp_x]`
  },

  // STATS Questions
  {
    id: "stats-1",
    title: "Mean and Variance",
    category: "stats",
    difficulty: "easy",
    description: `Implement functions to calculate mean and variance of a dataset.

Mean: sum(x) / n
Variance: sum((x_i - mean)^2) / n  (population variance)

Example:
data = [2, 4, 4, 4, 5, 5, 7, 9]
mean = 5.0
variance = 4.0`,
    hints: [
      "Mean is the sum divided by count",
      "For variance, first calculate the mean",
      "Then sum the squared differences from mean",
      "Divide by n for population variance"
    ],
    starterCode: `def mean(data):
    """Calculate the arithmetic mean."""
    # Your code here
    pass

def variance(data):
    """Calculate the population variance."""
    # Your code here
    pass

# Test
data = [2, 4, 4, 4, 5, 5, 7, 9]
print(f"Mean: {mean(data)}")  # Expected: 5.0
print(f"Variance: {variance(data)}")  # Expected: 4.0`,
    testCases: `# Test Cases
data = [2, 4, 4, 4, 5, 5, 7, 9]
assert mean(data) == 5.0
assert variance(data) == 4.0

assert mean([1, 2, 3, 4, 5]) == 3.0
assert abs(variance([1, 2, 3, 4, 5]) - 2.0) < 1e-6

print("All tests passed!")`,
    solution: `def mean(data):
    return sum(data) / len(data)

def variance(data):
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / len(data)`
  },
  {
    id: "stats-2",
    title: "Z-Score Normalization",
    category: "stats",
    difficulty: "easy",
    description: `Implement z-score normalization (standardization).

Z-score transforms data to have mean=0 and std=1.

Formula: z = (x - mean) / std

Where std = sqrt(variance)

Example:
data = [1, 2, 3, 4, 5]
mean = 3, std ≈ 1.41
z-scores ≈ [-1.41, -0.71, 0, 0.71, 1.41]`,
    hints: [
      "First calculate mean and standard deviation",
      "Standard deviation is sqrt(variance)",
      "Apply the formula to each element",
      "Handle the case where std is 0"
    ],
    starterCode: `import math

def z_score_normalize(data):
    """
    Normalize data using z-score standardization.

    Args:
        data: List[float] - Input data

    Returns:
        List[float] - Normalized data with mean=0, std=1
    """
    # Your code here
    pass

# Test
data = [1, 2, 3, 4, 5]
normalized = z_score_normalize(data)
print(normalized)
print(f"Mean: {sum(normalized)/len(normalized):.6f}")  # Should be ~0`,
    testCases: `# Test Cases
import math

normalized = z_score_normalize([1, 2, 3, 4, 5])
mean_norm = sum(normalized) / len(normalized)
var_norm = sum((x - mean_norm) ** 2 for x in normalized) / len(normalized)

assert abs(mean_norm) < 1e-6, "Mean should be ~0"
assert abs(var_norm - 1.0) < 1e-6, "Variance should be ~1"

print("All tests passed!")`,
    solution: `import math

def z_score_normalize(data):
    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    std = math.sqrt(variance)

    if std == 0:
        return [0] * n

    return [(x - mean) / std for x in data]`
  },
  {
    id: "stats-3",
    title: "Correlation Coefficient",
    category: "stats",
    difficulty: "medium",
    description: `Implement Pearson correlation coefficient between two variables.

Formula:
r = Σ((x_i - mean_x)(y_i - mean_y)) / (n * std_x * std_y)

Or equivalently:
r = cov(x,y) / (std_x * std_y)

Range: -1 (perfect negative) to +1 (perfect positive)

Example:
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
r = 1.0 (perfect positive correlation)`,
    hints: [
      "Calculate means of both x and y",
      "Calculate standard deviations of both",
      "Calculate the covariance",
      "r = covariance / (std_x * std_y)"
    ],
    starterCode: `import math

def correlation(x, y):
    """
    Calculate Pearson correlation coefficient.

    Args:
        x: List[float] - First variable
        y: List[float] - Second variable

    Returns:
        float - Correlation coefficient between -1 and 1
    """
    # Your code here
    pass

# Test
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
print(correlation(x, y))  # Expected: 1.0

x2 = [1, 2, 3, 4, 5]
y2 = [5, 4, 3, 2, 1]
print(correlation(x2, y2))  # Expected: -1.0`,
    testCases: `# Test Cases
assert abs(correlation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]) - 1.0) < 1e-6
assert abs(correlation([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) - (-1.0)) < 1e-6
assert abs(correlation([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-6

print("All tests passed!")`,
    solution: `import math

def correlation(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    var_x = sum((xi - mean_x) ** 2 for xi in x) / n
    var_y = sum((yi - mean_y) ** 2 for yi in y) / n

    std_x = math.sqrt(var_x)
    std_y = math.sqrt(var_y)

    covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n

    return covariance / (std_x * std_y)`
  },

  // ML Questions
  {
    id: "ml-1",
    title: "Linear Regression Prediction",
    category: "ml",
    difficulty: "easy",
    description: `Implement prediction for simple linear regression.

Given weights (slope) and bias (intercept), predict y for input x.

Formula: y = w * x + b

For multiple features: y = w1*x1 + w2*x2 + ... + b

Example:
weights = [2.0], bias = 1.0
x = [3.0]
y = 2.0 * 3.0 + 1.0 = 7.0`,
    hints: [
      "For single feature: y = weight * x + bias",
      "For multiple features: sum of (weight_i * x_i) + bias",
      "Use zip() to pair weights with features"
    ],
    starterCode: `def linear_predict(x, weights, bias):
    """
    Make a linear regression prediction.

    Args:
        x: List[float] - Input features
        weights: List[float] - Model weights
        bias: float - Model bias/intercept

    Returns:
        float - Predicted value
    """
    # Your code here
    pass

# Test
print(linear_predict([3.0], [2.0], 1.0))  # Expected: 7.0
print(linear_predict([1.0, 2.0, 3.0], [0.5, 1.0, 1.5], 0.5))  # Expected: 7.0`,
    testCases: `# Test Cases
assert linear_predict([3.0], [2.0], 1.0) == 7.0
assert linear_predict([1.0, 2.0, 3.0], [0.5, 1.0, 1.5], 0.5) == 7.0
assert linear_predict([0.0], [5.0], 3.0) == 3.0
assert linear_predict([1.0, 1.0], [1.0, 1.0], 0.0) == 2.0

print("All tests passed!")`,
    solution: `def linear_predict(x, weights, bias):
    return sum(w * xi for w, xi in zip(weights, x)) + bias`
  },
  {
    id: "ml-2",
    title: "Sigmoid Activation",
    category: "ml",
    difficulty: "easy",
    description: `Implement the sigmoid activation function.

Sigmoid maps any real number to (0, 1), useful for binary classification.

Formula: sigmoid(x) = 1 / (1 + exp(-x))

Properties:
- sigmoid(0) = 0.5
- sigmoid(large positive) → 1
- sigmoid(large negative) → 0

Example:
sigmoid(0) = 0.5
sigmoid(2) ≈ 0.88`,
    hints: [
      "Use math.exp() for the exponential",
      "For numerical stability, handle large negative values",
      "sigmoid(-x) = 1 - sigmoid(x)"
    ],
    starterCode: `import math

def sigmoid(x):
    """
    Compute sigmoid activation.

    Args:
        x: float - Input value

    Returns:
        float - Output in range (0, 1)
    """
    # Your code here
    pass

# Test
print(sigmoid(0))  # Expected: 0.5
print(sigmoid(2))  # Expected: ~0.88
print(sigmoid(-2))  # Expected: ~0.12`,
    testCases: `# Test Cases
import math

assert abs(sigmoid(0) - 0.5) < 1e-6
assert abs(sigmoid(2) - 0.8807970779778823) < 1e-6
assert abs(sigmoid(-2) - 0.11920292202211755) < 1e-6
assert sigmoid(100) > 0.99  # Should be close to 1
assert sigmoid(-100) < 0.01  # Should be close to 0

print("All tests passed!")`,
    solution: `import math

def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        # For numerical stability with negative values
        exp_x = math.exp(x)
        return exp_x / (1 + exp_x)`
  },
  {
    id: "ml-3",
    title: "K-Nearest Neighbors",
    category: "ml",
    difficulty: "medium",
    description: `Implement a simple KNN classifier for 2D points.

Given training points with labels, predict the label for a new point
by finding the k nearest neighbors and returning the most common label.

Use Euclidean distance: sqrt((x1-x2)^2 + (y1-y2)^2)

Example:
Training: [(0,0,'A'), (1,1,'A'), (5,5,'B'), (6,6,'B')]
Query: (0.5, 0.5), k=3
Nearest 3: (0,0,'A'), (1,1,'A'), (5,5,'B')
Prediction: 'A' (majority vote)`,
    hints: [
      "Calculate distance from query to all training points",
      "Sort by distance and take k nearest",
      "Count labels and return most common",
      "Use collections.Counter for counting"
    ],
    starterCode: `import math
from collections import Counter

def knn_predict(train_data, query_point, k):
    """
    Predict label using K-Nearest Neighbors.

    Args:
        train_data: List of (x, y, label) tuples
        query_point: (x, y) tuple to classify
        k: Number of neighbors

    Returns:
        Predicted label
    """
    # Your code here
    pass

# Test
train = [(0, 0, 'A'), (1, 1, 'A'), (5, 5, 'B'), (6, 6, 'B')]
print(knn_predict(train, (0.5, 0.5), 3))  # Expected: 'A'
print(knn_predict(train, (5.5, 5.5), 3))  # Expected: 'B'`,
    testCases: `# Test Cases
train = [(0, 0, 'A'), (1, 1, 'A'), (5, 5, 'B'), (6, 6, 'B')]
assert knn_predict(train, (0.5, 0.5), 3) == 'A'
assert knn_predict(train, (5.5, 5.5), 3) == 'B'
assert knn_predict(train, (3, 3), 4) in ['A', 'B']  # Tie case

print("All tests passed!")`,
    solution: `import math
from collections import Counter

def knn_predict(train_data, query_point, k):
    distances = []
    for x, y, label in train_data:
        dist = math.sqrt((x - query_point[0])**2 + (y - query_point[1])**2)
        distances.append((dist, label))

    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]

    labels = [label for _, label in k_nearest]
    return Counter(labels).most_common(1)[0][0]`
  },
  {
    id: "ml-4",
    title: "Cross-Entropy Loss",
    category: "ml",
    difficulty: "medium",
    description: `Implement binary cross-entropy loss.

Cross-entropy measures the difference between predicted probabilities
and actual labels. Used in classification tasks.

Formula: L = -[y * log(p) + (1-y) * log(1-p)]

Where:
- y is the true label (0 or 1)
- p is the predicted probability

For a batch, return the mean loss.

Example:
y_true = [1, 0, 1]
y_pred = [0.9, 0.1, 0.8]
Loss ≈ 0.146`,
    hints: [
      "Use math.log() for natural logarithm",
      "Add small epsilon (1e-15) to avoid log(0)",
      "Calculate loss for each sample, then average",
      "Clip predictions to avoid numerical issues"
    ],
    starterCode: `import math

def binary_cross_entropy(y_true, y_pred):
    """
    Calculate binary cross-entropy loss.

    Args:
        y_true: List[int] - True labels (0 or 1)
        y_pred: List[float] - Predicted probabilities

    Returns:
        float - Mean cross-entropy loss
    """
    # Your code here
    pass

# Test
y_true = [1, 0, 1]
y_pred = [0.9, 0.1, 0.8]
print(binary_cross_entropy(y_true, y_pred))  # Expected: ~0.146`,
    testCases: `# Test Cases
import math

loss = binary_cross_entropy([1, 0, 1], [0.9, 0.1, 0.8])
assert abs(loss - 0.14462152754328741) < 1e-6

# Perfect predictions should have very low loss
loss_perfect = binary_cross_entropy([1, 0], [0.99, 0.01])
assert loss_perfect < 0.05

# Bad predictions should have high loss
loss_bad = binary_cross_entropy([1, 0], [0.1, 0.9])
assert loss_bad > 2.0

print("All tests passed!")`,
    solution: `import math

def binary_cross_entropy(y_true, y_pred):
    eps = 1e-15
    total_loss = 0

    for y, p in zip(y_true, y_pred):
        p = max(min(p, 1 - eps), eps)  # Clip to avoid log(0)
        loss = -(y * math.log(p) + (1 - y) * math.log(1 - p))
        total_loss += loss

    return total_loss / len(y_true)`
  },
  {
    id: "ml-5",
    title: "Mini-Batch Gradient Descent",
    category: "ml",
    difficulty: "hard",
    description: `Implement mini-batch gradient descent for linear regression.

Given data (X, y), optimize weights to minimize MSE loss.

Steps:
1. Split data into mini-batches
2. For each batch, compute gradients
3. Update weights: w = w - lr * gradient

MSE Loss: (1/n) * sum((y_pred - y_true)^2)
Gradient: (2/n) * X.T @ (X @ w - y)

Implement one epoch of training.`,
    hints: [
      "Create batches by slicing the data",
      "For each batch, compute predictions",
      "Compute gradient of MSE w.r.t weights",
      "Update weights using learning rate"
    ],
    starterCode: `def mini_batch_gd(X, y, weights, learning_rate, batch_size):
    """
    Perform one epoch of mini-batch gradient descent.

    Args:
        X: List[List[float]] - Features (n_samples x n_features)
        y: List[float] - Targets
        weights: List[float] - Current weights
        learning_rate: float
        batch_size: int

    Returns:
        List[float] - Updated weights
    """
    # Your code here
    pass

# Test
X = [[1, 1], [2, 2], [3, 3], [4, 4]]
y = [2, 4, 6, 8]
weights = [0.0, 0.0]
new_weights = mini_batch_gd(X, y, weights, 0.01, 2)
print(new_weights)`,
    testCases: `# Test Cases
X = [[1, 1], [2, 2], [3, 3], [4, 4]]
y = [2, 4, 6, 8]
weights = [0.0, 0.0]

# After training, weights should move towards [1, 1]
new_weights = mini_batch_gd(X, y, weights, 0.01, 2)
assert new_weights[0] > 0 and new_weights[1] > 0, "Weights should increase"

# Multiple epochs should improve
for _ in range(100):
    weights = mini_batch_gd(X, y, weights, 0.01, 2)

# Weights should be close to [1, 1] (y = x1 + x2)
assert abs(weights[0] - 1.0) < 0.5
assert abs(weights[1] - 1.0) < 0.5

print("All tests passed!")`,
    solution: `def mini_batch_gd(X, y, weights, learning_rate, batch_size):
    n_samples = len(X)
    n_features = len(weights)
    weights = weights.copy()

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        X_batch = X[start:end]
        y_batch = y[start:end]
        batch_n = len(X_batch)

        # Compute predictions
        predictions = []
        for row in X_batch:
            pred = sum(w * x for w, x in zip(weights, row))
            predictions.append(pred)

        # Compute gradients
        gradients = [0.0] * n_features
        for i in range(batch_n):
            error = predictions[i] - y_batch[i]
            for j in range(n_features):
                gradients[j] += (2 / batch_n) * error * X_batch[i][j]

        # Update weights
        for j in range(n_features):
            weights[j] -= learning_rate * gradients[j]

    return weights`
  },
];

export const categoryLabels: Record<string, string> = {
  math: "MATHEMATICS",
  stats: "STATISTICS",
  ml: "MACHINE LEARNING",
};

export const difficultyColors: Record<string, string> = {
  easy: "border-terminal-accent text-terminal-accent",
  medium: "border-terminal-warning text-terminal-warning",
  hard: "border-red-500 text-red-500",
};

export function getQuestionsByCategory(category: string): Question[] {
  if (category === "all") return questions;
  return questions.filter(q => q.category === category);
}

export function getQuestionById(id: string): Question | undefined {
  return questions.find(q => q.id === id);
}
