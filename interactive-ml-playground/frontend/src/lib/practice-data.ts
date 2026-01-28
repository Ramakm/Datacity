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

  // Additional MATH Questions (Medium & Hard)
  {
    id: "math-4",
    title: "Eigenvalue Power Iteration",
    category: "math",
    difficulty: "hard",
    description: `Implement power iteration to find the dominant eigenvalue and eigenvector.

Power iteration is an algorithm to find the largest eigenvalue and its eigenvector.

Algorithm:
1. Start with random vector v
2. Repeat: v = A @ v, then normalize v
3. Eigenvalue λ = v.T @ A @ v

Stop when eigenvalue converges (change < tolerance).

Example:
A = [[2, 1], [1, 2]]
Dominant eigenvalue ≈ 3.0
Eigenvector ≈ [0.707, 0.707]`,
    hints: [
      "Start with a random or unit vector",
      "Multiply matrix by vector each iteration",
      "Normalize the result to prevent overflow",
      "Compute eigenvalue using Rayleigh quotient: v.T @ A @ v"
    ],
    starterCode: `import math

def power_iteration(A, num_iterations=100, tol=1e-6):
    """
    Find dominant eigenvalue and eigenvector using power iteration.

    Args:
        A: List[List[float]] - Square matrix
        num_iterations: int - Max iterations
        tol: float - Convergence tolerance

    Returns:
        Tuple[float, List[float]] - (eigenvalue, eigenvector)
    """
    # Your code here
    pass

# Test
A = [[2, 1], [1, 2]]
eigenval, eigenvec = power_iteration(A)
print(f"Eigenvalue: {eigenval}")  # Expected: ~3.0
print(f"Eigenvector: {eigenvec}")`,
    testCases: `# Test Cases
import math

eigenval, eigenvec = power_iteration([[2, 1], [1, 2]])
assert abs(eigenval - 3.0) < 0.01, f"Expected ~3.0, got {eigenval}"

# Test with identity matrix (eigenvalue = 1)
eigenval2, _ = power_iteration([[1, 0], [0, 1]])
assert abs(eigenval2 - 1.0) < 0.01

print("All tests passed!")`,
    solution: `import math

def power_iteration(A, num_iterations=100, tol=1e-6):
    n = len(A)
    # Initialize with unit vector
    v = [1.0 / math.sqrt(n)] * n
    eigenval = 0

    for _ in range(num_iterations):
        # Matrix-vector multiplication
        Av = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]

        # Normalize
        norm = math.sqrt(sum(x**2 for x in Av))
        v_new = [x / norm for x in Av]

        # Compute eigenvalue (Rayleigh quotient)
        Av_new = [sum(A[i][j] * v_new[j] for j in range(n)) for i in range(n)]
        eigenval_new = sum(v_new[i] * Av_new[i] for i in range(n))

        if abs(eigenval_new - eigenval) < tol:
            return eigenval_new, v_new

        eigenval = eigenval_new
        v = v_new

    return eigenval, v`
  },
  {
    id: "math-5",
    title: "SVD Approximation",
    category: "math",
    difficulty: "hard",
    description: `Implement rank-1 SVD approximation using power iteration.

For a matrix A, find the best rank-1 approximation A₁ = σ₁ * u₁ * v₁ᵀ

Steps:
1. Compute AᵀA
2. Use power iteration to find dominant eigenvector v₁
3. Compute u₁ = A @ v₁ / ||A @ v₁||
4. Compute σ₁ = ||A @ v₁||

Return the approximation matrix.`,
    hints: [
      "First compute A^T @ A",
      "Find the dominant eigenvector of A^T @ A",
      "This gives you the right singular vector v",
      "Compute u = A @ v and normalize",
      "Singular value σ = ||A @ v||"
    ],
    starterCode: `import math

def rank1_svd(A):
    """
    Compute rank-1 SVD approximation.

    Args:
        A: List[List[float]] - Input matrix (m x n)

    Returns:
        List[List[float]] - Rank-1 approximation
    """
    # Your code here
    pass

# Test
A = [[1, 2], [3, 4], [5, 6]]
A_approx = rank1_svd(A)
print("Original:")
for row in A: print(row)
print("Rank-1 approximation:")
for row in A_approx: print([round(x, 2) for x in row])`,
    testCases: `# Test Cases
import math

A = [[3, 0], [0, 3]]
A_approx = rank1_svd(A)

# For diagonal matrix, rank-1 should capture one direction
# Check that result is valid (not NaN or Inf)
assert all(math.isfinite(A_approx[i][j]) for i in range(2) for j in range(2))

print("All tests passed!")`,
    solution: `import math

def rank1_svd(A):
    m, n = len(A), len(A[0])

    # Compute A^T @ A
    AtA = [[sum(A[k][i] * A[k][j] for k in range(m)) for j in range(n)] for i in range(n)]

    # Power iteration for dominant eigenvector (v)
    v = [1.0 / math.sqrt(n)] * n
    for _ in range(100):
        Av = [sum(AtA[i][j] * v[j] for j in range(n)) for i in range(n)]
        norm = math.sqrt(sum(x**2 for x in Av))
        v = [x / norm for x in Av]

    # Compute u = A @ v
    Av = [sum(A[i][j] * v[j] for j in range(n)) for i in range(m)]
    sigma = math.sqrt(sum(x**2 for x in Av))
    u = [x / sigma for x in Av] if sigma > 0 else [0] * m

    # Rank-1 approximation: sigma * u @ v^T
    return [[sigma * u[i] * v[j] for j in range(n)] for i in range(m)]`
  },
  {
    id: "math-6",
    title: "Newton's Method Root Finding",
    category: "math",
    difficulty: "medium",
    description: `Implement Newton's method for finding roots of f(x) = x² - a (computing √a).

Newton's method: x_{n+1} = x_n - f(x_n) / f'(x_n)

For f(x) = x² - a:
- f'(x) = 2x
- Update: x_{n+1} = x_n - (x_n² - a) / (2 * x_n)
- Simplifies to: x_{n+1} = (x_n + a/x_n) / 2

Example:
sqrt(2) ≈ 1.414
Starting from x=1, iterations: 1.5, 1.4167, 1.4142...`,
    hints: [
      "Start with initial guess x = a or x = 1",
      "Apply the update formula iteratively",
      "Stop when |x_new - x| < tolerance",
      "The simplified formula is (x + a/x) / 2"
    ],
    starterCode: `def newton_sqrt(a, tol=1e-10, max_iter=100):
    """
    Compute square root using Newton's method.

    Args:
        a: float - Number to find square root of
        tol: float - Convergence tolerance
        max_iter: int - Maximum iterations

    Returns:
        float - Approximate square root of a
    """
    # Your code here
    pass

# Test
print(newton_sqrt(2))   # Expected: ~1.414
print(newton_sqrt(9))   # Expected: 3.0
print(newton_sqrt(100)) # Expected: 10.0`,
    testCases: `# Test Cases
import math

assert abs(newton_sqrt(2) - math.sqrt(2)) < 1e-6
assert abs(newton_sqrt(9) - 3.0) < 1e-6
assert abs(newton_sqrt(100) - 10.0) < 1e-6
assert abs(newton_sqrt(0.25) - 0.5) < 1e-6

print("All tests passed!")`,
    solution: `def newton_sqrt(a, tol=1e-10, max_iter=100):
    if a < 0:
        raise ValueError("Cannot compute sqrt of negative number")
    if a == 0:
        return 0

    x = a  # Initial guess
    for _ in range(max_iter):
        x_new = (x + a / x) / 2
        if abs(x_new - x) < tol:
            return x_new
        x = x_new

    return x`
  },

  // Additional STATS Questions (Medium & Hard)
  {
    id: "stats-4",
    title: "Covariance Matrix",
    category: "stats",
    difficulty: "medium",
    description: `Implement covariance matrix calculation for multivariate data.

Covariance matrix C where C[i][j] = Cov(X_i, X_j)

Cov(X, Y) = E[(X - μX)(Y - μY)] = Σ(x_i - μX)(y_i - μY) / n

For data with n samples and d features, result is d x d matrix.

Example:
Data: [[1,2], [3,4], [5,6]]
Cov matrix shows how each pair of features varies together.`,
    hints: [
      "First compute the mean of each feature",
      "Center the data by subtracting means",
      "Cov[i][j] = mean of (X_i - mean_i) * (X_j - mean_j)",
      "The diagonal contains variances"
    ],
    starterCode: `def covariance_matrix(data):
    """
    Compute covariance matrix.

    Args:
        data: List[List[float]] - n_samples x n_features

    Returns:
        List[List[float]] - n_features x n_features covariance matrix
    """
    # Your code here
    pass

# Test
data = [[1, 2], [3, 4], [5, 6]]
cov = covariance_matrix(data)
print("Covariance matrix:")
for row in cov:
    print([round(x, 4) for x in row])`,
    testCases: `# Test Cases
# Perfectly correlated data
data = [[1, 2], [2, 4], [3, 6]]
cov = covariance_matrix(data)
# Cov[0][1] should equal sqrt(Var[0] * Var[1]) for perfect correlation
var0 = cov[0][0]
var1 = cov[1][1]
cov01 = cov[0][1]
import math
assert abs(cov01 - math.sqrt(var0 * var1)) < 1e-6

print("All tests passed!")`,
    solution: `def covariance_matrix(data):
    n = len(data)
    d = len(data[0])

    # Compute means
    means = [sum(data[i][j] for i in range(n)) / n for j in range(d)]

    # Center data
    centered = [[data[i][j] - means[j] for j in range(d)] for i in range(n)]

    # Compute covariance matrix
    cov = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(d):
            cov[i][j] = sum(centered[k][i] * centered[k][j] for k in range(n)) / n

    return cov`
  },
  {
    id: "stats-5",
    title: "Bootstrap Confidence Interval",
    category: "stats",
    difficulty: "hard",
    description: `Implement bootstrap method to estimate confidence interval for the mean.

Bootstrap:
1. Resample with replacement from original data
2. Compute statistic (mean) for each resample
3. Use percentiles of bootstrap distribution for CI

For 95% CI, use 2.5th and 97.5th percentiles.

Example:
data = [1, 2, 3, 4, 5]
95% CI for mean ≈ [2.0, 4.0]`,
    hints: [
      "Use random.choices() for sampling with replacement",
      "Generate many (e.g., 1000) bootstrap samples",
      "Compute mean of each bootstrap sample",
      "Sort and find percentiles for CI bounds"
    ],
    starterCode: `import random

def bootstrap_ci(data, n_bootstrap=1000, confidence=0.95):
    """
    Compute bootstrap confidence interval for the mean.

    Args:
        data: List[float] - Original data
        n_bootstrap: int - Number of bootstrap samples
        confidence: float - Confidence level (e.g., 0.95)

    Returns:
        Tuple[float, float] - (lower_bound, upper_bound)
    """
    random.seed(42)  # For reproducibility
    # Your code here
    pass

# Test
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
lower, upper = bootstrap_ci(data)
print(f"95% CI: [{lower:.2f}, {upper:.2f}]")
print(f"Sample mean: {sum(data)/len(data)}")`,
    testCases: `# Test Cases
import random
random.seed(42)

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
lower, upper = bootstrap_ci(data)
mean = sum(data) / len(data)

# CI should contain the sample mean
assert lower < mean < upper, "CI should contain sample mean"

# CI should be reasonable width
assert upper - lower < 5, "CI too wide"
assert upper - lower > 0.5, "CI too narrow"

print("All tests passed!")`,
    solution: `import random

def bootstrap_ci(data, n_bootstrap=1000, confidence=0.95):
    random.seed(42)
    n = len(data)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = random.choices(data, k=n)
        bootstrap_means.append(sum(sample) / n)

    # Sort and find percentiles
    bootstrap_means.sort()
    alpha = 1 - confidence
    lower_idx = int((alpha / 2) * n_bootstrap)
    upper_idx = int((1 - alpha / 2) * n_bootstrap)

    return bootstrap_means[lower_idx], bootstrap_means[upper_idx]`
  },
  {
    id: "stats-6",
    title: "Welch's T-Test",
    category: "stats",
    difficulty: "hard",
    description: `Implement Welch's t-test for comparing means of two samples.

Welch's t-test doesn't assume equal variances.

t = (mean1 - mean2) / sqrt(s1²/n1 + s2²/n2)

Degrees of freedom (Welch-Satterthwaite):
df = (s1²/n1 + s2²/n2)² / ((s1²/n1)²/(n1-1) + (s2²/n2)²/(n2-1))

Return the t-statistic and approximate p-value.`,
    hints: [
      "Calculate means and sample variances for both groups",
      "Use sample variance: s² = Σ(x-mean)² / (n-1)",
      "Compute pooled standard error",
      "Use t-distribution for p-value (approximate with normal for large n)"
    ],
    starterCode: `import math

def welch_ttest(sample1, sample2):
    """
    Perform Welch's t-test.

    Args:
        sample1: List[float] - First sample
        sample2: List[float] - Second sample

    Returns:
        Tuple[float, float, float] - (t_statistic, degrees_of_freedom, approx_p_value)
    """
    # Your code here
    pass

# Test
group1 = [1, 2, 3, 4, 5]
group2 = [2, 3, 4, 5, 6]
t_stat, df, p_val = welch_ttest(group1, group2)
print(f"t-statistic: {t_stat:.4f}")
print(f"Degrees of freedom: {df:.2f}")
print(f"p-value (approx): {p_val:.4f}")`,
    testCases: `# Test Cases
import math

# Same means should give t close to 0
t, df, p = welch_ttest([1, 2, 3], [1, 2, 3])
assert abs(t) < 0.01, "Same samples should have t ≈ 0"

# Different means
t2, df2, p2 = welch_ttest([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
assert abs(t2) > 3, "Very different means should have large |t|"
assert p2 < 0.05, "Should be significant"

print("All tests passed!")`,
    solution: `import math

def welch_ttest(sample1, sample2):
    n1, n2 = len(sample1), len(sample2)
    mean1 = sum(sample1) / n1
    mean2 = sum(sample2) / n2

    # Sample variances (using n-1)
    var1 = sum((x - mean1)**2 for x in sample1) / (n1 - 1)
    var2 = sum((x - mean2)**2 for x in sample2) / (n2 - 1)

    # Standard error
    se = math.sqrt(var1/n1 + var2/n2)

    # t-statistic
    t_stat = (mean1 - mean2) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var1/n1 + var2/n2)**2
    denom = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
    df = num / denom

    # Approximate p-value using normal approximation (for simplicity)
    # For more accuracy, use t-distribution
    z = abs(t_stat)
    p_value = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))

    return t_stat, df, p_value`
  },
  {
    id: "stats-7",
    title: "Principal Component Analysis",
    category: "stats",
    difficulty: "hard",
    description: `Implement PCA to reduce dimensionality.

PCA steps:
1. Center the data (subtract mean)
2. Compute covariance matrix
3. Find eigenvectors (principal components)
4. Project data onto top k eigenvectors

Return the transformed data with reduced dimensions.`,
    hints: [
      "First center the data by subtracting the mean of each feature",
      "Compute the covariance matrix",
      "Use power iteration to find eigenvectors",
      "Project: X_reduced = X_centered @ V[:, :k]"
    ],
    starterCode: `import math

def pca(data, n_components=2):
    """
    Perform PCA dimensionality reduction.

    Args:
        data: List[List[float]] - n_samples x n_features
        n_components: int - Number of components to keep

    Returns:
        List[List[float]] - n_samples x n_components
    """
    # Your code here
    pass

# Test (3D data to 2D)
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
reduced = pca(data, n_components=2)
print("Reduced data:")
for row in reduced:
    print([round(x, 2) for x in row])`,
    testCases: `# Test Cases
import math

data = [[1, 2], [3, 4], [5, 6], [7, 8]]
reduced = pca(data, n_components=1)

# Should have 4 samples with 1 feature each
assert len(reduced) == 4
assert len(reduced[0]) == 1

# Variance should be preserved along principal component
values = [r[0] for r in reduced]
variance = sum((v - sum(values)/len(values))**2 for v in values) / len(values)
assert variance > 0, "Should have variance"

print("All tests passed!")`,
    solution: `import math

def pca(data, n_components=2):
    n = len(data)
    d = len(data[0])

    # Center data
    means = [sum(data[i][j] for i in range(n)) / n for j in range(d)]
    centered = [[data[i][j] - means[j] for j in range(d)] for i in range(n)]

    # Covariance matrix
    cov = [[sum(centered[k][i] * centered[k][j] for k in range(n)) / n
            for j in range(d)] for i in range(d)]

    # Find eigenvectors using power iteration (simplified)
    components = []
    cov_work = [row[:] for row in cov]

    for _ in range(min(n_components, d)):
        # Power iteration
        v = [1.0 / math.sqrt(d)] * d
        for _ in range(100):
            Av = [sum(cov_work[i][j] * v[j] for j in range(d)) for i in range(d)]
            norm = math.sqrt(sum(x**2 for x in Av))
            if norm < 1e-10:
                break
            v = [x / norm for x in Av]
        components.append(v)

        # Deflate (remove component)
        eigenval = sum(v[i] * sum(cov_work[i][j] * v[j] for j in range(d)) for i in range(d))
        for i in range(d):
            for j in range(d):
                cov_work[i][j] -= eigenval * v[i] * v[j]

    # Project data
    return [[sum(centered[i][j] * components[k][j] for j in range(d))
             for k in range(len(components))] for i in range(n)]`
  },

  // Additional ML Questions (Medium & Hard)
  {
    id: "ml-6",
    title: "Decision Tree Split",
    category: "ml",
    difficulty: "medium",
    description: `Implement the best split finder for a decision tree.

Find the feature and threshold that minimizes Gini impurity after split.

Gini = 1 - Σ(p_i²) where p_i is proportion of class i

For a split, compute weighted average of child Ginis.

Example:
Data: [(1, 'A'), (2, 'A'), (3, 'B'), (4, 'B')]
Best split at x=2.5 separates classes perfectly.`,
    hints: [
      "Try each unique value as potential threshold",
      "For each threshold, split data into left and right",
      "Compute Gini impurity for each child",
      "Weighted average: (n_left * gini_left + n_right * gini_right) / n"
    ],
    starterCode: `def gini_impurity(labels):
    """Compute Gini impurity."""
    # Your code here
    pass

def find_best_split(features, labels):
    """
    Find best split point for 1D feature.

    Args:
        features: List[float] - Feature values
        labels: List - Class labels

    Returns:
        Tuple[float, float] - (best_threshold, best_gini)
    """
    # Your code here
    pass

# Test
features = [1, 2, 3, 4]
labels = ['A', 'A', 'B', 'B']
threshold, gini = find_best_split(features, labels)
print(f"Best split at {threshold} with Gini={gini:.4f}")`,
    testCases: `# Test Cases
# Perfect split should have Gini = 0
features = [1, 2, 3, 4]
labels = ['A', 'A', 'B', 'B']
threshold, gini = find_best_split(features, labels)
assert gini < 0.01, "Should find near-perfect split"
assert 2 < threshold < 3, "Threshold should be between 2 and 3"

# Pure set has Gini = 0
assert gini_impurity(['A', 'A', 'A']) == 0

print("All tests passed!")`,
    solution: `def gini_impurity(labels):
    if len(labels) == 0:
        return 0
    from collections import Counter
    counts = Counter(labels)
    n = len(labels)
    return 1 - sum((c/n)**2 for c in counts.values())

def find_best_split(features, labels):
    combined = sorted(zip(features, labels))
    best_gini = float('inf')
    best_threshold = None

    for i in range(len(combined) - 1):
        threshold = (combined[i][0] + combined[i+1][0]) / 2
        left_labels = [l for f, l in combined if f <= threshold]
        right_labels = [l for f, l in combined if f > threshold]

        n = len(labels)
        weighted_gini = (len(left_labels) * gini_impurity(left_labels) +
                        len(right_labels) * gini_impurity(right_labels)) / n

        if weighted_gini < best_gini:
            best_gini = weighted_gini
            best_threshold = threshold

    return best_threshold, best_gini`
  },
  {
    id: "ml-7",
    title: "K-Means Clustering",
    category: "ml",
    difficulty: "medium",
    description: `Implement K-Means clustering algorithm.

Algorithm:
1. Initialize k centroids randomly
2. Assign each point to nearest centroid
3. Update centroids to mean of assigned points
4. Repeat until convergence

Return final cluster assignments and centroids.`,
    hints: [
      "Initialize centroids by randomly selecting k points",
      "Use Euclidean distance to find nearest centroid",
      "Update each centroid as mean of assigned points",
      "Converged when assignments don't change"
    ],
    starterCode: `import math
import random

def kmeans(data, k, max_iter=100):
    """
    Perform K-Means clustering.

    Args:
        data: List[List[float]] - n_samples x n_features
        k: int - Number of clusters
        max_iter: int - Maximum iterations

    Returns:
        Tuple[List[int], List[List[float]]] - (assignments, centroids)
    """
    random.seed(42)
    # Your code here
    pass

# Test
data = [[1, 1], [1, 2], [2, 1], [8, 8], [8, 9], [9, 8]]
assignments, centroids = kmeans(data, k=2)
print(f"Assignments: {assignments}")
print(f"Centroids: {centroids}")`,
    testCases: `# Test Cases
import random
random.seed(42)

data = [[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]]
assignments, centroids = kmeans(data, k=2)

# Check that we have 2 distinct clusters
assert len(set(assignments)) == 2, "Should have 2 clusters"

# First 3 points should be in same cluster
assert assignments[0] == assignments[1] == assignments[2]

# Last 3 points should be in same cluster
assert assignments[3] == assignments[4] == assignments[5]

# Different clusters
assert assignments[0] != assignments[3]

print("All tests passed!")`,
    solution: `import math
import random

def kmeans(data, k, max_iter=100):
    random.seed(42)
    n = len(data)
    d = len(data[0])

    # Initialize centroids randomly
    indices = random.sample(range(n), k)
    centroids = [data[i][:] for i in indices]

    def distance(p1, p2):
        return math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))

    assignments = [0] * n

    for _ in range(max_iter):
        # Assign points to nearest centroid
        new_assignments = []
        for point in data:
            distances = [distance(point, c) for c in centroids]
            new_assignments.append(distances.index(min(distances)))

        # Check convergence
        if new_assignments == assignments:
            break
        assignments = new_assignments

        # Update centroids
        for i in range(k):
            cluster_points = [data[j] for j in range(n) if assignments[j] == i]
            if cluster_points:
                centroids[i] = [sum(p[d] for p in cluster_points) / len(cluster_points)
                               for d in range(len(data[0]))]

    return assignments, centroids`
  },
  {
    id: "ml-8",
    title: "Logistic Regression Training",
    category: "ml",
    difficulty: "hard",
    description: `Implement logistic regression training with gradient descent.

Model: P(y=1|x) = sigmoid(w·x + b)

Loss: Binary Cross-Entropy
L = -[y*log(p) + (1-y)*log(1-p)]

Gradients:
∂L/∂w = (p - y) * x
∂L/∂b = (p - y)

Train the model and return learned weights.`,
    hints: [
      "Initialize weights to zeros",
      "For each sample, compute prediction using sigmoid",
      "Compute gradients for weights and bias",
      "Update: w = w - lr * gradient"
    ],
    starterCode: `import math

def train_logistic_regression(X, y, learning_rate=0.1, n_epochs=1000):
    """
    Train logistic regression using gradient descent.

    Args:
        X: List[List[float]] - Features (n_samples x n_features)
        y: List[int] - Labels (0 or 1)
        learning_rate: float
        n_epochs: int

    Returns:
        Tuple[List[float], float] - (weights, bias)
    """
    # Your code here
    pass

# Test
X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]
weights, bias = train_logistic_regression(X, y)
print(f"Weights: {weights}")
print(f"Bias: {bias}")`,
    testCases: `# Test Cases
import math

X = [[0], [1], [2], [3], [4], [5]]
y = [0, 0, 0, 1, 1, 1]
weights, bias = train_logistic_regression(X, y, learning_rate=0.5, n_epochs=1000)

# Predict
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

predictions = [sigmoid(X[i][0] * weights[0] + bias) for i in range(len(X))]

# First half should have low probability
assert all(p < 0.5 for p in predictions[:3]), "First samples should predict 0"

# Second half should have high probability
assert all(p > 0.5 for p in predictions[3:]), "Last samples should predict 1"

print("All tests passed!")`,
    solution: `import math

def train_logistic_regression(X, y, learning_rate=0.1, n_epochs=1000):
    n = len(X)
    d = len(X[0])

    weights = [0.0] * d
    bias = 0.0

    def sigmoid(z):
        if z >= 0:
            return 1 / (1 + math.exp(-z))
        else:
            exp_z = math.exp(z)
            return exp_z / (1 + exp_z)

    for _ in range(n_epochs):
        # Compute gradients
        dw = [0.0] * d
        db = 0.0

        for i in range(n):
            z = sum(weights[j] * X[i][j] for j in range(d)) + bias
            p = sigmoid(z)
            error = p - y[i]

            for j in range(d):
                dw[j] += error * X[i][j]
            db += error

        # Update
        for j in range(d):
            weights[j] -= learning_rate * dw[j] / n
        bias -= learning_rate * db / n

    return weights, bias`
  },
  {
    id: "ml-9",
    title: "Neural Network Forward Pass",
    category: "ml",
    difficulty: "hard",
    description: `Implement forward pass for a simple 2-layer neural network.

Architecture:
Input (d) -> Hidden (h) with ReLU -> Output (k) with Softmax

Forward pass:
1. z1 = x @ W1 + b1
2. a1 = ReLU(z1)
3. z2 = a1 @ W2 + b2
4. output = softmax(z2)

Return the output probabilities.`,
    hints: [
      "ReLU: max(0, x)",
      "Matrix multiplication: result[i][j] = sum(A[i][k] * B[k][j])",
      "Softmax with numerical stability: subtract max before exp",
      "Each step feeds into the next"
    ],
    starterCode: `import math

def forward_pass(x, W1, b1, W2, b2):
    """
    Perform forward pass through 2-layer neural network.

    Args:
        x: List[float] - Input (d features)
        W1: List[List[float]] - First layer weights (d x h)
        b1: List[float] - First layer bias (h)
        W2: List[List[float]] - Second layer weights (h x k)
        b2: List[float] - Second layer bias (k)

    Returns:
        List[float] - Output probabilities (k classes)
    """
    # Your code here
    pass

# Test
x = [1.0, 2.0]
W1 = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  # 2x3
b1 = [0.1, 0.1, 0.1]
W2 = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]  # 3x2
b2 = [0.1, 0.1]

output = forward_pass(x, W1, b1, W2, b2)
print(f"Output probabilities: {[round(p, 4) for p in output]}")
print(f"Sum: {sum(output)}")  # Should be 1.0`,
    testCases: `# Test Cases
import math

x = [1.0, 0.0]
W1 = [[1.0, 0.0], [0.0, 1.0]]
b1 = [0.0, 0.0]
W2 = [[1.0, 0.0], [0.0, 1.0]]
b2 = [0.0, 0.0]

output = forward_pass(x, W1, b1, W2, b2)

# Should be valid probability distribution
assert abs(sum(output) - 1.0) < 1e-6, "Should sum to 1"
assert all(0 <= p <= 1 for p in output), "All probs should be in [0,1]"

print("All tests passed!")`,
    solution: `import math

def forward_pass(x, W1, b1, W2, b2):
    d = len(x)
    h = len(b1)
    k = len(b2)

    # Layer 1: z1 = x @ W1 + b1
    z1 = [sum(x[i] * W1[i][j] for i in range(d)) + b1[j] for j in range(h)]

    # ReLU activation
    a1 = [max(0, z) for z in z1]

    # Layer 2: z2 = a1 @ W2 + b2
    z2 = [sum(a1[i] * W2[i][j] for i in range(h)) + b2[j] for j in range(k)]

    # Softmax
    max_z = max(z2)
    exp_z = [math.exp(z - max_z) for z in z2]
    sum_exp = sum(exp_z)
    output = [e / sum_exp for e in exp_z]

    return output`
  },
  {
    id: "ml-10",
    title: "Backpropagation",
    category: "ml",
    difficulty: "hard",
    description: `Implement backpropagation for a 2-layer neural network.

Given forward pass outputs and true labels, compute gradients.

For softmax + cross-entropy loss:
∂L/∂z2 = output - one_hot(y)

Then backpropagate:
∂L/∂W2 = a1.T @ ∂L/∂z2
∂L/∂a1 = ∂L/∂z2 @ W2.T
∂L/∂z1 = ∂L/∂a1 * (z1 > 0)  # ReLU derivative
∂L/∂W1 = x.T @ ∂L/∂z1`,
    hints: [
      "Start from the output layer gradient",
      "For softmax + cross-entropy: grad = predictions - one_hot(y)",
      "ReLU derivative: 1 if z > 0, else 0",
      "Chain rule: multiply gradients as you go backward"
    ],
    starterCode: `def backpropagation(x, y, W1, b1, W2, b2, output, z1, a1):
    """
    Compute gradients using backpropagation.

    Args:
        x: List[float] - Input
        y: int - True class label
        W1, b1, W2, b2: Network parameters
        output: List[float] - Forward pass output (softmax probs)
        z1: List[float] - Pre-activation of layer 1
        a1: List[float] - Activation of layer 1 (ReLU output)

    Returns:
        Tuple of gradients: (dW1, db1, dW2, db2)
    """
    # Your code here
    pass

# Test (would need forward pass first)
print("Implement backpropagation to compute gradients")`,
    testCases: `# Test Cases
# Simple test with known values
x = [1.0, 0.0]
y = 0
W1 = [[0.5, 0.5], [0.5, 0.5]]
b1 = [0.0, 0.0]
W2 = [[0.5, 0.5], [0.5, 0.5]]
b2 = [0.0, 0.0]

# Compute forward pass
z1 = [0.5, 0.5]  # x @ W1 + b1
a1 = [0.5, 0.5]  # ReLU
output = [0.5, 0.5]  # Softmax (equal for equal weights)

dW1, db1, dW2, db2 = backpropagation(x, y, W1, b1, W2, b2, output, z1, a1)

# Gradients should exist and be finite
assert all(all(abs(g) < 10 for g in row) for row in dW1)
assert all(abs(g) < 10 for g in db1)

print("All tests passed!")`,
    solution: `def backpropagation(x, y, W1, b1, W2, b2, output, z1, a1):
    d = len(x)
    h = len(b1)
    k = len(b2)

    # One-hot encode y
    one_hot = [0.0] * k
    one_hot[y] = 1.0

    # Output layer gradient (softmax + cross-entropy)
    dz2 = [output[i] - one_hot[i] for i in range(k)]

    # Gradients for W2 and b2
    dW2 = [[a1[i] * dz2[j] for j in range(k)] for i in range(h)]
    db2 = dz2[:]

    # Backprop to hidden layer
    da1 = [sum(dz2[j] * W2[i][j] for j in range(k)) for i in range(h)]

    # ReLU backward
    dz1 = [da1[i] if z1[i] > 0 else 0 for i in range(h)]

    # Gradients for W1 and b1
    dW1 = [[x[i] * dz1[j] for j in range(h)] for i in range(d)]
    db1 = dz1[:]

    return dW1, db1, dW2, db2`
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
