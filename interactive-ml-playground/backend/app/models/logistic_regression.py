import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from typing import Any


class LogisticRegressionTrainer:
    """Handles training and prediction for Logistic Regression models."""

    def __init__(self):
        self.model: LogisticRegression | None = None
        self.scaler: StandardScaler | None = None
        self.feature_columns: list[str] = []
        self.target_column: str = ""

    def train(
        self,
        data: list[dict],
        feature_columns: list[str],
        target_column: str,
        test_size: float = 0.2,
    ) -> dict[str, Any]:
        """
        Train a logistic regression model on the provided data.

        Args:
            data: List of dictionaries containing the dataset
            feature_columns: List of column names to use as features
            target_column: Name of the target column
            test_size: Fraction of data to use for testing

        Returns:
            Dictionary containing metrics, coefficients, and predictions
        """
        df = pd.DataFrame(data)

        # Validate columns exist
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Feature columns not found: {missing_features}")
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        # Prepare features and target
        X = df[feature_columns].values
        y = df[target_column].values

        # Handle any NaN values
        mask = ~(np.isnan(X).any(axis=1) | pd.isna(y))
        X = X[mask]
        y = y[mask]

        # Check for binary classification
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError(f"Logistic regression requires exactly 2 classes, found {len(unique_classes)}: {unique_classes.tolist()}")

        # Convert to binary (0 and 1)
        class_mapping = {unique_classes[0]: 0, unique_classes[1]: 1}
        y = np.array([class_mapping[val] for val in y])
        class_labels = {0: str(unique_classes[0]), 1: str(unique_classes[1])}

        if len(X) < 10:
            raise ValueError("Insufficient data after removing NaN values. Need at least 10 rows.")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(X_train_scaled, y_train)
        self.feature_columns = feature_columns
        self.target_column = target_column

        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        cm = confusion_matrix(y_test, y_pred)
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "confusion_matrix": {
                "true_negative": int(cm[0, 0]),
                "false_positive": int(cm[0, 1]),
                "false_negative": int(cm[1, 0]),
                "true_positive": int(cm[1, 1]),
            },
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

        # Prepare coefficients
        coefficients = {
            col: round(coef, 6)
            for col, coef in zip(feature_columns, self.model.coef_[0])
        }

        # Prepare scaler parameters for predictions
        scaler_params = {
            "mean": {col: float(m) for col, m in zip(feature_columns, self.scaler.mean_)},
            "scale": {col: float(s) for col, s in zip(feature_columns, self.scaler.scale_)},
        }

        # Prepare predictions for visualization
        predictions = []
        for i in range(len(y_test)):
            feature_dict = {col: float(X_test[i][j]) for j, col in enumerate(feature_columns)}
            predictions.append({
                "actual": int(y_test[i]),
                "predicted": int(y_pred[i]),
                "probability": float(y_prob[i]),
                "features": feature_dict,
            })

        # Generate Python code
        generated_code = self._generate_code(feature_columns, target_column, test_size)

        return {
            "metrics": metrics,
            "coefficients": coefficients,
            "intercept": round(float(self.model.intercept_[0]), 6),
            "scaler_params": scaler_params,
            "class_labels": class_labels,
            "predictions": predictions,
            "generated_code": generated_code,
        }

    def predict(
        self,
        data: list[dict],
        feature_columns: list[str],
        coefficients: dict[str, float],
        intercept: float,
        scaler_params: dict,
    ) -> list[dict]:
        """
        Make predictions using provided coefficients (no stored model needed).

        Args:
            data: List of dictionaries containing feature data
            feature_columns: List of feature column names
            coefficients: Dictionary mapping feature names to coefficients
            intercept: The model intercept
            scaler_params: Dictionary with mean and scale for standardization

        Returns:
            List of prediction dictionaries with class and probability
        """
        df = pd.DataFrame(data)
        predictions = []

        for _, row in df.iterrows():
            # Scale features
            scaled_features = []
            for col in feature_columns:
                mean = scaler_params["mean"].get(col, 0)
                scale = scaler_params["scale"].get(col, 1)
                scaled = (row[col] - mean) / scale
                scaled_features.append(scaled)

            # Calculate linear combination
            z = intercept
            for col, scaled_val in zip(feature_columns, scaled_features):
                z += coefficients.get(col, 0) * scaled_val

            # Apply sigmoid
            probability = 1 / (1 + np.exp(-z))
            predicted_class = 1 if probability >= 0.5 else 0

            predictions.append({
                "predicted_class": predicted_class,
                "probability": round(float(probability), 4),
            })

        return predictions

    def _generate_code(
        self,
        feature_columns: list[str],
        target_column: str,
        test_size: float,
    ) -> str:
        """Generate Python code that replicates the training process."""
        features_str = ", ".join(f'"{col}"' for col in feature_columns)

        code = f'''import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load your data
df = pd.read_csv("your_data.csv")

# Define features and target
feature_columns = [{features_str}]
target_column = "{target_column}"

# Prepare the data
X = df[feature_columns]
y = df[target_column]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={test_size}, random_state=42, stratify=y
)

# Scale features (important for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {{accuracy:.4f}}")
print(f"Precision: {{precision:.4f}}")
print(f"Recall: {{recall:.4f}}")
print(f"F1 Score: {{f1:.4f}}")
print(f"Confusion Matrix:\\n{{cm}}")
print(f"Coefficients: {{dict(zip(feature_columns, model.coef_[0]))}}")
print(f"Intercept: {{model.intercept_[0]:.4f}}")'''

        return code
