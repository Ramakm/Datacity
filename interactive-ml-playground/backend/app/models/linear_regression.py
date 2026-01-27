import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Any


class LinearRegressionTrainer:
    """Handles training and prediction for Linear Regression models."""

    def __init__(self):
        self.model: LinearRegression | None = None
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
        Train a linear regression model on the provided data.

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
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]

        if len(X) < 10:
            raise ValueError("Insufficient data after removing NaN values. Need at least 10 rows.")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Train model
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        self.feature_columns = feature_columns
        self.target_column = target_column

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        metrics = {
            "r2_score": round(r2_score(y_test, y_pred), 4),
            "mse": round(mean_squared_error(y_test, y_pred), 4),
            "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            "mae": round(mean_absolute_error(y_test, y_pred), 4),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

        # Prepare coefficients
        coefficients = {
            col: round(coef, 6)
            for col, coef in zip(feature_columns, self.model.coef_)
        }

        # Prepare predictions for visualization
        predictions = []
        for i in range(len(y_test)):
            feature_dict = {col: float(X_test[i][j]) for j, col in enumerate(feature_columns)}
            predictions.append({
                "actual": float(y_test[i]),
                "predicted": float(y_pred[i]),
                "features": feature_dict,
            })

        # Generate Python code
        generated_code = self._generate_code(feature_columns, target_column, test_size)

        return {
            "metrics": metrics,
            "coefficients": coefficients,
            "intercept": round(float(self.model.intercept_), 6),
            "predictions": predictions,
            "generated_code": generated_code,
        }

    def predict(
        self,
        data: list[dict],
        feature_columns: list[str],
        coefficients: dict[str, float],
        intercept: float,
    ) -> list[float]:
        """
        Make predictions using provided coefficients (no stored model needed).

        Args:
            data: List of dictionaries containing feature data
            feature_columns: List of feature column names
            coefficients: Dictionary mapping feature names to coefficients
            intercept: The model intercept

        Returns:
            List of predicted values
        """
        df = pd.DataFrame(data)
        predictions = []

        for _, row in df.iterrows():
            pred = intercept
            for col in feature_columns:
                pred += coefficients.get(col, 0) * row[col]
            predictions.append(round(float(pred), 4))

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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

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
    X, y, test_size={test_size}, random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² Score: {{r2:.4f}}")
print(f"RMSE: {{rmse:.4f}}")
print(f"Coefficients: {{dict(zip(feature_columns, model.coef_))}}")
print(f"Intercept: {{model.intercept_:.4f}}")'''

        return code
