import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Any


class KNNTrainer:
    """Handles training and prediction for K-Nearest Neighbors models."""

    def __init__(self):
        self.model: KNeighborsClassifier | None = None
        self.scaler: StandardScaler | None = None
        self.label_encoder: LabelEncoder | None = None
        self.feature_columns: list[str] = []
        self.target_column: str = ""

    def train(
        self,
        data: list[dict],
        feature_columns: list[str],
        target_column: str,
        n_neighbors: int = 5,
        test_size: float = 0.2,
    ) -> dict[str, Any]:
        """
        Train a KNN model on the provided data.

        Args:
            data: List of dictionaries containing the dataset
            feature_columns: List of column names to use as features
            target_column: Name of the target column
            n_neighbors: Number of neighbors to use
            test_size: Fraction of data to use for testing

        Returns:
            Dictionary containing metrics, parameters, and predictions
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

        # Handle any NaN values in features
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]

        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        class_labels = {int(i): str(label) for i, label in enumerate(self.label_encoder.classes_)}
        n_classes = len(class_labels)

        if len(X) < 10:
            raise ValueError("Insufficient data after removing NaN values. Need at least 10 rows.")

        # Validate n_neighbors
        if n_neighbors >= len(X):
            n_neighbors = max(1, len(X) // 2)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model.fit(X_train_scaled, y_train)
        self.feature_columns = feature_columns
        self.target_column = target_column

        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)

        # Calculate metrics
        average = 'binary' if n_classes == 2 else 'weighted'
        cm = confusion_matrix(y_test, y_pred)
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, average=average, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, average=average, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, average=average, zero_division=0), 4),
            "n_neighbors": n_neighbors,
            "n_classes": n_classes,
            "confusion_matrix": cm.tolist(),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

        # Prepare scaler parameters for predictions
        scaler_params = {
            "mean": {col: float(m) for col, m in zip(feature_columns, self.scaler.mean_)},
            "scale": {col: float(s) for col, s in zip(feature_columns, self.scaler.scale_)},
        }

        # Store training data for KNN (needed for predictions)
        training_data = {
            "X": X_train_scaled.tolist(),
            "y": y_train.tolist(),
        }

        # Prepare predictions for visualization
        predictions = []
        for i in range(len(y_test)):
            feature_dict = {col: float(X_test[i][j]) for j, col in enumerate(feature_columns)}
            prob_dict = {class_labels[j]: round(float(p), 4) for j, p in enumerate(y_prob[i])}
            predictions.append({
                "actual": int(y_test[i]),
                "actual_label": class_labels[int(y_test[i])],
                "predicted": int(y_pred[i]),
                "predicted_label": class_labels[int(y_pred[i])],
                "probabilities": prob_dict,
                "features": feature_dict,
            })

        # Generate Python code
        generated_code = self._generate_code(feature_columns, target_column, n_neighbors, test_size)

        return {
            "metrics": metrics,
            "scaler_params": scaler_params,
            "training_data": training_data,
            "class_labels": class_labels,
            "n_neighbors": n_neighbors,
            "predictions": predictions,
            "generated_code": generated_code,
        }

    def predict(
        self,
        data: list[dict],
        feature_columns: list[str],
        training_data: dict,
        scaler_params: dict,
        class_labels: dict,
        n_neighbors: int,
    ) -> list[dict]:
        """
        Make predictions using stored training data (KNN requires training data).

        Args:
            data: List of dictionaries containing feature data
            feature_columns: List of feature column names
            training_data: Dictionary with X and y training data
            scaler_params: Dictionary with mean and scale for standardization
            class_labels: Dictionary mapping class indices to labels
            n_neighbors: Number of neighbors to use

        Returns:
            List of prediction dictionaries with class and probabilities
        """
        df = pd.DataFrame(data)

        # Recreate the model with training data
        X_train = np.array(training_data["X"])
        y_train = np.array(training_data["y"])

        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)

        predictions = []

        for _, row in df.iterrows():
            # Scale features
            scaled_features = []
            for col in feature_columns:
                mean = scaler_params["mean"].get(col, 0)
                scale = scaler_params["scale"].get(col, 1)
                scaled = (row[col] - mean) / scale
                scaled_features.append(scaled)

            X_new = np.array([scaled_features])
            pred = model.predict(X_new)[0]
            prob = model.predict_proba(X_new)[0]

            # Convert class_labels keys to int for lookup
            class_labels_int = {int(k): v for k, v in class_labels.items()}
            prob_dict = {class_labels_int[j]: round(float(p), 4) for j, p in enumerate(prob)}

            predictions.append({
                "predicted_class": int(pred),
                "predicted_label": class_labels_int[int(pred)],
                "probabilities": prob_dict,
            })

        return predictions

    def _generate_code(
        self,
        feature_columns: list[str],
        target_column: str,
        n_neighbors: int,
        test_size: float,
    ) -> str:
        """Generate Python code that replicates the training process."""
        features_str = ", ".join(f'"{col}"' for col in feature_columns)

        code = f'''import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load your data
df = pd.read_csv("your_data.csv")

# Define features and target
feature_columns = [{features_str}]
target_column = "{target_column}"

# Prepare the data
X = df[feature_columns]
y = df[target_column]

# Encode labels (convert to numbers)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size={test_size}, random_state=42, stratify=y_encoded
)

# Scale features (important for KNN - distance-based algorithm)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
n_neighbors = {n_neighbors}
model = KNeighborsClassifier(n_neighbors=n_neighbors)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

print(f"K = {{n_neighbors}}")
print(f"Accuracy: {{accuracy:.4f}}")
print(f"Precision: {{precision:.4f}}")
print(f"Recall: {{recall:.4f}}")
print(f"F1 Score: {{f1:.4f}}")
print(f"Confusion Matrix:\\n{{cm}}")
print(f"Classes: {{label_encoder.classes_}}")'''

        return code
