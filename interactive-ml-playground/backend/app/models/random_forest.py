import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from typing import Any


class RandomForestTrainer:
    """Handles training and prediction for Random Forest models."""

    def __init__(self):
        self.model: RandomForestClassifier | None = None
        self.label_encoder: LabelEncoder | None = None
        self.feature_columns: list[str] = []
        self.target_column: str = ""

    def train(
        self,
        data: list[dict],
        feature_columns: list[str],
        target_column: str,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        test_size: float = 0.2,
    ) -> dict[str, Any]:
        """
        Train a Random Forest model on the provided data.

        Args:
            data: List of dictionaries containing the dataset
            feature_columns: List of column names to use as features
            target_column: Name of the target column
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of each tree (None for unlimited)
            min_samples_split: Minimum samples required to split a node
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

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )

        # Train model
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        self.feature_columns = feature_columns
        self.target_column = target_column

        # Make predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)

        # Calculate metrics
        average = 'binary' if n_classes == 2 else 'weighted'
        cm = confusion_matrix(y_test, y_pred)

        # Get feature importances
        feature_importances = {
            col: round(float(imp), 4)
            for col, imp in zip(feature_columns, self.model.feature_importances_)
        }

        # Calculate OOB score if available
        oob_score = None
        if hasattr(self.model, 'oob_score_'):
            oob_score = round(self.model.oob_score_, 4)

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, average=average, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, average=average, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, average=average, zero_division=0), 4),
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "n_classes": n_classes,
            "feature_importances": feature_importances,
            "confusion_matrix": cm.tolist(),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

        if oob_score:
            metrics["oob_score"] = oob_score

        # Forest parameters for storage
        forest_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "feature_columns": feature_columns,
            "class_labels": class_labels,
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
        generated_code = self._generate_code(
            feature_columns, target_column, n_estimators, max_depth, min_samples_split, test_size
        )

        return {
            "metrics": metrics,
            "forest_params": forest_params,
            "class_labels": class_labels,
            "predictions": predictions,
            "generated_code": generated_code,
        }

    def predict(
        self,
        data: list[dict],
        feature_columns: list[str],
        forest_params: dict,
        class_labels: dict,
    ) -> list[dict]:
        """
        Make predictions using the trained Random Forest model.
        """
        df = pd.DataFrame(data)
        predictions = []

        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")

        for _, row in df.iterrows():
            features = [[row[col] for col in feature_columns]]
            X_new = np.array(features)
            pred = self.model.predict(X_new)[0]
            prob = self.model.predict_proba(X_new)[0]

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
        n_estimators: int,
        max_depth: int | None,
        min_samples_split: int,
        test_size: float,
    ) -> str:
        """Generate Python code that replicates the training process."""
        features_str = ", ".join(f'"{col}"' for col in feature_columns)
        max_depth_str = str(max_depth) if max_depth else "None"

        code = f'''import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

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

# Create and train the Random Forest model
model = RandomForestClassifier(
    n_estimators={n_estimators},
    max_depth={max_depth_str},
    min_samples_split={min_samples_split},
    random_state=42,
    n_jobs=-1  # Use all available cores
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

print(f"Number of Trees: {n_estimators}")
print(f"Max Depth: {max_depth_str}")
print(f"Accuracy: {{accuracy:.4f}}")
print(f"Precision: {{precision:.4f}}")
print(f"Recall: {{recall:.4f}}")
print(f"F1 Score: {{f1:.4f}}")
print(f"Confusion Matrix:\\n{{cm}}")

# Feature Importances
print("\\nFeature Importances:")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
for i, idx in enumerate(indices):
    print(f"  {{i+1}}. {{feature_columns[idx]}}: {{importances[idx]:.4f}}")

# Visualize Feature Importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feature_columns[i] for i in indices], rotation=45)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.savefig("feature_importances.png", dpi=150)
plt.show()'''

        return code
