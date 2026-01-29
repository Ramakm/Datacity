from pydantic import BaseModel, Field
from typing import Optional


class DataUploadResponse(BaseModel):
    """Response after uploading and parsing CSV data."""
    success: bool
    columns: list[str]
    numeric_columns: list[str]
    row_count: int
    preview: list[dict]
    message: str


class TrainRequest(BaseModel):
    """Request to train a linear regression model."""
    data: list[dict]
    feature_columns: list[str] = Field(..., min_length=1)
    target_column: str
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)


class PredictionPoint(BaseModel):
    """Single prediction point with actual and predicted values."""
    actual: float
    predicted: float
    features: dict


class TrainResponse(BaseModel):
    """Response after training the model."""
    success: bool
    message: str
    metrics: dict
    coefficients: dict
    intercept: float
    predictions: list[PredictionPoint]
    generated_code: str


class PredictRequest(BaseModel):
    """Request to make predictions on new data."""
    data: list[dict]
    feature_columns: list[str]
    coefficients: dict
    intercept: float


class PredictResponse(BaseModel):
    """Response with predictions."""
    success: bool
    predictions: list[float]
    message: str


# Logistic Regression Schemas
class LogisticTrainRequest(BaseModel):
    """Request to train a logistic regression model."""
    data: list[dict]
    feature_columns: list[str] = Field(..., min_length=1)
    target_column: str
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)


class LogisticPredictionPoint(BaseModel):
    """Single prediction point for classification."""
    actual: int
    predicted: int
    probability: float
    features: dict


class LogisticTrainResponse(BaseModel):
    """Response after training logistic regression."""
    success: bool
    message: str
    metrics: dict
    coefficients: dict
    intercept: float
    scaler_params: dict
    class_labels: dict
    predictions: list[LogisticPredictionPoint]
    generated_code: str


class LogisticPredictRequest(BaseModel):
    """Request to make logistic regression predictions."""
    data: list[dict]
    feature_columns: list[str]
    coefficients: dict
    intercept: float
    scaler_params: dict


class LogisticPredictResponse(BaseModel):
    """Response with classification predictions."""
    success: bool
    predictions: list[dict]
    message: str


# KNN Schemas
class KNNTrainRequest(BaseModel):
    """Request to train a KNN model."""
    data: list[dict]
    feature_columns: list[str] = Field(..., min_length=1)
    target_column: str
    n_neighbors: int = Field(default=5, ge=1, le=50)
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)


class KNNPredictionPoint(BaseModel):
    """Single prediction point for KNN."""
    actual: int
    actual_label: str
    predicted: int
    predicted_label: str
    probabilities: dict
    features: dict


class KNNTrainResponse(BaseModel):
    """Response after training KNN."""
    success: bool
    message: str
    metrics: dict
    scaler_params: dict
    training_data: dict
    class_labels: dict
    n_neighbors: int
    predictions: list[KNNPredictionPoint]
    generated_code: str


class KNNPredictRequest(BaseModel):
    """Request to make KNN predictions."""
    data: list[dict]
    feature_columns: list[str]
    training_data: dict
    scaler_params: dict
    class_labels: dict
    n_neighbors: int


class KNNPredictResponse(BaseModel):
    """Response with KNN predictions."""
    success: bool
    predictions: list[dict]
    message: str


# K-Means Schemas
class KMeansTrainRequest(BaseModel):
    """Request to train a K-Means model."""
    data: list[dict]
    feature_columns: list[str] = Field(..., min_length=1)
    n_clusters: int = Field(default=3, ge=2, le=20)


class ClusterAssignment(BaseModel):
    """Single cluster assignment."""
    cluster: int
    features: dict
    original_index: int


class KMeansTrainResponse(BaseModel):
    """Response after training K-Means."""
    success: bool
    message: str
    metrics: dict
    centroids: list[dict]
    centroids_scaled: list[dict]
    scaler_params: dict
    assignments: list[ClusterAssignment]
    generated_code: str


class KMeansPredictRequest(BaseModel):
    """Request to assign new data to clusters."""
    data: list[dict]
    feature_columns: list[str]
    centroids_scaled: list[dict]
    scaler_params: dict


class KMeansPredictResponse(BaseModel):
    """Response with cluster assignments."""
    success: bool
    predictions: list[dict]
    message: str


# Decision Tree Schemas
class DecisionTreeTrainRequest(BaseModel):
    """Request to train a Decision Tree model."""
    data: list[dict]
    feature_columns: list[str] = Field(..., min_length=1)
    target_column: str
    max_depth: Optional[int] = Field(default=None, ge=1, le=50)
    min_samples_split: int = Field(default=2, ge=2, le=100)
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)


class DecisionTreePredictionPoint(BaseModel):
    """Single prediction point for Decision Tree."""
    actual: int
    actual_label: str
    predicted: int
    predicted_label: str
    probabilities: dict
    features: dict


class DecisionTreeTrainResponse(BaseModel):
    """Response after training Decision Tree."""
    success: bool
    message: str
    metrics: dict
    tree_params: dict
    class_labels: dict
    predictions: list[DecisionTreePredictionPoint]
    generated_code: str


class DecisionTreePredictRequest(BaseModel):
    """Request to make Decision Tree predictions."""
    data: list[dict]
    feature_columns: list[str]
    tree_params: dict
    class_labels: dict


class DecisionTreePredictResponse(BaseModel):
    """Response with Decision Tree predictions."""
    success: bool
    predictions: list[dict]
    message: str


# Random Forest Schemas
class RandomForestTrainRequest(BaseModel):
    """Request to train a Random Forest model."""
    data: list[dict]
    feature_columns: list[str] = Field(..., min_length=1)
    target_column: str
    n_estimators: int = Field(default=100, ge=10, le=500)
    max_depth: Optional[int] = Field(default=None, ge=1, le=50)
    min_samples_split: int = Field(default=2, ge=2, le=100)
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)


class RandomForestPredictionPoint(BaseModel):
    """Single prediction point for Random Forest."""
    actual: int
    actual_label: str
    predicted: int
    predicted_label: str
    probabilities: dict
    features: dict


class RandomForestTrainResponse(BaseModel):
    """Response after training Random Forest."""
    success: bool
    message: str
    metrics: dict
    forest_params: dict
    class_labels: dict
    predictions: list[RandomForestPredictionPoint]
    generated_code: str


class RandomForestPredictRequest(BaseModel):
    """Request to make Random Forest predictions."""
    data: list[dict]
    feature_columns: list[str]
    forest_params: dict
    class_labels: dict


class RandomForestPredictResponse(BaseModel):
    """Response with Random Forest predictions."""
    success: bool
    predictions: list[dict]
    message: str
