import io
import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File
from app.schemas.schemas import (
    DataUploadResponse,
    TrainRequest,
    TrainResponse,
    PredictRequest,
    PredictResponse,
    PredictionPoint,
    LogisticTrainRequest,
    LogisticTrainResponse,
    LogisticPredictRequest,
    LogisticPredictResponse,
    LogisticPredictionPoint,
    KNNTrainRequest,
    KNNTrainResponse,
    KNNPredictRequest,
    KNNPredictResponse,
    KNNPredictionPoint,
    KMeansTrainRequest,
    KMeansTrainResponse,
    KMeansPredictRequest,
    KMeansPredictResponse,
    ClusterAssignment,
    DecisionTreeTrainRequest,
    DecisionTreeTrainResponse,
    DecisionTreePredictionPoint,
    RandomForestTrainRequest,
    RandomForestTrainResponse,
    RandomForestPredictionPoint,
)
from app.models.linear_regression import LinearRegressionTrainer
from app.models.logistic_regression import LogisticRegressionTrainer
from app.models.knn import KNNTrainer
from app.models.kmeans import KMeansTrainer
from app.models.decision_tree import DecisionTreeTrainer
from app.models.random_forest import RandomForestTrainer

router = APIRouter(prefix="/api/ml", tags=["Machine Learning"])


@router.post("/upload-csv", response_model=DataUploadResponse)
async def upload_csv(file: UploadFile = File(...)) -> DataUploadResponse:
    """
    Upload and parse a CSV file.

    Returns column information and a preview of the data.
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")

        # Get numeric columns for ML
        numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

        if len(numeric_columns) < 2:
            raise HTTPException(
                status_code=400,
                detail="CSV must have at least 2 numeric columns for regression"
            )

        # Create preview (first 10 rows)
        preview = df.head(10).to_dict(orient="records")

        return DataUploadResponse(
            success=True,
            columns=df.columns.tolist(),
            numeric_columns=numeric_columns,
            row_count=len(df),
            preview=preview,
            message=f"Successfully loaded {len(df)} rows with {len(df.columns)} columns",
        )

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty or malformed")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File encoding not supported. Please use UTF-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")


@router.post("/parse-csv-text", response_model=DataUploadResponse)
async def parse_csv_text(payload: dict) -> DataUploadResponse:
    """
    Parse CSV data from text input.

    Accepts a JSON payload with 'csv_text' field containing CSV-formatted string.
    """
    csv_text = payload.get("csv_text", "").strip()

    if not csv_text:
        raise HTTPException(status_code=400, detail="No CSV text provided")

    try:
        df = pd.read_csv(io.StringIO(csv_text))

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV data is empty")

        numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

        if len(numeric_columns) < 2:
            raise HTTPException(
                status_code=400,
                detail="CSV must have at least 2 numeric columns for regression"
            )

        preview = df.head(10).to_dict(orient="records")

        return DataUploadResponse(
            success=True,
            columns=df.columns.tolist(),
            numeric_columns=numeric_columns,
            row_count=len(df),
            preview=preview,
            message=f"Successfully parsed {len(df)} rows with {len(df.columns)} columns",
        )

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV data is empty or malformed")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")


@router.post("/train/linear-regression", response_model=TrainResponse)
async def train_linear_regression(request: TrainRequest) -> TrainResponse:
    """
    Train a Linear Regression model on the provided data.

    Returns model metrics, coefficients, predictions, and generated Python code.
    """
    try:
        trainer = LinearRegressionTrainer()
        result = trainer.train(
            data=request.data,
            feature_columns=request.feature_columns,
            target_column=request.target_column,
            test_size=request.test_size,
        )

        predictions = [
            PredictionPoint(
                actual=p["actual"],
                predicted=p["predicted"],
                features=p["features"],
            )
            for p in result["predictions"]
        ]

        return TrainResponse(
            success=True,
            message="Model trained successfully",
            metrics=result["metrics"],
            coefficients=result["coefficients"],
            intercept=result["intercept"],
            predictions=predictions,
            generated_code=result["generated_code"],
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/predict/linear-regression", response_model=PredictResponse)
async def predict_linear_regression(request: PredictRequest) -> PredictResponse:
    """
    Make predictions using a trained Linear Regression model.

    Uses the provided coefficients and intercept to make predictions.
    """
    try:
        trainer = LinearRegressionTrainer()
        predictions = trainer.predict(
            data=request.data,
            feature_columns=request.feature_columns,
            coefficients=request.coefficients,
            intercept=request.intercept,
        )

        return PredictResponse(
            success=True,
            predictions=predictions,
            message=f"Generated {len(predictions)} predictions",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Logistic Regression Endpoints
@router.post("/train/logistic-regression", response_model=LogisticTrainResponse)
async def train_logistic_regression(request: LogisticTrainRequest) -> LogisticTrainResponse:
    """
    Train a Logistic Regression model on the provided data.

    Returns model metrics, coefficients, predictions, and generated Python code.
    """
    try:
        trainer = LogisticRegressionTrainer()
        result = trainer.train(
            data=request.data,
            feature_columns=request.feature_columns,
            target_column=request.target_column,
            test_size=request.test_size,
        )

        predictions = [
            LogisticPredictionPoint(
                actual=p["actual"],
                predicted=p["predicted"],
                probability=p["probability"],
                features=p["features"],
            )
            for p in result["predictions"]
        ]

        return LogisticTrainResponse(
            success=True,
            message="Model trained successfully",
            metrics=result["metrics"],
            coefficients=result["coefficients"],
            intercept=result["intercept"],
            scaler_params=result["scaler_params"],
            class_labels=result["class_labels"],
            predictions=predictions,
            generated_code=result["generated_code"],
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/predict/logistic-regression", response_model=LogisticPredictResponse)
async def predict_logistic_regression(request: LogisticPredictRequest) -> LogisticPredictResponse:
    """
    Make predictions using a trained Logistic Regression model.
    """
    try:
        trainer = LogisticRegressionTrainer()
        predictions = trainer.predict(
            data=request.data,
            feature_columns=request.feature_columns,
            coefficients=request.coefficients,
            intercept=request.intercept,
            scaler_params=request.scaler_params,
        )

        return LogisticPredictResponse(
            success=True,
            predictions=predictions,
            message=f"Generated {len(predictions)} predictions",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# KNN Endpoints
@router.post("/train/knn", response_model=KNNTrainResponse)
async def train_knn(request: KNNTrainRequest) -> KNNTrainResponse:
    """
    Train a K-Nearest Neighbors model on the provided data.

    Returns model metrics, predictions, and generated Python code.
    """
    try:
        trainer = KNNTrainer()
        result = trainer.train(
            data=request.data,
            feature_columns=request.feature_columns,
            target_column=request.target_column,
            n_neighbors=request.n_neighbors,
            test_size=request.test_size,
        )

        predictions = [
            KNNPredictionPoint(
                actual=p["actual"],
                actual_label=p["actual_label"],
                predicted=p["predicted"],
                predicted_label=p["predicted_label"],
                probabilities=p["probabilities"],
                features=p["features"],
            )
            for p in result["predictions"]
        ]

        return KNNTrainResponse(
            success=True,
            message="Model trained successfully",
            metrics=result["metrics"],
            scaler_params=result["scaler_params"],
            training_data=result["training_data"],
            class_labels=result["class_labels"],
            n_neighbors=result["n_neighbors"],
            predictions=predictions,
            generated_code=result["generated_code"],
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/predict/knn", response_model=KNNPredictResponse)
async def predict_knn(request: KNNPredictRequest) -> KNNPredictResponse:
    """
    Make predictions using a trained KNN model.
    """
    try:
        trainer = KNNTrainer()
        predictions = trainer.predict(
            data=request.data,
            feature_columns=request.feature_columns,
            training_data=request.training_data,
            scaler_params=request.scaler_params,
            class_labels=request.class_labels,
            n_neighbors=request.n_neighbors,
        )

        return KNNPredictResponse(
            success=True,
            predictions=predictions,
            message=f"Generated {len(predictions)} predictions",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# K-Means Endpoints
@router.post("/train/kmeans", response_model=KMeansTrainResponse)
async def train_kmeans(request: KMeansTrainRequest) -> KMeansTrainResponse:
    """
    Train a K-Means Clustering model on the provided data.

    Returns metrics, centroids, cluster assignments, and generated Python code.
    """
    try:
        trainer = KMeansTrainer()
        result = trainer.train(
            data=request.data,
            feature_columns=request.feature_columns,
            n_clusters=request.n_clusters,
        )

        assignments = [
            ClusterAssignment(
                cluster=a["cluster"],
                features=a["features"],
                original_index=a["original_index"],
            )
            for a in result["assignments"]
        ]

        return KMeansTrainResponse(
            success=True,
            message="Clustering completed successfully",
            metrics=result["metrics"],
            centroids=result["centroids"],
            centroids_scaled=result["centroids_scaled"],
            scaler_params=result["scaler_params"],
            assignments=assignments,
            generated_code=result["generated_code"],
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")


@router.post("/predict/kmeans", response_model=KMeansPredictResponse)
async def predict_kmeans(request: KMeansPredictRequest) -> KMeansPredictResponse:
    """
    Assign new data points to clusters.
    """
    try:
        trainer = KMeansTrainer()
        predictions = trainer.predict(
            data=request.data,
            feature_columns=request.feature_columns,
            centroids_scaled=request.centroids_scaled,
            scaler_params=request.scaler_params,
        )

        return KMeansPredictResponse(
            success=True,
            predictions=predictions,
            message=f"Assigned {len(predictions)} points to clusters",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Decision Tree Endpoints
@router.post("/train/decision-tree", response_model=DecisionTreeTrainResponse)
async def train_decision_tree(request: DecisionTreeTrainRequest) -> DecisionTreeTrainResponse:
    """
    Train a Decision Tree model on the provided data.

    Returns model metrics, feature importances, predictions, and generated Python code.
    """
    try:
        trainer = DecisionTreeTrainer()
        result = trainer.train(
            data=request.data,
            feature_columns=request.feature_columns,
            target_column=request.target_column,
            max_depth=request.max_depth,
            min_samples_split=request.min_samples_split,
            test_size=request.test_size,
        )

        predictions = [
            DecisionTreePredictionPoint(
                actual=p["actual"],
                actual_label=p["actual_label"],
                predicted=p["predicted"],
                predicted_label=p["predicted_label"],
                probabilities=p["probabilities"],
                features=p["features"],
            )
            for p in result["predictions"]
        ]

        return DecisionTreeTrainResponse(
            success=True,
            message="Model trained successfully",
            metrics=result["metrics"],
            tree_params=result["tree_params"],
            class_labels=result["class_labels"],
            predictions=predictions,
            generated_code=result["generated_code"],
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# Random Forest Endpoints
@router.post("/train/random-forest", response_model=RandomForestTrainResponse)
async def train_random_forest(request: RandomForestTrainRequest) -> RandomForestTrainResponse:
    """
    Train a Random Forest model on the provided data.

    Returns model metrics, feature importances, predictions, and generated Python code.
    """
    try:
        trainer = RandomForestTrainer()
        result = trainer.train(
            data=request.data,
            feature_columns=request.feature_columns,
            target_column=request.target_column,
            n_estimators=request.n_estimators,
            max_depth=request.max_depth,
            min_samples_split=request.min_samples_split,
            test_size=request.test_size,
        )

        predictions = [
            RandomForestPredictionPoint(
                actual=p["actual"],
                actual_label=p["actual_label"],
                predicted=p["predicted"],
                predicted_label=p["predicted_label"],
                probabilities=p["probabilities"],
                features=p["features"],
            )
            for p in result["predictions"]
        ]

        return RandomForestTrainResponse(
            success=True,
            message="Model trained successfully",
            metrics=result["metrics"],
            forest_params=result["forest_params"],
            class_labels=result["class_labels"],
            predictions=predictions,
            generated_code=result["generated_code"],
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/models")
async def list_models():
    """List available ML models."""
    return {
        "models": [
            {
                "id": "linear-regression",
                "name": "Linear Regression",
                "description": "A fundamental algorithm that models the relationship between variables using a linear equation.",
                "category": "Regression",
                "difficulty": "Beginner",
                "tags": ["supervised", "regression", "interpretable"],
            },
            {
                "id": "logistic-regression",
                "name": "Logistic Regression",
                "description": "A classification algorithm for predicting binary outcomes using the sigmoid function.",
                "category": "Classification",
                "difficulty": "Beginner",
                "tags": ["supervised", "classification", "interpretable"],
            },
            {
                "id": "knn",
                "name": "K-Nearest Neighbors",
                "description": "A simple algorithm that classifies based on the majority class of nearest neighbors.",
                "category": "Classification",
                "difficulty": "Beginner",
                "tags": ["supervised", "classification", "instance-based"],
            },
            {
                "id": "kmeans",
                "name": "K-Means Clustering",
                "description": "An unsupervised algorithm that groups data into K clusters based on similarity.",
                "category": "Clustering",
                "difficulty": "Beginner",
                "tags": ["unsupervised", "clustering", "centroid-based"],
            },
            {
                "id": "decision-tree",
                "name": "Decision Tree",
                "description": "A tree-based algorithm that makes predictions by learning decision rules from features.",
                "category": "Classification",
                "difficulty": "Beginner",
                "tags": ["supervised", "classification", "interpretable", "tree-based"],
            },
            {
                "id": "random-forest",
                "name": "Random Forest",
                "description": "An ensemble of decision trees that improves accuracy through voting.",
                "category": "Classification",
                "difficulty": "Intermediate",
                "tags": ["supervised", "classification", "ensemble", "tree-based"],
            },
        ]
    }
