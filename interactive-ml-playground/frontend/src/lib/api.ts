const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface DataUploadResponse {
  success: boolean;
  columns: string[];
  numeric_columns: string[];
  row_count: number;
  preview: Record<string, unknown>[];
  message: string;
}

export interface PredictionPoint {
  actual: number;
  predicted: number;
  features: Record<string, number>;
}

export interface TrainResponse {
  success: boolean;
  message: string;
  metrics: {
    r2_score: number;
    mse: number;
    rmse: number;
    mae: number;
    train_samples: number;
    test_samples: number;
  };
  coefficients: Record<string, number>;
  intercept: number;
  predictions: PredictionPoint[];
  generated_code: string;
}

// Logistic Regression types
export interface LogisticPredictionPoint {
  actual: number;
  predicted: number;
  probability: number;
  features: Record<string, number>;
}

export interface LogisticTrainResponse {
  success: boolean;
  message: string;
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    confusion_matrix: {
      true_negative: number;
      false_positive: number;
      false_negative: number;
      true_positive: number;
    };
    train_samples: number;
    test_samples: number;
  };
  coefficients: Record<string, number>;
  intercept: number;
  scaler_params: {
    mean: Record<string, number>;
    scale: Record<string, number>;
  };
  class_labels: Record<string, string>;
  predictions: LogisticPredictionPoint[];
  generated_code: string;
}

// KNN types
export interface KNNPredictionPoint {
  actual: number;
  actual_label: string;
  predicted: number;
  predicted_label: string;
  probabilities: Record<string, number>;
  features: Record<string, number>;
}

export interface KNNTrainResponse {
  success: boolean;
  message: string;
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    n_neighbors: number;
    n_classes: number;
    confusion_matrix: number[][];
    train_samples: number;
    test_samples: number;
  };
  scaler_params: {
    mean: Record<string, number>;
    scale: Record<string, number>;
  };
  training_data: {
    X: number[][];
    y: number[];
  };
  class_labels: Record<string, string>;
  n_neighbors: number;
  predictions: KNNPredictionPoint[];
  generated_code: string;
}

// K-Means types
export interface ClusterAssignment {
  cluster: number;
  features: Record<string, number>;
  original_index: number;
}

export interface KMeansTrainResponse {
  success: boolean;
  message: string;
  metrics: {
    silhouette_score: number;
    calinski_harabasz_score: number;
    davies_bouldin_score: number;
    inertia: number;
    n_clusters: number;
    n_samples: number;
    cluster_stats: Array<{
      cluster_id: number;
      size: number;
      percentage: number;
      feature_means: Record<string, number>;
      feature_stds: Record<string, number>;
    }>;
  };
  centroids: Record<string, number>[];
  centroids_scaled: Record<string, number>[];
  scaler_params: {
    mean: Record<string, number>;
    scale: Record<string, number>;
  };
  assignments: ClusterAssignment[];
  generated_code: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  description: string;
  category: string;
  difficulty: string;
  tags: string[];
  coming_soon?: boolean;
}

class ApiClient {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_BASE_URL;
  }

  async uploadCsv(file: File): Promise<DataUploadResponse> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${this.baseUrl}/api/ml/upload-csv`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to upload CSV");
    }

    return response.json();
  }

  async parseCsvText(csvText: string): Promise<DataUploadResponse> {
    const response = await fetch(`${this.baseUrl}/api/ml/parse-csv-text`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ csv_text: csvText }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to parse CSV");
    }

    return response.json();
  }

  async trainLinearRegression(
    data: Record<string, unknown>[],
    featureColumns: string[],
    targetColumn: string,
    testSize: number = 0.2
  ): Promise<TrainResponse> {
    const response = await fetch(
      `${this.baseUrl}/api/ml/train/linear-regression`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          data,
          feature_columns: featureColumns,
          target_column: targetColumn,
          test_size: testSize,
        }),
      }
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to train model");
    }

    return response.json();
  }

  async getModels(): Promise<{ models: ModelInfo[] }> {
    const response = await fetch(`${this.baseUrl}/api/ml/models`);

    if (!response.ok) {
      throw new Error("Failed to fetch models");
    }

    return response.json();
  }

  async healthCheck(): Promise<{ status: string }> {
    const response = await fetch(`${this.baseUrl}/health`);
    return response.json();
  }

  async trainLogisticRegression(
    data: Record<string, unknown>[],
    featureColumns: string[],
    targetColumn: string,
    testSize: number = 0.2
  ): Promise<LogisticTrainResponse> {
    const response = await fetch(
      `${this.baseUrl}/api/ml/train/logistic-regression`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          data,
          feature_columns: featureColumns,
          target_column: targetColumn,
          test_size: testSize,
        }),
      }
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to train model");
    }

    return response.json();
  }

  async trainKNN(
    data: Record<string, unknown>[],
    featureColumns: string[],
    targetColumn: string,
    nNeighbors: number = 5,
    testSize: number = 0.2
  ): Promise<KNNTrainResponse> {
    const response = await fetch(`${this.baseUrl}/api/ml/train/knn`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        data,
        feature_columns: featureColumns,
        target_column: targetColumn,
        n_neighbors: nNeighbors,
        test_size: testSize,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to train model");
    }

    return response.json();
  }

  async trainKMeans(
    data: Record<string, unknown>[],
    featureColumns: string[],
    nClusters: number = 3
  ): Promise<KMeansTrainResponse> {
    const response = await fetch(`${this.baseUrl}/api/ml/train/kmeans`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        data,
        feature_columns: featureColumns,
        n_clusters: nClusters,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to train model");
    }

    return response.json();
  }
}

export const apiClient = new ApiClient();
