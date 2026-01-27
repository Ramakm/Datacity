"use client";

import { useState, useRef } from "react";
import {
  Upload,
  FileText,
  Play,
  AlertCircle,
  CheckCircle,
  Loader2,
  Table,
  Code,
  BarChart3,
  X,
} from "lucide-react";
import clsx from "clsx";
import { apiClient, DataUploadResponse, LogisticTrainResponse } from "@/lib/api";
import CodeBlock from "./CodeBlock";

type Step = "upload" | "configure" | "results";

const SAMPLE_CSV = `age,bmi,glucose,blood_pressure,insulin,diabetes
25,22.5,85,70,80,0
45,30.2,140,85,150,1
32,24.8,95,75,90,0
55,33.1,160,92,180,1
28,21.0,82,68,75,0
50,31.5,155,88,170,1
38,26.3,105,78,100,0
62,35.0,175,95,200,1
29,23.2,88,72,82,0
48,29.8,135,84,140,1
35,25.5,98,76,95,0
58,32.8,165,90,185,1
27,22.0,84,69,78,0
52,30.9,150,87,165,1
40,27.1,110,80,105,0`;

export default function LogisticTryItTab() {
  const [step, setStep] = useState<Step>("upload");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Data state
  const [uploadedData, setUploadedData] = useState<DataUploadResponse | null>(null);
  const [csvText, setCsvText] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Configuration state
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [testSize, setTestSize] = useState(0.2);

  // Results state
  const [trainResult, setTrainResult] = useState<LogisticTrainResponse | null>(null);
  const [activeResultTab, setActiveResultTab] = useState<"metrics" | "code" | "predictions">("metrics");

  const resetAll = () => {
    setStep("upload");
    setUploadedData(null);
    setCsvText("");
    setSelectedFeatures([]);
    setTargetColumn("");
    setTrainResult(null);
    setError(null);
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const response = await apiClient.uploadCsv(file);
      setUploadedData(response);
      setStep("configure");

      const text = await file.text();
      setCsvText(text);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to upload file");
    } finally {
      setLoading(false);
    }
  };

  const handlePasteCSV = async () => {
    if (!csvText.trim()) {
      setError("Please paste some CSV data");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await apiClient.parseCsvText(csvText);
      setUploadedData(response);
      setStep("configure");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to parse CSV");
    } finally {
      setLoading(false);
    }
  };

  const loadSampleData = () => {
    setCsvText(SAMPLE_CSV);
  };

  const toggleFeature = (column: string) => {
    if (column === targetColumn) return;

    setSelectedFeatures((prev) =>
      prev.includes(column)
        ? prev.filter((c) => c !== column)
        : [...prev, column]
    );
  };

  const setTarget = (column: string) => {
    setTargetColumn(column);
    setSelectedFeatures((prev) => prev.filter((c) => c !== column));
  };

  const handleTrain = async () => {
    if (selectedFeatures.length === 0) {
      setError("Please select at least one feature column");
      return;
    }
    if (!targetColumn) {
      setError("Please select a target column");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const lines = csvText.trim().split("\n");
      const headers = lines[0].split(",").map((h) => h.trim());
      const data: Record<string, unknown>[] = [];

      for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(",").map((v) => v.trim());
        const row: Record<string, unknown> = {};
        headers.forEach((h, idx) => {
          const val = values[idx];
          row[h] = isNaN(Number(val)) ? val : Number(val);
        });
        data.push(row);
      }

      const result = await apiClient.trainLogisticRegression(
        data,
        selectedFeatures,
        targetColumn,
        testSize
      );

      setTrainResult(result);
      setStep("results");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Training failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Progress Steps */}
      <div className="flex items-center justify-center gap-4">
        {[
          { id: "upload", label: "Upload Data", icon: Upload },
          { id: "configure", label: "Configure", icon: Table },
          { id: "results", label: "Results", icon: BarChart3 },
        ].map((s, index) => (
          <div key={s.id} className="flex items-center gap-2">
            <div
              className={clsx(
                "w-10 h-10 rounded-full flex items-center justify-center transition-colors",
                step === s.id
                  ? "bg-purple-500 text-white"
                  : ["configure", "results"].indexOf(step) > ["upload", "configure", "results"].indexOf(s.id)
                  ? "bg-purple-500/30 text-purple-300"
                  : "bg-slate-700 text-slate-400"
              )}
            >
              <s.icon className="w-5 h-5" />
            </div>
            <span
              className={clsx(
                "text-sm font-medium hidden sm:block",
                step === s.id ? "text-purple-400" : "text-slate-500"
              )}
            >
              {s.label}
            </span>
            {index < 2 && (
              <div className="w-8 h-px bg-slate-700 hidden sm:block" />
            )}
          </div>
        ))}
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-red-400 font-medium">Error</p>
            <p className="text-red-300 text-sm">{error}</p>
          </div>
          <button
            onClick={() => setError(null)}
            className="ml-auto text-red-400 hover:text-red-300"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      )}

      {/* Step 1: Upload */}
      {step === "upload" && (
        <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700/50 p-6">
          <h3 className="text-lg font-semibold text-white mb-6">
            Load Your Data
          </h3>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* File Upload */}
            <div className="space-y-4">
              <div
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-slate-600 rounded-xl p-8 text-center cursor-pointer hover:border-purple-500/50 hover:bg-purple-500/5 transition-colors"
              >
                <Upload className="w-12 h-12 text-slate-500 mx-auto mb-4" />
                <p className="text-white font-medium mb-2">
                  Click to upload a CSV file
                </p>
                <p className="text-slate-400 text-sm">
                  or drag and drop
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </div>
            </div>

            {/* Paste CSV */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <p className="text-white font-medium">Or paste CSV data:</p>
                <button
                  onClick={loadSampleData}
                  className="text-sm text-purple-400 hover:text-purple-300 font-medium"
                >
                  Load sample data
                </button>
              </div>
              <textarea
                value={csvText}
                onChange={(e) => setCsvText(e.target.value)}
                placeholder="Paste your CSV data here..."
                className="w-full h-48 px-4 py-3 bg-slate-900/50 border border-slate-600 rounded-xl text-white placeholder-slate-500 focus:ring-2 focus:ring-purple-500 focus:border-purple-500 font-mono text-sm resize-none"
              />
              <button
                onClick={handlePasteCSV}
                disabled={loading || !csvText.trim()}
                className={clsx(
                  "w-full py-3 rounded-xl font-medium transition-colors flex items-center justify-center gap-2",
                  csvText.trim()
                    ? "bg-purple-500 text-white hover:bg-purple-600"
                    : "bg-slate-700 text-slate-500 cursor-not-allowed"
                )}
              >
                {loading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <FileText className="w-5 h-5" />
                )}
                Parse CSV
              </button>
            </div>
          </div>

          <div className="mt-6 p-4 bg-purple-500/10 border border-purple-500/20 rounded-lg">
            <p className="text-sm text-purple-300">
              <strong>Tip:</strong> For classification, your target column should have
              binary values (0/1, Yes/No, True/False). The sample data includes a diabetes
              prediction dataset.
            </p>
          </div>
        </div>
      )}

      {/* Step 2: Configure */}
      {step === "configure" && uploadedData && (
        <div className="space-y-6">
          {/* Data Preview */}
          <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700/50 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">
                Data Preview
              </h3>
              <span className="text-sm text-slate-400">
                {uploadedData.row_count} rows, {uploadedData.columns.length} columns
              </span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-slate-900/50">
                  <tr>
                    {uploadedData.columns.map((col) => (
                      <th
                        key={col}
                        className="px-4 py-2 text-left text-slate-300 font-semibold whitespace-nowrap"
                      >
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-700/50">
                  {uploadedData.preview.slice(0, 5).map((row, idx) => (
                    <tr key={idx}>
                      {uploadedData.columns.map((col) => (
                        <td
                          key={col}
                          className="px-4 py-2 text-slate-400 whitespace-nowrap"
                        >
                          {String(row[col] ?? "")}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Column Selection */}
          <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700/50 p-6">
            <h3 className="text-lg font-semibold text-white mb-4">
              Configure Model
            </h3>

            <div className="space-y-6">
              {/* Feature Selection */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-3">
                  Select Feature Columns (inputs)
                </label>
                <div className="flex flex-wrap gap-2">
                  {uploadedData.numeric_columns.map((col) => (
                    <button
                      key={col}
                      onClick={() => toggleFeature(col)}
                      disabled={col === targetColumn}
                      className={clsx(
                        "px-4 py-2 rounded-lg text-sm font-medium transition-colors",
                        col === targetColumn
                          ? "bg-slate-700 text-slate-500 cursor-not-allowed"
                          : selectedFeatures.includes(col)
                          ? "bg-purple-500 text-white"
                          : "bg-slate-700 text-slate-300 hover:bg-slate-600"
                      )}
                    >
                      {col}
                      {selectedFeatures.includes(col) && (
                        <CheckCircle className="inline w-4 h-4 ml-1" />
                      )}
                    </button>
                  ))}
                </div>
                <p className="text-sm text-slate-500 mt-2">
                  Selected: {selectedFeatures.length} feature(s)
                </p>
              </div>

              {/* Target Selection */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-3">
                  Select Target Column (what to predict - should be 0/1)
                </label>
                <div className="flex flex-wrap gap-2">
                  {uploadedData.numeric_columns.map((col) => (
                    <button
                      key={col}
                      onClick={() => setTarget(col)}
                      disabled={selectedFeatures.includes(col)}
                      className={clsx(
                        "px-4 py-2 rounded-lg text-sm font-medium transition-colors",
                        selectedFeatures.includes(col)
                          ? "bg-slate-700 text-slate-500 cursor-not-allowed"
                          : col === targetColumn
                          ? "bg-green-500 text-white"
                          : "bg-slate-700 text-slate-300 hover:bg-slate-600"
                      )}
                    >
                      {col}
                      {col === targetColumn && (
                        <CheckCircle className="inline w-4 h-4 ml-1" />
                      )}
                    </button>
                  ))}
                </div>
              </div>

              {/* Test Size */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-3">
                  Test Set Size: {Math.round(testSize * 100)}%
                </label>
                <input
                  type="range"
                  min="10"
                  max="50"
                  value={testSize * 100}
                  onChange={(e) => setTestSize(Number(e.target.value) / 100)}
                  className="w-full max-w-xs accent-purple-500"
                />
                <p className="text-sm text-slate-500 mt-1">
                  {Math.round((1 - testSize) * 100)}% for training, {Math.round(testSize * 100)}% for testing
                </p>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-4">
            <button
              onClick={resetAll}
              className="px-6 py-3 rounded-xl font-medium border border-slate-600 text-slate-300 hover:bg-slate-800"
            >
              Start Over
            </button>
            <button
              onClick={handleTrain}
              disabled={loading || selectedFeatures.length === 0 || !targetColumn}
              className={clsx(
                "flex-1 py-3 rounded-xl font-medium transition-colors flex items-center justify-center gap-2",
                selectedFeatures.length > 0 && targetColumn
                  ? "bg-purple-500 text-white hover:bg-purple-600"
                  : "bg-slate-700 text-slate-500 cursor-not-allowed"
              )}
            >
              {loading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Play className="w-5 h-5" />
              )}
              Train Model
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Results */}
      {step === "results" && trainResult && (
        <div className="space-y-6">
          {/* Success Banner */}
          <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4 flex items-center gap-3">
            <CheckCircle className="w-6 h-6 text-green-400" />
            <div>
              <p className="text-green-400 font-medium">Model trained successfully!</p>
              <p className="text-green-300 text-sm">{trainResult.message}</p>
            </div>
          </div>

          {/* Result Tabs */}
          <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700/50 overflow-hidden">
            <div className="border-b border-slate-700/50 flex">
              {[
                { id: "metrics", label: "Metrics", icon: BarChart3 },
                { id: "predictions", label: "Predictions", icon: Table },
                { id: "code", label: "Generated Code", icon: Code },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveResultTab(tab.id as typeof activeResultTab)}
                  className={clsx(
                    "flex items-center gap-2 px-6 py-4 text-sm font-medium border-b-2 transition-colors",
                    activeResultTab === tab.id
                      ? "border-purple-500 text-purple-400 bg-purple-500/10"
                      : "border-transparent text-slate-400 hover:text-white hover:bg-slate-700/50"
                  )}
                >
                  <tab.icon className="w-4 h-4" />
                  {tab.label}
                </button>
              ))}
            </div>

            <div className="p-6">
              {activeResultTab === "metrics" && (
                <div className="space-y-6">
                  {/* Main Metrics */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-slate-900/50 rounded-lg p-4 text-center">
                      <div className="text-2xl font-bold text-purple-400">
                        {(trainResult.metrics.accuracy * 100).toFixed(1)}%
                      </div>
                      <div className="text-sm text-slate-400">Accuracy</div>
                    </div>
                    <div className="bg-slate-900/50 rounded-lg p-4 text-center">
                      <div className="text-2xl font-bold text-blue-400">
                        {(trainResult.metrics.precision * 100).toFixed(1)}%
                      </div>
                      <div className="text-sm text-slate-400">Precision</div>
                    </div>
                    <div className="bg-slate-900/50 rounded-lg p-4 text-center">
                      <div className="text-2xl font-bold text-green-400">
                        {(trainResult.metrics.recall * 100).toFixed(1)}%
                      </div>
                      <div className="text-sm text-slate-400">Recall</div>
                    </div>
                    <div className="bg-slate-900/50 rounded-lg p-4 text-center">
                      <div className="text-2xl font-bold text-amber-400">
                        {(trainResult.metrics.f1_score * 100).toFixed(1)}%
                      </div>
                      <div className="text-sm text-slate-400">F1 Score</div>
                    </div>
                  </div>

                  {/* Confusion Matrix */}
                  <div>
                    <h4 className="text-lg font-semibold text-white mb-4">Confusion Matrix</h4>
                    <div className="grid grid-cols-2 gap-2 max-w-xs">
                      <div className="bg-green-500/20 border border-green-500/30 rounded-lg p-4 text-center">
                        <div className="text-xl font-bold text-green-400">
                          {trainResult.metrics.confusion_matrix.true_negative}
                        </div>
                        <div className="text-xs text-slate-400">True Negative</div>
                      </div>
                      <div className="bg-red-500/20 border border-red-500/30 rounded-lg p-4 text-center">
                        <div className="text-xl font-bold text-red-400">
                          {trainResult.metrics.confusion_matrix.false_positive}
                        </div>
                        <div className="text-xs text-slate-400">False Positive</div>
                      </div>
                      <div className="bg-red-500/20 border border-red-500/30 rounded-lg p-4 text-center">
                        <div className="text-xl font-bold text-red-400">
                          {trainResult.metrics.confusion_matrix.false_negative}
                        </div>
                        <div className="text-xs text-slate-400">False Negative</div>
                      </div>
                      <div className="bg-green-500/20 border border-green-500/30 rounded-lg p-4 text-center">
                        <div className="text-xl font-bold text-green-400">
                          {trainResult.metrics.confusion_matrix.true_positive}
                        </div>
                        <div className="text-xs text-slate-400">True Positive</div>
                      </div>
                    </div>
                  </div>

                  {/* Coefficients */}
                  <div>
                    <h4 className="text-lg font-semibold text-white mb-4">Coefficients</h4>
                    <div className="space-y-2">
                      {Object.entries(trainResult.coefficients).map(([feature, coef]) => (
                        <div key={feature} className="flex items-center justify-between">
                          <span className="text-slate-300">{feature}</span>
                          <span className={clsx(
                            "font-mono",
                            coef > 0 ? "text-green-400" : "text-red-400"
                          )}>
                            {coef > 0 ? "+" : ""}{coef.toFixed(4)}
                          </span>
                        </div>
                      ))}
                      <div className="flex items-center justify-between pt-2 border-t border-slate-700">
                        <span className="text-slate-300">Intercept</span>
                        <span className="font-mono text-amber-400">
                          {trainResult.intercept.toFixed(4)}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Sample Info */}
                  <div className="flex gap-4 text-sm text-slate-400">
                    <span>Train samples: {trainResult.metrics.train_samples}</span>
                    <span>Test samples: {trainResult.metrics.test_samples}</span>
                  </div>
                </div>
              )}

              {activeResultTab === "predictions" && (
                <div className="space-y-4">
                  <h4 className="text-lg font-semibold text-white">Test Predictions</h4>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead className="bg-slate-900/50">
                        <tr>
                          <th className="px-4 py-2 text-left text-slate-300">Actual</th>
                          <th className="px-4 py-2 text-left text-slate-300">Predicted</th>
                          <th className="px-4 py-2 text-left text-slate-300">Probability</th>
                          <th className="px-4 py-2 text-left text-slate-300">Result</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-700/50">
                        {trainResult.predictions.slice(0, 10).map((pred, idx) => (
                          <tr key={idx}>
                            <td className="px-4 py-2 text-slate-400">{pred.actual}</td>
                            <td className="px-4 py-2 text-slate-400">{pred.predicted}</td>
                            <td className="px-4 py-2 text-slate-400">
                              {(pred.probability * 100).toFixed(1)}%
                            </td>
                            <td className="px-4 py-2">
                              {pred.actual === pred.predicted ? (
                                <span className="text-green-400 flex items-center gap-1">
                                  <CheckCircle className="w-4 h-4" /> Correct
                                </span>
                              ) : (
                                <span className="text-red-400 flex items-center gap-1">
                                  <X className="w-4 h-4" /> Wrong
                                </span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {trainResult.predictions.length > 10 && (
                    <p className="text-sm text-slate-500">
                      Showing 10 of {trainResult.predictions.length} predictions
                    </p>
                  )}
                </div>
              )}

              {activeResultTab === "code" && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-white">
                      Python Code
                    </h3>
                    <span className="text-sm text-slate-500">
                      Copy and run this in your own environment
                    </span>
                  </div>
                  <CodeBlock code={trainResult.generated_code} language="python" />
                </div>
              )}
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-4">
            <button
              onClick={resetAll}
              className="px-6 py-3 rounded-xl font-medium border border-slate-600 text-slate-300 hover:bg-slate-800"
            >
              Try Different Data
            </button>
            <button
              onClick={() => setStep("configure")}
              className="px-6 py-3 rounded-xl font-medium bg-purple-500 text-white hover:bg-purple-600"
            >
              Adjust Configuration
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
