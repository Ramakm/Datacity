"use client";

import { useState, useRef, useCallback } from "react";
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
import { apiClient, DataUploadResponse, TrainResponse } from "@/lib/api";
import CodeBlock from "./CodeBlock";
import ScatterPlot from "./ScatterPlot";
import MetricsDisplay from "./MetricsDisplay";

type Step = "upload" | "configure" | "results";

const SAMPLE_CSV = `size_sqft,bedrooms,age_years,distance_to_center,price
1500,3,10,5.2,250000
1800,4,5,3.1,320000
1200,2,15,8.5,180000
2200,4,2,2.0,420000
1000,2,20,10.0,150000
1650,3,8,4.5,280000
2000,4,3,1.5,380000
1400,3,12,6.0,230000
1750,3,6,3.8,300000
900,1,25,12.0,120000
2500,5,1,1.0,500000
1300,2,18,7.5,190000
1900,4,4,2.5,350000
1100,2,22,9.0,160000
2100,4,7,3.0,390000`;

export default function TryItTab() {
  const [step, setStep] = useState<Step>("upload");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Data state
  const [uploadedData, setUploadedData] = useState<DataUploadResponse | null>(null);
  const [rawData, setRawData] = useState<Record<string, unknown>[]>([]);
  const [csvText, setCsvText] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Configuration state
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [testSize, setTestSize] = useState(0.2);

  // Results state
  const [trainResult, setTrainResult] = useState<TrainResponse | null>(null);
  const [activeResultTab, setActiveResultTab] = useState<"metrics" | "code" | "predictions">("metrics");

  const resetAll = () => {
    setStep("upload");
    setUploadedData(null);
    setRawData([]);
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
      setRawData(response.preview);
      setStep("configure");

      // Read full file for training
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
      setRawData(response.preview);
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
      // Parse full CSV for training
      const response = await apiClient.parseCsvText(csvText);
      const fullData = response.preview;

      // Actually we need to get all data, not just preview
      // Let's re-parse and use it
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

      const result = await apiClient.trainLinearRegression(
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
                "w-10 h-10 rounded-full flex items-center justify-center",
                step === s.id
                  ? "bg-primary-500 text-white"
                  : ["configure", "results"].indexOf(step) > ["upload", "configure", "results"].indexOf(s.id)
                  ? "bg-primary-100 text-primary-600"
                  : "bg-slate-200 text-slate-500"
              )}
            >
              <s.icon className="w-5 h-5" />
            </div>
            <span
              className={clsx(
                "text-sm font-medium hidden sm:block",
                step === s.id ? "text-primary-600" : "text-slate-500"
              )}
            >
              {s.label}
            </span>
            {index < 2 && (
              <div className="w-8 h-px bg-slate-200 hidden sm:block" />
            )}
          </div>
        ))}
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-red-800 font-medium">Error</p>
            <p className="text-red-700 text-sm">{error}</p>
          </div>
          <button
            onClick={() => setError(null)}
            className="ml-auto text-red-500 hover:text-red-700"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      )}

      {/* Step 1: Upload */}
      {step === "upload" && (
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-6">
            Load Your Data
          </h3>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* File Upload */}
            <div className="space-y-4">
              <div
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-slate-300 rounded-xl p-8 text-center cursor-pointer hover:border-primary-400 hover:bg-primary-50 transition-colors"
              >
                <Upload className="w-12 h-12 text-slate-400 mx-auto mb-4" />
                <p className="text-slate-700 font-medium mb-2">
                  Click to upload a CSV file
                </p>
                <p className="text-slate-500 text-sm">
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
                <p className="text-slate-700 font-medium">Or paste CSV data:</p>
                <button
                  onClick={loadSampleData}
                  className="text-sm text-primary-600 hover:text-primary-700 font-medium"
                >
                  Load sample data
                </button>
              </div>
              <textarea
                value={csvText}
                onChange={(e) => setCsvText(e.target.value)}
                placeholder="Paste your CSV data here..."
                className="w-full h-48 px-4 py-3 border border-slate-300 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-primary-500 font-mono text-sm resize-none"
              />
              <button
                onClick={handlePasteCSV}
                disabled={loading || !csvText.trim()}
                className={clsx(
                  "w-full py-3 rounded-xl font-medium transition-colors flex items-center justify-center gap-2",
                  csvText.trim()
                    ? "bg-primary-500 text-white hover:bg-primary-600"
                    : "bg-slate-200 text-slate-500 cursor-not-allowed"
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

          <div className="mt-6 p-4 bg-slate-50 rounded-lg">
            <p className="text-sm text-slate-600">
              <strong>Tip:</strong> Your CSV should have numeric columns for regression.
              The first row should contain column headers.
            </p>
          </div>
        </div>
      )}

      {/* Step 2: Configure */}
      {step === "configure" && uploadedData && (
        <div className="space-y-6">
          {/* Data Preview */}
          <div className="bg-white rounded-xl border border-slate-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-slate-800">
                Data Preview
              </h3>
              <span className="text-sm text-slate-500">
                {uploadedData.row_count} rows, {uploadedData.columns.length} columns
              </span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-slate-50">
                  <tr>
                    {uploadedData.columns.map((col) => (
                      <th
                        key={col}
                        className="px-4 py-2 text-left text-slate-700 font-semibold whitespace-nowrap"
                      >
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {uploadedData.preview.slice(0, 5).map((row, idx) => (
                    <tr key={idx}>
                      {uploadedData.columns.map((col) => (
                        <td
                          key={col}
                          className="px-4 py-2 text-slate-600 whitespace-nowrap"
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
          <div className="bg-white rounded-xl border border-slate-200 p-6">
            <h3 className="text-lg font-semibold text-slate-800 mb-4">
              Configure Model
            </h3>

            <div className="space-y-6">
              {/* Feature Selection */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-3">
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
                          ? "bg-slate-100 text-slate-400 cursor-not-allowed"
                          : selectedFeatures.includes(col)
                          ? "bg-primary-500 text-white"
                          : "bg-slate-100 text-slate-700 hover:bg-slate-200"
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
                <label className="block text-sm font-medium text-slate-700 mb-3">
                  Select Target Column (what to predict)
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
                          ? "bg-slate-100 text-slate-400 cursor-not-allowed"
                          : col === targetColumn
                          ? "bg-green-500 text-white"
                          : "bg-slate-100 text-slate-700 hover:bg-slate-200"
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
                <label className="block text-sm font-medium text-slate-700 mb-3">
                  Test Set Size: {Math.round(testSize * 100)}%
                </label>
                <input
                  type="range"
                  min="10"
                  max="50"
                  value={testSize * 100}
                  onChange={(e) => setTestSize(Number(e.target.value) / 100)}
                  className="w-full max-w-xs"
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
              className="px-6 py-3 rounded-xl font-medium border border-slate-300 text-slate-700 hover:bg-slate-50"
            >
              Start Over
            </button>
            <button
              onClick={handleTrain}
              disabled={loading || selectedFeatures.length === 0 || !targetColumn}
              className={clsx(
                "flex-1 py-3 rounded-xl font-medium transition-colors flex items-center justify-center gap-2",
                selectedFeatures.length > 0 && targetColumn
                  ? "bg-primary-500 text-white hover:bg-primary-600"
                  : "bg-slate-200 text-slate-500 cursor-not-allowed"
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
          <div className="bg-green-50 border border-green-200 rounded-lg p-4 flex items-center gap-3">
            <CheckCircle className="w-6 h-6 text-green-500" />
            <div>
              <p className="text-green-800 font-medium">Model trained successfully!</p>
              <p className="text-green-700 text-sm">{trainResult.message}</p>
            </div>
          </div>

          {/* Result Tabs */}
          <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
            <div className="border-b border-slate-200 flex">
              {[
                { id: "metrics", label: "Metrics & Coefficients", icon: BarChart3 },
                { id: "predictions", label: "Predictions Chart", icon: Table },
                { id: "code", label: "Generated Code", icon: Code },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveResultTab(tab.id as typeof activeResultTab)}
                  className={clsx(
                    "flex items-center gap-2 px-6 py-4 text-sm font-medium border-b-2 transition-colors",
                    activeResultTab === tab.id
                      ? "border-primary-500 text-primary-600 bg-primary-50"
                      : "border-transparent text-slate-600 hover:text-slate-800 hover:bg-slate-50"
                  )}
                >
                  <tab.icon className="w-4 h-4" />
                  {tab.label}
                </button>
              ))}
            </div>

            <div className="p-6">
              {activeResultTab === "metrics" && (
                <MetricsDisplay
                  metrics={trainResult.metrics}
                  coefficients={trainResult.coefficients}
                  intercept={trainResult.intercept}
                />
              )}

              {activeResultTab === "predictions" && (
                <ScatterPlot predictions={trainResult.predictions} />
              )}

              {activeResultTab === "code" && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-slate-800">
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
              className="px-6 py-3 rounded-xl font-medium border border-slate-300 text-slate-700 hover:bg-slate-50"
            >
              Try Different Data
            </button>
            <button
              onClick={() => setStep("configure")}
              className="px-6 py-3 rounded-xl font-medium bg-primary-500 text-white hover:bg-primary-600"
            >
              Adjust Configuration
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
