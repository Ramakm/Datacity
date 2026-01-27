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
      // Parse CSV data for training
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

  const getStepIndex = (s: Step) => {
    const steps: Step[] = ["upload", "configure", "results"];
    return steps.indexOf(s);
  };

  return (
    <div className="space-y-6">
      {/* Progress Steps */}
      <div className="flex items-center justify-center gap-2">
        {[
          { id: "upload" as Step, label: "01_UPLOAD", icon: Upload },
          { id: "configure" as Step, label: "02_CONFIG", icon: Table },
          { id: "results" as Step, label: "03_RESULTS", icon: BarChart3 },
        ].map((s, index) => (
          <div key={s.id} className="flex items-center gap-2">
            <div
              className={clsx(
                "w-10 h-10 border-2 flex items-center justify-center transition-colors",
                step === s.id
                  ? "bg-terminal-black border-terminal-black text-terminal-mint"
                  : getStepIndex(step) > getStepIndex(s.id)
                  ? "bg-terminal-accent border-terminal-accent text-terminal-mint"
                  : "bg-terminal-panel border-terminal-grid text-terminal-grid"
              )}
            >
              <s.icon className="w-5 h-5" />
            </div>
            <span
              className={clsx(
                "font-mono text-xs uppercase tracking-terminal hidden sm:block",
                step === s.id ? "text-terminal-black font-bold" : "text-terminal-black/50"
              )}
            >
              {s.label}
            </span>
            {index < 2 && (
              <div className="w-8 h-0.5 bg-terminal-grid hidden sm:block" />
            )}
          </div>
        ))}
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border-2 border-red-500 p-4 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-mono text-xs font-bold uppercase tracking-terminal text-red-800">
              ERROR
            </p>
            <p className="font-mono text-xs text-red-700">{error}</p>
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
        <div className="bg-terminal-panel border-2 border-terminal-black p-6">
          <h3 className="font-mono text-sm font-bold uppercase tracking-terminal text-terminal-black mb-6">
            {"//"} DATA INPUT
          </h3>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* File Upload */}
            <div className="space-y-4">
              <div
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-terminal-black p-8 text-center cursor-pointer hover:bg-terminal-black hover:border-terminal-black group transition-colors"
              >
                <Upload className="w-12 h-12 text-terminal-black group-hover:text-terminal-mint mx-auto mb-4" />
                <p className="font-mono text-xs font-bold uppercase tracking-terminal text-terminal-black group-hover:text-terminal-mint mb-2">
                  CLICK TO UPLOAD CSV
                </p>
                <p className="font-mono text-xs text-terminal-black/50 group-hover:text-terminal-mint/70">
                  OR DRAG AND DROP
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
                <p className="font-mono text-xs font-bold uppercase tracking-terminal text-terminal-black">
                  OR PASTE CSV DATA:
                </p>
                <button
                  onClick={loadSampleData}
                  className="font-mono text-xs uppercase tracking-terminal text-terminal-accent hover:text-terminal-black font-bold"
                >
                  [LOAD SAMPLE]
                </button>
              </div>
              <textarea
                value={csvText}
                onChange={(e) => setCsvText(e.target.value)}
                placeholder="PASTE CSV DATA HERE..."
                className="w-full h-48 px-4 py-3 border-2 border-terminal-black bg-terminal-bg font-mono text-xs resize-none focus:outline-none focus:bg-terminal-panel placeholder:text-terminal-grid"
              />
              <button
                onClick={handlePasteCSV}
                disabled={loading || !csvText.trim()}
                className={clsx(
                  "w-full py-3 font-mono text-xs font-bold uppercase tracking-terminal border-2 transition-colors flex items-center justify-center gap-2",
                  csvText.trim()
                    ? "bg-terminal-black border-terminal-black text-terminal-mint hover:bg-terminal-accent hover:border-terminal-accent"
                    : "bg-terminal-grid border-terminal-grid text-terminal-panel cursor-not-allowed"
                )}
              >
                {loading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <FileText className="w-5 h-5" />
                )}
                PARSE CSV
              </button>
            </div>
          </div>

          <div className="mt-6 p-4 bg-terminal-black text-terminal-mint">
            <p className="font-mono text-xs">
              <span className="text-terminal-accent font-bold">TIP:</span> CSV SHOULD HAVE NUMERIC COLUMNS FOR REGRESSION.
              FIRST ROW = COLUMN HEADERS.
            </p>
          </div>
        </div>
      )}

      {/* Step 2: Configure */}
      {step === "configure" && uploadedData && (
        <div className="space-y-6">
          {/* Data Preview */}
          <div className="bg-terminal-panel border-2 border-terminal-black p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-mono text-sm font-bold uppercase tracking-terminal text-terminal-black">
                {"//"} DATA PREVIEW
              </h3>
              <span className="font-mono text-xs text-terminal-black/50">
                {uploadedData.row_count} ROWS // {uploadedData.columns.length} COLS
              </span>
            </div>
            <div className="overflow-x-auto border-2 border-terminal-black">
              <table className="w-full font-mono text-xs">
                <thead className="bg-terminal-black text-terminal-mint">
                  <tr>
                    {uploadedData.columns.map((col) => (
                      <th
                        key={col}
                        className="px-4 py-2 text-left font-bold uppercase tracking-terminal whitespace-nowrap"
                      >
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-terminal-grid">
                  {uploadedData.preview.slice(0, 5).map((row, idx) => (
                    <tr key={idx} className="hover:bg-terminal-accent/10">
                      {uploadedData.columns.map((col) => (
                        <td
                          key={col}
                          className="px-4 py-2 text-terminal-black whitespace-nowrap"
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
          <div className="bg-terminal-panel border-2 border-terminal-black p-6">
            <h3 className="font-mono text-sm font-bold uppercase tracking-terminal text-terminal-black mb-6">
              {"//"} MODEL CONFIGURATION
            </h3>

            <div className="space-y-6">
              {/* Feature Selection */}
              <div>
                <label className="block font-mono text-xs font-bold uppercase tracking-terminal text-terminal-black mb-3">
                  SELECT FEATURE COLUMNS (INPUTS)
                </label>
                <div className="flex flex-wrap gap-2">
                  {uploadedData.numeric_columns.map((col) => (
                    <button
                      key={col}
                      onClick={() => toggleFeature(col)}
                      disabled={col === targetColumn}
                      className={clsx(
                        "px-4 py-2 border-2 font-mono text-xs font-bold uppercase tracking-terminal transition-colors",
                        col === targetColumn
                          ? "border-terminal-grid bg-terminal-grid/30 text-terminal-grid cursor-not-allowed"
                          : selectedFeatures.includes(col)
                          ? "border-terminal-accent bg-terminal-accent text-terminal-mint"
                          : "border-terminal-black text-terminal-black hover:bg-terminal-black hover:text-terminal-mint"
                      )}
                    >
                      {col}
                      {selectedFeatures.includes(col) && (
                        <CheckCircle className="inline w-4 h-4 ml-1" />
                      )}
                    </button>
                  ))}
                </div>
                <p className="font-mono text-xs text-terminal-black/50 mt-2">
                  SELECTED: {selectedFeatures.length} FEATURE(S)
                </p>
              </div>

              {/* Target Selection */}
              <div>
                <label className="block font-mono text-xs font-bold uppercase tracking-terminal text-terminal-black mb-3">
                  SELECT TARGET COLUMN (OUTPUT TO PREDICT)
                </label>
                <div className="flex flex-wrap gap-2">
                  {uploadedData.numeric_columns.map((col) => (
                    <button
                      key={col}
                      onClick={() => setTarget(col)}
                      disabled={selectedFeatures.includes(col)}
                      className={clsx(
                        "px-4 py-2 border-2 font-mono text-xs font-bold uppercase tracking-terminal transition-colors",
                        selectedFeatures.includes(col)
                          ? "border-terminal-grid bg-terminal-grid/30 text-terminal-grid cursor-not-allowed"
                          : col === targetColumn
                          ? "border-terminal-warning bg-terminal-warning text-terminal-black"
                          : "border-terminal-black text-terminal-black hover:bg-terminal-black hover:text-terminal-mint"
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
                <label className="block font-mono text-xs font-bold uppercase tracking-terminal text-terminal-black mb-3">
                  TEST SET SIZE: {Math.round(testSize * 100)}%
                </label>
                <input
                  type="range"
                  min="10"
                  max="50"
                  value={testSize * 100}
                  onChange={(e) => setTestSize(Number(e.target.value) / 100)}
                  className="w-full max-w-xs accent-terminal-accent"
                />
                <p className="font-mono text-xs text-terminal-black/50 mt-1">
                  TRAIN: {Math.round((1 - testSize) * 100)}% // TEST: {Math.round(testSize * 100)}%
                </p>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-4">
            <button
              onClick={resetAll}
              className="px-6 py-3 font-mono text-xs font-bold uppercase tracking-terminal border-2 border-terminal-black text-terminal-black hover:bg-terminal-black hover:text-terminal-mint transition-colors"
            >
              RESTART
            </button>
            <button
              onClick={handleTrain}
              disabled={loading || selectedFeatures.length === 0 || !targetColumn}
              className={clsx(
                "flex-1 py-3 font-mono text-xs font-bold uppercase tracking-terminal border-2 transition-colors flex items-center justify-center gap-2",
                selectedFeatures.length > 0 && targetColumn
                  ? "bg-terminal-black border-terminal-black text-terminal-mint hover:bg-terminal-accent hover:border-terminal-accent"
                  : "bg-terminal-grid border-terminal-grid text-terminal-panel cursor-not-allowed"
              )}
            >
              {loading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Play className="w-5 h-5" />
              )}
              EXECUTE TRAINING
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Results */}
      {step === "results" && trainResult && (
        <div className="space-y-6">
          {/* Success Banner */}
          <div className="bg-terminal-accent/20 border-2 border-terminal-accent p-4 flex items-center gap-3">
            <CheckCircle className="w-6 h-6 text-terminal-accent" />
            <div>
              <p className="font-mono text-xs font-bold uppercase tracking-terminal text-terminal-black">
                MODEL TRAINED SUCCESSFULLY
              </p>
              <p className="font-mono text-xs text-terminal-black/70">{trainResult.message}</p>
            </div>
          </div>

          {/* Result Tabs */}
          <div className="bg-terminal-panel border-2 border-terminal-black overflow-hidden">
            <div className="border-b-2 border-terminal-black flex">
              {[
                { id: "metrics", label: "METRICS", icon: BarChart3 },
                { id: "predictions", label: "PREDICTIONS", icon: Table },
                { id: "code", label: "CODE", icon: Code },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveResultTab(tab.id as typeof activeResultTab)}
                  className={clsx(
                    "flex items-center gap-2 px-6 py-4 font-mono text-xs font-bold uppercase tracking-terminal border-b-2 transition-colors",
                    activeResultTab === tab.id
                      ? "border-terminal-black text-terminal-black bg-terminal-bg"
                      : "border-transparent text-terminal-black/50 hover:text-terminal-black hover:bg-terminal-panel/50"
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
                    <h3 className="font-mono text-sm font-bold uppercase tracking-terminal text-terminal-black">
                      {"//"} GENERATED PYTHON CODE
                    </h3>
                    <span className="font-mono text-xs text-terminal-black/50">
                      COPY AND EXECUTE IN YOUR ENVIRONMENT
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
              className="px-6 py-3 font-mono text-xs font-bold uppercase tracking-terminal border-2 border-terminal-black text-terminal-black hover:bg-terminal-black hover:text-terminal-mint transition-colors"
            >
              NEW DATA
            </button>
            <button
              onClick={() => setStep("configure")}
              className="px-6 py-3 font-mono text-xs font-bold uppercase tracking-terminal border-2 bg-terminal-black border-terminal-black text-terminal-mint hover:bg-terminal-accent hover:border-terminal-accent transition-colors"
            >
              RECONFIGURE
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
