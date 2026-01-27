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

  const [uploadedData, setUploadedData] = useState<DataUploadResponse | null>(null);
  const [csvText, setCsvText] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [testSize, setTestSize] = useState(0.2);

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
      prev.includes(column) ? prev.filter((c) => c !== column) : [...prev, column]
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

      const result = await apiClient.trainLogisticRegression(data, selectedFeatures, targetColumn, testSize);
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
          { id: "upload", label: "DATA_INPUT", icon: Upload },
          { id: "configure", label: "CONFIGURE", icon: Table },
          { id: "results", label: "RESULTS", icon: BarChart3 },
        ].map((s, index) => (
          <div key={s.id} className="flex items-center gap-2">
            <div
              className={clsx(
                "w-10 h-10 flex items-center justify-center transition-colors border-2",
                step === s.id
                  ? "bg-terminal-black text-terminal-mint border-terminal-black"
                  : ["configure", "results"].indexOf(step) > ["upload", "configure", "results"].indexOf(s.id)
                  ? "bg-terminal-accent/20 text-terminal-accent border-terminal-accent"
                  : "bg-terminal-panel text-terminal-black/50 border-terminal-grid"
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
            {index < 2 && <div className="w-8 h-0.5 bg-terminal-grid hidden sm:block" />}
          </div>
        ))}
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-500/10 border-2 border-red-500 p-4 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-mono text-xs font-bold text-red-600">ERROR</p>
            <p className="font-mono text-xs text-red-600">{error}</p>
          </div>
          <button onClick={() => setError(null)} className="ml-auto text-red-500 hover:text-red-600">
            <X className="w-5 h-5" />
          </button>
        </div>
      )}

      {/* Step 1: Upload */}
      {step === "upload" && (
        <div className="bg-terminal-panel border-2 border-terminal-black p-6">
          <h3 className="font-mono text-sm font-bold uppercase tracking-terminal text-terminal-black mb-6">
            LOAD DATA
          </h3>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-terminal-black p-8 text-center cursor-pointer hover:bg-terminal-black hover:text-terminal-mint transition-colors"
              >
                <Upload className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p className="font-mono text-xs font-bold uppercase tracking-terminal mb-2">
                  CLICK TO UPLOAD CSV
                </p>
                <p className="font-mono text-xs opacity-50">OR DRAG AND DROP</p>
                <input ref={fileInputRef} type="file" accept=".csv" onChange={handleFileUpload} className="hidden" />
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <p className="font-mono text-xs font-bold uppercase tracking-terminal text-terminal-black">
                  OR PASTE CSV:
                </p>
                <button
                  onClick={loadSampleData}
                  className="font-mono text-xs font-bold uppercase tracking-terminal text-terminal-accent hover:underline"
                >
                  LOAD SAMPLE
                </button>
              </div>
              <textarea
                value={csvText}
                onChange={(e) => setCsvText(e.target.value)}
                placeholder="PASTE CSV DATA HERE..."
                className="w-full h-48 px-4 py-3 bg-terminal-panel border-2 border-terminal-black text-terminal-black placeholder-terminal-black/30 font-mono text-xs resize-none focus:outline-none focus:border-terminal-accent"
              />
              <button
                onClick={handlePasteCSV}
                disabled={loading || !csvText.trim()}
                className={clsx(
                  "w-full py-3 font-mono text-xs font-bold uppercase tracking-terminal border-2 transition-colors flex items-center justify-center gap-2",
                  csvText.trim()
                    ? "bg-terminal-black text-terminal-mint border-terminal-black hover:bg-terminal-accent hover:border-terminal-accent"
                    : "bg-terminal-grid text-terminal-black/50 border-terminal-grid cursor-not-allowed"
                )}
              >
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <FileText className="w-4 h-4" />}
                PARSE CSV
              </button>
            </div>
          </div>

          <div className="mt-6 p-4 bg-terminal-accent/10 border-2 border-terminal-accent">
            <p className="font-mono text-xs text-terminal-accent">
              <span className="font-bold">NOTE:</span> TARGET COLUMN SHOULD CONTAIN BINARY VALUES (0/1).
              SAMPLE DATA: DIABETES PREDICTION DATASET.
            </p>
          </div>
        </div>
      )}

      {/* Step 2: Configure */}
      {step === "configure" && uploadedData && (
        <div className="space-y-6">
          <div className="bg-terminal-panel border-2 border-terminal-black p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-mono text-sm font-bold uppercase tracking-terminal text-terminal-black">
                DATA PREVIEW
              </h3>
              <span className="font-mono text-xs text-terminal-black/50">
                {uploadedData.row_count} ROWS // {uploadedData.columns.length} COLS
              </span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr>
                    {uploadedData.columns.map((col) => (
                      <th key={col} className="px-4 py-2 text-left font-mono text-xs font-bold uppercase tracking-terminal">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {uploadedData.preview.slice(0, 5).map((row, idx) => (
                    <tr key={idx}>
                      {uploadedData.columns.map((col) => (
                        <td key={col} className="px-4 py-2 font-mono text-xs">
                          {String(row[col] ?? "")}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="bg-terminal-panel border-2 border-terminal-black p-6">
            <h3 className="font-mono text-sm font-bold uppercase tracking-terminal text-terminal-black mb-4">
              CONFIGURE MODEL
            </h3>

            <div className="space-y-6">
              <div>
                <label className="block font-mono text-xs font-bold uppercase tracking-terminal text-terminal-black mb-3">
                  SELECT FEATURES (INPUTS)
                </label>
                <div className="flex flex-wrap gap-2">
                  {uploadedData.numeric_columns.map((col) => (
                    <button
                      key={col}
                      onClick={() => toggleFeature(col)}
                      disabled={col === targetColumn}
                      className={clsx(
                        "px-4 py-2 font-mono text-xs font-bold uppercase tracking-terminal border-2 transition-colors",
                        col === targetColumn
                          ? "bg-terminal-grid text-terminal-black/50 border-terminal-grid cursor-not-allowed"
                          : selectedFeatures.includes(col)
                          ? "bg-terminal-black text-terminal-mint border-terminal-black"
                          : "bg-terminal-panel text-terminal-black border-terminal-black hover:bg-terminal-black hover:text-terminal-mint"
                      )}
                    >
                      {col}
                      {selectedFeatures.includes(col) && <CheckCircle className="inline w-3 h-3 ml-1" />}
                    </button>
                  ))}
                </div>
                <p className="font-mono text-xs text-terminal-black/50 mt-2">
                  SELECTED: {selectedFeatures.length} FEATURE(S)
                </p>
              </div>

              <div>
                <label className="block font-mono text-xs font-bold uppercase tracking-terminal text-terminal-black mb-3">
                  SELECT TARGET (OUTPUT - BINARY 0/1)
                </label>
                <div className="flex flex-wrap gap-2">
                  {uploadedData.numeric_columns.map((col) => (
                    <button
                      key={col}
                      onClick={() => setTarget(col)}
                      disabled={selectedFeatures.includes(col)}
                      className={clsx(
                        "px-4 py-2 font-mono text-xs font-bold uppercase tracking-terminal border-2 transition-colors",
                        selectedFeatures.includes(col)
                          ? "bg-terminal-grid text-terminal-black/50 border-terminal-grid cursor-not-allowed"
                          : col === targetColumn
                          ? "bg-terminal-accent text-white border-terminal-accent"
                          : "bg-terminal-panel text-terminal-black border-terminal-black hover:bg-terminal-accent hover:text-white hover:border-terminal-accent"
                      )}
                    >
                      {col}
                      {col === targetColumn && <CheckCircle className="inline w-3 h-3 ml-1" />}
                    </button>
                  ))}
                </div>
              </div>

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
                  className="w-full max-w-xs"
                />
                <p className="font-mono text-xs text-terminal-black/50 mt-1">
                  TRAIN: {Math.round((1 - testSize) * 100)}% // TEST: {Math.round(testSize * 100)}%
                </p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <button
              onClick={resetAll}
              className="px-6 py-3 font-mono text-xs font-bold uppercase tracking-terminal border-2 border-terminal-black text-terminal-black hover:bg-terminal-black hover:text-terminal-mint transition-colors"
            >
              RESET
            </button>
            <button
              onClick={handleTrain}
              disabled={loading || selectedFeatures.length === 0 || !targetColumn}
              className={clsx(
                "flex-1 py-3 font-mono text-xs font-bold uppercase tracking-terminal border-2 transition-colors flex items-center justify-center gap-2",
                selectedFeatures.length > 0 && targetColumn
                  ? "bg-terminal-black text-terminal-mint border-terminal-black hover:bg-terminal-accent hover:border-terminal-accent"
                  : "bg-terminal-grid text-terminal-black/50 border-terminal-grid cursor-not-allowed"
              )}
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
              EXECUTE TRAINING
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Results */}
      {step === "results" && trainResult && (
        <div className="space-y-6">
          <div className="bg-terminal-accent/10 border-2 border-terminal-accent p-4 flex items-center gap-3">
            <CheckCircle className="w-6 h-6 text-terminal-accent" />
            <div>
              <p className="font-mono text-xs font-bold text-terminal-accent">TRAINING COMPLETE</p>
              <p className="font-mono text-xs text-terminal-accent">{trainResult.message}</p>
            </div>
          </div>

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
                    "flex items-center gap-2 px-6 py-4 font-mono text-xs font-bold uppercase tracking-terminal border-b-3 transition-colors",
                    activeResultTab === tab.id
                      ? "border-terminal-black text-terminal-black bg-terminal-panel"
                      : "border-transparent text-terminal-black/50 hover:text-terminal-black hover:bg-terminal-grid/50"
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
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-terminal-black p-4 text-center">
                      <div className="text-2xl font-mono font-bold text-terminal-mint">
                        {(trainResult.metrics.accuracy * 100).toFixed(1)}%
                      </div>
                      <div className="font-mono text-xs text-terminal-grid">ACCURACY</div>
                    </div>
                    <div className="bg-terminal-black p-4 text-center">
                      <div className="text-2xl font-mono font-bold text-terminal-mint">
                        {(trainResult.metrics.precision * 100).toFixed(1)}%
                      </div>
                      <div className="font-mono text-xs text-terminal-grid">PRECISION</div>
                    </div>
                    <div className="bg-terminal-black p-4 text-center">
                      <div className="text-2xl font-mono font-bold text-terminal-mint">
                        {(trainResult.metrics.recall * 100).toFixed(1)}%
                      </div>
                      <div className="font-mono text-xs text-terminal-grid">RECALL</div>
                    </div>
                    <div className="bg-terminal-black p-4 text-center">
                      <div className="text-2xl font-mono font-bold text-terminal-warning">
                        {(trainResult.metrics.f1_score * 100).toFixed(1)}%
                      </div>
                      <div className="font-mono text-xs text-terminal-grid">F1 SCORE</div>
                    </div>
                  </div>

                  <div>
                    <h4 className="font-mono text-sm font-bold uppercase tracking-terminal text-terminal-black mb-4">
                      CONFUSION MATRIX
                    </h4>
                    <div className="grid grid-cols-2 gap-2 max-w-xs">
                      <div className="bg-terminal-accent/20 border-2 border-terminal-accent p-4 text-center">
                        <div className="text-xl font-mono font-bold text-terminal-accent">
                          {trainResult.metrics.confusion_matrix.true_negative}
                        </div>
                        <div className="font-mono text-xs text-terminal-black/50">TRUE_NEG</div>
                      </div>
                      <div className="bg-red-500/20 border-2 border-red-500 p-4 text-center">
                        <div className="text-xl font-mono font-bold text-red-500">
                          {trainResult.metrics.confusion_matrix.false_positive}
                        </div>
                        <div className="font-mono text-xs text-terminal-black/50">FALSE_POS</div>
                      </div>
                      <div className="bg-red-500/20 border-2 border-red-500 p-4 text-center">
                        <div className="text-xl font-mono font-bold text-red-500">
                          {trainResult.metrics.confusion_matrix.false_negative}
                        </div>
                        <div className="font-mono text-xs text-terminal-black/50">FALSE_NEG</div>
                      </div>
                      <div className="bg-terminal-accent/20 border-2 border-terminal-accent p-4 text-center">
                        <div className="text-xl font-mono font-bold text-terminal-accent">
                          {trainResult.metrics.confusion_matrix.true_positive}
                        </div>
                        <div className="font-mono text-xs text-terminal-black/50">TRUE_POS</div>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h4 className="font-mono text-sm font-bold uppercase tracking-terminal text-terminal-black mb-4">
                      COEFFICIENTS
                    </h4>
                    <div className="space-y-2">
                      {Object.entries(trainResult.coefficients).map(([feature, coef]) => (
                        <div key={feature} className="flex items-center justify-between font-mono text-xs">
                          <span className="text-terminal-black">{feature}</span>
                          <span className={clsx("font-bold", coef > 0 ? "text-terminal-accent" : "text-red-500")}>
                            {coef > 0 ? "+" : ""}{coef.toFixed(4)}
                          </span>
                        </div>
                      ))}
                      <div className="flex items-center justify-between pt-2 border-t-2 border-terminal-black font-mono text-xs">
                        <span className="text-terminal-black">INTERCEPT</span>
                        <span className="font-bold text-terminal-warning">{trainResult.intercept.toFixed(4)}</span>
                      </div>
                    </div>
                  </div>

                  <div className="flex gap-4 font-mono text-xs text-terminal-black/50">
                    <span>TRAIN_SAMPLES: {trainResult.metrics.train_samples}</span>
                    <span>TEST_SAMPLES: {trainResult.metrics.test_samples}</span>
                  </div>
                </div>
              )}

              {activeResultTab === "predictions" && (
                <div className="space-y-4">
                  <h4 className="font-mono text-sm font-bold uppercase tracking-terminal text-terminal-black">
                    TEST PREDICTIONS
                  </h4>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr>
                          <th className="px-4 py-2 text-left font-mono text-xs font-bold uppercase tracking-terminal">ACTUAL</th>
                          <th className="px-4 py-2 text-left font-mono text-xs font-bold uppercase tracking-terminal">PREDICTED</th>
                          <th className="px-4 py-2 text-left font-mono text-xs font-bold uppercase tracking-terminal">PROBABILITY</th>
                          <th className="px-4 py-2 text-left font-mono text-xs font-bold uppercase tracking-terminal">STATUS</th>
                        </tr>
                      </thead>
                      <tbody>
                        {trainResult.predictions.slice(0, 10).map((pred, idx) => (
                          <tr key={idx}>
                            <td className="px-4 py-2 font-mono text-xs">{pred.actual}</td>
                            <td className="px-4 py-2 font-mono text-xs">{pred.predicted}</td>
                            <td className="px-4 py-2 font-mono text-xs">{(pred.probability * 100).toFixed(1)}%</td>
                            <td className="px-4 py-2 font-mono text-xs">
                              {pred.actual === pred.predicted ? (
                                <span className="text-terminal-accent flex items-center gap-1">
                                  <CheckCircle className="w-3 h-3" /> CORRECT
                                </span>
                              ) : (
                                <span className="text-red-500 flex items-center gap-1">
                                  <X className="w-3 h-3" /> WRONG
                                </span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {trainResult.predictions.length > 10 && (
                    <p className="font-mono text-xs text-terminal-black/50">
                      SHOWING 10 OF {trainResult.predictions.length} PREDICTIONS
                    </p>
                  )}
                </div>
              )}

              {activeResultTab === "code" && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="font-mono text-sm font-bold uppercase tracking-terminal text-terminal-black">
                      PYTHON CODE
                    </h3>
                    <span className="font-mono text-xs text-terminal-black/50">COPY AND RUN IN YOUR ENVIRONMENT</span>
                  </div>
                  <CodeBlock code={trainResult.generated_code} language="python" />
                </div>
              )}
            </div>
          </div>

          <div className="flex items-center gap-4">
            <button
              onClick={resetAll}
              className="px-6 py-3 font-mono text-xs font-bold uppercase tracking-terminal border-2 border-terminal-black text-terminal-black hover:bg-terminal-black hover:text-terminal-mint transition-colors"
            >
              NEW DATA
            </button>
            <button
              onClick={() => setStep("configure")}
              className="px-6 py-3 font-mono text-xs font-bold uppercase tracking-terminal bg-terminal-black text-terminal-mint border-2 border-terminal-black hover:bg-terminal-accent hover:border-terminal-accent transition-colors"
            >
              RECONFIGURE
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
