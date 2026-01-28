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
import { apiClient, DataUploadResponse, KMeansTrainResponse } from "@/lib/api";
import CodeBlock from "./CodeBlock";

type Step = "upload" | "configure" | "results";

const SAMPLE_CSV = `annual_income,spending_score,age
15,39,20
16,81,21
17,6,22
18,77,23
19,40,24
20,76,25
21,6,26
22,94,27
23,3,28
24,72,29
25,14,30
26,99,31
27,15,32
28,77,33
29,13,34
30,79,35
31,35,36
32,66,37
75,5,45
76,10,46
77,7,47
78,14,48
79,7,49
80,12,50
81,8,51
82,10,52
83,9,53`;

export default function KMeansTryItTab() {
  const [step, setStep] = useState<Step>("upload");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Data state
  const [uploadedData, setUploadedData] = useState<DataUploadResponse | null>(null);
  const [csvText, setCsvText] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Configuration state
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);
  const [nClusters, setNClusters] = useState(3);

  // Results state
  const [trainResult, setTrainResult] = useState<KMeansTrainResponse | null>(null);
  const [activeResultTab, setActiveResultTab] = useState<"metrics" | "code" | "clusters">("metrics");

  const resetAll = () => {
    setStep("upload");
    setUploadedData(null);
    setCsvText("");
    setSelectedFeatures([]);
    setTrainResult(null);
    setError(null);
    setNClusters(3);
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
    setSelectedFeatures((prev) =>
      prev.includes(column)
        ? prev.filter((c) => c !== column)
        : [...prev, column]
    );
  };

  const handleTrain = async () => {
    if (selectedFeatures.length < 2) {
      setError("Please select at least two feature columns for clustering");
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

      const result = await apiClient.trainKMeans(
        data,
        selectedFeatures,
        nClusters
      );

      setTrainResult(result);
      setStep("results");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Clustering failed");
    } finally {
      setLoading(false);
    }
  };

  const getStepIndex = (s: Step) => {
    const steps: Step[] = ["upload", "configure", "results"];
    return steps.indexOf(s);
  };

  const getClusterColor = (cluster: number) => {
    const colors = ["#1a5c3a", "#c4a000", "#dc2626", "#2563eb", "#7c3aed", "#db2777"];
    return colors[cluster % colors.length];
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
                  [LOAD CUSTOMER SAMPLE]
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
              <span className="text-terminal-accent font-bold">TIP:</span> K-MEANS REQUIRES NUMERIC FEATURES ONLY.
              NO TARGET COLUMN NEEDED - THIS IS UNSUPERVISED LEARNING.
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
              {"//"} CLUSTERING CONFIGURATION
            </h3>

            <div className="space-y-6">
              {/* Feature Selection */}
              <div>
                <label className="block font-mono text-xs font-bold uppercase tracking-terminal text-terminal-black mb-3">
                  SELECT FEATURE COLUMNS FOR CLUSTERING (MIN 2)
                </label>
                <div className="flex flex-wrap gap-2">
                  {uploadedData.numeric_columns.map((col) => (
                    <button
                      key={col}
                      onClick={() => toggleFeature(col)}
                      className={clsx(
                        "px-4 py-2 border-2 font-mono text-xs font-bold uppercase tracking-terminal transition-colors",
                        selectedFeatures.includes(col)
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

              {/* Number of Clusters */}
              <div>
                <label className="block font-mono text-xs font-bold uppercase tracking-terminal text-terminal-black mb-3">
                  NUMBER OF CLUSTERS (K): {nClusters}
                </label>
                <input
                  type="range"
                  min="2"
                  max="10"
                  value={nClusters}
                  onChange={(e) => setNClusters(Number(e.target.value))}
                  className="w-full max-w-xs accent-terminal-accent"
                />
                <p className="font-mono text-xs text-terminal-black/50 mt-1">
                  USE ELBOW METHOD OR SILHOUETTE ANALYSIS TO FIND OPTIMAL K
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
              disabled={loading || selectedFeatures.length < 2}
              className={clsx(
                "flex-1 py-3 font-mono text-xs font-bold uppercase tracking-terminal border-2 transition-colors flex items-center justify-center gap-2",
                selectedFeatures.length >= 2
                  ? "bg-terminal-black border-terminal-black text-terminal-mint hover:bg-terminal-accent hover:border-terminal-accent"
                  : "bg-terminal-grid border-terminal-grid text-terminal-panel cursor-not-allowed"
              )}
            >
              {loading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Play className="w-5 h-5" />
              )}
              EXECUTE K-MEANS CLUSTERING
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
                K-MEANS CLUSTERING COMPLETE
              </p>
              <p className="font-mono text-xs text-terminal-black/70">
                K={trainResult.metrics.n_clusters} | {trainResult.metrics.n_samples} SAMPLES CLUSTERED
              </p>
            </div>
          </div>

          {/* Result Tabs */}
          <div className="bg-terminal-panel border-2 border-terminal-black overflow-hidden">
            <div className="border-b-2 border-terminal-black flex">
              {[
                { id: "metrics", label: "METRICS", icon: BarChart3 },
                { id: "clusters", label: "CLUSTERS", icon: Table },
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
                <div className="space-y-6">
                  {/* Main metrics */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-terminal-black p-4 text-center">
                      <div className="text-2xl font-bold text-terminal-accent">
                        {trainResult.metrics.silhouette_score.toFixed(3)}
                      </div>
                      <div className="font-mono text-xs text-terminal-grid uppercase">SILHOUETTE</div>
                    </div>
                    <div className="bg-terminal-black p-4 text-center">
                      <div className="text-2xl font-bold text-terminal-mint">
                        {trainResult.metrics.inertia.toFixed(1)}
                      </div>
                      <div className="font-mono text-xs text-terminal-grid uppercase">INERTIA</div>
                    </div>
                    <div className="bg-terminal-black p-4 text-center">
                      <div className="text-2xl font-bold text-terminal-mint">
                        {trainResult.metrics.calinski_harabasz_score.toFixed(1)}
                      </div>
                      <div className="font-mono text-xs text-terminal-grid uppercase">CALINSKI-HARABASZ</div>
                    </div>
                    <div className="bg-terminal-black p-4 text-center">
                      <div className="text-2xl font-bold text-terminal-warning">
                        {trainResult.metrics.davies_bouldin_score.toFixed(3)}
                      </div>
                      <div className="font-mono text-xs text-terminal-grid uppercase">DAVIES-BOULDIN</div>
                    </div>
                  </div>

                  {/* Cluster Statistics */}
                  <div className="border-2 border-terminal-black p-4">
                    <h4 className="font-mono text-xs font-bold uppercase tracking-terminal text-terminal-black mb-4">
                      {"//"} CLUSTER STATISTICS
                    </h4>
                    <div className="space-y-3">
                      {trainResult.metrics.cluster_stats.map((stat) => (
                        <div key={stat.cluster_id} className="flex items-center gap-4">
                          <div
                            className="w-4 h-4 flex-shrink-0"
                            style={{ backgroundColor: getClusterColor(stat.cluster_id) }}
                          />
                          <span className="font-mono text-xs font-bold text-terminal-black w-24">
                            CLUSTER_{stat.cluster_id}
                          </span>
                          <div className="flex-1 h-4 bg-terminal-grid">
                            <div
                              className="h-full"
                              style={{
                                width: `${stat.percentage}%`,
                                backgroundColor: getClusterColor(stat.cluster_id),
                              }}
                            />
                          </div>
                          <span className="font-mono text-xs text-terminal-black/70 w-20 text-right">
                            {stat.size} ({stat.percentage.toFixed(1)}%)
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Centroids */}
                  <div className="border-2 border-terminal-black p-4">
                    <h4 className="font-mono text-xs font-bold uppercase tracking-terminal text-terminal-black mb-4">
                      {"//"} CLUSTER CENTROIDS
                    </h4>
                    <div className="overflow-x-auto">
                      <table className="w-full font-mono text-xs">
                        <thead className="bg-terminal-black text-terminal-mint">
                          <tr>
                            <th className="px-4 py-2 text-left">CLUSTER</th>
                            {Object.keys(trainResult.centroids[0] || {}).map((key) => (
                              <th key={key} className="px-4 py-2 text-left uppercase">{key}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-terminal-grid">
                          {trainResult.centroids.map((centroid, idx) => (
                            <tr key={idx}>
                              <td className="px-4 py-2">
                                <span
                                  className="inline-block w-3 h-3 mr-2"
                                  style={{ backgroundColor: getClusterColor(idx) }}
                                />
                                {idx}
                              </td>
                              {Object.values(centroid).map((val, vidx) => (
                                <td key={vidx} className="px-4 py-2">
                                  {typeof val === 'number' ? val.toFixed(2) : val}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}

              {activeResultTab === "clusters" && (
                <div className="overflow-x-auto border-2 border-terminal-black">
                  <table className="w-full font-mono text-xs">
                    <thead className="bg-terminal-black text-terminal-mint">
                      <tr>
                        <th className="px-4 py-2 text-left font-bold uppercase">INDEX</th>
                        <th className="px-4 py-2 text-left font-bold uppercase">CLUSTER</th>
                        {Object.keys(trainResult.assignments[0]?.features || {}).map((key) => (
                          <th key={key} className="px-4 py-2 text-left font-bold uppercase">{key}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-terminal-grid">
                      {trainResult.assignments.slice(0, 20).map((assignment, idx) => (
                        <tr key={idx} className="hover:bg-terminal-accent/10">
                          <td className="px-4 py-2 text-terminal-black">{assignment.original_index}</td>
                          <td className="px-4 py-2">
                            <span
                              className="inline-block px-2 py-1 text-white font-bold"
                              style={{ backgroundColor: getClusterColor(assignment.cluster) }}
                            >
                              CLUSTER_{assignment.cluster}
                            </span>
                          </td>
                          {Object.values(assignment.features).map((val, vidx) => (
                            <td key={vidx} className="px-4 py-2 text-terminal-black">
                              {typeof val === 'number' ? val.toFixed(2) : val}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {trainResult.assignments.length > 20 && (
                    <p className="p-4 text-center font-mono text-xs text-terminal-black/50">
                      SHOWING 20 OF {trainResult.assignments.length} SAMPLES
                    </p>
                  )}
                </div>
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
