"use client";

import { TrendingUp, Target, BarChart3, Activity } from "lucide-react";
import clsx from "clsx";

interface Metrics {
  r2_score: number;
  mse: number;
  rmse: number;
  mae: number;
  train_samples: number;
  test_samples: number;
}

interface Coefficients {
  [key: string]: number;
}

interface MetricsDisplayProps {
  metrics: Metrics;
  coefficients: Coefficients;
  intercept: number;
}

function getR2Quality(r2: number): { label: string; color: string } {
  if (r2 >= 0.9) return { label: "Excellent", color: "text-green-600" };
  if (r2 >= 0.7) return { label: "Good", color: "text-primary-600" };
  if (r2 >= 0.5) return { label: "Moderate", color: "text-amber-600" };
  return { label: "Poor", color: "text-red-600" };
}

export default function MetricsDisplay({
  metrics,
  coefficients,
  intercept,
}: MetricsDisplayProps) {
  const r2Quality = getR2Quality(metrics.r2_score);

  return (
    <div className="space-y-6">
      {/* Main metrics */}
      <div>
        <h3 className="text-lg font-semibold text-slate-800 mb-4">
          Model Performance
        </h3>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-gradient-to-br from-primary-50 to-primary-100 rounded-xl p-4 border border-primary-200">
            <div className="flex items-center gap-2 mb-2">
              <Target className="w-5 h-5 text-primary-600" />
              <span className="text-sm text-primary-700 font-medium">R² Score</span>
            </div>
            <div className="text-2xl font-bold text-primary-800">
              {metrics.r2_score.toFixed(4)}
            </div>
            <div className={clsx("text-sm font-medium mt-1", r2Quality.color)}>
              {r2Quality.label}
            </div>
          </div>

          <div className="bg-white rounded-xl p-4 border border-slate-200">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-5 h-5 text-slate-600" />
              <span className="text-sm text-slate-600 font-medium">RMSE</span>
            </div>
            <div className="text-2xl font-bold text-slate-800">
              {metrics.rmse.toLocaleString()}
            </div>
            <div className="text-sm text-slate-500 mt-1">Root Mean Sq. Error</div>
          </div>

          <div className="bg-white rounded-xl p-4 border border-slate-200">
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 className="w-5 h-5 text-slate-600" />
              <span className="text-sm text-slate-600 font-medium">MAE</span>
            </div>
            <div className="text-2xl font-bold text-slate-800">
              {metrics.mae.toLocaleString()}
            </div>
            <div className="text-sm text-slate-500 mt-1">Mean Absolute Error</div>
          </div>

          <div className="bg-white rounded-xl p-4 border border-slate-200">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-5 h-5 text-slate-600" />
              <span className="text-sm text-slate-600 font-medium">MSE</span>
            </div>
            <div className="text-2xl font-bold text-slate-800">
              {metrics.mse.toLocaleString()}
            </div>
            <div className="text-sm text-slate-500 mt-1">Mean Squared Error</div>
          </div>
        </div>
      </div>

      {/* Dataset info */}
      <div className="flex gap-4 text-sm text-slate-600 bg-slate-50 rounded-lg p-3">
        <span>
          Training samples: <strong>{metrics.train_samples}</strong>
        </span>
        <span className="text-slate-300">|</span>
        <span>
          Test samples: <strong>{metrics.test_samples}</strong>
        </span>
      </div>

      {/* Coefficients */}
      <div>
        <h3 className="text-lg font-semibold text-slate-800 mb-4">
          Model Coefficients
        </h3>
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
          <table className="w-full">
            <thead className="bg-slate-50">
              <tr>
                <th className="text-left px-4 py-3 text-sm font-semibold text-slate-700">
                  Feature
                </th>
                <th className="text-right px-4 py-3 text-sm font-semibold text-slate-700">
                  Coefficient
                </th>
                <th className="text-center px-4 py-3 text-sm font-semibold text-slate-700">
                  Impact
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {Object.entries(coefficients).map(([feature, coef]) => (
                <tr key={feature} className="hover:bg-slate-50">
                  <td className="px-4 py-3 text-slate-800 font-medium">
                    {feature}
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-slate-700">
                    {coef.toFixed(6)}
                  </td>
                  <td className="px-4 py-3 text-center">
                    <span
                      className={clsx(
                        "inline-flex items-center px-2 py-1 rounded text-xs font-medium",
                        coef >= 0
                          ? "bg-green-100 text-green-700"
                          : "bg-red-100 text-red-700"
                      )}
                    >
                      {coef >= 0 ? "+" : "-"} {coef >= 0 ? "Increase" : "Decrease"}
                    </span>
                  </td>
                </tr>
              ))}
              <tr className="bg-slate-50">
                <td className="px-4 py-3 text-slate-800 font-medium">
                  Intercept (baseline)
                </td>
                <td className="px-4 py-3 text-right font-mono text-slate-700">
                  {intercept.toFixed(6)}
                </td>
                <td className="px-4 py-3 text-center">
                  <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-slate-200 text-slate-700">
                    Base value
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="text-sm text-slate-500 mt-2">
          Each coefficient shows how much the target changes when that feature increases by 1 unit.
        </p>
      </div>
    </div>
  );
}
