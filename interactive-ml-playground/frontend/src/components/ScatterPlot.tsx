"use client";

import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

interface PredictionPoint {
  actual: number;
  predicted: number;
}

interface ScatterPlotProps {
  predictions: PredictionPoint[];
  title?: string;
}

export default function ScatterPlot({
  predictions,
  title = "Actual vs Predicted Values",
}: ScatterPlotProps) {
  // Prepare data for the chart
  const data = predictions.map((p, index) => ({
    x: p.actual,
    y: p.predicted,
    index,
  }));

  // Calculate min and max for the reference line
  const allValues = [...predictions.map((p) => p.actual), ...predictions.map((p) => p.predicted)];
  const minVal = Math.min(...allValues);
  const maxVal = Math.max(...allValues);
  const padding = (maxVal - minVal) * 0.1;

  // Perfect prediction line data
  const perfectLine = [
    { x: minVal - padding, y: minVal - padding },
    { x: maxVal + padding, y: maxVal + padding },
  ];

  return (
    <div className="w-full">
      <h3 className="text-lg font-semibold text-slate-800 mb-4">{title}</h3>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 20, right: 20, bottom: 40, left: 60 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              type="number"
              dataKey="x"
              name="Actual"
              domain={[minVal - padding, maxVal + padding]}
              tickFormatter={(value) => value.toLocaleString()}
              label={{
                value: "Actual Values",
                position: "bottom",
                offset: 20,
                style: { fill: "#64748b", fontSize: 12 },
              }}
              stroke="#94a3b8"
            />
            <YAxis
              type="number"
              dataKey="y"
              name="Predicted"
              domain={[minVal - padding, maxVal + padding]}
              tickFormatter={(value) => value.toLocaleString()}
              label={{
                value: "Predicted Values",
                angle: -90,
                position: "insideLeft",
                offset: -45,
                style: { fill: "#64748b", fontSize: 12 },
              }}
              stroke="#94a3b8"
            />
            <Tooltip
              cursor={{ strokeDasharray: "3 3" }}
              content={({ payload }) => {
                if (!payload || payload.length === 0) return null;
                const point = payload[0].payload;
                return (
                  <div className="bg-white border border-slate-200 rounded-lg p-3 shadow-lg">
                    <p className="text-sm text-slate-600">
                      Actual: <span className="font-semibold">{point.x.toLocaleString()}</span>
                    </p>
                    <p className="text-sm text-slate-600">
                      Predicted: <span className="font-semibold">{point.y.toLocaleString()}</span>
                    </p>
                    <p className="text-sm text-slate-500 mt-1">
                      Error: {(point.y - point.x).toLocaleString()}
                    </p>
                  </div>
                );
              }}
            />
            <Legend
              verticalAlign="top"
              height={36}
            />
            {/* Perfect prediction reference line */}
            <Scatter
              name="Perfect Prediction"
              data={perfectLine}
              line={{ stroke: "#22c55e", strokeWidth: 2, strokeDasharray: "5 5" }}
              shape={() => <></>}
              legendType="line"
            />
            {/* Actual predictions */}
            <Scatter
              name="Predictions"
              data={data}
              fill="#0ea5e9"
              opacity={0.7}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <p className="text-sm text-slate-500 text-center mt-2">
        Points closer to the green dashed line indicate better predictions
      </p>
    </div>
  );
}
