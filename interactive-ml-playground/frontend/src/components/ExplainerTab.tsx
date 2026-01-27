"use client";

import { useState } from "react";
import {
  ChevronRight,
  ChevronLeft,
  TrendingUp,
  Target,
  Calculator,
  CheckCircle,
} from "lucide-react";
import clsx from "clsx";

interface StoryStep {
  id: number;
  title: string;
  icon: React.ReactNode;
  content: React.ReactNode;
  visual: React.ReactNode;
}

const storySteps: StoryStep[] = [
  {
    id: 1,
    title: "The Problem",
    icon: <Target className="w-5 h-5" />,
    content: (
      <div className="space-y-4">
        <p className="text-lg text-slate-300 leading-relaxed">
          Imagine you&apos;re a real estate agent. Every day, clients ask:{" "}
          <span className="font-semibold text-cyan-400">
            &quot;How much is this house worth?&quot;
          </span>
        </p>
        <p className="text-slate-400">
          You notice a pattern: bigger houses tend to cost more. But how can you
          turn this intuition into a precise prediction?
        </p>
        <div className="bg-cyan-500/10 rounded-lg p-4 border border-cyan-500/20">
          <p className="text-cyan-300 font-medium">
            This is a regression problem: predicting a continuous number (price)
            from input features (size, bedrooms, etc.)
          </p>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-slate-800/50 rounded-xl p-6 h-full flex items-center justify-center">
        <div className="space-y-3 w-full max-w-xs">
          <div className="flex items-center justify-between p-3 bg-slate-700/50 rounded-lg border border-slate-600/50">
            <span className="text-slate-400">1,000 sqft</span>
            <span className="font-semibold text-white">$150,000</span>
          </div>
          <div className="flex items-center justify-between p-3 bg-slate-700/50 rounded-lg border border-slate-600/50">
            <span className="text-slate-400">1,500 sqft</span>
            <span className="font-semibold text-white">$225,000</span>
          </div>
          <div className="flex items-center justify-between p-3 bg-slate-700/50 rounded-lg border border-slate-600/50">
            <span className="text-slate-400">2,000 sqft</span>
            <span className="font-semibold text-white">$300,000</span>
          </div>
          <div className="flex items-center justify-between p-3 bg-cyan-500/10 rounded-lg border-2 border-cyan-500/30 border-dashed">
            <span className="text-cyan-400">1,750 sqft</span>
            <span className="font-semibold text-cyan-400">???</span>
          </div>
        </div>
      </div>
    ),
  },
  {
    id: 2,
    title: "The Line of Best Fit",
    icon: <TrendingUp className="w-5 h-5" />,
    content: (
      <div className="space-y-4">
        <p className="text-lg text-slate-300 leading-relaxed">
          Linear Regression finds the{" "}
          <span className="font-semibold text-cyan-400">
            best straight line
          </span>{" "}
          through your data points.
        </p>
        <p className="text-slate-400">
          This line minimizes the total distance from all points to the line.
          It&apos;s the line that &quot;best fits&quot; your data.
        </p>
        <div className="bg-slate-800/80 rounded-lg p-4 font-mono text-center text-lg border border-slate-700/50">
          <span className="text-cyan-400 font-bold">y</span> ={" "}
          <span className="text-green-400 font-bold">m</span>
          <span className="text-white">x</span> +{" "}
          <span className="text-amber-400 font-bold">b</span>
        </div>
        <ul className="space-y-2 text-slate-400">
          <li className="flex items-start gap-2">
            <span className="text-cyan-400 font-bold">y</span> = predicted
            value (house price)
          </li>
          <li className="flex items-start gap-2">
            <span className="text-white font-bold">x</span> = input feature
            (house size)
          </li>
          <li className="flex items-start gap-2">
            <span className="text-green-400 font-bold">m</span> = slope
            (coefficient) - how much y changes for each unit of x
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-400 font-bold">b</span> = intercept -
            the starting point when x is 0
          </li>
        </ul>
      </div>
    ),
    visual: (
      <div className="bg-slate-800/50 rounded-xl p-6 h-full">
        <svg viewBox="0 0 300 200" className="w-full h-full">
          {/* Axes */}
          <line x1="40" y1="160" x2="280" y2="160" stroke="#475569" strokeWidth="2" />
          <line x1="40" y1="160" x2="40" y2="20" stroke="#475569" strokeWidth="2" />

          {/* Axis labels */}
          <text x="160" y="185" textAnchor="middle" className="fill-slate-500 text-xs">
            House Size (sqft)
          </text>
          <text x="15" y="90" textAnchor="middle" transform="rotate(-90, 15, 90)" className="fill-slate-500 text-xs">
            Price ($)
          </text>

          {/* Data points */}
          <circle cx="60" cy="140" r="6" fill="#06b6d4" opacity="0.8" />
          <circle cx="100" cy="120" r="6" fill="#06b6d4" opacity="0.8" />
          <circle cx="120" cy="105" r="6" fill="#06b6d4" opacity="0.8" />
          <circle cx="160" cy="85" r="6" fill="#06b6d4" opacity="0.8" />
          <circle cx="200" cy="70" r="6" fill="#06b6d4" opacity="0.8" />
          <circle cx="220" cy="55" r="6" fill="#06b6d4" opacity="0.8" />
          <circle cx="260" cy="40" r="6" fill="#06b6d4" opacity="0.8" />

          {/* Best fit line */}
          <line x1="50" y1="145" x2="270" y2="35" stroke="#22c55e" strokeWidth="3" strokeDasharray="5,3" />

          {/* Legend */}
          <circle cx="200" cy="175" r="4" fill="#06b6d4" />
          <text x="210" y="179" className="fill-slate-400 text-xs">Data</text>
          <line x1="235" y1="175" x2="255" y2="175" stroke="#22c55e" strokeWidth="2" strokeDasharray="3,2" />
          <text x="260" y="179" className="fill-slate-400 text-xs">Fit</text>
        </svg>
      </div>
    ),
  },
  {
    id: 3,
    title: "How It Learns",
    icon: <Calculator className="w-5 h-5" />,
    content: (
      <div className="space-y-4">
        <p className="text-lg text-slate-300 leading-relaxed">
          The algorithm finds the best line by{" "}
          <span className="font-semibold text-cyan-400">
            minimizing errors
          </span>
          . It uses a method called &quot;Ordinary Least Squares&quot; (OLS).
        </p>
        <div className="bg-slate-800/80 rounded-lg p-4 space-y-3 border border-slate-700/50">
          <p className="font-medium text-white">The process:</p>
          <ol className="space-y-2 text-slate-400 list-decimal list-inside">
            <li>Start with a random line</li>
            <li>Calculate the error for each point (predicted - actual)</li>
            <li>Square each error (so negative errors count too)</li>
            <li>Sum all squared errors</li>
            <li>Adjust the line to minimize this sum</li>
          </ol>
        </div>
        <p className="text-slate-400">
          The math finds the exact slope (m) and intercept (b) that make the
          total squared error as small as possible.
        </p>
      </div>
    ),
    visual: (
      <div className="bg-slate-800/50 rounded-xl p-6 h-full">
        <svg viewBox="0 0 300 200" className="w-full h-full">
          {/* Axes */}
          <line x1="40" y1="160" x2="280" y2="160" stroke="#475569" strokeWidth="2" />
          <line x1="40" y1="160" x2="40" y2="20" stroke="#475569" strokeWidth="2" />

          {/* Best fit line */}
          <line x1="50" y1="145" x2="270" y2="35" stroke="#22c55e" strokeWidth="2" />

          {/* Data points with error lines */}
          <circle cx="80" cy="125" r="5" fill="#06b6d4" />
          <line x1="80" y1="125" x2="80" y2="135" stroke="#ef4444" strokeWidth="2" strokeDasharray="3,2" />

          <circle cx="120" cy="95" r="5" fill="#06b6d4" />
          <line x1="120" y1="95" x2="120" y2="115" stroke="#ef4444" strokeWidth="2" strokeDasharray="3,2" />

          <circle cx="160" cy="100" r="5" fill="#06b6d4" />
          <line x1="160" y1="100" x2="160" y2="90" stroke="#ef4444" strokeWidth="2" strokeDasharray="3,2" />

          <circle cx="200" cy="55" r="5" fill="#06b6d4" />
          <line x1="200" y1="55" x2="200" y2="70" stroke="#ef4444" strokeWidth="2" strokeDasharray="3,2" />

          <circle cx="240" cy="50" r="5" fill="#06b6d4" />
          <line x1="240" y1="50" x2="240" y2="48" stroke="#ef4444" strokeWidth="2" strokeDasharray="3,2" />

          {/* Error label */}
          <text x="140" y="75" className="fill-red-400 text-xs font-medium">errors</text>

          {/* Goal annotation */}
          <text x="150" y="185" textAnchor="middle" className="fill-slate-500 text-xs">
            Goal: minimize total squared error
          </text>
        </svg>
      </div>
    ),
  },
  {
    id: 4,
    title: "Multiple Features",
    icon: <TrendingUp className="w-5 h-5" />,
    content: (
      <div className="space-y-4">
        <p className="text-lg text-slate-300 leading-relaxed">
          Real predictions often use{" "}
          <span className="font-semibold text-cyan-400">
            multiple features
          </span>
          . House price depends on size AND bedrooms AND location AND age...
        </p>
        <div className="bg-slate-800/80 rounded-lg p-4 font-mono text-sm overflow-x-auto border border-slate-700/50">
          <p className="text-white">
            <span className="text-cyan-400">price</span> ={" "}
            <span className="text-green-400">150</span> × size +{" "}
            <span className="text-green-400">5000</span> × bedrooms +{" "}
            <span className="text-red-400">-2000</span> × age +{" "}
            <span className="text-amber-400">50000</span>
          </p>
        </div>
        <p className="text-slate-400">
          Each feature gets its own coefficient (weight). A positive coefficient
          means the feature increases the prediction; negative decreases it.
        </p>
        <div className="bg-green-500/10 rounded-lg p-4 border border-green-500/20">
          <p className="text-green-300 font-medium">
            The coefficients tell you which features matter most and in what
            direction!
          </p>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-slate-800/50 rounded-xl p-6 h-full flex items-center justify-center">
        <div className="space-y-4 w-full max-w-sm">
          <div className="text-center mb-4">
            <span className="text-lg font-semibold text-white">
              Feature Importance
            </span>
          </div>
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-slate-400">Size (sqft)</span>
                <span className="font-medium text-green-400">+$150/sqft</span>
              </div>
              <div className="h-4 bg-slate-700 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-green-500 to-green-400 rounded-full" style={{ width: "85%" }}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-slate-400">Bedrooms</span>
                <span className="font-medium text-green-400">+$5,000/room</span>
              </div>
              <div className="h-4 bg-slate-700 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-green-500 to-green-400 rounded-full" style={{ width: "60%" }}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-slate-400">Age (years)</span>
                <span className="font-medium text-red-400">-$2,000/year</span>
              </div>
              <div className="h-4 bg-slate-700 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-red-500 to-red-400 rounded-full" style={{ width: "40%" }}></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    ),
  },
  {
    id: 5,
    title: "Measuring Success",
    icon: <CheckCircle className="w-5 h-5" />,
    content: (
      <div className="space-y-4">
        <p className="text-lg text-slate-300 leading-relaxed">
          How do we know if our model is good? We use{" "}
          <span className="font-semibold text-cyan-400">metrics</span> to
          measure prediction accuracy.
        </p>
        <div className="space-y-3">
          <div className="bg-slate-800/80 rounded-lg p-4 border border-slate-700/50">
            <p className="font-semibold text-white mb-1">
              R² Score (0 to 1)
            </p>
            <p className="text-sm text-slate-400">
              How much variance in the data our model explains. 1.0 = perfect,
              0.0 = useless.
            </p>
          </div>
          <div className="bg-slate-800/80 rounded-lg p-4 border border-slate-700/50">
            <p className="font-semibold text-white mb-1">RMSE</p>
            <p className="text-sm text-slate-400">
              Root Mean Squared Error - average prediction error in the same
              units as our target (dollars).
            </p>
          </div>
          <div className="bg-slate-800/80 rounded-lg p-4 border border-slate-700/50">
            <p className="font-semibold text-white mb-1">MAE</p>
            <p className="text-sm text-slate-400">
              Mean Absolute Error - simpler average of all absolute errors.
            </p>
          </div>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-slate-800/50 rounded-xl p-6 h-full flex items-center justify-center">
        <div className="space-y-6 w-full max-w-xs">
          <div className="text-center">
            <div className="text-4xl font-bold text-cyan-400 mb-1">0.92</div>
            <div className="text-slate-400">R² Score</div>
            <div className="text-sm text-green-400 font-medium">Excellent!</div>
          </div>
          <div className="h-px bg-slate-700" />
          <div className="grid grid-cols-2 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-white">$12,450</div>
              <div className="text-sm text-slate-400">RMSE</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-white">$9,200</div>
              <div className="text-sm text-slate-400">MAE</div>
            </div>
          </div>
          <div className="bg-cyan-500/10 rounded-lg p-3 text-center border border-cyan-500/20">
            <p className="text-sm text-cyan-300">
              On average, predictions are within $9,200 of actual prices
            </p>
          </div>
        </div>
      </div>
    ),
  },
];

export default function ExplainerTab() {
  const [currentStep, setCurrentStep] = useState(0);

  const goToStep = (step: number) => {
    if (step >= 0 && step < storySteps.length) {
      setCurrentStep(step);
    }
  };

  const currentStory = storySteps[currentStep];

  return (
    <div className="space-y-6">
      {/* Progress indicator */}
      <div className="flex items-center justify-center gap-2">
        {storySteps.map((step, index) => (
          <button
            key={step.id}
            onClick={() => goToStep(index)}
            className={clsx(
              "w-3 h-3 rounded-full transition-all",
              index === currentStep
                ? "bg-cyan-500 scale-125 shadow-lg shadow-cyan-500/50"
                : index < currentStep
                ? "bg-cyan-500/50"
                : "bg-slate-600"
            )}
            aria-label={`Go to step ${index + 1}: ${step.title}`}
          />
        ))}
      </div>

      {/* Step header */}
      <div className="text-center">
        <div className="inline-flex items-center gap-2 bg-cyan-500/10 text-cyan-400 px-4 py-2 rounded-full text-sm font-medium mb-2 border border-cyan-500/20">
          {currentStory.icon}
          Step {currentStep + 1} of {storySteps.length}
        </div>
        <h2 className="text-2xl font-bold text-white">
          {currentStory.title}
        </h2>
      </div>

      {/* Content grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-fade-in" key={currentStep}>
        {/* Text content */}
        <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700/50 p-6">
          {currentStory.content}
        </div>

        {/* Visual */}
        <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700/50 p-2 min-h-[300px]">
          {currentStory.visual}
        </div>
      </div>

      {/* Navigation */}
      <div className="flex items-center justify-between">
        <button
          onClick={() => goToStep(currentStep - 1)}
          disabled={currentStep === 0}
          className={clsx(
            "flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors",
            currentStep === 0
              ? "text-slate-600 cursor-not-allowed"
              : "text-slate-300 hover:bg-slate-800"
          )}
        >
          <ChevronLeft className="w-5 h-5" />
          Previous
        </button>

        <span className="text-sm text-slate-500">
          {currentStep + 1} / {storySteps.length}
        </span>

        <button
          onClick={() => goToStep(currentStep + 1)}
          disabled={currentStep === storySteps.length - 1}
          className={clsx(
            "flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors",
            currentStep === storySteps.length - 1
              ? "text-slate-600 cursor-not-allowed"
              : "bg-gradient-to-r from-cyan-500 to-blue-600 text-white hover:shadow-lg hover:shadow-cyan-500/25"
          )}
        >
          Next
          <ChevronRight className="w-5 h-5" />
        </button>
      </div>

      {/* End CTA */}
      {currentStep === storySteps.length - 1 && (
        <div className="text-center py-6 bg-gradient-to-r from-cyan-500/10 to-purple-500/10 rounded-xl border border-cyan-500/20">
          <p className="text-lg font-semibold text-white mb-2">
            Ready to try it yourself?
          </p>
          <p className="text-slate-400 mb-4">
            Switch to the &quot;Try It With Data&quot; tab to upload your own dataset!
          </p>
        </div>
      )}
    </div>
  );
}
