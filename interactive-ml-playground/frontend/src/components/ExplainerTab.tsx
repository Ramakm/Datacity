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
    title: "PROBLEM DEFINITION",
    icon: <Target className="w-4 h-4" />,
    content: (
      <div className="space-y-4">
        <p className="font-mono text-sm text-terminal-black leading-relaxed">
          SCENARIO: Real estate price prediction. Input: property features. Output: continuous value (price).
        </p>
        <p className="font-mono text-xs text-terminal-black/70">
          Observation: Larger properties correlate with higher prices. Objective: Transform this pattern into precise predictions.
        </p>
        <div className="bg-terminal-accent/10 border-2 border-terminal-accent p-4">
          <p className="font-mono text-xs text-terminal-accent font-bold">
            CLASSIFICATION: REGRESSION PROBLEM // CONTINUOUS OUTPUT // SUPERVISED LEARNING
          </p>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full flex items-center justify-center">
        <div className="space-y-2 w-full max-w-xs font-mono text-xs">
          <div className="flex items-center justify-between p-2 border border-terminal-grid text-terminal-mint">
            <span>1,000 SQFT</span>
            <span className="font-bold">$150,000</span>
          </div>
          <div className="flex items-center justify-between p-2 border border-terminal-grid text-terminal-mint">
            <span>1,500 SQFT</span>
            <span className="font-bold">$225,000</span>
          </div>
          <div className="flex items-center justify-between p-2 border border-terminal-grid text-terminal-mint">
            <span>2,000 SQFT</span>
            <span className="font-bold">$300,000</span>
          </div>
          <div className="flex items-center justify-between p-2 border-2 border-dashed border-terminal-warning text-terminal-warning">
            <span>1,750 SQFT</span>
            <span className="font-bold">???</span>
          </div>
        </div>
      </div>
    ),
  },
  {
    id: 2,
    title: "LINE OF BEST FIT",
    icon: <TrendingUp className="w-4 h-4" />,
    content: (
      <div className="space-y-4">
        <p className="font-mono text-sm text-terminal-black leading-relaxed">
          Linear Regression computes the optimal straight line through data points.
        </p>
        <div className="bg-terminal-black p-4 font-mono text-center text-base border-2 border-terminal-black">
          <span className="text-terminal-accent font-bold">y</span>
          <span className="text-terminal-mint"> = </span>
          <span className="text-terminal-warning font-bold">m</span>
          <span className="text-terminal-mint">x + </span>
          <span className="text-red-400 font-bold">b</span>
        </div>
        <ul className="space-y-2 font-mono text-xs text-terminal-black/80">
          <li><span className="text-terminal-accent font-bold">y</span> = predicted output (price)</li>
          <li><span className="text-terminal-black font-bold">x</span> = input feature (size)</li>
          <li><span className="text-terminal-warning font-bold">m</span> = slope coefficient (rate of change)</li>
          <li><span className="text-red-400 font-bold">b</span> = y-intercept (baseline value)</li>
        </ul>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full">
        <svg viewBox="0 0 300 200" className="w-full h-full">
          <line x1="40" y1="160" x2="280" y2="160" stroke="#d4d4c8" strokeWidth="1" />
          <line x1="40" y1="160" x2="40" y2="20" stroke="#d4d4c8" strokeWidth="1" />
          <text x="160" y="185" textAnchor="middle" fill="#d4d4c8" className="text-xs" style={{ fontFamily: 'monospace' }}>SIZE (SQFT)</text>
          <text x="15" y="90" textAnchor="middle" transform="rotate(-90, 15, 90)" fill="#d4d4c8" className="text-xs" style={{ fontFamily: 'monospace' }}>PRICE ($)</text>
          <circle cx="60" cy="140" r="4" fill="#cfeee3" />
          <circle cx="100" cy="120" r="4" fill="#cfeee3" />
          <circle cx="120" cy="105" r="4" fill="#cfeee3" />
          <circle cx="160" cy="85" r="4" fill="#cfeee3" />
          <circle cx="200" cy="70" r="4" fill="#cfeee3" />
          <circle cx="220" cy="55" r="4" fill="#cfeee3" />
          <circle cx="260" cy="40" r="4" fill="#cfeee3" />
          <line x1="50" y1="145" x2="270" y2="35" stroke="#1a5c3a" strokeWidth="2" />
        </svg>
      </div>
    ),
  },
  {
    id: 3,
    title: "LEARNING ALGORITHM",
    icon: <Calculator className="w-4 h-4" />,
    content: (
      <div className="space-y-4">
        <p className="font-mono text-sm text-terminal-black leading-relaxed">
          Optimization via Ordinary Least Squares (OLS). Objective: minimize total squared error.
        </p>
        <div className="bg-terminal-panel border-2 border-terminal-black p-4 space-y-2">
          <p className="font-mono text-xs font-bold text-terminal-black">PROCEDURE:</p>
          <ol className="space-y-1 font-mono text-xs text-terminal-black/80 list-decimal list-inside">
            <li>Initialize random line parameters</li>
            <li>Compute error: predicted - actual</li>
            <li>Square each error value</li>
            <li>Sum all squared errors</li>
            <li>Optimize to minimize sum</li>
          </ol>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full">
        <svg viewBox="0 0 300 200" className="w-full h-full">
          <line x1="40" y1="160" x2="280" y2="160" stroke="#d4d4c8" strokeWidth="1" />
          <line x1="40" y1="160" x2="40" y2="20" stroke="#d4d4c8" strokeWidth="1" />
          <line x1="50" y1="145" x2="270" y2="35" stroke="#1a5c3a" strokeWidth="2" />
          <circle cx="80" cy="125" r="4" fill="#cfeee3" />
          <line x1="80" y1="125" x2="80" y2="135" stroke="#c4a000" strokeWidth="2" />
          <circle cx="120" cy="95" r="4" fill="#cfeee3" />
          <line x1="120" y1="95" x2="120" y2="115" stroke="#c4a000" strokeWidth="2" />
          <circle cx="160" cy="100" r="4" fill="#cfeee3" />
          <line x1="160" y1="100" x2="160" y2="90" stroke="#c4a000" strokeWidth="2" />
          <circle cx="200" cy="55" r="4" fill="#cfeee3" />
          <line x1="200" y1="55" x2="200" y2="70" stroke="#c4a000" strokeWidth="2" />
          <text x="150" y="185" textAnchor="middle" fill="#c4a000" className="text-xs" style={{ fontFamily: 'monospace' }}>MINIMIZE ERROR</text>
        </svg>
      </div>
    ),
  },
  {
    id: 4,
    title: "MULTIVARIATE ANALYSIS",
    icon: <TrendingUp className="w-4 h-4" />,
    content: (
      <div className="space-y-4">
        <p className="font-mono text-sm text-terminal-black leading-relaxed">
          Real-world predictions utilize multiple input features simultaneously.
        </p>
        <div className="bg-terminal-black p-4 font-mono text-xs overflow-x-auto border-2 border-terminal-black">
          <p className="text-terminal-mint">
            <span className="text-terminal-accent">price</span> =
            <span className="text-terminal-warning"> 150</span>*size +
            <span className="text-terminal-warning"> 5000</span>*beds +
            <span className="text-red-400"> -2000</span>*age +
            <span className="text-terminal-mint"> 50000</span>
          </p>
        </div>
        <div className="bg-terminal-accent/10 border-2 border-terminal-accent p-3">
          <p className="font-mono text-xs text-terminal-accent font-bold">
            COEFFICIENTS INDICATE FEATURE IMPORTANCE AND DIRECTION OF INFLUENCE
          </p>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full flex items-center justify-center">
        <div className="space-y-3 w-full max-w-sm font-mono text-xs">
          <div className="text-terminal-mint text-center font-bold mb-4">FEATURE WEIGHTS</div>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-terminal-grid">SIZE_SQFT</span>
              <span className="text-terminal-accent font-bold">+$150/sqft</span>
            </div>
            <div className="h-2 bg-terminal-grid"><div className="h-full bg-terminal-accent" style={{ width: "85%" }}></div></div>
            <div className="flex justify-between items-center">
              <span className="text-terminal-grid">BEDROOMS</span>
              <span className="text-terminal-accent font-bold">+$5,000/room</span>
            </div>
            <div className="h-2 bg-terminal-grid"><div className="h-full bg-terminal-accent" style={{ width: "60%" }}></div></div>
            <div className="flex justify-between items-center">
              <span className="text-terminal-grid">AGE_YEARS</span>
              <span className="text-red-400 font-bold">-$2,000/year</span>
            </div>
            <div className="h-2 bg-terminal-grid"><div className="h-full bg-red-400" style={{ width: "40%" }}></div></div>
          </div>
        </div>
      </div>
    ),
  },
  {
    id: 5,
    title: "PERFORMANCE METRICS",
    icon: <CheckCircle className="w-4 h-4" />,
    content: (
      <div className="space-y-4">
        <p className="font-mono text-sm text-terminal-black leading-relaxed">
          Model evaluation through standardized metrics.
        </p>
        <div className="space-y-2">
          <div className="bg-terminal-panel border-2 border-terminal-black p-3">
            <p className="font-mono text-xs font-bold text-terminal-black">R² SCORE (0-1)</p>
            <p className="font-mono text-xs text-terminal-black/70">Variance explained by model. 1.0 = perfect fit.</p>
          </div>
          <div className="bg-terminal-panel border-2 border-terminal-black p-3">
            <p className="font-mono text-xs font-bold text-terminal-black">RMSE</p>
            <p className="font-mono text-xs text-terminal-black/70">Root Mean Squared Error. Same units as target.</p>
          </div>
          <div className="bg-terminal-panel border-2 border-terminal-black p-3">
            <p className="font-mono text-xs font-bold text-terminal-black">MAE</p>
            <p className="font-mono text-xs text-terminal-black/70">Mean Absolute Error. Average prediction error.</p>
          </div>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full flex items-center justify-center">
        <div className="space-y-4 w-full max-w-xs font-mono">
          <div className="text-center">
            <div className="text-4xl font-bold text-terminal-accent">0.92</div>
            <div className="text-xs text-terminal-grid">R² SCORE</div>
            <div className="text-xs text-terminal-accent font-bold">EXCELLENT</div>
          </div>
          <div className="border-t border-terminal-grid/30 pt-4 grid grid-cols-2 gap-4 text-center">
            <div>
              <div className="text-xl font-bold text-terminal-mint">$12,450</div>
              <div className="text-xs text-terminal-grid">RMSE</div>
            </div>
            <div>
              <div className="text-xl font-bold text-terminal-mint">$9,200</div>
              <div className="text-xs text-terminal-grid">MAE</div>
            </div>
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
              "w-8 h-2 transition-all",
              index === currentStep
                ? "bg-terminal-black"
                : index < currentStep
                ? "bg-terminal-accent"
                : "bg-terminal-grid"
            )}
            aria-label={`Go to step ${index + 1}: ${step.title}`}
          />
        ))}
      </div>

      {/* Step header */}
      <div className="text-center">
        <div className="inline-flex items-center gap-2 bg-terminal-black text-terminal-mint px-4 py-2 font-mono text-xs uppercase tracking-terminal mb-4">
          {currentStory.icon}
          STEP {currentStep + 1} OF {storySteps.length}
        </div>
        <h2 className="heading-terminal text-2xl text-terminal-black">
          {currentStory.title}
        </h2>
      </div>

      {/* Content grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 animate-fade-in" key={currentStep}>
        {/* Text content */}
        <div className="bg-terminal-panel border-2 border-terminal-black p-6">
          {currentStory.content}
        </div>

        {/* Visual */}
        <div className="border-2 border-terminal-black min-h-[300px]">
          {currentStory.visual}
        </div>
      </div>

      {/* Navigation */}
      <div className="flex items-center justify-between">
        <button
          onClick={() => goToStep(currentStep - 1)}
          disabled={currentStep === 0}
          className={clsx(
            "flex items-center gap-2 px-4 py-2 font-mono text-xs uppercase tracking-terminal border-2 transition-colors",
            currentStep === 0
              ? "border-terminal-grid text-terminal-grid cursor-not-allowed"
              : "border-terminal-black text-terminal-black hover:bg-terminal-black hover:text-terminal-mint"
          )}
        >
          <ChevronLeft className="w-4 h-4" />
          PREV
        </button>

        <span className="font-mono text-xs text-terminal-black/50">
          {currentStep + 1} / {storySteps.length}
        </span>

        <button
          onClick={() => goToStep(currentStep + 1)}
          disabled={currentStep === storySteps.length - 1}
          className={clsx(
            "flex items-center gap-2 px-4 py-2 font-mono text-xs uppercase tracking-terminal border-2 transition-colors",
            currentStep === storySteps.length - 1
              ? "border-terminal-grid text-terminal-grid cursor-not-allowed"
              : "border-terminal-black bg-terminal-black text-terminal-mint hover:bg-terminal-accent hover:border-terminal-accent"
          )}
        >
          NEXT
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>

      {/* End CTA */}
      {currentStep === storySteps.length - 1 && (
        <div className="text-center py-6 bg-terminal-accent/10 border-2 border-terminal-accent">
          <p className="font-mono text-sm font-bold text-terminal-black mb-2">
            THEORY COMPLETE
          </p>
          <p className="font-mono text-xs text-terminal-black/70">
            PROCEED TO &quot;EXECUTE MODEL&quot; TAB TO TRAIN WITH YOUR DATA
          </p>
        </div>
      )}
    </div>
  );
}
