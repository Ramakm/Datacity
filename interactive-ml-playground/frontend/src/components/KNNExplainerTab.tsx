"use client";

import { useState } from "react";
import {
  ChevronRight,
  ChevronLeft,
  Users,
  Target,
  Ruler,
  Vote,
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
          SCENARIO: Species classification. Input: physical measurements. Output: discrete class label.
        </p>
        <p className="font-mono text-xs text-terminal-black/70">
          Given a new sample, classify it based on similarity to existing labeled data. No explicit model training - decisions made at prediction time.
        </p>
        <div className="bg-terminal-accent/10 border-2 border-terminal-accent p-4">
          <p className="font-mono text-xs text-terminal-accent font-bold">
            CLASSIFICATION: INSTANCE-BASED LEARNING // LAZY ALGORITHM // NON-PARAMETRIC METHOD
          </p>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full flex items-center justify-center">
        <div className="space-y-2 w-full max-w-xs font-mono text-xs">
          <div className="flex items-center justify-between p-2 border border-terminal-grid text-terminal-mint">
            <span>SAMPLE A</span>
            <span className="font-bold text-terminal-accent">CLASS: SETOSA</span>
          </div>
          <div className="flex items-center justify-between p-2 border border-terminal-grid text-terminal-mint">
            <span>SAMPLE B</span>
            <span className="font-bold text-terminal-accent">CLASS: VERSICOLOR</span>
          </div>
          <div className="flex items-center justify-between p-2 border border-terminal-grid text-terminal-mint">
            <span>SAMPLE C</span>
            <span className="font-bold text-terminal-accent">CLASS: VIRGINICA</span>
          </div>
          <div className="flex items-center justify-between p-2 border-2 border-dashed border-terminal-warning text-terminal-warning">
            <span>NEW SAMPLE</span>
            <span className="font-bold">CLASS: ???</span>
          </div>
        </div>
      </div>
    ),
  },
  {
    id: 2,
    title: "DISTANCE CALCULATION",
    icon: <Ruler className="w-4 h-4" />,
    content: (
      <div className="space-y-4">
        <p className="font-mono text-sm text-terminal-black leading-relaxed">
          KNN measures similarity using distance metrics. Most common: Euclidean distance.
        </p>
        <div className="bg-terminal-black p-4 font-mono text-center text-base border-2 border-terminal-black">
          <span className="text-terminal-accent font-bold">d</span>
          <span className="text-terminal-mint"> = </span>
          <span className="text-terminal-mint">√(</span>
          <span className="text-terminal-warning">Σ(xᵢ - yᵢ)²</span>
          <span className="text-terminal-mint">)</span>
        </div>
        <ul className="space-y-2 font-mono text-xs text-terminal-black/80">
          <li><span className="text-terminal-accent font-bold">d</span> = distance between two points</li>
          <li><span className="text-terminal-warning font-bold">xᵢ, yᵢ</span> = feature values of two samples</li>
          <li>Lower distance = more similar samples</li>
        </ul>
        <p className="font-mono text-xs text-terminal-black/70">
          Other metrics: Manhattan, Minkowski, Cosine similarity
        </p>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full">
        <svg viewBox="0 0 300 200" className="w-full h-full">
          <line x1="40" y1="160" x2="280" y2="160" stroke="#d4d4c8" strokeWidth="1" />
          <line x1="40" y1="160" x2="40" y2="20" stroke="#d4d4c8" strokeWidth="1" />
          <text x="160" y="185" textAnchor="middle" fill="#d4d4c8" className="text-xs" style={{ fontFamily: 'monospace' }}>FEATURE_1</text>
          <text x="15" y="90" textAnchor="middle" transform="rotate(-90, 15, 90)" fill="#d4d4c8" className="text-xs" style={{ fontFamily: 'monospace' }}>FEATURE_2</text>

          {/* Class A points (green) */}
          <circle cx="80" cy="120" r="6" fill="#1a5c3a" />
          <circle cx="100" cy="130" r="6" fill="#1a5c3a" />
          <circle cx="90" cy="110" r="6" fill="#1a5c3a" />

          {/* Class B points (yellow) */}
          <circle cx="200" cy="60" r="6" fill="#c4a000" />
          <circle cx="220" cy="70" r="6" fill="#c4a000" />
          <circle cx="210" cy="50" r="6" fill="#c4a000" />

          {/* New point */}
          <circle cx="150" cy="90" r="8" fill="none" stroke="#cfeee3" strokeWidth="2" strokeDasharray="4" />
          <text x="150" y="93" textAnchor="middle" fill="#cfeee3" className="text-xs" style={{ fontFamily: 'monospace' }}>?</text>

          {/* Distance lines */}
          <line x1="150" y1="90" x2="100" y2="130" stroke="#cfeee3" strokeWidth="1" strokeDasharray="2" opacity="0.5" />
          <line x1="150" y1="90" x2="200" y2="60" stroke="#cfeee3" strokeWidth="1" strokeDasharray="2" opacity="0.5" />
        </svg>
      </div>
    ),
  },
  {
    id: 3,
    title: "FINDING K NEIGHBORS",
    icon: <Users className="w-4 h-4" />,
    content: (
      <div className="space-y-4">
        <p className="font-mono text-sm text-terminal-black leading-relaxed">
          The algorithm identifies the K closest training samples to the query point.
        </p>
        <div className="bg-terminal-panel border-2 border-terminal-black p-4 space-y-2">
          <p className="font-mono text-xs font-bold text-terminal-black">PROCEDURE:</p>
          <ol className="space-y-1 font-mono text-xs text-terminal-black/80 list-decimal list-inside">
            <li>Calculate distance to all training points</li>
            <li>Sort distances in ascending order</li>
            <li>Select K nearest neighbors</li>
            <li>Examine class labels of neighbors</li>
          </ol>
        </div>
        <div className="bg-terminal-warning/10 border-2 border-terminal-warning p-3">
          <p className="font-mono text-xs text-terminal-black">
            <span className="font-bold text-terminal-warning">K VALUE:</span> Hyperparameter that controls model behavior. Small K = sensitive to noise. Large K = smoother boundaries.
          </p>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full">
        <svg viewBox="0 0 300 200" className="w-full h-full">
          {/* Grid */}
          <line x1="40" y1="160" x2="280" y2="160" stroke="#d4d4c8" strokeWidth="1" />
          <line x1="40" y1="160" x2="40" y2="20" stroke="#d4d4c8" strokeWidth="1" />

          {/* Class A points */}
          <circle cx="80" cy="120" r="6" fill="#1a5c3a" />
          <circle cx="100" cy="130" r="6" fill="#1a5c3a" />
          <circle cx="90" cy="110" r="6" fill="#1a5c3a" />
          <circle cx="120" cy="100" r="6" fill="#1a5c3a" />

          {/* Class B points */}
          <circle cx="200" cy="60" r="6" fill="#c4a000" />
          <circle cx="220" cy="70" r="6" fill="#c4a000" />
          <circle cx="210" cy="50" r="6" fill="#c4a000" />
          <circle cx="180" cy="80" r="6" fill="#c4a000" />

          {/* New point */}
          <circle cx="150" cy="90" r="8" fill="#cfeee3" />

          {/* K=3 circle */}
          <circle cx="150" cy="90" r="50" fill="none" stroke="#cfeee3" strokeWidth="2" strokeDasharray="4" />

          {/* K=3 neighbors highlighted */}
          <circle cx="120" cy="100" r="10" fill="none" stroke="#cfeee3" strokeWidth="2" />
          <circle cx="180" cy="80" r="10" fill="none" stroke="#cfeee3" strokeWidth="2" />
          <circle cx="100" cy="130" r="10" fill="none" stroke="#cfeee3" strokeWidth="2" />

          <text x="150" y="185" textAnchor="middle" fill="#cfeee3" className="text-xs" style={{ fontFamily: 'monospace' }}>K = 3 NEIGHBORS</text>
        </svg>
      </div>
    ),
  },
  {
    id: 4,
    title: "MAJORITY VOTING",
    icon: <Vote className="w-4 h-4" />,
    content: (
      <div className="space-y-4">
        <p className="font-mono text-sm text-terminal-black leading-relaxed">
          Classification via democratic vote among K neighbors. The class with most representatives wins.
        </p>
        <div className="bg-terminal-black p-4 font-mono text-xs overflow-x-auto border-2 border-terminal-black">
          <p className="text-terminal-mint mb-2">
            <span className="text-terminal-accent">K = 5 NEIGHBORS:</span>
          </p>
          <p className="text-terminal-mint">
            CLASS_A: 3 votes <span className="text-terminal-accent">■■■</span>
          </p>
          <p className="text-terminal-mint">
            CLASS_B: 2 votes <span className="text-terminal-warning">■■</span>
          </p>
          <p className="text-terminal-mint mt-2">
            <span className="text-terminal-accent font-bold">PREDICTION: CLASS_A</span>
          </p>
        </div>
        <div className="bg-terminal-accent/10 border-2 border-terminal-accent p-3">
          <p className="font-mono text-xs text-terminal-accent font-bold">
            WEIGHTED VOTING: CLOSER NEIGHBORS CAN HAVE MORE INFLUENCE (1/DISTANCE)
          </p>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full flex items-center justify-center">
        <div className="space-y-4 w-full max-w-sm font-mono text-xs">
          <div className="text-terminal-mint text-center font-bold mb-4">VOTE COUNT (K=5)</div>
          <div className="space-y-3">
            <div>
              <div className="flex justify-between items-center mb-1">
                <span className="text-terminal-grid">CLASS_A</span>
                <span className="text-terminal-accent font-bold">3 / 5</span>
              </div>
              <div className="h-4 bg-terminal-grid border border-terminal-grid">
                <div className="h-full bg-terminal-accent" style={{ width: "60%" }}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between items-center mb-1">
                <span className="text-terminal-grid">CLASS_B</span>
                <span className="text-terminal-warning font-bold">2 / 5</span>
              </div>
              <div className="h-4 bg-terminal-grid border border-terminal-grid">
                <div className="h-full bg-terminal-warning" style={{ width: "40%" }}></div>
              </div>
            </div>
          </div>
          <div className="border-t border-terminal-grid/30 pt-4 text-center">
            <p className="text-terminal-mint">WINNER: <span className="text-terminal-accent font-bold">CLASS_A</span></p>
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
          Model evaluation for classification tasks.
        </p>
        <div className="space-y-2">
          <div className="bg-terminal-panel border-2 border-terminal-black p-3">
            <p className="font-mono text-xs font-bold text-terminal-black">ACCURACY</p>
            <p className="font-mono text-xs text-terminal-black/70">Correct predictions / Total predictions</p>
          </div>
          <div className="bg-terminal-panel border-2 border-terminal-black p-3">
            <p className="font-mono text-xs font-bold text-terminal-black">PRECISION</p>
            <p className="font-mono text-xs text-terminal-black/70">True positives / Predicted positives</p>
          </div>
          <div className="bg-terminal-panel border-2 border-terminal-black p-3">
            <p className="font-mono text-xs font-bold text-terminal-black">RECALL</p>
            <p className="font-mono text-xs text-terminal-black/70">True positives / Actual positives</p>
          </div>
          <div className="bg-terminal-panel border-2 border-terminal-black p-3">
            <p className="font-mono text-xs font-bold text-terminal-black">F1 SCORE</p>
            <p className="font-mono text-xs text-terminal-black/70">Harmonic mean of precision and recall</p>
          </div>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full flex items-center justify-center">
        <div className="space-y-4 w-full max-w-xs font-mono">
          <div className="text-center">
            <div className="text-4xl font-bold text-terminal-accent">96.7%</div>
            <div className="text-xs text-terminal-grid">ACCURACY</div>
            <div className="text-xs text-terminal-accent font-bold">EXCELLENT</div>
          </div>
          <div className="border-t border-terminal-grid/30 pt-4 grid grid-cols-2 gap-4 text-center">
            <div>
              <div className="text-xl font-bold text-terminal-mint">0.95</div>
              <div className="text-xs text-terminal-grid">PRECISION</div>
            </div>
            <div>
              <div className="text-xl font-bold text-terminal-mint">0.97</div>
              <div className="text-xs text-terminal-grid">RECALL</div>
            </div>
          </div>
          <div className="border-t border-terminal-grid/30 pt-4 text-center">
            <div className="text-xl font-bold text-terminal-warning">0.96</div>
            <div className="text-xs text-terminal-grid">F1 SCORE</div>
          </div>
        </div>
      </div>
    ),
  },
];

export default function KNNExplainerTab() {
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
