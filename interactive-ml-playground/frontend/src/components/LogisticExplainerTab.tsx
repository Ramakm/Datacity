"use client";

import { useState } from "react";
import {
  ChevronRight,
  ChevronLeft,
  Binary,
  Target,
  Sigma,
  BarChart3,
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
          SCENARIO: Medical diagnosis prediction. Input: patient metrics. Output: binary classification (0 or 1).
        </p>
        <p className="font-mono text-xs text-terminal-black/70">
          Objective: Predict categorical outcomes. Not continuous values - discrete class membership.
        </p>
        <div className="bg-terminal-accent/10 border-2 border-terminal-accent p-4">
          <p className="font-mono text-xs text-terminal-accent font-bold">
            CLASSIFICATION: BINARY PROBLEM // CATEGORICAL OUTPUT // SUPERVISED LEARNING
          </p>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full flex items-center justify-center">
        <div className="space-y-2 w-full max-w-xs font-mono text-xs">
          <div className="flex items-center justify-between p-2 border border-terminal-grid text-terminal-mint">
            <span>BMI: 32, AGE: 55</span>
            <span className="font-bold text-red-400">CLASS_1</span>
          </div>
          <div className="flex items-center justify-between p-2 border border-terminal-grid text-terminal-mint">
            <span>BMI: 22, AGE: 28</span>
            <span className="font-bold text-terminal-accent">CLASS_0</span>
          </div>
          <div className="flex items-center justify-between p-2 border border-terminal-grid text-terminal-mint">
            <span>BMI: 28, AGE: 45</span>
            <span className="font-bold text-red-400">CLASS_1</span>
          </div>
          <div className="flex items-center justify-between p-2 border-2 border-dashed border-terminal-warning text-terminal-warning">
            <span>BMI: 26, AGE: 40</span>
            <span className="font-bold">???</span>
          </div>
        </div>
      </div>
    ),
  },
  {
    id: 2,
    title: "LINEAR LIMITATION",
    icon: <Binary className="w-4 h-4" />,
    content: (
      <div className="space-y-4">
        <p className="font-mono text-sm text-terminal-black leading-relaxed">
          Linear regression outputs unbounded values. Classification requires probability range [0, 1].
        </p>
        <div className="bg-terminal-panel border-2 border-terminal-black p-4 space-y-2">
          <p className="font-mono text-xs font-bold text-terminal-black">REQUIREMENTS:</p>
          <ul className="space-y-1 font-mono text-xs text-terminal-black/80">
            <li className="flex items-start gap-2">
              <CheckCircle className="w-3 h-3 text-terminal-accent mt-0.5 flex-shrink-0" />
              Output bounded: 0 ≤ p ≤ 1
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="w-3 h-3 text-terminal-accent mt-0.5 flex-shrink-0" />
              Smooth transition curve
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="w-3 h-3 text-terminal-accent mt-0.5 flex-shrink-0" />
              Interpretable as probability
            </li>
          </ul>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full">
        <svg viewBox="0 0 300 200" className="w-full h-full">
          <line x1="40" y1="160" x2="280" y2="160" stroke="#d4d4c8" strokeWidth="1" />
          <line x1="40" y1="160" x2="40" y2="20" stroke="#d4d4c8" strokeWidth="1" />
          <line x1="40" y1="140" x2="280" y2="140" stroke="#d4d4c8" strokeWidth="1" strokeDasharray="4,4" />
          <line x1="40" y1="40" x2="280" y2="40" stroke="#d4d4c8" strokeWidth="1" strokeDasharray="4,4" />
          <text x="25" y="145" fill="#d4d4c8" style={{ fontFamily: 'monospace', fontSize: '10px' }}>0</text>
          <text x="25" y="45" fill="#d4d4c8" style={{ fontFamily: 'monospace', fontSize: '10px' }}>1</text>
          <line x1="50" y1="170" x2="270" y2="10" stroke="#ef4444" strokeWidth="2" opacity="0.6" />
          <rect x="50" y="5" width="60" height="35" fill="#ef4444" opacity="0.2" />
          <rect x="210" y="160" width="60" height="35" fill="#ef4444" opacity="0.2" />
          <text x="80" y="25" textAnchor="middle" fill="#ef4444" style={{ fontFamily: 'monospace', fontSize: '10px' }}>&gt;1?</text>
          <text x="240" y="180" textAnchor="middle" fill="#ef4444" style={{ fontFamily: 'monospace', fontSize: '10px' }}>&lt;0?</text>
          <text x="200" y="195" fill="#ef4444" style={{ fontFamily: 'monospace', fontSize: '9px' }}>LINEAR: INVALID</text>
        </svg>
      </div>
    ),
  },
  {
    id: 3,
    title: "SIGMOID FUNCTION",
    icon: <Sigma className="w-4 h-4" />,
    content: (
      <div className="space-y-4">
        <p className="font-mono text-sm text-terminal-black leading-relaxed">
          Sigmoid function transforms linear output to probability space [0, 1].
        </p>
        <div className="bg-terminal-black p-4 font-mono text-center text-base border-2 border-terminal-black">
          <span className="text-terminal-accent font-bold">p</span>
          <span className="text-terminal-mint"> = 1 / (1 + e</span>
          <sup className="text-terminal-warning">-z</sup>
          <span className="text-terminal-mint">)</span>
        </div>
        <ul className="space-y-2 font-mono text-xs text-terminal-black/80">
          <li><span className="text-terminal-accent font-bold">p</span> = probability output (0 to 1)</li>
          <li><span className="text-terminal-warning font-bold">z</span> = linear combination (w₁x₁ + w₂x₂ + b)</li>
          <li><span className="text-terminal-black font-bold">e</span> = Euler constant (~2.718)</li>
        </ul>
        <div className="bg-terminal-accent/10 border-2 border-terminal-accent p-3">
          <p className="font-mono text-xs text-terminal-accent font-bold">
            z → -∞: p → 0 // z → +∞: p → 1
          </p>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full">
        <svg viewBox="0 0 300 200" className="w-full h-full">
          <line x1="40" y1="100" x2="280" y2="100" stroke="#d4d4c8" strokeWidth="1" />
          <line x1="160" y1="180" x2="160" y2="20" stroke="#d4d4c8" strokeWidth="1" />
          <line x1="40" y1="170" x2="280" y2="170" stroke="#d4d4c8" strokeWidth="1" strokeDasharray="4,4" />
          <line x1="40" y1="30" x2="280" y2="30" stroke="#d4d4c8" strokeWidth="1" strokeDasharray="4,4" />
          <text x="25" y="175" fill="#d4d4c8" style={{ fontFamily: 'monospace', fontSize: '10px' }}>0</text>
          <text x="25" y="35" fill="#d4d4c8" style={{ fontFamily: 'monospace', fontSize: '10px' }}>1</text>
          <text x="25" y="105" fill="#d4d4c8" style={{ fontFamily: 'monospace', fontSize: '10px' }}>0.5</text>
          <path
            d="M 40 168 Q 80 165 100 160 Q 120 150 140 120 Q 160 100 160 100 Q 160 100 180 80 Q 200 50 220 40 Q 260 32 280 32"
            fill="none"
            stroke="#1a5c3a"
            strokeWidth="3"
          />
          <circle cx="160" cy="100" r="5" fill="#c4a000" />
          <text x="175" y="95" fill="#c4a000" style={{ fontFamily: 'monospace', fontSize: '9px' }}>THRESHOLD</text>
          <text x="160" y="195" textAnchor="middle" fill="#d4d4c8" style={{ fontFamily: 'monospace', fontSize: '9px' }}>Z (LINEAR COMBINATION)</text>
        </svg>
      </div>
    ),
  },
  {
    id: 4,
    title: "DECISION BOUNDARY",
    icon: <BarChart3 className="w-4 h-4" />,
    content: (
      <div className="space-y-4">
        <p className="font-mono text-sm text-terminal-black leading-relaxed">
          Decision boundary separates classes based on probability threshold.
        </p>
        <div className="bg-terminal-panel border-2 border-terminal-black p-4 space-y-2">
          <p className="font-mono text-xs font-bold text-terminal-black">DECISION RULE:</p>
          <div className="space-y-2 font-mono text-xs text-terminal-black/80">
            <div className="flex items-center gap-3">
              <span className="text-terminal-accent">IF p ≥ 0.5</span>
              <span>→</span>
              <span className="bg-terminal-accent/20 text-terminal-accent px-2 py-1 border border-terminal-accent">
                PREDICT CLASS_1
              </span>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-red-600">IF p &lt; 0.5</span>
              <span>→</span>
              <span className="bg-red-500/20 text-red-600 px-2 py-1 border border-red-600">
                PREDICT CLASS_0
              </span>
            </div>
          </div>
        </div>
        <p className="font-mono text-xs text-terminal-black/70">
          Threshold adjustable based on error cost analysis.
        </p>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full">
        <svg viewBox="0 0 300 200" className="w-full h-full">
          <line x1="40" y1="160" x2="280" y2="160" stroke="#d4d4c8" strokeWidth="1" />
          <line x1="40" y1="160" x2="40" y2="20" stroke="#d4d4c8" strokeWidth="1" />
          <line x1="160" y1="20" x2="160" y2="160" stroke="#c4a000" strokeWidth="2" strokeDasharray="5,5" />
          <rect x="40" y="20" width="120" height="140" fill="#ef4444" opacity="0.15" />
          <rect x="160" y="20" width="120" height="140" fill="#1a5c3a" opacity="0.15" />
          <circle cx="70" cy="60" r="5" fill="#ef4444" />
          <circle cx="90" cy="100" r="5" fill="#ef4444" />
          <circle cx="60" cy="130" r="5" fill="#ef4444" />
          <circle cx="120" cy="80" r="5" fill="#ef4444" />
          <circle cx="100" cy="140" r="5" fill="#ef4444" />
          <circle cx="200" cy="50" r="5" fill="#1a5c3a" />
          <circle cx="230" cy="90" r="5" fill="#1a5c3a" />
          <circle cx="250" cy="60" r="5" fill="#1a5c3a" />
          <circle cx="180" cy="120" r="5" fill="#1a5c3a" />
          <circle cx="220" cy="140" r="5" fill="#1a5c3a" />
          <text x="100" y="180" textAnchor="middle" fill="#ef4444" style={{ fontFamily: 'monospace', fontSize: '10px', fontWeight: 'bold' }}>CLASS_0</text>
          <text x="220" y="180" textAnchor="middle" fill="#1a5c3a" style={{ fontFamily: 'monospace', fontSize: '10px', fontWeight: 'bold' }}>CLASS_1</text>
          <text x="160" y="15" textAnchor="middle" fill="#c4a000" style={{ fontFamily: 'monospace', fontSize: '9px' }}>BOUNDARY</text>
        </svg>
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
          Classification metrics differ from regression evaluation.
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
          <div className="text-center text-terminal-mint font-bold text-xs mb-4">CONFUSION MATRIX</div>
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-terminal-accent/20 border-2 border-terminal-accent p-3 text-center">
              <div className="text-xl font-bold text-terminal-accent">45</div>
              <div className="text-xs text-terminal-grid">TRUE_NEG</div>
            </div>
            <div className="bg-red-500/20 border-2 border-red-500 p-3 text-center">
              <div className="text-xl font-bold text-red-400">5</div>
              <div className="text-xs text-terminal-grid">FALSE_POS</div>
            </div>
            <div className="bg-red-500/20 border-2 border-red-500 p-3 text-center">
              <div className="text-xl font-bold text-red-400">3</div>
              <div className="text-xs text-terminal-grid">FALSE_NEG</div>
            </div>
            <div className="bg-terminal-accent/20 border-2 border-terminal-accent p-3 text-center">
              <div className="text-xl font-bold text-terminal-accent">47</div>
              <div className="text-xs text-terminal-grid">TRUE_POS</div>
            </div>
          </div>
          <div className="border-t border-terminal-grid/30 pt-3 text-center">
            <div className="text-2xl font-bold text-terminal-mint">92%</div>
            <div className="text-xs text-terminal-grid">ACCURACY</div>
          </div>
        </div>
      </div>
    ),
  },
];

export default function LogisticExplainerTab() {
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
