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
    title: "The Problem",
    icon: <Target className="w-5 h-5" />,
    content: (
      <div className="space-y-4">
        <p className="text-lg text-slate-300 leading-relaxed">
          Imagine you&apos;re a doctor analyzing patient data. You need to answer:{" "}
          <span className="font-semibold text-purple-400">
            &quot;Will this patient develop diabetes?&quot;
          </span>
        </p>
        <p className="text-slate-400">
          This isn&apos;t about predicting a number - it&apos;s about predicting a{" "}
          <strong className="text-white">category</strong>: Yes or No, True or
          False, 1 or 0.
        </p>
        <div className="bg-purple-500/10 rounded-lg p-4 border border-purple-500/20">
          <p className="text-purple-300 font-medium">
            This is a classification problem: predicting which category
            something belongs to based on its features.
          </p>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-slate-800/50 rounded-xl p-6 h-full flex items-center justify-center">
        <div className="space-y-3 w-full max-w-xs">
          <div className="flex items-center justify-between p-3 bg-slate-700/50 rounded-lg border border-slate-600/50">
            <span className="text-slate-400">BMI: 32, Age: 55</span>
            <span className="font-semibold text-red-400">Diabetic</span>
          </div>
          <div className="flex items-center justify-between p-3 bg-slate-700/50 rounded-lg border border-slate-600/50">
            <span className="text-slate-400">BMI: 22, Age: 28</span>
            <span className="font-semibold text-green-400">Healthy</span>
          </div>
          <div className="flex items-center justify-between p-3 bg-slate-700/50 rounded-lg border border-slate-600/50">
            <span className="text-slate-400">BMI: 28, Age: 45</span>
            <span className="font-semibold text-red-400">Diabetic</span>
          </div>
          <div className="flex items-center justify-between p-3 bg-purple-500/10 rounded-lg border-2 border-purple-500/30 border-dashed">
            <span className="text-purple-400">BMI: 26, Age: 40</span>
            <span className="font-semibold text-purple-400">???</span>
          </div>
        </div>
      </div>
    ),
  },
  {
    id: 2,
    title: "Why Not Linear Regression?",
    icon: <Binary className="w-5 h-5" />,
    content: (
      <div className="space-y-4">
        <p className="text-lg text-slate-300 leading-relaxed">
          Linear regression predicts{" "}
          <span className="font-semibold text-purple-400">
            any number
          </span>{" "}
          - but we need a probability between 0 and 1!
        </p>
        <p className="text-slate-400">
          A straight line can predict values like -0.5 or 1.7, which don&apos;t
          make sense as probabilities.
        </p>
        <div className="bg-slate-800/80 rounded-lg p-4 border border-slate-700/50">
          <p className="font-medium text-white mb-2">We need a function that:</p>
          <ul className="space-y-2 text-slate-400">
            <li className="flex items-start gap-2">
              <CheckCircle className="w-4 h-4 text-green-400 mt-1 flex-shrink-0" />
              Always outputs values between 0 and 1
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="w-4 h-4 text-green-400 mt-1 flex-shrink-0" />
              Creates a smooth S-shaped curve
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="w-4 h-4 text-green-400 mt-1 flex-shrink-0" />
              Can represent probability of class membership
            </li>
          </ul>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-slate-800/50 rounded-xl p-6 h-full">
        <svg viewBox="0 0 300 200" className="w-full h-full">
          {/* Axes */}
          <line x1="40" y1="160" x2="280" y2="160" stroke="#475569" strokeWidth="2" />
          <line x1="40" y1="160" x2="40" y2="20" stroke="#475569" strokeWidth="2" />

          {/* 0 and 1 lines */}
          <line x1="40" y1="140" x2="280" y2="140" stroke="#475569" strokeWidth="1" strokeDasharray="4,4" />
          <line x1="40" y1="40" x2="280" y2="40" stroke="#475569" strokeWidth="1" strokeDasharray="4,4" />
          <text x="25" y="145" className="fill-slate-500 text-xs">0</text>
          <text x="25" y="45" className="fill-slate-500 text-xs">1</text>

          {/* Linear regression line (bad) */}
          <line x1="50" y1="170" x2="270" y2="10" stroke="#ef4444" strokeWidth="2" opacity="0.6" />

          {/* Problem areas */}
          <rect x="50" y="5" width="60" height="35" fill="#ef4444" opacity="0.2" rx="4" />
          <rect x="210" y="160" width="60" height="35" fill="#ef4444" opacity="0.2" rx="4" />

          {/* Labels */}
          <text x="80" y="25" textAnchor="middle" className="fill-red-400 text-xs">&gt;1?</text>
          <text x="240" y="180" textAnchor="middle" className="fill-red-400 text-xs">&lt;0?</text>

          {/* Legend */}
          <line x1="120" y1="185" x2="150" y2="185" stroke="#ef4444" strokeWidth="2" />
          <text x="155" y="189" className="fill-slate-400 text-xs">Linear (invalid)</text>
        </svg>
      </div>
    ),
  },
  {
    id: 3,
    title: "The Sigmoid Function",
    icon: <Sigma className="w-5 h-5" />,
    content: (
      <div className="space-y-4">
        <p className="text-lg text-slate-300 leading-relaxed">
          Logistic Regression uses the{" "}
          <span className="font-semibold text-purple-400">
            sigmoid function
          </span>{" "}
          to squash any input into a probability between 0 and 1.
        </p>
        <div className="bg-slate-800/80 rounded-lg p-4 font-mono text-center text-lg border border-slate-700/50">
          <span className="text-purple-400 font-bold">p</span> ={" "}
          <span className="text-white">1 / (1 + e</span>
          <sup className="text-amber-400">-z</sup>
          <span className="text-white">)</span>
        </div>
        <ul className="space-y-2 text-slate-400">
          <li className="flex items-start gap-2">
            <span className="text-purple-400 font-bold">p</span> = probability
            (always between 0 and 1)
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-400 font-bold">z</span> = linear
            combination (w₁x₁ + w₂x₂ + ... + b)
          </li>
          <li className="flex items-start gap-2">
            <span className="text-white font-bold">e</span> = Euler&apos;s number
            (~2.718)
          </li>
        </ul>
        <div className="bg-green-500/10 rounded-lg p-4 border border-green-500/20">
          <p className="text-green-300 font-medium">
            When z is very negative → p ≈ 0 (unlikely)
            <br />
            When z is very positive → p ≈ 1 (likely)
          </p>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-slate-800/50 rounded-xl p-6 h-full">
        <svg viewBox="0 0 300 200" className="w-full h-full">
          {/* Axes */}
          <line x1="40" y1="100" x2="280" y2="100" stroke="#475569" strokeWidth="2" />
          <line x1="160" y1="180" x2="160" y2="20" stroke="#475569" strokeWidth="2" />

          {/* 0 and 1 lines */}
          <line x1="40" y1="170" x2="280" y2="170" stroke="#475569" strokeWidth="1" strokeDasharray="4,4" />
          <line x1="40" y1="30" x2="280" y2="30" stroke="#475569" strokeWidth="1" strokeDasharray="4,4" />
          <text x="25" y="175" className="fill-slate-500 text-xs">0</text>
          <text x="25" y="35" className="fill-slate-500 text-xs">1</text>
          <text x="25" y="105" className="fill-slate-500 text-xs">0.5</text>

          {/* Sigmoid curve */}
          <path
            d="M 40 168 Q 80 165 100 160 Q 120 150 140 120 Q 160 100 160 100 Q 160 100 180 80 Q 200 50 220 40 Q 260 32 280 32"
            fill="none"
            stroke="#a855f7"
            strokeWidth="3"
          />

          {/* Decision threshold at 0.5 */}
          <circle cx="160" cy="100" r="5" fill="#22c55e" />
          <text x="175" y="95" className="fill-green-400 text-xs">threshold</text>

          {/* Labels */}
          <text x="160" y="195" textAnchor="middle" className="fill-slate-500 text-xs">
            z (linear combination)
          </text>
          <text x="60" y="155" className="fill-slate-500 text-xs">Class 0</text>
          <text x="240" y="55" className="fill-slate-500 text-xs">Class 1</text>
        </svg>
      </div>
    ),
  },
  {
    id: 4,
    title: "Making Decisions",
    icon: <BarChart3 className="w-5 h-5" />,
    content: (
      <div className="space-y-4">
        <p className="text-lg text-slate-300 leading-relaxed">
          Once we have a probability, we need a{" "}
          <span className="font-semibold text-purple-400">
            decision boundary
          </span>{" "}
          to make a final prediction.
        </p>
        <div className="bg-slate-800/80 rounded-lg p-4 space-y-3 border border-slate-700/50">
          <p className="font-medium text-white">The decision rule:</p>
          <div className="space-y-2 text-slate-400">
            <div className="flex items-center gap-3">
              <span className="text-green-400">If p ≥ 0.5</span>
              <span>→</span>
              <span className="bg-green-500/20 text-green-400 px-2 py-1 rounded">
                Predict Class 1
              </span>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-red-400">If p &lt; 0.5</span>
              <span>→</span>
              <span className="bg-red-500/20 text-red-400 px-2 py-1 rounded">
                Predict Class 0
              </span>
            </div>
          </div>
        </div>
        <p className="text-slate-400">
          The threshold of 0.5 is common, but you can adjust it based on the
          cost of different types of errors (false positives vs false negatives).
        </p>
      </div>
    ),
    visual: (
      <div className="bg-slate-800/50 rounded-xl p-6 h-full">
        <svg viewBox="0 0 300 200" className="w-full h-full">
          {/* Axes */}
          <line x1="40" y1="160" x2="280" y2="160" stroke="#475569" strokeWidth="2" />
          <line x1="40" y1="160" x2="40" y2="20" stroke="#475569" strokeWidth="2" />

          {/* Decision boundary line */}
          <line x1="160" y1="20" x2="160" y2="160" stroke="#a855f7" strokeWidth="2" strokeDasharray="5,5" />

          {/* Class 0 region */}
          <rect x="40" y="20" width="120" height="140" fill="#ef4444" opacity="0.1" />
          {/* Class 1 region */}
          <rect x="160" y="20" width="120" height="140" fill="#22c55e" opacity="0.1" />

          {/* Data points - Class 0 (red) */}
          <circle cx="70" cy="60" r="6" fill="#ef4444" />
          <circle cx="90" cy="100" r="6" fill="#ef4444" />
          <circle cx="60" cy="130" r="6" fill="#ef4444" />
          <circle cx="120" cy="80" r="6" fill="#ef4444" />
          <circle cx="100" cy="140" r="6" fill="#ef4444" />

          {/* Data points - Class 1 (green) */}
          <circle cx="200" cy="50" r="6" fill="#22c55e" />
          <circle cx="230" cy="90" r="6" fill="#22c55e" />
          <circle cx="250" cy="60" r="6" fill="#22c55e" />
          <circle cx="180" cy="120" r="6" fill="#22c55e" />
          <circle cx="220" cy="140" r="6" fill="#22c55e" />

          {/* Labels */}
          <text x="100" y="180" textAnchor="middle" className="fill-red-400 text-xs font-medium">
            Class 0
          </text>
          <text x="220" y="180" textAnchor="middle" className="fill-green-400 text-xs font-medium">
            Class 1
          </text>
          <text x="160" y="15" textAnchor="middle" className="fill-purple-400 text-xs">
            Decision Boundary
          </text>
        </svg>
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
          For classification, we use different{" "}
          <span className="font-semibold text-purple-400">metrics</span> than
          regression to evaluate performance.
        </p>
        <div className="space-y-3">
          <div className="bg-slate-800/80 rounded-lg p-4 border border-slate-700/50">
            <p className="font-semibold text-white mb-1">Accuracy</p>
            <p className="text-sm text-slate-400">
              Percentage of correct predictions overall.
            </p>
          </div>
          <div className="bg-slate-800/80 rounded-lg p-4 border border-slate-700/50">
            <p className="font-semibold text-white mb-1">Precision</p>
            <p className="text-sm text-slate-400">
              Of all positive predictions, how many were actually positive?
            </p>
          </div>
          <div className="bg-slate-800/80 rounded-lg p-4 border border-slate-700/50">
            <p className="font-semibold text-white mb-1">Recall</p>
            <p className="text-sm text-slate-400">
              Of all actual positives, how many did we correctly identify?
            </p>
          </div>
          <div className="bg-slate-800/80 rounded-lg p-4 border border-slate-700/50">
            <p className="font-semibold text-white mb-1">F1 Score</p>
            <p className="text-sm text-slate-400">
              Harmonic mean of precision and recall - balances both.
            </p>
          </div>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-slate-800/50 rounded-xl p-6 h-full flex items-center justify-center">
        <div className="space-y-4 w-full max-w-xs">
          <div className="text-center mb-2">
            <span className="text-lg font-semibold text-white">
              Confusion Matrix
            </span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-green-500/20 border border-green-500/30 rounded-lg p-3 text-center">
              <div className="text-xl font-bold text-green-400">45</div>
              <div className="text-xs text-slate-400">True Negative</div>
            </div>
            <div className="bg-red-500/20 border border-red-500/30 rounded-lg p-3 text-center">
              <div className="text-xl font-bold text-red-400">5</div>
              <div className="text-xs text-slate-400">False Positive</div>
            </div>
            <div className="bg-red-500/20 border border-red-500/30 rounded-lg p-3 text-center">
              <div className="text-xl font-bold text-red-400">3</div>
              <div className="text-xs text-slate-400">False Negative</div>
            </div>
            <div className="bg-green-500/20 border border-green-500/30 rounded-lg p-3 text-center">
              <div className="text-xl font-bold text-green-400">47</div>
              <div className="text-xs text-slate-400">True Positive</div>
            </div>
          </div>
          <div className="bg-purple-500/10 rounded-lg p-3 text-center border border-purple-500/20">
            <p className="text-sm text-purple-300">
              Accuracy: 92% (92/100 correct)
            </p>
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
              "w-3 h-3 rounded-full transition-all",
              index === currentStep
                ? "bg-purple-500 scale-125 shadow-lg shadow-purple-500/50"
                : index < currentStep
                ? "bg-purple-500/50"
                : "bg-slate-600"
            )}
            aria-label={`Go to step ${index + 1}: ${step.title}`}
          />
        ))}
      </div>

      {/* Step header */}
      <div className="text-center">
        <div className="inline-flex items-center gap-2 bg-purple-500/10 text-purple-400 px-4 py-2 rounded-full text-sm font-medium mb-2 border border-purple-500/20">
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
              : "bg-gradient-to-r from-purple-500 to-pink-600 text-white hover:shadow-lg hover:shadow-purple-500/25"
          )}
        >
          Next
          <ChevronRight className="w-5 h-5" />
        </button>
      </div>

      {/* End CTA */}
      {currentStep === storySteps.length - 1 && (
        <div className="text-center py-6 bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-xl border border-purple-500/20">
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
