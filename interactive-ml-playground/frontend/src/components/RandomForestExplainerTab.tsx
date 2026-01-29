"use client";

import { useState } from "react";
import {
  ChevronRight,
  ChevronLeft,
  Target,
  Trees,
  Shuffle,
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
    title: "ENSEMBLE LEARNING",
    icon: <Target className="w-4 h-4" />,
    content: (
      <div className="space-y-4">
        <p className="font-mono text-sm text-terminal-black leading-relaxed">
          SCENARIO: Random Forest combines multiple Decision Trees to make more accurate and robust predictions.
        </p>
        <p className="font-mono text-xs text-terminal-black/70">
          Instead of relying on a single tree, we build many trees and aggregate their predictions. This reduces overfitting and improves generalization.
        </p>
        <div className="bg-terminal-accent/10 border-2 border-terminal-accent p-4">
          <p className="font-mono text-xs text-terminal-accent font-bold">
            CLASSIFICATION: ENSEMBLE METHOD // BAGGING // VARIANCE REDUCTION
          </p>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full flex items-center justify-center">
        <div className="space-y-3 w-full max-w-xs font-mono text-xs">
          <div className="text-terminal-mint text-center font-bold mb-4">SINGLE TREE vs FOREST</div>
          <div className="flex items-center justify-between p-2 border border-terminal-grid text-terminal-mint">
            <span>1 TREE</span>
            <span className="text-terminal-warning">HIGH VARIANCE</span>
          </div>
          <div className="flex items-center justify-between p-2 border border-terminal-grid text-terminal-mint">
            <span>10 TREES</span>
            <span className="text-terminal-mint">MEDIUM VARIANCE</span>
          </div>
          <div className="flex items-center justify-between p-2 border border-terminal-grid text-terminal-mint">
            <span>100 TREES</span>
            <span className="text-terminal-accent font-bold">LOW VARIANCE</span>
          </div>
          <div className="mt-4 p-2 border-2 border-terminal-accent text-terminal-accent text-center">
            WISDOM OF THE CROWD
          </div>
        </div>
      </div>
    ),
  },
  {
    id: 2,
    title: "BOOTSTRAP AGGREGATING",
    icon: <Shuffle className="w-4 h-4" />,
    content: (
      <div className="space-y-4">
        <p className="font-mono text-sm text-terminal-black leading-relaxed">
          Each tree is trained on a random subset of the data, sampled with replacement (bootstrap sampling).
        </p>
        <div className="bg-terminal-black p-4 font-mono text-center text-sm border-2 border-terminal-black">
          <span className="text-terminal-accent font-bold">BAGGING</span>
          <span className="text-terminal-mint"> = </span>
          <span className="text-terminal-warning">Bootstrap + Aggregating</span>
        </div>
        <ul className="space-y-2 font-mono text-xs text-terminal-black/80">
          <li><span className="text-terminal-accent font-bold">Bootstrap</span> = random sampling with replacement</li>
          <li><span className="text-terminal-warning font-bold">~63%</span> of original data in each sample</li>
          <li>Remaining ~37% = Out-of-Bag (OOB) samples</li>
        </ul>
        <p className="font-mono text-xs text-terminal-black/70">
          OOB samples can be used for validation without a separate test set
        </p>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full">
        <svg viewBox="0 0 300 200" className="w-full h-full">
          {/* Original dataset */}
          <text x="150" y="20" textAnchor="middle" fill="#cfeee3" className="text-xs" style={{ fontFamily: 'monospace', fontWeight: 'bold' }}>ORIGINAL DATA</text>
          <rect x="50" y="25" width="200" height="30" fill="none" stroke="#cfeee3" strokeWidth="2" />

          {/* Data points */}
          {[60, 80, 100, 120, 140, 160, 180, 200, 220, 240].map((x, i) => (
            <circle key={i} cx={x} cy={40} r="6" fill="#cfeee3" />
          ))}

          {/* Arrows */}
          <line x1="100" y1="60" x2="70" y2="85" stroke="#d4d4c8" strokeWidth="1" />
          <line x1="150" y1="60" x2="150" y2="85" stroke="#d4d4c8" strokeWidth="1" />
          <line x1="200" y1="60" x2="230" y2="85" stroke="#d4d4c8" strokeWidth="1" />

          {/* Bootstrap samples */}
          <text x="70" y="100" textAnchor="middle" fill="#c4a000" className="text-xs" style={{ fontFamily: 'monospace' }}>SAMPLE 1</text>
          <rect x="30" y="105" width="80" height="20" fill="none" stroke="#c4a000" strokeWidth="1" />
          {[40, 55, 55, 70, 85, 100].map((x, i) => (
            <circle key={i} cx={x} cy={115} r="4" fill="#c4a000" />
          ))}

          <text x="150" y="100" textAnchor="middle" fill="#1a5c3a" className="text-xs" style={{ fontFamily: 'monospace' }}>SAMPLE 2</text>
          <rect x="110" y="105" width="80" height="20" fill="none" stroke="#1a5c3a" strokeWidth="1" />
          {[120, 135, 150, 150, 165, 180].map((x, i) => (
            <circle key={i} cx={x} cy={115} r="4" fill="#1a5c3a" />
          ))}

          <text x="230" y="100" textAnchor="middle" fill="#cfeee3" className="text-xs" style={{ fontFamily: 'monospace' }}>SAMPLE N</text>
          <rect x="190" y="105" width="80" height="20" fill="none" stroke="#cfeee3" strokeWidth="1" />
          {[200, 215, 230, 230, 245, 260].map((x, i) => (
            <circle key={i} cx={x} cy={115} r="4" fill="#cfeee3" />
          ))}

          {/* Note about duplicates */}
          <text x="150" y="150" textAnchor="middle" fill="#d4d4c8" className="text-xs" style={{ fontFamily: 'monospace' }}>DUPLICATES ALLOWED</text>
          <text x="150" y="170" textAnchor="middle" fill="#d4d4c8" className="text-xs" style={{ fontFamily: 'monospace' }}>(SAMPLING WITH REPLACEMENT)</text>
        </svg>
      </div>
    ),
  },
  {
    id: 3,
    title: "FEATURE RANDOMNESS",
    icon: <Trees className="w-4 h-4" />,
    content: (
      <div className="space-y-4">
        <p className="font-mono text-sm text-terminal-black leading-relaxed">
          At each split, only a random subset of features is considered. This decorrelates the trees.
        </p>
        <div className="bg-terminal-panel border-2 border-terminal-black p-4 space-y-2">
          <p className="font-mono text-xs font-bold text-terminal-black">FEATURE SAMPLING:</p>
          <ol className="space-y-1 font-mono text-xs text-terminal-black/80 list-decimal list-inside">
            <li>For classification: sqrt(n_features)</li>
            <li>For regression: n_features / 3</li>
            <li>Each split considers different features</li>
            <li>Prevents dominant features from always being chosen</li>
          </ol>
        </div>
        <div className="bg-terminal-warning/10 border-2 border-terminal-warning p-3">
          <p className="font-mono text-xs text-terminal-black">
            <span className="font-bold text-terminal-warning">KEY INSIGHT:</span> Combining weak, diverse learners creates a strong ensemble.
          </p>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full">
        <svg viewBox="0 0 300 200" className="w-full h-full">
          {/* All features */}
          <text x="150" y="20" textAnchor="middle" fill="#cfeee3" className="text-xs" style={{ fontFamily: 'monospace', fontWeight: 'bold' }}>ALL FEATURES</text>
          <rect x="40" y="25" width="50" height="25" fill="#cfeee3" stroke="#cfeee3" strokeWidth="1" />
          <text x="65" y="42" textAnchor="middle" fill="#1e1e1e" className="text-xs" style={{ fontFamily: 'monospace' }}>X1</text>
          <rect x="100" y="25" width="50" height="25" fill="#cfeee3" stroke="#cfeee3" strokeWidth="1" />
          <text x="125" y="42" textAnchor="middle" fill="#1e1e1e" className="text-xs" style={{ fontFamily: 'monospace' }}>X2</text>
          <rect x="160" y="25" width="50" height="25" fill="#cfeee3" stroke="#cfeee3" strokeWidth="1" />
          <text x="185" y="42" textAnchor="middle" fill="#1e1e1e" className="text-xs" style={{ fontFamily: 'monospace' }}>X3</text>
          <rect x="220" y="25" width="50" height="25" fill="#cfeee3" stroke="#cfeee3" strokeWidth="1" />
          <text x="245" y="42" textAnchor="middle" fill="#1e1e1e" className="text-xs" style={{ fontFamily: 'monospace' }}>X4</text>

          {/* Tree 1 - uses X1, X3 */}
          <text x="80" y="85" textAnchor="middle" fill="#c4a000" className="text-xs" style={{ fontFamily: 'monospace' }}>TREE 1</text>
          <rect x="40" y="90" width="35" height="20" fill="#c4a000" stroke="#c4a000" strokeWidth="1" />
          <text x="58" y="104" textAnchor="middle" fill="#1e1e1e" className="text-xs" style={{ fontFamily: 'monospace' }}>X1</text>
          <rect x="85" y="90" width="35" height="20" fill="#c4a000" stroke="#c4a000" strokeWidth="1" />
          <text x="103" y="104" textAnchor="middle" fill="#1e1e1e" className="text-xs" style={{ fontFamily: 'monospace' }}>X3</text>

          {/* Tree 2 - uses X2, X4 */}
          <text x="220" y="85" textAnchor="middle" fill="#1a5c3a" className="text-xs" style={{ fontFamily: 'monospace' }}>TREE 2</text>
          <rect x="180" y="90" width="35" height="20" fill="#1a5c3a" stroke="#1a5c3a" strokeWidth="1" />
          <text x="198" y="104" textAnchor="middle" fill="#cfeee3" className="text-xs" style={{ fontFamily: 'monospace' }}>X2</text>
          <rect x="225" y="90" width="35" height="20" fill="#1a5c3a" stroke="#1a5c3a" strokeWidth="1" />
          <text x="243" y="104" textAnchor="middle" fill="#cfeee3" className="text-xs" style={{ fontFamily: 'monospace' }}>X4</text>

          {/* Arrows */}
          <line x1="80" y1="55" x2="80" y2="80" stroke="#d4d4c8" strokeWidth="1" strokeDasharray="3" />
          <line x1="220" y1="55" x2="220" y2="80" stroke="#d4d4c8" strokeWidth="1" strokeDasharray="3" />

          {/* Result */}
          <text x="150" y="145" textAnchor="middle" fill="#cfeee3" className="text-xs" style={{ fontFamily: 'monospace' }}>RANDOM SUBSETS AT EACH SPLIT</text>
          <text x="150" y="165" textAnchor="middle" fill="#d4d4c8" className="text-xs" style={{ fontFamily: 'monospace' }}>= DECORRELATED TREES</text>
          <text x="150" y="185" textAnchor="middle" fill="#d4d4c8" className="text-xs" style={{ fontFamily: 'monospace' }}>= REDUCED VARIANCE</text>
        </svg>
      </div>
    ),
  },
  {
    id: 4,
    title: "AGGREGATION (VOTING)",
    icon: <Vote className="w-4 h-4" />,
    content: (
      <div className="space-y-4">
        <p className="font-mono text-sm text-terminal-black leading-relaxed">
          For classification: majority vote across all trees. For regression: average of all predictions.
        </p>
        <div className="bg-terminal-black p-4 font-mono text-xs overflow-x-auto border-2 border-terminal-black">
          <p className="text-terminal-mint mb-2">
            <span className="text-terminal-accent">100 TREES VOTE:</span>
          </p>
          <p className="text-terminal-mint">
            CLASS_A: 67 votes <span className="text-terminal-accent">■■■■■■■</span>
          </p>
          <p className="text-terminal-mint">
            CLASS_B: 28 votes <span className="text-terminal-warning">■■■</span>
          </p>
          <p className="text-terminal-mint">
            CLASS_C: 5 votes <span className="text-terminal-grid">■</span>
          </p>
          <p className="text-terminal-mint mt-2">
            <span className="text-terminal-accent font-bold">PREDICTION: CLASS_A (67%)</span>
          </p>
        </div>
        <div className="bg-terminal-accent/10 border-2 border-terminal-accent p-3">
          <p className="font-mono text-xs text-terminal-accent font-bold">
            SOFT VOTING: PROBABILITIES CAN BE AVERAGED FOR CONFIDENCE SCORES
          </p>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full flex items-center justify-center">
        <div className="space-y-4 w-full max-w-sm font-mono text-xs">
          <div className="text-terminal-mint text-center font-bold mb-4">ENSEMBLE VOTING (N=100)</div>
          <div className="space-y-3">
            <div>
              <div className="flex justify-between items-center mb-1">
                <span className="text-terminal-grid">CLASS_A</span>
                <span className="text-terminal-accent font-bold">67 / 100</span>
              </div>
              <div className="h-4 bg-terminal-grid border border-terminal-grid">
                <div className="h-full bg-terminal-accent" style={{ width: "67%" }}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between items-center mb-1">
                <span className="text-terminal-grid">CLASS_B</span>
                <span className="text-terminal-warning font-bold">28 / 100</span>
              </div>
              <div className="h-4 bg-terminal-grid border border-terminal-grid">
                <div className="h-full bg-terminal-warning" style={{ width: "28%" }}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between items-center mb-1">
                <span className="text-terminal-grid">CLASS_C</span>
                <span className="text-terminal-grid font-bold">5 / 100</span>
              </div>
              <div className="h-4 bg-terminal-grid border border-terminal-grid">
                <div className="h-full bg-terminal-mint" style={{ width: "5%" }}></div>
              </div>
            </div>
          </div>
          <div className="border-t border-terminal-grid/30 pt-4 text-center">
            <p className="text-terminal-mint">WINNER: <span className="text-terminal-accent font-bold">CLASS_A</span></p>
            <p className="text-terminal-grid text-xs mt-1">CONFIDENCE: 67%</p>
          </div>
        </div>
      </div>
    ),
  },
  {
    id: 5,
    title: "ADVANTAGES & METRICS",
    icon: <CheckCircle className="w-4 h-4" />,
    content: (
      <div className="space-y-4">
        <p className="font-mono text-sm text-terminal-black leading-relaxed">
          Random Forests provide robust performance with built-in feature importance and resistance to overfitting.
        </p>
        <div className="space-y-2">
          <div className="bg-terminal-panel border-2 border-terminal-black p-3">
            <p className="font-mono text-xs font-bold text-terminal-black">PROS</p>
            <p className="font-mono text-xs text-terminal-black/70">Works well out-of-the-box, handles high dimensions, provides feature importance</p>
          </div>
          <div className="bg-terminal-panel border-2 border-terminal-black p-3">
            <p className="font-mono text-xs font-bold text-terminal-black">CONS</p>
            <p className="font-mono text-xs text-terminal-black/70">Less interpretable than single tree, slower for large forests, memory intensive</p>
          </div>
          <div className="bg-terminal-panel border-2 border-terminal-black p-3">
            <p className="font-mono text-xs font-bold text-terminal-black">HYPERPARAMETERS</p>
            <p className="font-mono text-xs text-terminal-black/70">n_estimators, max_depth, min_samples_split, max_features</p>
          </div>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full flex items-center justify-center">
        <div className="space-y-4 w-full max-w-xs font-mono">
          <div className="text-center">
            <div className="text-4xl font-bold text-terminal-accent">98.3%</div>
            <div className="text-xs text-terminal-grid">ACCURACY</div>
            <div className="text-xs text-terminal-accent font-bold">RANDOM FOREST</div>
          </div>
          <div className="border-t border-terminal-grid/30 pt-4">
            <div className="text-center mb-2">
              <div className="text-xl font-bold text-terminal-warning">93.7%</div>
              <div className="text-xs text-terminal-grid">SINGLE TREE</div>
            </div>
          </div>
          <div className="border-t border-terminal-grid/30 pt-4 text-center">
            <p className="text-terminal-mint text-xs">IMPROVEMENT: <span className="text-terminal-accent font-bold">+4.6%</span></p>
            <p className="text-terminal-grid text-xs mt-1">VARIANCE REDUCTION THROUGH AVERAGING</p>
          </div>
        </div>
      </div>
    ),
  },
];

export default function RandomForestExplainerTab() {
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
