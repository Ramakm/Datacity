"use client";

import { useState } from "react";
import {
  ChevronRight,
  ChevronLeft,
  Target,
  GitBranch,
  Layers,
  Leaf,
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
          SCENARIO: Classification or regression task. Decision Trees learn a hierarchy of if-else questions to make predictions.
        </p>
        <p className="font-mono text-xs text-terminal-black/70">
          Given features, the tree asks a series of questions to arrive at a prediction. Each question splits the data based on a feature threshold.
        </p>
        <div className="bg-terminal-accent/10 border-2 border-terminal-accent p-4">
          <p className="font-mono text-xs text-terminal-accent font-bold">
            CLASSIFICATION: SUPERVISED LEARNING // INTERPRETABLE MODEL // RULE-BASED DECISIONS
          </p>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full flex items-center justify-center">
        <div className="space-y-2 w-full max-w-xs font-mono text-xs">
          <div className="flex items-center justify-between p-2 border border-terminal-grid text-terminal-mint">
            <span>IF petal_length &lt; 2.5</span>
            <span className="font-bold text-terminal-accent">SETOSA</span>
          </div>
          <div className="flex items-center justify-between p-2 border border-terminal-grid text-terminal-mint">
            <span>IF petal_width &lt; 1.75</span>
            <span className="font-bold text-terminal-warning">VERSICOLOR</span>
          </div>
          <div className="flex items-center justify-between p-2 border border-terminal-grid text-terminal-mint">
            <span>ELSE</span>
            <span className="font-bold text-terminal-mint">VIRGINICA</span>
          </div>
          <div className="flex items-center justify-between p-2 border-2 border-dashed border-terminal-warning text-terminal-warning mt-4">
            <span>NEW SAMPLE</span>
            <span className="font-bold">FOLLOW THE TREE</span>
          </div>
        </div>
      </div>
    ),
  },
  {
    id: 2,
    title: "SPLITTING CRITERIA",
    icon: <GitBranch className="w-4 h-4" />,
    content: (
      <div className="space-y-4">
        <p className="font-mono text-sm text-terminal-black leading-relaxed">
          The tree finds the best question to ask at each node by measuring information gain or impurity reduction.
        </p>
        <div className="bg-terminal-black p-4 font-mono text-center text-base border-2 border-terminal-black">
          <span className="text-terminal-accent font-bold">Gini</span>
          <span className="text-terminal-mint"> = 1 - </span>
          <span className="text-terminal-warning">Σpᵢ²</span>
        </div>
        <ul className="space-y-2 font-mono text-xs text-terminal-black/80">
          <li><span className="text-terminal-accent font-bold">Gini Impurity</span> = measures class heterogeneity</li>
          <li><span className="text-terminal-warning font-bold">pᵢ</span> = proportion of class i in node</li>
          <li>Goal: minimize impurity after split</li>
        </ul>
        <p className="font-mono text-xs text-terminal-black/70">
          Alternative: Entropy / Information Gain
        </p>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full">
        <svg viewBox="0 0 300 200" className="w-full h-full">
          {/* Root node */}
          <rect x="110" y="10" width="80" height="35" fill="none" stroke="#cfeee3" strokeWidth="2" />
          <text x="150" y="32" textAnchor="middle" fill="#cfeee3" className="text-xs" style={{ fontFamily: 'monospace' }}>ROOT</text>

          {/* Split lines */}
          <line x1="150" y1="45" x2="80" y2="80" stroke="#d4d4c8" strokeWidth="1" />
          <line x1="150" y1="45" x2="220" y2="80" stroke="#d4d4c8" strokeWidth="1" />

          {/* Left child - mixed */}
          <rect x="40" y="80" width="80" height="35" fill="none" stroke="#c4a000" strokeWidth="2" />
          <text x="80" y="95" textAnchor="middle" fill="#c4a000" className="text-xs" style={{ fontFamily: 'monospace' }}>GINI=0.5</text>
          <text x="80" y="108" textAnchor="middle" fill="#d4d4c8" className="text-xs" style={{ fontFamily: 'monospace' }}>MIXED</text>

          {/* Right child - pure */}
          <rect x="180" y="80" width="80" height="35" fill="none" stroke="#1a5c3a" strokeWidth="2" />
          <text x="220" y="95" textAnchor="middle" fill="#1a5c3a" className="text-xs" style={{ fontFamily: 'monospace' }}>GINI=0.0</text>
          <text x="220" y="108" textAnchor="middle" fill="#d4d4c8" className="text-xs" style={{ fontFamily: 'monospace' }}>PURE</text>

          {/* Legend */}
          <text x="150" y="160" textAnchor="middle" fill="#cfeee3" className="text-xs" style={{ fontFamily: 'monospace' }}>LOWER GINI = PURER NODE</text>
          <text x="150" y="180" textAnchor="middle" fill="#d4d4c8" className="text-xs" style={{ fontFamily: 'monospace' }}>SPLIT TO REDUCE IMPURITY</text>
        </svg>
      </div>
    ),
  },
  {
    id: 3,
    title: "TREE CONSTRUCTION",
    icon: <Layers className="w-4 h-4" />,
    content: (
      <div className="space-y-4">
        <p className="font-mono text-sm text-terminal-black leading-relaxed">
          The tree grows by recursively splitting nodes until stopping criteria are met.
        </p>
        <div className="bg-terminal-panel border-2 border-terminal-black p-4 space-y-2">
          <p className="font-mono text-xs font-bold text-terminal-black">STOPPING CONDITIONS:</p>
          <ol className="space-y-1 font-mono text-xs text-terminal-black/80 list-decimal list-inside">
            <li>Maximum depth reached</li>
            <li>Minimum samples per node</li>
            <li>Node is pure (all same class)</li>
            <li>No improvement from split</li>
          </ol>
        </div>
        <div className="bg-terminal-warning/10 border-2 border-terminal-warning p-3">
          <p className="font-mono text-xs text-terminal-black">
            <span className="font-bold text-terminal-warning">OVERFITTING RISK:</span> Deep trees memorize training data. Use max_depth and min_samples_split to regularize.
          </p>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full">
        <svg viewBox="0 0 300 200" className="w-full h-full">
          {/* Level 0 - Root */}
          <rect x="130" y="5" width="40" height="25" fill="#cfeee3" stroke="#cfeee3" strokeWidth="1" />
          <text x="150" y="21" textAnchor="middle" fill="#1e1e1e" className="text-xs" style={{ fontFamily: 'monospace', fontWeight: 'bold' }}>X1&lt;5</text>

          {/* Level 1 */}
          <line x1="150" y1="30" x2="80" y2="50" stroke="#d4d4c8" strokeWidth="1" />
          <line x1="150" y1="30" x2="220" y2="50" stroke="#d4d4c8" strokeWidth="1" />

          <rect x="55" y="50" width="50" height="25" fill="#c4a000" stroke="#c4a000" strokeWidth="1" />
          <text x="80" y="66" textAnchor="middle" fill="#1e1e1e" className="text-xs" style={{ fontFamily: 'monospace', fontWeight: 'bold' }}>X2&lt;3</text>

          <rect x="195" y="50" width="50" height="25" fill="#1a5c3a" stroke="#1a5c3a" strokeWidth="1" />
          <text x="220" y="66" textAnchor="middle" fill="#cfeee3" className="text-xs" style={{ fontFamily: 'monospace', fontWeight: 'bold' }}>A</text>

          {/* Level 2 */}
          <line x1="80" y1="75" x2="50" y2="95" stroke="#d4d4c8" strokeWidth="1" />
          <line x1="80" y1="75" x2="110" y2="95" stroke="#d4d4c8" strokeWidth="1" />

          <rect x="30" y="95" width="40" height="25" fill="#1a5c3a" stroke="#1a5c3a" strokeWidth="1" />
          <text x="50" y="111" textAnchor="middle" fill="#cfeee3" className="text-xs" style={{ fontFamily: 'monospace', fontWeight: 'bold' }}>B</text>

          <rect x="90" y="95" width="40" height="25" fill="#c4a000" stroke="#c4a000" strokeWidth="1" />
          <text x="110" y="111" textAnchor="middle" fill="#1e1e1e" className="text-xs" style={{ fontFamily: 'monospace', fontWeight: 'bold' }}>C</text>

          {/* Labels */}
          <text x="40" y="145" fill="#d4d4c8" className="text-xs" style={{ fontFamily: 'monospace' }}>DEPTH: 0</text>
          <text x="115" y="145" fill="#d4d4c8" className="text-xs" style={{ fontFamily: 'monospace' }}>1</text>
          <text x="190" y="145" fill="#d4d4c8" className="text-xs" style={{ fontFamily: 'monospace' }}>2</text>

          <text x="150" y="175" textAnchor="middle" fill="#cfeee3" className="text-xs" style={{ fontFamily: 'monospace' }}>RECURSIVE BINARY SPLITTING</text>
        </svg>
      </div>
    ),
  },
  {
    id: 4,
    title: "MAKING PREDICTIONS",
    icon: <Leaf className="w-4 h-4" />,
    content: (
      <div className="space-y-4">
        <p className="font-mono text-sm text-terminal-black leading-relaxed">
          To classify a new sample, traverse the tree from root to leaf following the decision rules.
        </p>
        <div className="bg-terminal-black p-4 font-mono text-xs overflow-x-auto border-2 border-terminal-black">
          <p className="text-terminal-mint mb-2">
            <span className="text-terminal-accent">NEW SAMPLE:</span> petal_length=4.5
          </p>
          <p className="text-terminal-mint">
            ROOT: petal_length &lt; 2.5? <span className="text-red-400">NO</span>
          </p>
          <p className="text-terminal-mint">
            &nbsp;&nbsp;→ petal_width &lt; 1.75? <span className="text-terminal-accent">YES</span>
          </p>
          <p className="text-terminal-mint mt-2">
            <span className="text-terminal-warning font-bold">PREDICTION: VERSICOLOR</span>
          </p>
        </div>
        <div className="bg-terminal-accent/10 border-2 border-terminal-accent p-3">
          <p className="font-mono text-xs text-terminal-accent font-bold">
            LEAF NODE PREDICTION: MAJORITY CLASS FOR CLASSIFICATION // MEAN VALUE FOR REGRESSION
          </p>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full flex items-center justify-center">
        <div className="space-y-4 w-full max-w-sm font-mono text-xs">
          <div className="text-terminal-mint text-center font-bold mb-4">TRAVERSAL PATH</div>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-terminal-accent flex items-center justify-center text-terminal-black font-bold">1</div>
              <div className="flex-1 p-2 border border-terminal-grid text-terminal-mint">
                petal_length &lt; 2.5?
              </div>
              <div className="text-red-400 font-bold">NO</div>
            </div>
            <div className="ml-4 border-l-2 border-terminal-grid h-4"></div>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-terminal-warning flex items-center justify-center text-terminal-black font-bold">2</div>
              <div className="flex-1 p-2 border border-terminal-grid text-terminal-mint">
                petal_width &lt; 1.75?
              </div>
              <div className="text-terminal-accent font-bold">YES</div>
            </div>
            <div className="ml-4 border-l-2 border-terminal-grid h-4"></div>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-terminal-mint flex items-center justify-center text-terminal-black font-bold">L</div>
              <div className="flex-1 p-2 border-2 border-terminal-warning bg-terminal-warning/20 text-terminal-warning font-bold">
                VERSICOLOR (LEAF)
              </div>
            </div>
          </div>
        </div>
      </div>
    ),
  },
  {
    id: 5,
    title: "FEATURE IMPORTANCE",
    icon: <CheckCircle className="w-4 h-4" />,
    content: (
      <div className="space-y-4">
        <p className="font-mono text-sm text-terminal-black leading-relaxed">
          Decision Trees provide built-in feature importance based on impurity reduction.
        </p>
        <div className="space-y-2">
          <div className="bg-terminal-panel border-2 border-terminal-black p-3">
            <p className="font-mono text-xs font-bold text-terminal-black">IMPORTANCE CALCULATION</p>
            <p className="font-mono text-xs text-terminal-black/70">Sum of impurity decrease weighted by samples at each split</p>
          </div>
          <div className="bg-terminal-panel border-2 border-terminal-black p-3">
            <p className="font-mono text-xs font-bold text-terminal-black">INTERPRETABILITY</p>
            <p className="font-mono text-xs text-terminal-black/70">Easy to visualize and explain decisions</p>
          </div>
          <div className="bg-terminal-panel border-2 border-terminal-black p-3">
            <p className="font-mono text-xs font-bold text-terminal-black">METRICS</p>
            <p className="font-mono text-xs text-terminal-black/70">Accuracy, Precision, Recall, F1-Score</p>
          </div>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full flex items-center justify-center">
        <div className="space-y-4 w-full max-w-xs font-mono">
          <div className="text-center mb-4">
            <div className="text-xs text-terminal-grid uppercase mb-2">FEATURE IMPORTANCE</div>
          </div>
          <div className="space-y-3">
            <div>
              <div className="flex justify-between items-center mb-1">
                <span className="text-terminal-grid text-xs">petal_width</span>
                <span className="text-terminal-accent font-bold text-xs">0.45</span>
              </div>
              <div className="h-3 bg-terminal-grid border border-terminal-grid">
                <div className="h-full bg-terminal-accent" style={{ width: "90%" }}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between items-center mb-1">
                <span className="text-terminal-grid text-xs">petal_length</span>
                <span className="text-terminal-mint font-bold text-xs">0.38</span>
              </div>
              <div className="h-3 bg-terminal-grid border border-terminal-grid">
                <div className="h-full bg-terminal-mint" style={{ width: "76%" }}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between items-center mb-1">
                <span className="text-terminal-grid text-xs">sepal_length</span>
                <span className="text-terminal-warning font-bold text-xs">0.12</span>
              </div>
              <div className="h-3 bg-terminal-grid border border-terminal-grid">
                <div className="h-full bg-terminal-warning" style={{ width: "24%" }}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between items-center mb-1">
                <span className="text-terminal-grid text-xs">sepal_width</span>
                <span className="text-terminal-grid font-bold text-xs">0.05</span>
              </div>
              <div className="h-3 bg-terminal-grid border border-terminal-grid">
                <div className="h-full bg-terminal-grid" style={{ width: "10%" }}></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    ),
  },
];

export default function DecisionTreeExplainerTab() {
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
