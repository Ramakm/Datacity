"use client";

import { useState } from "react";
import {
  ChevronRight,
  ChevronLeft,
  Layers,
  Target,
  RefreshCw,
  Move,
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
          SCENARIO: Customer segmentation. Input: behavioral metrics. Output: group assignments WITHOUT labels.
        </p>
        <p className="font-mono text-xs text-terminal-black/70">
          Discover natural groupings in data without prior knowledge of categories. The algorithm finds structure autonomously.
        </p>
        <div className="bg-terminal-accent/10 border-2 border-terminal-accent p-4">
          <p className="font-mono text-xs text-terminal-accent font-bold">
            CLASSIFICATION: UNSUPERVISED LEARNING // CENTROID-BASED CLUSTERING // PARTITIONING METHOD
          </p>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full flex items-center justify-center">
        <div className="space-y-2 w-full max-w-xs font-mono text-xs">
          <div className="flex items-center justify-between p-2 border border-terminal-grid text-terminal-mint">
            <span>CUSTOMER_001</span>
            <span className="font-bold text-terminal-warning">CLUSTER: ???</span>
          </div>
          <div className="flex items-center justify-between p-2 border border-terminal-grid text-terminal-mint">
            <span>CUSTOMER_002</span>
            <span className="font-bold text-terminal-warning">CLUSTER: ???</span>
          </div>
          <div className="flex items-center justify-between p-2 border border-terminal-grid text-terminal-mint">
            <span>CUSTOMER_003</span>
            <span className="font-bold text-terminal-warning">CLUSTER: ???</span>
          </div>
          <div className="text-center mt-4 text-terminal-accent">
            NO LABELS PROVIDED // FIND NATURAL GROUPS
          </div>
        </div>
      </div>
    ),
  },
  {
    id: 2,
    title: "CENTROID INITIALIZATION",
    icon: <Target className="w-4 h-4" />,
    content: (
      <div className="space-y-4">
        <p className="font-mono text-sm text-terminal-black leading-relaxed">
          Initialize K cluster centers (centroids) randomly in the feature space.
        </p>
        <div className="bg-terminal-panel border-2 border-terminal-black p-4 space-y-2">
          <p className="font-mono text-xs font-bold text-terminal-black">INITIALIZATION METHODS:</p>
          <ul className="space-y-1 font-mono text-xs text-terminal-black/80 list-disc list-inside">
            <li>Random: Select K random points as initial centroids</li>
            <li>K-Means++: Smart initialization for better convergence</li>
            <li>Manual: User-defined starting positions</li>
          </ul>
        </div>
        <div className="bg-terminal-warning/10 border-2 border-terminal-warning p-3">
          <p className="font-mono text-xs text-terminal-black">
            <span className="font-bold text-terminal-warning">K VALUE:</span> Number of clusters must be specified in advance. Common methods: Elbow method, Silhouette analysis.
          </p>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full">
        <svg viewBox="0 0 300 200" className="w-full h-full">
          <line x1="40" y1="160" x2="280" y2="160" stroke="#d4d4c8" strokeWidth="1" />
          <line x1="40" y1="160" x2="40" y2="20" stroke="#d4d4c8" strokeWidth="1" />
          <text x="160" y="185" textAnchor="middle" fill="#d4d4c8" className="text-xs" style={{ fontFamily: 'monospace' }}>FEATURE_1</text>
          <text x="15" y="90" textAnchor="middle" transform="rotate(-90, 15, 90)" fill="#d4d4c8" className="text-xs" style={{ fontFamily: 'monospace' }}>FEATURE_2</text>

          {/* Data points (unassigned - gray) */}
          <circle cx="70" cy="130" r="4" fill="#d4d4c8" />
          <circle cx="90" cy="120" r="4" fill="#d4d4c8" />
          <circle cx="80" cy="140" r="4" fill="#d4d4c8" />
          <circle cx="100" cy="135" r="4" fill="#d4d4c8" />

          <circle cx="180" cy="50" r="4" fill="#d4d4c8" />
          <circle cx="200" cy="60" r="4" fill="#d4d4c8" />
          <circle cx="190" cy="40" r="4" fill="#d4d4c8" />
          <circle cx="210" cy="55" r="4" fill="#d4d4c8" />

          <circle cx="240" cy="130" r="4" fill="#d4d4c8" />
          <circle cx="250" cy="140" r="4" fill="#d4d4c8" />
          <circle cx="260" cy="125" r="4" fill="#d4d4c8" />

          {/* Initial centroids (K=3) */}
          <circle cx="100" cy="80" r="8" fill="#1a5c3a" stroke="#cfeee3" strokeWidth="2" />
          <circle cx="180" cy="100" r="8" fill="#c4a000" stroke="#cfeee3" strokeWidth="2" />
          <circle cx="220" cy="70" r="8" fill="#dc2626" stroke="#cfeee3" strokeWidth="2" />

          <text x="100" y="70" textAnchor="middle" fill="#cfeee3" className="text-xs" style={{ fontFamily: 'monospace' }}>C1</text>
          <text x="180" y="90" textAnchor="middle" fill="#cfeee3" className="text-xs" style={{ fontFamily: 'monospace' }}>C2</text>
          <text x="220" y="60" textAnchor="middle" fill="#cfeee3" className="text-xs" style={{ fontFamily: 'monospace' }}>C3</text>
        </svg>
      </div>
    ),
  },
  {
    id: 3,
    title: "ASSIGNMENT STEP",
    icon: <Layers className="w-4 h-4" />,
    content: (
      <div className="space-y-4">
        <p className="font-mono text-sm text-terminal-black leading-relaxed">
          Assign each data point to the nearest centroid based on distance.
        </p>
        <div className="bg-terminal-black p-4 font-mono text-center text-base border-2 border-terminal-black">
          <span className="text-terminal-accent font-bold">cluster(x)</span>
          <span className="text-terminal-mint"> = </span>
          <span className="text-terminal-warning">argmin</span>
          <span className="text-terminal-mint"> ||x - cᵢ||²</span>
        </div>
        <div className="bg-terminal-panel border-2 border-terminal-black p-4 space-y-2">
          <p className="font-mono text-xs font-bold text-terminal-black">PROCEDURE:</p>
          <ol className="space-y-1 font-mono text-xs text-terminal-black/80 list-decimal list-inside">
            <li>Calculate distance from point to each centroid</li>
            <li>Find minimum distance</li>
            <li>Assign point to that cluster</li>
            <li>Repeat for all data points</li>
          </ol>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full">
        <svg viewBox="0 0 300 200" className="w-full h-full">
          <line x1="40" y1="160" x2="280" y2="160" stroke="#d4d4c8" strokeWidth="1" />
          <line x1="40" y1="160" x2="40" y2="20" stroke="#d4d4c8" strokeWidth="1" />

          {/* Cluster 1 points (green) */}
          <circle cx="70" cy="130" r="4" fill="#1a5c3a" />
          <circle cx="90" cy="120" r="4" fill="#1a5c3a" />
          <circle cx="80" cy="140" r="4" fill="#1a5c3a" />
          <circle cx="100" cy="135" r="4" fill="#1a5c3a" />

          {/* Cluster 2 points (yellow) */}
          <circle cx="180" cy="50" r="4" fill="#c4a000" />
          <circle cx="200" cy="60" r="4" fill="#c4a000" />
          <circle cx="190" cy="40" r="4" fill="#c4a000" />
          <circle cx="210" cy="55" r="4" fill="#c4a000" />

          {/* Cluster 3 points (red) */}
          <circle cx="240" cy="130" r="4" fill="#dc2626" />
          <circle cx="250" cy="140" r="4" fill="#dc2626" />
          <circle cx="260" cy="125" r="4" fill="#dc2626" />

          {/* Centroids */}
          <circle cx="85" cy="130" r="10" fill="none" stroke="#1a5c3a" strokeWidth="2" strokeDasharray="4" />
          <circle cx="195" cy="50" r="10" fill="none" stroke="#c4a000" strokeWidth="2" strokeDasharray="4" />
          <circle cx="250" cy="130" r="10" fill="none" stroke="#dc2626" strokeWidth="2" strokeDasharray="4" />

          {/* Distance lines */}
          <line x1="70" y1="130" x2="85" y2="130" stroke="#cfeee3" strokeWidth="1" strokeDasharray="2" opacity="0.5" />
          <line x1="180" y1="50" x2="195" y2="50" stroke="#cfeee3" strokeWidth="1" strokeDasharray="2" opacity="0.5" />

          <text x="150" y="185" textAnchor="middle" fill="#cfeee3" className="text-xs" style={{ fontFamily: 'monospace' }}>POINTS ASSIGNED TO NEAREST CENTROID</text>
        </svg>
      </div>
    ),
  },
  {
    id: 4,
    title: "UPDATE STEP",
    icon: <Move className="w-4 h-4" />,
    content: (
      <div className="space-y-4">
        <p className="font-mono text-sm text-terminal-black leading-relaxed">
          Recalculate centroid positions as the mean of all assigned points.
        </p>
        <div className="bg-terminal-black p-4 font-mono text-center text-base border-2 border-terminal-black">
          <span className="text-terminal-accent font-bold">cᵢ</span>
          <span className="text-terminal-mint"> = </span>
          <span className="text-terminal-warning">1/|Sᵢ|</span>
          <span className="text-terminal-mint"> Σ x</span>
        </div>
        <ul className="space-y-2 font-mono text-xs text-terminal-black/80">
          <li><span className="text-terminal-accent font-bold">cᵢ</span> = new centroid position for cluster i</li>
          <li><span className="text-terminal-warning font-bold">|Sᵢ|</span> = number of points in cluster i</li>
          <li><span className="text-terminal-mint font-bold">Σ x</span> = sum of all point coordinates in cluster</li>
        </ul>
        <div className="bg-terminal-accent/10 border-2 border-terminal-accent p-3">
          <p className="font-mono text-xs text-terminal-accent font-bold">
            CENTROIDS MOVE TOWARD THE CENTER OF THEIR ASSIGNED POINTS
          </p>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full">
        <svg viewBox="0 0 300 200" className="w-full h-full">
          <line x1="40" y1="160" x2="280" y2="160" stroke="#d4d4c8" strokeWidth="1" />
          <line x1="40" y1="160" x2="40" y2="20" stroke="#d4d4c8" strokeWidth="1" />

          {/* Cluster points */}
          <circle cx="70" cy="130" r="4" fill="#1a5c3a" />
          <circle cx="90" cy="120" r="4" fill="#1a5c3a" />
          <circle cx="80" cy="140" r="4" fill="#1a5c3a" />
          <circle cx="100" cy="135" r="4" fill="#1a5c3a" />

          <circle cx="180" cy="50" r="4" fill="#c4a000" />
          <circle cx="200" cy="60" r="4" fill="#c4a000" />
          <circle cx="190" cy="40" r="4" fill="#c4a000" />
          <circle cx="210" cy="55" r="4" fill="#c4a000" />

          <circle cx="240" cy="130" r="4" fill="#dc2626" />
          <circle cx="250" cy="140" r="4" fill="#dc2626" />
          <circle cx="260" cy="125" r="4" fill="#dc2626" />

          {/* Old centroids (faded) */}
          <circle cx="85" cy="130" r="6" fill="#1a5c3a" opacity="0.3" />
          <circle cx="195" cy="50" r="6" fill="#c4a000" opacity="0.3" />
          <circle cx="250" cy="130" r="6" fill="#dc2626" opacity="0.3" />

          {/* Movement arrows */}
          <line x1="85" y1="130" x2="85" y2="131" stroke="#cfeee3" strokeWidth="2" markerEnd="url(#arrow)" />
          <line x1="195" y1="50" x2="195" y2="51" stroke="#cfeee3" strokeWidth="2" />

          {/* New centroids */}
          <circle cx="85" cy="131" r="8" fill="#1a5c3a" stroke="#cfeee3" strokeWidth="2" />
          <circle cx="195" cy="51" r="8" fill="#c4a000" stroke="#cfeee3" strokeWidth="2" />
          <circle cx="250" cy="132" r="8" fill="#dc2626" stroke="#cfeee3" strokeWidth="2" />

          <text x="150" y="185" textAnchor="middle" fill="#cfeee3" className="text-xs" style={{ fontFamily: 'monospace' }}>CENTROIDS UPDATED TO CLUSTER MEANS</text>
        </svg>
      </div>
    ),
  },
  {
    id: 5,
    title: "ITERATE TO CONVERGENCE",
    icon: <RefreshCw className="w-4 h-4" />,
    content: (
      <div className="space-y-4">
        <p className="font-mono text-sm text-terminal-black leading-relaxed">
          Repeat assignment and update steps until convergence.
        </p>
        <div className="bg-terminal-panel border-2 border-terminal-black p-4 space-y-2">
          <p className="font-mono text-xs font-bold text-terminal-black">CONVERGENCE CRITERIA:</p>
          <ul className="space-y-1 font-mono text-xs text-terminal-black/80 list-disc list-inside">
            <li>No points change clusters</li>
            <li>Centroids move less than threshold</li>
            <li>Maximum iterations reached</li>
          </ul>
        </div>
        <div className="bg-terminal-black p-4 font-mono text-xs text-terminal-mint">
          <p>ITER 1: INERTIA = 245.32</p>
          <p>ITER 2: INERTIA = 189.45</p>
          <p>ITER 3: INERTIA = 156.23</p>
          <p>ITER 4: INERTIA = 152.10</p>
          <p className="text-terminal-accent font-bold">ITER 5: CONVERGED</p>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full flex items-center justify-center">
        <div className="space-y-4 w-full max-w-sm font-mono text-xs">
          <div className="text-terminal-mint text-center font-bold mb-4">CONVERGENCE PROGRESS</div>
          <div className="space-y-2">
            {[
              { iter: 1, inertia: 100, label: "INITIAL" },
              { iter: 2, inertia: 75, label: "IMPROVING" },
              { iter: 3, inertia: 55, label: "IMPROVING" },
              { iter: 4, inertia: 45, label: "STABILIZING" },
              { iter: 5, inertia: 42, label: "CONVERGED" },
            ].map((step) => (
              <div key={step.iter} className="flex items-center gap-2">
                <span className="text-terminal-grid w-16">ITER {step.iter}</span>
                <div className="flex-1 h-3 bg-terminal-grid">
                  <div
                    className={clsx(
                      "h-full transition-all",
                      step.label === "CONVERGED" ? "bg-terminal-accent" : "bg-terminal-warning"
                    )}
                    style={{ width: `${step.inertia}%` }}
                  />
                </div>
                <span className={clsx(
                  "w-20 text-right",
                  step.label === "CONVERGED" ? "text-terminal-accent font-bold" : "text-terminal-grid"
                )}>
                  {step.label}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    ),
  },
  {
    id: 6,
    title: "EVALUATION METRICS",
    icon: <CheckCircle className="w-4 h-4" />,
    content: (
      <div className="space-y-4">
        <p className="font-mono text-sm text-terminal-black leading-relaxed">
          Evaluate clustering quality without ground truth labels.
        </p>
        <div className="space-y-2">
          <div className="bg-terminal-panel border-2 border-terminal-black p-3">
            <p className="font-mono text-xs font-bold text-terminal-black">SILHOUETTE SCORE (-1 to 1)</p>
            <p className="font-mono text-xs text-terminal-black/70">Measures cluster cohesion vs separation. Higher = better.</p>
          </div>
          <div className="bg-terminal-panel border-2 border-terminal-black p-3">
            <p className="font-mono text-xs font-bold text-terminal-black">INERTIA</p>
            <p className="font-mono text-xs text-terminal-black/70">Sum of squared distances to centroids. Lower = tighter clusters.</p>
          </div>
          <div className="bg-terminal-panel border-2 border-terminal-black p-3">
            <p className="font-mono text-xs font-bold text-terminal-black">CALINSKI-HARABASZ INDEX</p>
            <p className="font-mono text-xs text-terminal-black/70">Ratio of between/within cluster dispersion. Higher = better.</p>
          </div>
        </div>
      </div>
    ),
    visual: (
      <div className="bg-terminal-black p-6 h-full flex items-center justify-center">
        <div className="space-y-4 w-full max-w-xs font-mono">
          <div className="text-center">
            <div className="text-4xl font-bold text-terminal-accent">0.72</div>
            <div className="text-xs text-terminal-grid">SILHOUETTE SCORE</div>
            <div className="text-xs text-terminal-accent font-bold">GOOD SEPARATION</div>
          </div>
          <div className="border-t border-terminal-grid/30 pt-4 grid grid-cols-2 gap-4 text-center">
            <div>
              <div className="text-xl font-bold text-terminal-mint">152.3</div>
              <div className="text-xs text-terminal-grid">INERTIA</div>
            </div>
            <div>
              <div className="text-xl font-bold text-terminal-mint">456.7</div>
              <div className="text-xs text-terminal-grid">CALINSKI-HARABASZ</div>
            </div>
          </div>
          <div className="border-t border-terminal-grid/30 pt-4 text-center">
            <div className="text-lg font-bold text-terminal-warning">K = 3</div>
            <div className="text-xs text-terminal-grid">OPTIMAL CLUSTERS</div>
          </div>
        </div>
      </div>
    ),
  },
];

export default function KMeansExplainerTab() {
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
            PROCEED TO &quot;EXECUTE MODEL&quot; TAB TO CLUSTER YOUR DATA
          </p>
        </div>
      )}
    </div>
  );
}
