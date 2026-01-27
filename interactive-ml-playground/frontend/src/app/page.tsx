"use client";

import { Terminal, Database, Cpu, ArrowRight, AlertTriangle } from "lucide-react";
import ModelCard from "@/components/ModelCard";
import HeroAnimation from "@/components/HeroAnimation";

const models = [
  {
    id: "linear-regression",
    name: "Linear Regression",
    description: "Predictive modeling through best-fit line analysis. Foundation of regression algorithms.",
    category: "REGRESSION",
    difficulty: "Beginner",
    tags: ["supervised", "regression", "interpretable"],
  },
  {
    id: "logistic-regression",
    name: "Logistic Regression",
    description: "Binary classification using sigmoid probability function. Gateway to classification systems.",
    category: "CLASSIFICATION",
    difficulty: "Beginner",
    tags: ["supervised", "classification", "probabilities"],
  },
  {
    id: "knn",
    name: "K-Nearest Neighbors",
    description: "Instance-based learning through proximity analysis. Lazy evaluation methodology.",
    category: "CLASSIFICATION",
    difficulty: "Beginner",
    tags: ["supervised", "instance-based", "lazy-learning"],
    comingSoon: true,
  },
  {
    id: "kmeans",
    name: "K-Means Clustering",
    description: "Unsupervised pattern discovery through centroid-based partitioning.",
    category: "CLUSTERING",
    difficulty: "Intermediate",
    tags: ["unsupervised", "clustering", "partitioning"],
    comingSoon: true,
  },
];

const procedures = [
  {
    code: "01",
    label: "SELECT MODEL",
    description: "Choose algorithm from available modules. Each serves distinct analytical purpose.",
    icon: Terminal,
  },
  {
    code: "02",
    label: "INPUT DATA",
    description: "Upload CSV dataset or utilize provided sample data for analysis.",
    icon: Database,
  },
  {
    code: "03",
    label: "EXECUTE",
    description: "Train model, analyze metrics, export reproducible Python implementation.",
    icon: Cpu,
  },
];

function HeroSection() {
  return (
    <section className="relative py-12 md:py-20 px-4">
      <div className="max-w-7xl mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12 items-center">
          {/* Left Content */}
          <div>
            {/* System Status */}
            <div className="flex items-center gap-3 mb-8">
              <div className="w-3 h-3 bg-terminal-accent animate-pulse" />
              <span className="font-mono text-xs uppercase tracking-terminal text-terminal-black">
                SYSTEM ONLINE // ALL MODULES OPERATIONAL
              </span> <br></br>
              <span className="font-mono text-xs uppercase tracking-terminal text-terminal-black">
    <a
      href="https://x.com/techwith_ram"
      target="_blank"
      className="
        font-mono text-xs uppercase tracking-terminal
        text-terminal-black
        px-2 py-1
        border-2 border-transparent
        transition-all duration-150
        hover:bg-terminal-black
        hover:text-terminal-mint
        hover:border-terminal-black
        hover:shadow-[inset_0_0_0_1px_rgba(0,0,0,0.3)]
        cursor-crosshair
      "
      >
  DESIGNED BY RAMAKRUSHNA
</a>
              </span>
            </div>

            {/* Main Title */}
            <h1 className="heading-terminal text-4xl sm:text-5xl md:text-6xl lg:text-7xl text-terminal-black mb-6 leading-tight">
              MACHINE
              <br />
              LEARNING
              <br />
              <span className="text-terminal-accent">TERMINAL</span>
            </h1>

            {/* Subtitle */}
            <p className="font-mono text-sm md:text-base text-terminal-black/70 max-w-xl mb-10 leading-relaxed">
              INTERACTIVE RESEARCH INTERFACE FOR ALGORITHMIC ANALYSIS.
              <br />
              TRAIN MODELS. ANALYZE DATA. GENERATE CODE.
              <br />
              NO ACCOUNTS REQUIRED. NO DATA RETENTION.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4">
              <a
                href="#models"
                className="btn-terminal text-base px-6 py-3"
              >
                ACCESS MODELS
                <ArrowRight className="w-4 h-4" />
              </a>
              <a
                href="#procedures"
                className="inline-flex items-center justify-center gap-2 px-6 py-3 font-mono font-bold text-xs uppercase tracking-terminal text-terminal-black border-2 border-terminal-black hover:bg-terminal-black hover:text-terminal-mint transition-all"
              >
                VIEW PROCEDURES
              </a>
            </div>
          </div>

          {/* Right Animation */}
          <div className="hidden lg:block">
            <div className="border-2 border-terminal-black bg-terminal-panel/30 h-[450px] relative overflow-hidden">
              <HeroAnimation />
            </div>
          </div>
        </div>

        {/* Warning Notice - Full Width Below */}
        <div className="mt-12 p-4 border-2 border-terminal-warning bg-terminal-warning/10 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-terminal-warning flex-shrink-0 mt-0.5" />
          <p className="font-mono text-xs text-terminal-black leading-relaxed">
            <span className="font-bold">NOTICE:</span> THIS IS AN EDUCATIONAL RESEARCH TOOL.
            ALL COMPUTATIONS ARE PERFORMED LOCALLY. NO USER DATA IS TRANSMITTED OR STORED.
          </p>
        </div>
      </div>
    </section>
  );
}

function ProceduresSection() {
  return (
    <section id="procedures" className="py-20 px-4">
      <div className="max-w-5xl mx-auto">
        {/* Section Header */}
        <div className="mb-12">
          <span className="font-mono text-xs uppercase tracking-terminal text-terminal-accent mb-2 block">
            {"//"} OPERATIONAL PROCEDURES
          </span>
          <h2 className="heading-terminal text-3xl md:text-4xl text-terminal-black">
            SYSTEM WORKFLOW
          </h2>
        </div>

        {/* Procedure Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {procedures.map((proc, index) => (
            <div
              key={proc.code}
              className="bg-terminal-panel border-2 border-terminal-black p-6 relative"
            >
              {/* Procedure Number */}
              <div className="text-6xl font-mono font-bold text-terminal-black/10 absolute top-4 right-4">
                {proc.code}
              </div>

              {/* Icon */}
              <div className="w-12 h-12 bg-terminal-black flex items-center justify-center mb-4">
                <proc.icon className="w-6 h-6 text-terminal-mint" />
              </div>

              {/* Content */}
              <h3 className="font-mono font-bold text-sm uppercase tracking-terminal text-terminal-black mb-2">
                {proc.label}
              </h3>
              <p className="font-mono text-xs text-terminal-black/70 leading-relaxed">
                {proc.description}
              </p>

              {/* Connector */}
              {index < procedures.length - 1 && (
                <div className="hidden md:block absolute top-1/2 -right-6 w-6 border-t-2 border-dashed border-terminal-black/30" />
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function ModelsSection() {
  return (
    <section id="models" className="py-20 px-4 bg-terminal-panel/50">
      <div className="max-w-6xl mx-auto">
        {/* Section Header */}
        <div className="mb-12">
          <span className="font-mono text-xs uppercase tracking-terminal text-terminal-accent mb-2 block">
            {"//"} AVAILABLE MODULES
          </span>
          <h2 className="heading-terminal text-3xl md:text-4xl text-terminal-black mb-4">
            MODEL REGISTRY
          </h2>
          <p className="font-mono text-xs text-terminal-black/70 max-w-2xl">
            SELECT A MODULE TO BEGIN ANALYSIS. EACH MODEL INCLUDES INTERACTIVE TRAINING,
            VISUALIZATION, AND CODE GENERATION CAPABILITIES.
          </p>
        </div>

        {/* Model Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {models.map((model) => (
            <ModelCard
              key={model.id}
              {...model}
            />
          ))}
        </div>
      </div>
    </section>
  );
}

function SystemInfoSection() {
  return (
    <section className="py-20 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="bg-terminal-black text-terminal-mint p-8 md:p-12 border-2 border-terminal-black">
          {/* Terminal Header */}
          <div className="flex items-center gap-2 mb-6 pb-4 border-b border-terminal-mint/30">
            <div className="w-3 h-3 bg-red-500" />
            <div className="w-3 h-3 bg-terminal-warning" />
            <div className="w-3 h-3 bg-terminal-accent" />
            <span className="ml-4 font-mono text-xs opacity-60">SYSTEM_INFO.sh</span>
          </div>

          {/* Terminal Content */}
          <div className="font-mono text-sm space-y-2">
            <p><span className="text-terminal-accent">$</span> cat /etc/ml-terminal/info</p>
            <p className="opacity-70">---</p>
            <p className="opacity-70">VERSION: 1.0.0</p>
            <p className="opacity-70">STATUS: OPERATIONAL</p>
            <p className="opacity-70">MODELS_AVAILABLE: 4</p>
            <p className="opacity-70">MODELS_ACTIVE: 2</p>
            <p className="opacity-70">---</p>
            <p className="mt-4"><span className="text-terminal-accent">$</span> echo $PRIVACY_POLICY</p>
            <p className="opacity-70">NO_TRACKING // NO_COOKIES // NO_DATA_RETENTION</p>
            <p className="mt-4"><span className="text-terminal-accent">$</span> _<span className="animate-pulse">|</span></p>
          </div>

          {/* CTA */}
          <div className="mt-8 pt-6 border-t border-terminal-mint/30">
            <a
              href="#models"
              className="inline-flex items-center gap-2 px-6 py-3 bg-terminal-mint text-terminal-black font-mono font-bold text-xs uppercase tracking-terminal border-2 border-terminal-mint hover:bg-transparent hover:text-terminal-mint transition-all"
            >
              INITIALIZE SESSION
              <ArrowRight className="w-4 h-4" />
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  return (
    <main className="relative">
      <HeroSection />
      <ProceduresSection />
      <ModelsSection />
      <SystemInfoSection />
    </main>
  );
}
