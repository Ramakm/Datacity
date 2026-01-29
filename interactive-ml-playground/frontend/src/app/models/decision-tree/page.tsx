"use client";

import { useState } from "react";
import { ArrowLeft, BookOpen, FlaskConical, GitBranch } from "lucide-react";
import Link from "next/link";
import clsx from "clsx";
import DecisionTreeExplainerTab from "@/components/DecisionTreeExplainerTab";
import DecisionTreeTryItTab from "@/components/DecisionTreeTryItTab";

type Tab = "explainer" | "tryit";

export default function DecisionTreePage() {
  const [activeTab, setActiveTab] = useState<Tab>("explainer");

  return (
    <div className="min-h-screen bg-terminal-bg">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <Link
            href="/"
            className="inline-flex items-center gap-2 font-mono text-xs uppercase tracking-terminal text-terminal-black hover:text-terminal-accent transition-colors mb-6"
          >
            <ArrowLeft className="w-4 h-4" />
            RETURN TO REGISTRY
          </Link>

          <div className="bg-terminal-panel border-2 border-terminal-black p-6">
            <div className="flex items-start justify-between">
              <div className="flex items-start gap-4">
                {/* Icon */}
                <div className="w-12 h-12 bg-terminal-black flex items-center justify-center">
                  <GitBranch className="w-6 h-6 text-terminal-mint" />
                </div>

                <div>
                  <h1 className="heading-terminal text-2xl md:text-3xl text-terminal-black mb-2">
                    DECISION_TREE
                  </h1>
                  <p className="font-mono text-xs text-terminal-black/70 max-w-2xl leading-relaxed">
                    TREE-BASED CLASSIFICATION // HIERARCHICAL DECISION RULES //
                    INTERPRETABLE FEATURE-BASED PREDICTIONS
                  </p>
                </div>
              </div>

              <div className="hidden sm:flex items-center gap-2">
                <span className="text-xs font-mono font-bold px-2 py-1 border-2 border-terminal-accent text-terminal-accent uppercase tracking-terminal">
                  LVL-1
                </span>
                <span className="text-xs font-mono font-bold px-2 py-1 border-2 border-terminal-black text-terminal-black uppercase tracking-terminal">
                  CLASSIFICATION
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="border-b-2 border-terminal-black mb-8">
          <nav className="flex">
            <button
              onClick={() => setActiveTab("explainer")}
              className={clsx(
                "flex items-center gap-2 px-6 py-4 font-mono font-bold text-xs uppercase tracking-terminal border-b-3 transition-colors",
                activeTab === "explainer"
                  ? "border-terminal-black text-terminal-black bg-terminal-panel"
                  : "border-transparent text-terminal-black/50 hover:text-terminal-black hover:bg-terminal-panel/50"
              )}
            >
              <BookOpen className="w-4 h-4" />
              THEORY
            </button>
            <button
              onClick={() => setActiveTab("tryit")}
              className={clsx(
                "flex items-center gap-2 px-6 py-4 font-mono font-bold text-xs uppercase tracking-terminal border-b-3 transition-colors",
                activeTab === "tryit"
                  ? "border-terminal-black text-terminal-black bg-terminal-panel"
                  : "border-transparent text-terminal-black/50 hover:text-terminal-black hover:bg-terminal-panel/50"
              )}
            >
              <FlaskConical className="w-4 h-4" />
              EXECUTE MODEL
            </button>
          </nav>
        </div>

        {/* Tab Content */}
        <div className="pb-12">
          {activeTab === "explainer" && <DecisionTreeExplainerTab />}
          {activeTab === "tryit" && <DecisionTreeTryItTab />}
        </div>
      </div>
    </div>
  );
}
