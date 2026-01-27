"use client";

import { useState } from "react";
import { ArrowLeft, BookOpen, FlaskConical, Binary } from "lucide-react";
import Link from "next/link";
import clsx from "clsx";
import LogisticExplainerTab from "@/components/LogisticExplainerTab";
import LogisticTryItTab from "@/components/LogisticTryItTab";

type Tab = "explainer" | "tryit";

export default function LogisticRegressionPage() {
  const [activeTab, setActiveTab] = useState<Tab>("explainer");

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-900 to-slate-950">
      {/* Background effects */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-pink-500/10 rounded-full blur-3xl" />
      </div>

      <div className="relative max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <Link
            href="/"
            className="inline-flex items-center gap-2 text-slate-400 hover:text-purple-400 transition-colors mb-6"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Models
          </Link>

          <div className="flex items-start justify-between">
            <div className="flex items-start gap-4">
              {/* Icon */}
              <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-purple-500/20 to-purple-500/5 border border-purple-500/30 flex items-center justify-center">
                <Binary className="w-7 h-7 text-purple-400" />
              </div>

              <div>
                <h1 className="text-3xl font-bold text-white mb-2">
                  Logistic Regression
                </h1>
                <p className="text-slate-400 max-w-2xl">
                  A classification algorithm that predicts probabilities for
                  binary outcomes using the sigmoid function.
                </p>
              </div>
            </div>

            <div className="hidden sm:flex items-center gap-2">
              <span className="text-xs font-medium px-3 py-1.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                Beginner
              </span>
              <span className="text-xs font-medium px-3 py-1.5 rounded-full bg-slate-700/50 text-slate-300 border border-slate-600/50">
                Classification
              </span>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="border-b border-slate-700/50 mb-8">
          <nav className="flex gap-8">
            <button
              onClick={() => setActiveTab("explainer")}
              className={clsx(
                "flex items-center gap-2 py-4 border-b-2 font-medium transition-colors",
                activeTab === "explainer"
                  ? "border-purple-500 text-purple-400"
                  : "border-transparent text-slate-400 hover:text-white hover:border-slate-600"
              )}
            >
              <BookOpen className="w-5 h-5" />
              Explainer
            </button>
            <button
              onClick={() => setActiveTab("tryit")}
              className={clsx(
                "flex items-center gap-2 py-4 border-b-2 font-medium transition-colors",
                activeTab === "tryit"
                  ? "border-purple-500 text-purple-400"
                  : "border-transparent text-slate-400 hover:text-white hover:border-slate-600"
              )}
            >
              <FlaskConical className="w-5 h-5" />
              Try It With Data
            </button>
          </nav>
        </div>

        {/* Tab Content */}
        <div className="pb-12">
          {activeTab === "explainer" && <LogisticExplainerTab />}
          {activeTab === "tryit" && <LogisticTryItTab />}
        </div>
      </div>
    </div>
  );
}
