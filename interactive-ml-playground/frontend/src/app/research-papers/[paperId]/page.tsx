"use client";

import { useState } from "react";
import { useParams, notFound } from "next/navigation";
import { ArrowLeft, FileText, Code, ChevronRight, ExternalLink } from "lucide-react";
import Link from "next/link";
import clsx from "clsx";
import { getPaperById, categoryColors, PaperComponent } from "@/lib/papers-data";

export default function PaperBreakdownPage() {
  const params = useParams();
  const paperId = params.paperId as string;
  const paper = getPaperById(paperId);

  const [selectedIndex, setSelectedIndex] = useState(0);

  if (!paper || !paper.hasBreakdown || !paper.components) {
    notFound();
  }

  const components: PaperComponent[] = paper.components;
  const selectedComponent = components[selectedIndex];

  return (
    <div className="min-h-screen bg-terminal-bg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <Link
            href="/research-papers"
            className="inline-flex items-center gap-2 font-mono text-xs uppercase tracking-terminal text-terminal-black hover:text-terminal-accent transition-colors mb-6"
          >
            <ArrowLeft className="w-4 h-4" />
            BACK TO PAPERS
          </Link>

          <div className="bg-terminal-panel border-2 border-terminal-black p-6">
            <div className="flex items-start justify-between">
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 bg-terminal-black flex items-center justify-center">
                  <Code className="w-6 h-6 text-terminal-mint" />
                </div>
                <div>
                  <div className="flex items-center gap-3 mb-2">
                    <h1 className="heading-terminal text-xl md:text-2xl text-terminal-black">
                      {paper.shortTitle.toUpperCase()}_BREAKDOWN
                    </h1>
                    <span className={clsx(
                      "text-xs font-mono font-bold px-2 py-0.5 border uppercase tracking-terminal",
                      categoryColors[paper.category]
                    )}>
                      {paper.year}
                    </span>
                  </div>
                  <p className="font-mono text-xs text-terminal-black/70 max-w-2xl leading-relaxed">
                    {paper.title}
                  </p>
                  <p className="font-mono text-xs text-terminal-black/50 mt-1">
                    {paper.authors}
                  </p>
                </div>
              </div>
              <a
                href={paper.link}
                target="_blank"
                rel="noopener noreferrer"
                className="hidden sm:flex items-center gap-2 px-3 py-2 font-mono text-xs uppercase tracking-terminal border-2 border-terminal-black text-terminal-black hover:bg-terminal-black hover:text-terminal-mint transition-all"
              >
                <FileText className="w-4 h-4" />
                VIEW PDF
                <ExternalLink className="w-4 h-4" />
              </a>
            </div>
          </div>
        </div>

        {/* Implementation Section */}
        <div className="bg-terminal-panel border-2 border-terminal-black">
          <div className="border-b-2 border-terminal-black p-4 flex items-center gap-3">
            <Code className="w-5 h-5 text-terminal-black" />
            <h2 className="font-mono font-bold text-sm uppercase tracking-terminal text-terminal-black">
              IMPLEMENTATION_BREAKDOWN // {components.length} COMPONENTS
            </h2>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 divide-y-2 lg:divide-y-0 lg:divide-x-2 divide-terminal-black">
            {/* Component List */}
            <div className="p-4">
              <h3 className="font-mono text-xs font-bold uppercase tracking-terminal text-terminal-black/50 mb-4">
                SELECT COMPONENT
              </h3>
              <div className="space-y-2">
                {components.map((component, index) => (
                  <button
                    key={component.name}
                    onClick={() => setSelectedIndex(index)}
                    className={clsx(
                      "w-full flex items-center gap-3 p-3 font-mono text-left transition-all border-2",
                      selectedIndex === index
                        ? "bg-terminal-black text-terminal-mint border-terminal-black"
                        : "bg-transparent text-terminal-black border-terminal-black/20 hover:border-terminal-black"
                    )}
                  >
                    <span className="text-xs opacity-50">{String(index + 1).padStart(2, '0')}</span>
                    <span className="font-bold text-sm uppercase tracking-terminal flex-1">
                      {component.name}
                    </span>
                    <ChevronRight className={clsx(
                      "w-4 h-4 transition-transform",
                      selectedIndex === index ? "rotate-90" : ""
                    )} />
                  </button>
                ))}
              </div>
            </div>

            {/* Code Display */}
            <div className="p-4 bg-terminal-black/5">
              <div className="mb-4">
                <h3 className="font-mono text-sm font-bold uppercase tracking-terminal text-terminal-black mb-2">
                  {selectedComponent.name}
                </h3>
                <p className="font-mono text-xs text-terminal-black/70 leading-relaxed">
                  {selectedComponent.description}
                </p>
              </div>

              <div className="bg-terminal-black p-4 overflow-x-auto">
                <div className="flex items-center gap-2 mb-3 pb-2 border-b border-terminal-mint/30">
                  <div className="w-2 h-2 bg-red-500" />
                  <div className="w-2 h-2 bg-terminal-warning" />
                  <div className="w-2 h-2 bg-terminal-accent" />
                  <span className="ml-2 font-mono text-xs text-terminal-mint/60">
                    {selectedComponent.name.toLowerCase().replace(/ /g, '_')}.py
                  </span>
                </div>
                <pre className="font-mono text-xs text-terminal-mint whitespace-pre-wrap leading-relaxed">
                  {selectedComponent.code}
                </pre>
              </div>

              <div className="mt-4 p-3 border-2 border-dashed border-terminal-black/30">
                <p className="font-mono text-xs text-terminal-black/60">
                  <span className="font-bold text-terminal-accent">TIP:</span> Run this code in a Jupyter notebook
                  or Python environment with PyTorch installed.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <div className="mt-6 flex justify-between items-center">
          <button
            onClick={() => setSelectedIndex(Math.max(0, selectedIndex - 1))}
            disabled={selectedIndex === 0}
            className={clsx(
              "px-4 py-2 font-mono text-xs uppercase tracking-terminal border-2 transition-all",
              selectedIndex === 0
                ? "border-terminal-black/20 text-terminal-black/30 cursor-not-allowed"
                : "border-terminal-black text-terminal-black hover:bg-terminal-black hover:text-terminal-mint"
            )}
          >
            ← PREVIOUS
          </button>

          <span className="font-mono text-xs text-terminal-black/50">
            {selectedIndex + 1} / {components.length}
          </span>

          <button
            onClick={() => setSelectedIndex(Math.min(components.length - 1, selectedIndex + 1))}
            disabled={selectedIndex === components.length - 1}
            className={clsx(
              "px-4 py-2 font-mono text-xs uppercase tracking-terminal border-2 transition-all",
              selectedIndex === components.length - 1
                ? "border-terminal-black/20 text-terminal-black/30 cursor-not-allowed"
                : "border-terminal-black text-terminal-black hover:bg-terminal-black hover:text-terminal-mint"
            )}
          >
            NEXT →
          </button>
        </div>
      </div>
    </div>
  );
}
