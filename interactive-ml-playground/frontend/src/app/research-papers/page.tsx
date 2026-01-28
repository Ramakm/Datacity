"use client";

import { useState } from "react";
import { ArrowLeft, FileText, ExternalLink, BookOpen, Code } from "lucide-react";
import Link from "next/link";
import clsx from "clsx";
import { papers, categoryLabels, categoryColors } from "@/lib/papers-data";

export default function ResearchPapersPage() {
  const [filterCategory, setFilterCategory] = useState<string>("all");

  const filteredPapers = filterCategory === "all"
    ? papers
    : papers.filter(p => p.category === filterCategory);

  const categories = ["all", "foundational", "embeddings", "rnn", "transformer", "llm"];

  return (
    <div className="min-h-screen bg-terminal-bg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <Link
            href="/"
            className="inline-flex items-center gap-2 font-mono text-xs uppercase tracking-terminal text-terminal-black hover:text-terminal-accent transition-colors mb-6"
          >
            <ArrowLeft className="w-4 h-4" />
            RETURN TO TERMINAL
          </Link>

          <div className="bg-terminal-panel border-2 border-terminal-black p-6">
            <div className="flex items-start justify-between">
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 bg-terminal-black flex items-center justify-center">
                  <FileText className="w-6 h-6 text-terminal-mint" />
                </div>
                <div>
                  <h1 className="heading-terminal text-2xl md:text-3xl text-terminal-black mb-2">
                    RESEARCH_PAPERS
                  </h1>
                  <p className="font-mono text-xs text-terminal-black/70 max-w-2xl leading-relaxed">
                    CURATED COLLECTION OF FOUNDATIONAL ML/NLP PAPERS //
                    FROM INFORMATION THEORY TO LARGE LANGUAGE MODELS
                  </p>
                </div>
              </div>
              <div className="hidden sm:flex items-center gap-2">
                <span className="text-xs font-mono font-bold px-2 py-1 border-2 border-terminal-accent text-terminal-accent uppercase tracking-terminal">
                  {papers.length} PAPERS
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Category Filter */}
        <div className="mb-6 flex flex-wrap gap-2">
          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setFilterCategory(cat)}
              className={clsx(
                "px-3 py-1 font-mono text-xs uppercase tracking-terminal border-2 transition-all",
                filterCategory === cat
                  ? "bg-terminal-black text-terminal-mint border-terminal-black"
                  : "bg-transparent text-terminal-black border-terminal-black/30 hover:border-terminal-black"
              )}
            >
              {cat === "all" ? "ALL" : categoryLabels[cat]}
            </button>
          ))}
        </div>

        {/* Papers List */}
        <div className="bg-terminal-panel border-2 border-terminal-black mb-8">
          <div className="border-b-2 border-terminal-black p-4 flex items-center gap-3">
            <BookOpen className="w-5 h-5 text-terminal-black" />
            <h2 className="font-mono font-bold text-sm uppercase tracking-terminal text-terminal-black">
              PAPER_REGISTRY // {filteredPapers.length} ENTRIES
            </h2>
          </div>

          <div className="divide-y-2 divide-terminal-black/20">
            {filteredPapers.map((paper) => (
              <div
                key={paper.id}
                className="p-4 hover:bg-terminal-black/5 transition-colors"
              >
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 w-12 text-center">
                    <span className="font-mono text-xs text-terminal-black/50">{paper.year}</span>
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between gap-4 mb-2">
                      <div>
                        <h3 className="font-mono text-sm font-bold text-terminal-black">
                          {paper.title}
                        </h3>
                        <p className="font-mono text-xs text-terminal-black/60 mt-1">
                          {paper.authors}
                        </p>
                        <p className="font-mono text-xs text-terminal-black/50 mt-1">
                          {paper.description}
                        </p>
                      </div>
                      <span className={clsx(
                        "text-xs font-mono font-bold px-2 py-0.5 border uppercase tracking-terminal flex-shrink-0",
                        categoryColors[paper.category]
                      )}>
                        {paper.shortTitle}
                      </span>
                    </div>

                    {/* Action Links */}
                    <div className="flex items-center gap-3 mt-3">
                      <a
                        href={paper.link}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-2 px-3 py-1.5 font-mono text-xs uppercase tracking-terminal border-2 border-terminal-black text-terminal-black hover:bg-terminal-black hover:text-terminal-mint transition-all"
                      >
                        <FileText className="w-3 h-3" />
                        VIEW PDF
                        <ExternalLink className="w-3 h-3" />
                      </a>

                      {paper.hasBreakdown && (
                        <Link
                          href={`/research-papers/${paper.id}`}
                          className="inline-flex items-center gap-2 px-3 py-1.5 font-mono text-xs uppercase tracking-terminal border-2 border-terminal-accent text-terminal-accent hover:bg-terminal-accent hover:text-terminal-black transition-all"
                        >
                          <Code className="w-3 h-3" />
                          IMPLEMENTATION
                        </Link>
                      )}

                      {!paper.hasBreakdown && (
                        <span className="inline-flex items-center gap-2 px-3 py-1.5 font-mono text-xs uppercase tracking-terminal border-2 border-terminal-black/20 text-terminal-black/40">
                          <Code className="w-3 h-3" />
                          COMING SOON
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Reading Order */}
        <div className="bg-terminal-black text-terminal-mint p-6 border-2 border-terminal-black">
          <div className="flex items-center gap-2 mb-4 pb-3 border-b border-terminal-mint/30">
            <div className="w-3 h-3 bg-red-500" />
            <div className="w-3 h-3 bg-terminal-warning" />
            <div className="w-3 h-3 bg-terminal-accent" />
            <span className="ml-4 font-mono text-xs opacity-60">RECOMMENDED_READING_ORDER.md</span>
          </div>

          <div className="font-mono text-sm space-y-2">
            <p><span className="text-terminal-accent">$</span> cat ./reading_path.txt</p>
            <p className="opacity-70 mt-2">SUGGESTED PROGRESSION:</p>
            <p className="opacity-70">1. Information Theory (Shannon) → Neural LM (Bengio)</p>
            <p className="opacity-70">2. Word2Vec → GloVe → FastText</p>
            <p className="opacity-70">3. LSTM → ELMo → Attention</p>
            <p className="opacity-70">4. Transformer → BERT → GPT Series</p>
            <p className="opacity-70">5. RLHF → Modern LLMs</p>
            <p className="mt-3"><span className="text-terminal-accent">$</span> _<span className="animate-pulse">|</span></p>
          </div>
        </div>
      </div>
    </div>
  );
}
