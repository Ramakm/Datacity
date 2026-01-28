"use client";

import { useState } from "react";
import { ArrowLeft, Code, Play, RotateCcw, Lightbulb, Check, ChevronDown, ChevronUp } from "lucide-react";
import Link from "next/link";
import clsx from "clsx";
import { questions, categoryLabels, difficultyColors, getQuestionsByCategory, Question } from "@/lib/practice-data";

export default function PracticePage() {
  const [filterCategory, setFilterCategory] = useState<string>("all");
  const [selectedQuestion, setSelectedQuestion] = useState<Question>(questions[0]);
  const [code, setCode] = useState<string>(questions[0].starterCode);
  const [output, setOutput] = useState<string>("");
  const [showHints, setShowHints] = useState<boolean>(false);
  const [showSolution, setShowSolution] = useState<boolean>(false);
  const [currentHintIndex, setCurrentHintIndex] = useState<number>(0);

  const filteredQuestions = getQuestionsByCategory(filterCategory);
  const categories = ["all", "math", "stats", "ml"];

  const handleQuestionSelect = (question: Question) => {
    setSelectedQuestion(question);
    setCode(question.starterCode);
    setOutput("");
    setShowHints(false);
    setShowSolution(false);
    setCurrentHintIndex(0);
  };

  const handleReset = () => {
    setCode(selectedQuestion.starterCode);
    setOutput("");
  };

  const handleRunCode = () => {
    setOutput("// Code execution is simulated in this demo\n// In production, this would connect to a Python backend\n\n" +
      "To test your code:\n" +
      "1. Copy your code to a Python environment\n" +
      "2. Run the test cases below\n\n" +
      "--- TEST CASES ---\n" + selectedQuestion.testCases);
  };

  const handleShowSolution = () => {
    setShowSolution(!showSolution);
    if (!showSolution) {
      setCode(selectedQuestion.solution);
    } else {
      setCode(selectedQuestion.starterCode);
    }
  };

  const handleNextHint = () => {
    if (currentHintIndex < selectedQuestion.hints.length - 1) {
      setCurrentHintIndex(currentHintIndex + 1);
    }
  };

  return (
    <div className="min-h-screen bg-terminal-bg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-6">
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
                  <Code className="w-6 h-6 text-terminal-mint" />
                </div>
                <div>
                  <h1 className="heading-terminal text-2xl md:text-3xl text-terminal-black mb-2">
                    PRACTICE_TERMINAL
                  </h1>
                  <p className="font-mono text-xs text-terminal-black/70 max-w-2xl leading-relaxed">
                    CODING CHALLENGES FOR MATH, STATISTICS & ML //
                    BUILD YOUR ALGORITHMIC FOUNDATIONS
                  </p>
                </div>
              </div>
              <div className="hidden sm:flex items-center gap-2">
                <span className="text-xs font-mono font-bold px-2 py-1 border-2 border-terminal-accent text-terminal-accent uppercase tracking-terminal">
                  {questions.length} PROBLEMS
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Category Filter */}
        <div className="mb-4 flex flex-wrap gap-2">
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

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Questions List */}
          <div className="lg:col-span-1 bg-terminal-panel border-2 border-terminal-black">
            <div className="border-b-2 border-terminal-black p-3">
              <h2 className="font-mono font-bold text-xs uppercase tracking-terminal text-terminal-black">
                PROBLEMS // {filteredQuestions.length}
              </h2>
            </div>
            <div className="max-h-[600px] overflow-y-auto">
              {filteredQuestions.map((question) => (
                <button
                  key={question.id}
                  onClick={() => handleQuestionSelect(question)}
                  className={clsx(
                    "w-full p-3 text-left border-b border-terminal-black/20 transition-colors",
                    selectedQuestion.id === question.id
                      ? "bg-terminal-black text-terminal-mint"
                      : "hover:bg-terminal-black/5"
                  )}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <h3 className="font-mono text-xs font-bold truncate">
                        {question.title}
                      </h3>
                      <div className="flex items-center gap-2 mt-1">
                        <span className={clsx(
                          "text-[10px] font-mono font-bold px-1.5 py-0.5 border uppercase",
                          selectedQuestion.id === question.id
                            ? "border-terminal-mint/50 text-terminal-mint/80"
                            : difficultyColors[question.difficulty]
                        )}>
                          {question.difficulty}
                        </span>
                        <span className={clsx(
                          "text-[10px] font-mono uppercase",
                          selectedQuestion.id === question.id
                            ? "text-terminal-mint/60"
                            : "text-terminal-black/50"
                        )}>
                          {question.category}
                        </span>
                      </div>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Code Editor Section */}
          <div className="lg:col-span-2 flex flex-col gap-4">
            {/* Question Description */}
            <div className="bg-terminal-panel border-2 border-terminal-black">
              <div className="border-b-2 border-terminal-black p-3 flex items-center justify-between">
                <h2 className="font-mono font-bold text-sm uppercase tracking-terminal text-terminal-black">
                  {selectedQuestion.title}
                </h2>
                <span className={clsx(
                  "text-xs font-mono font-bold px-2 py-0.5 border uppercase tracking-terminal",
                  difficultyColors[selectedQuestion.difficulty]
                )}>
                  {selectedQuestion.difficulty}
                </span>
              </div>
              <div className="p-4">
                <pre className="font-mono text-xs text-terminal-black/80 whitespace-pre-wrap leading-relaxed">
                  {selectedQuestion.description}
                </pre>
              </div>

              {/* Hints Section */}
              <div className="border-t-2 border-terminal-black/20 p-3">
                <button
                  onClick={() => setShowHints(!showHints)}
                  className="flex items-center gap-2 font-mono text-xs uppercase tracking-terminal text-terminal-accent hover:text-terminal-black transition-colors"
                >
                  <Lightbulb className="w-4 h-4" />
                  {showHints ? "HIDE HINTS" : "SHOW HINTS"}
                  {showHints ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>
                {showHints && (
                  <div className="mt-3 space-y-2">
                    {selectedQuestion.hints.slice(0, currentHintIndex + 1).map((hint, index) => (
                      <div key={index} className="flex items-start gap-2 p-2 bg-terminal-accent/10 border border-terminal-accent/30">
                        <span className="font-mono text-xs font-bold text-terminal-accent">
                          {index + 1}.
                        </span>
                        <span className="font-mono text-xs text-terminal-black/70">
                          {hint}
                        </span>
                      </div>
                    ))}
                    {currentHintIndex < selectedQuestion.hints.length - 1 && (
                      <button
                        onClick={handleNextHint}
                        className="font-mono text-xs text-terminal-accent hover:underline"
                      >
                        Show next hint →
                      </button>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Code Editor */}
            <div className="bg-terminal-black border-2 border-terminal-black flex-1">
              <div className="flex items-center justify-between p-2 border-b border-terminal-mint/30">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-red-500" />
                  <div className="w-2 h-2 bg-terminal-warning" />
                  <div className="w-2 h-2 bg-terminal-accent" />
                  <span className="ml-2 font-mono text-xs text-terminal-mint/60">
                    solution.py
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleReset}
                    className="flex items-center gap-1 px-2 py-1 font-mono text-xs text-terminal-mint/70 hover:text-terminal-mint transition-colors"
                  >
                    <RotateCcw className="w-3 h-3" />
                    RESET
                  </button>
                  <button
                    onClick={handleShowSolution}
                    className={clsx(
                      "flex items-center gap-1 px-2 py-1 font-mono text-xs transition-colors",
                      showSolution
                        ? "text-terminal-warning"
                        : "text-terminal-mint/70 hover:text-terminal-mint"
                    )}
                  >
                    <Check className="w-3 h-3" />
                    {showSolution ? "HIDE SOLUTION" : "SHOW SOLUTION"}
                  </button>
                  <button
                    onClick={handleRunCode}
                    className="flex items-center gap-1 px-3 py-1 bg-terminal-accent text-terminal-black font-mono text-xs font-bold uppercase"
                  >
                    <Play className="w-3 h-3" />
                    RUN
                  </button>
                </div>
              </div>
              <textarea
                value={code}
                onChange={(e) => setCode(e.target.value)}
                className="w-full h-64 p-4 bg-transparent text-terminal-mint font-mono text-xs resize-none focus:outline-none"
                spellCheck={false}
              />
            </div>

            {/* Output */}
            {output && (
              <div className="bg-terminal-panel border-2 border-terminal-black">
                <div className="border-b-2 border-terminal-black p-2">
                  <span className="font-mono text-xs font-bold uppercase tracking-terminal text-terminal-black">
                    OUTPUT
                  </span>
                </div>
                <pre className="p-4 font-mono text-xs text-terminal-black/80 whitespace-pre-wrap max-h-48 overflow-y-auto">
                  {output}
                </pre>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
