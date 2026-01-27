"use client";

import { useState } from "react";
import { Copy, Check } from "lucide-react";

interface CodeBlockProps {
  code: string;
  language?: string;
}

function highlightPython(code: string): string {
  // Simple syntax highlighting for Python
  const keywords = [
    "import",
    "from",
    "def",
    "class",
    "return",
    "if",
    "else",
    "elif",
    "for",
    "while",
    "in",
    "not",
    "and",
    "or",
    "True",
    "False",
    "None",
    "as",
    "with",
    "try",
    "except",
    "finally",
    "raise",
    "pass",
    "break",
    "continue",
    "lambda",
    "yield",
    "global",
    "nonlocal",
    "assert",
    "del",
    "is",
  ];

  let highlighted = code
    // Escape HTML
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // Highlight strings (handle both single and double quotes)
  highlighted = highlighted.replace(
    /(["'])((?:\\.|(?!\1)[^\\])*?)\1/g,
    '<span class="string">$1$2$1</span>'
  );

  // Highlight comments
  highlighted = highlighted.replace(
    /(#.*)$/gm,
    '<span class="comment">$1</span>'
  );

  // Highlight keywords
  keywords.forEach((keyword) => {
    const regex = new RegExp(`\\b(${keyword})\\b`, "g");
    highlighted = highlighted.replace(
      regex,
      '<span class="keyword">$1</span>'
    );
  });

  // Highlight function calls
  highlighted = highlighted.replace(
    /\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/g,
    '<span class="function">$1</span>('
  );

  // Highlight numbers
  highlighted = highlighted.replace(
    /\b(\d+\.?\d*)\b/g,
    '<span class="number">$1</span>'
  );

  return highlighted;
}

export default function CodeBlock({ code, language = "python" }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative group">
      <button
        onClick={handleCopy}
        className="absolute top-3 right-3 p-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors opacity-0 group-hover:opacity-100"
        aria-label="Copy code"
      >
        {copied ? (
          <Check className="w-4 h-4 text-green-400" />
        ) : (
          <Copy className="w-4 h-4 text-slate-300" />
        )}
      </button>
      <div className="code-block overflow-x-auto">
        <pre>
          <code
            dangerouslySetInnerHTML={{
              __html: language === "python" ? highlightPython(code) : code,
            }}
          />
        </pre>
      </div>
    </div>
  );
}
