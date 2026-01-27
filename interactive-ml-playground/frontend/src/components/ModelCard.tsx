"use client";

import Link from "next/link";
import { ArrowRight, Lock } from "lucide-react";
import clsx from "clsx";

interface ModelCardProps {
  id: string;
  name: string;
  description: string;
  category: string;
  difficulty: string;
  tags: string[];
  comingSoon?: boolean;
}

const difficultyLabels: Record<string, { label: string; class: string }> = {
  Beginner: { label: "LVL-1", class: "border-terminal-accent text-terminal-accent" },
  Intermediate: { label: "LVL-2", class: "border-terminal-warning text-terminal-warning" },
  Advanced: { label: "LVL-3", class: "border-red-600 text-red-600" },
};

export default function ModelCard({
  id,
  name,
  description,
  category,
  difficulty,
  tags,
  comingSoon = false,
}: ModelCardProps) {
  const difficultyInfo = difficultyLabels[difficulty] || { label: "LVL-?", class: "border-terminal-black text-terminal-black" };

  const cardContent = (
    <>
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="font-mono font-bold text-sm uppercase tracking-terminal flex items-center gap-2">
            {name.toUpperCase().replace(/ /g, "_")}
            {comingSoon && <Lock className="w-4 h-4" />}
          </h3>
          <span className="text-xs font-mono uppercase tracking-wide opacity-60">{category}</span>
        </div>
        <span
          className={clsx(
            "text-xs font-mono font-bold px-2 py-1 border-2 uppercase tracking-terminal",
            difficultyInfo.class
          )}
        >
          {difficultyInfo.label}
        </span>
      </div>

      {/* Description */}
      <p className="text-xs font-mono leading-relaxed mb-4 line-clamp-2 opacity-80">{description}</p>

      {/* Tags */}
      <div className="flex flex-wrap gap-2 mb-4">
        {tags.map((tag) => (
          <span
            key={tag}
            className="text-xs font-mono px-2 py-1 border border-terminal-black/30 uppercase tracking-wide"
          >
            {tag}
          </span>
        ))}
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between pt-4 border-t-2 border-current/20">
        {comingSoon ? (
          <span className="text-xs font-mono uppercase tracking-terminal opacity-50">[PENDING]</span>
        ) : (
          <>
            <span className="text-xs font-mono font-bold uppercase tracking-terminal">
              ACCESS MODULE
            </span>
            <ArrowRight className="w-4 h-4" />
          </>
        )}
      </div>
    </>
  );

  const cardClassName = clsx(
    "model-card block bg-terminal-panel border-2 border-terminal-black p-5",
    comingSoon ? "opacity-50 cursor-not-allowed" : "cursor-pointer"
  );

  if (comingSoon) {
    return <div className={cardClassName}>{cardContent}</div>;
  }

  return (
    <Link href={`/models/${id}`} className={cardClassName}>
      {cardContent}
    </Link>
  );
}
