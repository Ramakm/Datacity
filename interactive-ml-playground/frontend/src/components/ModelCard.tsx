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

const difficultyColors: Record<string, string> = {
  Beginner: "bg-green-100 text-green-700",
  Intermediate: "bg-yellow-100 text-yellow-700",
  Advanced: "bg-red-100 text-red-700",
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
  const cardContent = (
    <>
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="text-lg font-semibold text-slate-800 flex items-center gap-2">
            {name}
            {comingSoon && <Lock className="w-4 h-4 text-slate-400" />}
          </h3>
          <span className="text-sm text-slate-500">{category}</span>
        </div>
        <span
          className={clsx(
            "text-xs font-medium px-2.5 py-1 rounded-full",
            difficultyColors[difficulty] || "bg-slate-100 text-slate-600"
          )}
        >
          {difficulty}
        </span>
      </div>

      <p className="text-slate-600 text-sm mb-4 line-clamp-2">{description}</p>

      <div className="flex flex-wrap gap-2 mb-4">
        {tags.map((tag) => (
          <span
            key={tag}
            className="text-xs bg-slate-100 text-slate-600 px-2 py-1 rounded"
          >
            {tag}
          </span>
        ))}
      </div>

      <div className="flex items-center justify-between pt-4 border-t border-slate-100">
        {comingSoon ? (
          <span className="text-sm text-slate-400">Coming Soon</span>
        ) : (
          <>
            <span className="text-sm text-primary-600 font-medium">
              Start Learning
            </span>
            <ArrowRight className="w-4 h-4 text-primary-600" />
          </>
        )}
      </div>
    </>
  );

  const cardClassName = clsx(
    "model-card block bg-white rounded-xl border border-slate-200 p-6",
    comingSoon ? "opacity-60 cursor-not-allowed" : "cursor-pointer"
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
