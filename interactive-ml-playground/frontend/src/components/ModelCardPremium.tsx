"use client";

import { useState, useRef } from "react";
import Link from "next/link";
import { motion, useMotionValue, useSpring, useTransform } from "framer-motion";
import { ArrowRight, Lock, TrendingUp, GitBranch, Target, Layers } from "lucide-react";

interface ModelCardProps {
  id: string;
  name: string;
  description: string;
  category: string;
  difficulty: string;
  tags: string[];
  comingSoon?: boolean;
  icon: "linear" | "logistic" | "knn" | "kmeans";
  color: string;
  index: number;
}

const iconMap = {
  linear: TrendingUp,
  logistic: GitBranch,
  knn: Target,
  kmeans: Layers,
};

// Mini animated graph components
function LinearGraph({ color }: { color: string }) {
  return (
    <svg viewBox="0 0 100 60" className="w-full h-full">
      {/* Grid lines */}
      <g stroke="rgba(255,255,255,0.1)" strokeWidth="0.5">
        {[15, 30, 45].map((y) => (
          <line key={y} x1="10" y1={y} x2="90" y2={y} />
        ))}
        {[25, 50, 75].map((x) => (
          <line key={x} x1={x} y1="5" x2={x} y2="55" />
        ))}
      </g>
      {/* Data points */}
      <motion.g>
        {[
          [15, 48], [25, 42], [35, 38], [45, 32], [55, 28], [65, 22], [75, 18], [85, 12]
        ].map(([cx, cy], i) => (
          <motion.circle
            key={i}
            cx={cx}
            cy={cy}
            r="3"
            fill={color}
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: i * 0.1, duration: 0.3 }}
          />
        ))}
      </motion.g>
      {/* Trend line */}
      <motion.line
        x1="10" y1="52" x2="90" y2="8"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 1, delay: 0.5 }}
      />
    </svg>
  );
}

function LogisticGraph({ color }: { color: string }) {
  return (
    <svg viewBox="0 0 100 60" className="w-full h-full">
      {/* S-curve */}
      <motion.path
        d="M 10 50 Q 30 50, 50 30 Q 70 10, 90 10"
        fill="none"
        stroke={color}
        strokeWidth="2.5"
        strokeLinecap="round"
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 1.2 }}
      />
      {/* Classification line */}
      <motion.line
        x1="50" y1="5" x2="50" y2="55"
        stroke="rgba(255,255,255,0.3)"
        strokeWidth="1"
        strokeDasharray="4 2"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
      />
      {/* Points */}
      {[[20, 48], [30, 45], [40, 38], [60, 18], [70, 12], [80, 10]].map(([cx, cy], i) => (
        <motion.circle
          key={i}
          cx={cx}
          cy={cy}
          r="3"
          fill={i < 3 ? "#ef4444" : "#22c55e"}
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.3 + i * 0.1 }}
        />
      ))}
    </svg>
  );
}

function KNNGraph({ color }: { color: string }) {
  const points = [
    { x: 25, y: 20, group: 0 }, { x: 35, y: 15, group: 0 }, { x: 30, y: 28, group: 0 },
    { x: 70, y: 45, group: 1 }, { x: 75, y: 38, group: 1 }, { x: 65, y: 50, group: 1 },
    { x: 50, y: 32, group: -1 }, // Query point
  ];

  return (
    <svg viewBox="0 0 100 60" className="w-full h-full">
      {/* Connection lines to query */}
      {points.slice(0, 6).map((p, i) => (
        <motion.line
          key={i}
          x1={50} y1={32}
          x2={p.x} y2={p.y}
          stroke="rgba(255,255,255,0.2)"
          strokeWidth="1"
          strokeDasharray="3 2"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ delay: 0.5 + i * 0.1, duration: 0.3 }}
        />
      ))}
      {/* Group points */}
      {points.slice(0, 6).map((p, i) => (
        <motion.circle
          key={i}
          cx={p.x}
          cy={p.y}
          r="4"
          fill={p.group === 0 ? "#3b82f6" : "#22c55e"}
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: i * 0.08 }}
        />
      ))}
      {/* Query point */}
      <motion.circle
        cx={50} cy={32} r="5"
        fill={color}
        stroke="white"
        strokeWidth="2"
        initial={{ scale: 0 }}
        animate={{ scale: [0, 1.3, 1] }}
        transition={{ delay: 0.8, duration: 0.4 }}
      />
    </svg>
  );
}

function KMeansGraph({ color }: { color: string }) {
  const clusters = [
    { cx: 25, cy: 25, points: [[20, 20], [30, 18], [22, 32], [28, 28]], color: "#3b82f6" },
    { cx: 70, cy: 20, points: [[65, 15], [75, 18], [72, 25], [68, 22]], color: "#22c55e" },
    { cx: 55, cy: 45, points: [[50, 42], [60, 48], [52, 50], [58, 40]], color: "#f59e0b" },
  ];

  return (
    <svg viewBox="0 0 100 60" className="w-full h-full">
      {clusters.map((cluster, ci) => (
        <g key={ci}>
          {/* Cluster circle */}
          <motion.circle
            cx={cluster.cx}
            cy={cluster.cy}
            r="18"
            fill={cluster.color}
            fillOpacity="0.15"
            stroke={cluster.color}
            strokeWidth="1"
            strokeOpacity="0.4"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.5 + ci * 0.2, duration: 0.4 }}
          />
          {/* Points */}
          {cluster.points.map(([x, y], pi) => (
            <motion.circle
              key={pi}
              cx={x}
              cy={y}
              r="3"
              fill={cluster.color}
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: ci * 0.1 + pi * 0.05 }}
            />
          ))}
          {/* Centroid */}
          <motion.circle
            cx={cluster.cx}
            cy={cluster.cy}
            r="4"
            fill={cluster.color}
            stroke="white"
            strokeWidth="1.5"
            initial={{ scale: 0 }}
            animate={{ scale: [0, 1.2, 1] }}
            transition={{ delay: 0.8 + ci * 0.1 }}
          />
        </g>
      ))}
    </svg>
  );
}

const graphComponents = {
  linear: LinearGraph,
  logistic: LogisticGraph,
  knn: KNNGraph,
  kmeans: KMeansGraph,
};

export default function ModelCardPremium({
  id,
  name,
  description,
  category,
  difficulty,
  tags,
  comingSoon = false,
  icon,
  color,
  index,
}: ModelCardProps) {
  const cardRef = useRef<HTMLDivElement>(null);
  const [isHovered, setIsHovered] = useState(false);

  const x = useMotionValue(0);
  const y = useMotionValue(0);

  const mouseXSpring = useSpring(x);
  const mouseYSpring = useSpring(y);

  const rotateX = useTransform(mouseYSpring, [-0.5, 0.5], ["10deg", "-10deg"]);
  const rotateY = useTransform(mouseXSpring, [-0.5, 0.5], ["-10deg", "10deg"]);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const xPct = mouseX / width - 0.5;
    const yPct = mouseY / height - 0.5;
    x.set(xPct);
    y.set(yPct);
  };

  const handleMouseLeave = () => {
    x.set(0);
    y.set(0);
    setIsHovered(false);
  };

  const Icon = iconMap[icon];
  const Graph = graphComponents[icon];

  const content = (
    <motion.div
      ref={cardRef}
      onMouseMove={handleMouseMove}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={handleMouseLeave}
      style={{
        rotateY,
        rotateX,
        transformStyle: "preserve-3d",
      }}
      initial={{ opacity: 0, y: 50 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay: index * 0.1 }}
      className="relative group cursor-pointer"
    >
      {/* Glow effect */}
      <motion.div
        className="absolute -inset-0.5 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 blur-xl"
        style={{
          background: `linear-gradient(135deg, ${color}40, ${color}20, transparent)`,
        }}
      />

      {/* Card */}
      <div
        className="relative bg-slate-900/80 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 overflow-hidden"
        style={{
          transform: "translateZ(0)",
          boxShadow: isHovered
            ? `0 25px 50px -12px ${color}30, 0 0 0 1px ${color}20`
            : "0 4px 6px -1px rgba(0, 0, 0, 0.3)",
        }}
      >
        {/* Top glow line */}
        <motion.div
          className="absolute top-0 left-0 right-0 h-px"
          style={{
            background: `linear-gradient(90deg, transparent, ${color}, transparent)`,
          }}
          initial={{ scaleX: 0, opacity: 0 }}
          animate={isHovered ? { scaleX: 1, opacity: 1 } : { scaleX: 0, opacity: 0 }}
          transition={{ duration: 0.3 }}
        />

        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div
            className="w-12 h-12 rounded-xl flex items-center justify-center"
            style={{
              background: `linear-gradient(135deg, ${color}30, ${color}10)`,
              border: `1px solid ${color}40`,
            }}
          >
            <Icon className="w-6 h-6" style={{ color }} />
          </div>

          <div className="flex items-center gap-2">
            <span
              className="text-xs font-medium px-2.5 py-1 rounded-full"
              style={{
                background: `${color}20`,
                color: color,
              }}
            >
              {difficulty}
            </span>
            {comingSoon && <Lock className="w-4 h-4 text-slate-500" />}
          </div>
        </div>

        {/* Title & Description */}
        <h3 className="text-xl font-bold text-white mb-2">{name}</h3>
        <p className="text-slate-400 text-sm mb-4 line-clamp-2">{description}</p>

        {/* Mini Graph */}
        <div
          className="h-20 mb-4 rounded-lg overflow-hidden"
          style={{
            background: "rgba(0,0,0,0.3)",
            border: "1px solid rgba(255,255,255,0.05)",
          }}
        >
          {isHovered && <Graph color={color} />}
          {!isHovered && (
            <div className="w-full h-full flex items-center justify-center text-slate-600 text-xs">
              Hover to preview
            </div>
          )}
        </div>

        {/* Tags */}
        <div className="flex flex-wrap gap-2 mb-4">
          {tags.map((tag) => (
            <span
              key={tag}
              className="text-xs px-2 py-1 rounded-md bg-slate-800/50 text-slate-400 border border-slate-700/50"
            >
              {tag}
            </span>
          ))}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between pt-4 border-t border-slate-700/50">
          {comingSoon ? (
            <span className="text-sm text-slate-500">Coming Soon</span>
          ) : (
            <>
              <span
                className="text-sm font-medium"
                style={{ color }}
              >
                Explore Model
              </span>
              <motion.div
                animate={isHovered ? { x: 5 } : { x: 0 }}
                transition={{ duration: 0.2 }}
              >
                <ArrowRight className="w-5 h-5" style={{ color }} />
              </motion.div>
            </>
          )}
        </div>
      </div>
    </motion.div>
  );

  if (comingSoon) {
    return <div className="opacity-60">{content}</div>;
  }

  return <Link href={`/models/${id}`}>{content}</Link>;
}
