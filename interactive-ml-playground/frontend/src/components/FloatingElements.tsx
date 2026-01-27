"use client";

import { motion } from "framer-motion";

const floatingData = [
  { x: "10%", y: "20%", delay: 0, symbol: "y = mx + b", color: "#00f2fe" },
  { x: "85%", y: "15%", delay: 0.5, symbol: "R² = 0.94", color: "#a855f7" },
  { x: "75%", y: "70%", delay: 1, symbol: "loss↓", color: "#22c55e" },
  { x: "15%", y: "75%", delay: 1.5, symbol: "∇f(x)", color: "#f59e0b" },
  { x: "50%", y: "85%", delay: 2, symbol: "σ(z)", color: "#ec4899" },
  { x: "90%", y: "45%", delay: 0.8, symbol: "Σwᵢxᵢ", color: "#06b6d4" },
  { x: "5%", y: "50%", delay: 1.2, symbol: "μ, σ²", color: "#8b5cf6" },
];

export function FloatingSymbols() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {floatingData.map((item, i) => (
        <motion.div
          key={i}
          className="absolute font-mono text-sm font-medium opacity-40"
          style={{
            left: item.x,
            top: item.y,
            color: item.color,
            textShadow: `0 0 20px ${item.color}`,
          }}
          initial={{ opacity: 0, y: 20 }}
          animate={{
            opacity: [0.2, 0.5, 0.2],
            y: [0, -15, 0],
          }}
          transition={{
            delay: item.delay,
            duration: 4,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        >
          {item.symbol}
        </motion.div>
      ))}
    </div>
  );
}

const dataPoints = Array.from({ length: 20 }, (_, i) => ({
  x: Math.random() * 100,
  y: Math.random() * 100,
  size: Math.random() * 4 + 2,
  delay: Math.random() * 2,
  duration: 3 + Math.random() * 2,
  color: ["#00f2fe", "#4facfe", "#a855f7", "#22c55e", "#f59e0b"][Math.floor(Math.random() * 5)],
}));

export function FloatingDataPoints() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {dataPoints.map((point, i) => (
        <motion.div
          key={i}
          className="absolute rounded-full"
          style={{
            left: `${point.x}%`,
            top: `${point.y}%`,
            width: point.size,
            height: point.size,
            backgroundColor: point.color,
            boxShadow: `0 0 ${point.size * 3}px ${point.color}`,
          }}
          animate={{
            y: [0, -30, 0],
            opacity: [0.3, 0.8, 0.3],
            scale: [1, 1.2, 1],
          }}
          transition={{
            delay: point.delay,
            duration: point.duration,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
      ))}
    </div>
  );
}

export function GlowOrbs() {
  return (
    <>
      <motion.div
        className="absolute w-[600px] h-[600px] rounded-full pointer-events-none"
        style={{
          left: "10%",
          top: "20%",
          background: "radial-gradient(circle, rgba(59, 130, 246, 0.15) 0%, transparent 70%)",
          filter: "blur(40px)",
        }}
        animate={{
          scale: [1, 1.2, 1],
          opacity: [0.3, 0.5, 0.3],
        }}
        transition={{
          duration: 8,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />
      <motion.div
        className="absolute w-[500px] h-[500px] rounded-full pointer-events-none"
        style={{
          right: "5%",
          top: "40%",
          background: "radial-gradient(circle, rgba(168, 85, 247, 0.15) 0%, transparent 70%)",
          filter: "blur(40px)",
        }}
        animate={{
          scale: [1.2, 1, 1.2],
          opacity: [0.4, 0.2, 0.4],
        }}
        transition={{
          duration: 10,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />
      <motion.div
        className="absolute w-[400px] h-[400px] rounded-full pointer-events-none"
        style={{
          left: "50%",
          bottom: "10%",
          transform: "translateX(-50%)",
          background: "radial-gradient(circle, rgba(6, 182, 212, 0.12) 0%, transparent 70%)",
          filter: "blur(40px)",
        }}
        animate={{
          scale: [1, 1.3, 1],
          opacity: [0.3, 0.5, 0.3],
        }}
        transition={{
          duration: 6,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />
    </>
  );
}
