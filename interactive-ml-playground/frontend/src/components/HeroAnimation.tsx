"use client";

import { useEffect, useRef } from "react";

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  color: string;
  type: "data" | "neuron" | "gradient";
  angle?: number;
  orbitRadius?: number;
  orbitSpeed?: number;
  centerX?: number;
  centerY?: number;
}

interface Connection {
  from: number;
  to: number;
  weight: number;
  pulse: number;
}

export default function HeroAnimation() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);
  const particlesRef = useRef<Particle[]>([]);
  const connectionsRef = useRef<Connection[]>([]);
  const timeRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size
    const resizeCanvas = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    };
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    const width = canvas.getBoundingClientRect().width;
    const height = canvas.getBoundingClientRect().height;

    // Initialize neural network layers
    const layers = [3, 5, 4, 2]; // Input, hidden1, hidden2, output
    const layerSpacing = width / (layers.length + 1);
    const particles: Particle[] = [];
    const connections: Connection[] = [];

    // Create neurons for each layer
    let neuronIndex = 0;
    const neuronPositions: { x: number; y: number; layer: number }[] = [];

    layers.forEach((count, layerIdx) => {
      const x = layerSpacing * (layerIdx + 1);
      const verticalSpacing = height / (count + 1);

      for (let i = 0; i < count; i++) {
        const y = verticalSpacing * (i + 1);
        neuronPositions.push({ x, y, layer: layerIdx });

        particles.push({
          x,
          y,
          vx: 0,
          vy: 0,
          radius: layerIdx === 0 || layerIdx === layers.length - 1 ? 8 : 6,
          color: layerIdx === 0 ? "#1a5c3a" : layerIdx === layers.length - 1 ? "#c4a000" : "#0a0a0a",
          type: "neuron",
        });
        neuronIndex++;
      }
    });

    // Create connections between layers
    let startIdx = 0;
    for (let l = 0; l < layers.length - 1; l++) {
      const currentLayerStart = startIdx;
      const nextLayerStart = startIdx + layers[l];

      for (let i = 0; i < layers[l]; i++) {
        for (let j = 0; j < layers[l + 1]; j++) {
          connections.push({
            from: currentLayerStart + i,
            to: nextLayerStart + j,
            weight: Math.random() * 0.8 + 0.2,
            pulse: Math.random() * Math.PI * 2,
          });
        }
      }
      startIdx += layers[l];
    }

    // Add orbiting data particles
    const orbitCenters = [
      { x: width * 0.15, y: height * 0.3 },
      { x: width * 0.85, y: height * 0.7 },
    ];

    for (let i = 0; i < 8; i++) {
      const center = orbitCenters[i % 2];
      particles.push({
        x: center.x,
        y: center.y,
        vx: 0,
        vy: 0,
        radius: 3,
        color: "#1a5c3a",
        type: "data",
        angle: (Math.PI * 2 * i) / 4,
        orbitRadius: 30 + Math.random() * 20,
        orbitSpeed: 0.02 + Math.random() * 0.01,
        centerX: center.x,
        centerY: center.y,
      });
    }

    // Add gradient descent particles
    for (let i = 0; i < 5; i++) {
      particles.push({
        x: width * 0.1 + Math.random() * width * 0.2,
        y: height * 0.8,
        vx: 1 + Math.random(),
        vy: -0.5 - Math.random() * 0.5,
        radius: 2,
        color: "#c4a000",
        type: "gradient",
      });
    }

    particlesRef.current = particles;
    connectionsRef.current = connections;

    // Animation loop
    const animate = () => {
      const w = canvas.getBoundingClientRect().width;
      const h = canvas.getBoundingClientRect().height;

      ctx.clearRect(0, 0, w, h);
      timeRef.current += 0.016;

      // Draw grid pattern (subtle)
      ctx.strokeStyle = "rgba(10, 10, 10, 0.03)";
      ctx.lineWidth = 1;
      const gridSize = 30;
      for (let x = 0; x < w; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }
      for (let y = 0; y < h; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      // Draw connections with pulse effect
      connectionsRef.current.forEach((conn) => {
        const fromP = particlesRef.current[conn.from];
        const toP = particlesRef.current[conn.to];

        const pulseIntensity = (Math.sin(timeRef.current * 3 + conn.pulse) + 1) / 2;
        const alpha = 0.1 + pulseIntensity * 0.15 * conn.weight;

        ctx.strokeStyle = `rgba(26, 92, 58, ${alpha})`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(fromP.x, fromP.y);
        ctx.lineTo(toP.x, toP.y);
        ctx.stroke();

        // Draw pulse traveling along connection
        if (pulseIntensity > 0.7) {
          const progress = (timeRef.current * 2 + conn.pulse) % 1;
          const px = fromP.x + (toP.x - fromP.x) * progress;
          const py = fromP.y + (toP.y - fromP.y) * progress;

          ctx.fillStyle = "rgba(26, 92, 58, 0.6)";
          ctx.beginPath();
          ctx.arc(px, py, 2, 0, Math.PI * 2);
          ctx.fill();
        }
      });

      // Update and draw particles
      particlesRef.current.forEach((p, idx) => {
        // Update based on type
        if (p.type === "data" && p.angle !== undefined && p.orbitRadius && p.orbitSpeed && p.centerX && p.centerY) {
          p.angle += p.orbitSpeed;
          p.x = p.centerX + Math.cos(p.angle) * p.orbitRadius;
          p.y = p.centerY + Math.sin(p.angle) * p.orbitRadius;
        } else if (p.type === "gradient") {
          // Gradient descent - oscillating path toward minimum
          p.x += p.vx;
          p.y += Math.sin(timeRef.current * 2 + idx) * 0.5;

          // Reset when reaching the network
          if (p.x > w * 0.3) {
            p.x = w * 0.05;
            p.y = h * 0.5 + Math.random() * h * 0.3;
          }
        } else if (p.type === "neuron") {
          // Subtle breathing effect for neurons
          const breathe = Math.sin(timeRef.current * 2 + idx * 0.5) * 0.5;
          p.radius = (idx < 3 || idx >= particlesRef.current.length - 5 ? 8 : 6) + breathe;
        }

        // Draw particle
        ctx.fillStyle = p.color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fill();

        // Add glow effect for neurons
        if (p.type === "neuron") {
          const gradient = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.radius * 2);
          gradient.addColorStop(0, `${p.color}33`);
          gradient.addColorStop(1, "transparent");
          ctx.fillStyle = gradient;
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.radius * 2, 0, Math.PI * 2);
          ctx.fill();
        }
      });

      // Draw axis labels
      ctx.fillStyle = "rgba(10, 10, 10, 0.3)";
      ctx.font = "10px monospace";
      ctx.textAlign = "center";

      // Layer labels
      const layerLabels = ["INPUT", "HIDDEN_1", "HIDDEN_2", "OUTPUT"];
      layers.forEach((_, idx) => {
        const x = layerSpacing * (idx + 1);
        ctx.fillText(layerLabels[idx], x, h - 10);
      });

      // Draw formulas
      ctx.fillStyle = "rgba(10, 10, 10, 0.15)";
      ctx.font = "italic 12px monospace";
      ctx.fillText("y = f(Wx + b)", w * 0.5, 25);
      ctx.fillText("L = -log(p)", w * 0.2, h - 30);
      ctx.fillText("w = w - η∇L", w * 0.8, h - 30);

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener("resize", resizeCanvas);
      cancelAnimationFrame(animationRef.current);
    };
  }, []);

  return (
    <div className="relative w-full h-full min-h-[400px]">
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ display: "block" }}
      />
      {/* Overlay labels */}
      <div className="absolute top-4 right-4 font-mono text-xs text-terminal-black/30 uppercase tracking-terminal">
        NEURAL_NETWORK_VIZ
      </div>
      <div className="absolute bottom-4 left-4 font-mono text-xs text-terminal-accent/50">
        <span className="animate-pulse">●</span> LIVE SIMULATION
      </div>
    </div>
  );
}
