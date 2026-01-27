"use client";

import { useRef } from "react";
import { motion, useScroll, useTransform, useInView } from "framer-motion";
import { Play, ChevronDown, Sparkles, Database, Brain, MousePointer, Zap } from "lucide-react";
import AnimatedBackground from "@/components/AnimatedBackground";
import { FloatingSymbols, FloatingDataPoints, GlowOrbs } from "@/components/FloatingElements";
import ModelCardPremium from "@/components/ModelCardPremium";

const models = [
  {
    id: "linear-regression",
    name: "Linear Regression",
    description: "Find the best-fit line through your data. The foundation of predictive modeling.",
    category: "Regression",
    difficulty: "Beginner",
    tags: ["supervised", "regression", "interpretable"],
    icon: "linear" as const,
    color: "#00f2fe",
  },
  {
    id: "logistic-regression",
    name: "Logistic Regression",
    description: "Classify data into categories. The gateway to classification algorithms.",
    category: "Classification",
    difficulty: "Beginner",
    tags: ["supervised", "classification", "probabilities"],
    icon: "logistic" as const,
    color: "#a855f7",
  },
  {
    id: "knn",
    name: "K-Nearest Neighbors",
    description: "Predict based on similarity. Let your data vote on the outcome.",
    category: "Classification",
    difficulty: "Beginner",
    tags: ["supervised", "instance-based", "lazy-learning"],
    icon: "knn" as const,
    color: "#22c55e",
  },
  {
    id: "kmeans",
    name: "K-Means Clustering",
    description: "Discover hidden patterns. Group similar data points automatically.",
    category: "Clustering",
    difficulty: "Intermediate",
    tags: ["unsupervised", "clustering", "partitioning"],
    icon: "kmeans" as const,
    color: "#f59e0b",
  },
];

const steps = [
  {
    number: "01",
    title: "Pick a Model",
    description: "Choose from our collection of ML algorithms. Each one tells a different story.",
    icon: MousePointer,
    color: "#00f2fe",
  },
  {
    number: "02",
    title: "Play with Data",
    description: "Upload your own CSV or use sample data. Watch the algorithm learn in real-time.",
    icon: Database,
    color: "#a855f7",
  },
  {
    number: "03",
    title: "See Intelligence Emerge",
    description: "Visualize predictions, understand coefficients, and export working Python code.",
    icon: Brain,
    color: "#22c55e",
  },
];

function HeroSection() {
  const ref = useRef(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start start", "end start"],
  });

  const opacity = useTransform(scrollYProgress, [0, 0.5], [1, 0]);
  const scale = useTransform(scrollYProgress, [0, 0.5], [1, 0.8]);
  const y = useTransform(scrollYProgress, [0, 0.5], [0, -100]);

  return (
    <section ref={ref} className="relative min-h-screen flex items-center justify-center overflow-hidden">
      <FloatingSymbols />
      <FloatingDataPoints />
      <GlowOrbs />

      <motion.div
        style={{ opacity, scale, y }}
        className="relative z-10 text-center px-4 max-w-5xl mx-auto"
      >
        {/* Badge */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 backdrop-blur-sm mb-8"
        >
          <Sparkles className="w-4 h-4 text-amber-400" />
          <span className="text-sm text-slate-300">The future of ML education</span>
        </motion.div>

        {/* Main Headline */}
        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="text-5xl md:text-7xl lg:text-8xl font-bold mb-6 leading-tight"
        >
          <span className="text-white">Train AI.</span>
          <br />
          <span className="bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-500 bg-clip-text text-transparent">
            Break it.
          </span>
          <br />
          <span className="text-white">Understand it.</span>
        </motion.h1>

        {/* Subheading */}
        <motion.p
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="text-xl md:text-2xl text-slate-400 max-w-3xl mx-auto mb-12"
        >
          The world&apos;s first interactive Machine Learning playground.
          <br className="hidden sm:block" />
          <span className="text-slate-300">No lectures. Just real models you can touch.</span>
        </motion.p>

        {/* CTAs */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="flex flex-col sm:flex-row items-center justify-center gap-4"
        >
          <motion.a
            href="#models"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="group relative px-8 py-4 rounded-xl font-semibold text-lg overflow-hidden"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 to-blue-600" />
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-400 to-blue-500 opacity-0 group-hover:opacity-100 transition-opacity" />
            <div className="absolute inset-[1px] bg-gradient-to-r from-cyan-500 to-blue-600 rounded-[10px] group-hover:bg-transparent transition-colors" />
            <span className="relative flex items-center gap-2 text-white">
              <Zap className="w-5 h-5" />
              Start Playing
            </span>
          </motion.a>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="group flex items-center gap-2 px-8 py-4 rounded-xl font-semibold text-lg border border-white/20 text-white hover:bg-white/5 transition-colors"
          >
            <Play className="w-5 h-5 text-cyan-400 group-hover:text-cyan-300 transition-colors" />
            Watch 30-second Demo
          </motion.button>
        </motion.div>

        {/* Scroll indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5 }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
        >
          <motion.div
            animate={{ y: [0, 10, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="flex flex-col items-center gap-2 text-slate-500"
          >
            <span className="text-sm">Scroll to explore</span>
            <ChevronDown className="w-5 h-5" />
          </motion.div>
        </motion.div>
      </motion.div>
    </section>
  );
}

function HowItWorksSection() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <section ref={ref} className="relative py-32 px-4">
      <div className="max-w-6xl mx-auto">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
          className="text-center mb-20"
        >
          <span className="text-cyan-400 font-medium text-sm tracking-wider uppercase mb-4 block">
            How It Works
          </span>
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
            Three steps to{" "}
            <span className="bg-gradient-to-r from-cyan-400 to-purple-500 bg-clip-text text-transparent">
              understanding
            </span>
          </h2>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto">
            We believe the best way to learn ML is to experience it. No theory overload, just hands-on discovery.
          </p>
        </motion.div>

        {/* Steps */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {steps.map((step, index) => (
            <motion.div
              key={step.number}
              initial={{ opacity: 0, y: 50 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.6, delay: index * 0.2 }}
              className="relative group"
            >
              {/* Connector line */}
              {index < steps.length - 1 && (
                <div className="hidden md:block absolute top-16 left-[60%] w-[80%] h-px bg-gradient-to-r from-slate-700 to-transparent" />
              )}

              <div className="relative">
                {/* Glow */}
                <div
                  className="absolute -inset-4 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 blur-2xl"
                  style={{ background: `${step.color}20` }}
                />

                {/* Card */}
                <div className="relative bg-slate-900/50 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-8 hover:border-slate-600/50 transition-colors">
                  {/* Number */}
                  <div
                    className="text-6xl font-bold mb-6 opacity-20"
                    style={{ color: step.color }}
                  >
                    {step.number}
                  </div>

                  {/* Icon */}
                  <div
                    className="w-14 h-14 rounded-xl flex items-center justify-center mb-6"
                    style={{
                      background: `linear-gradient(135deg, ${step.color}30, ${step.color}10)`,
                      border: `1px solid ${step.color}40`,
                    }}
                  >
                    <step.icon className="w-7 h-7" style={{ color: step.color }} />
                  </div>

                  {/* Content */}
                  <h3 className="text-2xl font-bold text-white mb-3">{step.title}</h3>
                  <p className="text-slate-400 leading-relaxed">{step.description}</p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

function ModelGallerySection() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <section id="models" ref={ref} className="relative py-32 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <span className="text-purple-400 font-medium text-sm tracking-wider uppercase mb-4 block">
            Model Gallery
          </span>
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
            Choose your{" "}
            <span className="bg-gradient-to-r from-purple-400 to-pink-500 bg-clip-text text-transparent">
              adventure
            </span>
          </h2>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto">
            Each model is a journey. Start with the basics or dive into clustering.
            We&apos;ll make it make sense.
          </p>
        </motion.div>

        {/* Model Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {models.map((model, index) => (
            <ModelCardPremium
              key={model.id}
              {...model}
              index={index}
            />
          ))}
        </div>
      </div>
    </section>
  );
}

function CTASection() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <section ref={ref} className="relative py-32 px-4">
      <div className="max-w-4xl mx-auto">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={isInView ? { opacity: 1, scale: 1 } : {}}
          transition={{ duration: 0.6 }}
          className="relative"
        >
          {/* Glow background */}
          <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 via-purple-500/20 to-pink-500/20 blur-3xl" />

          {/* Card */}
          <div className="relative bg-slate-900/80 backdrop-blur-xl border border-slate-700/50 rounded-3xl p-12 md:p-16 text-center overflow-hidden">
            {/* Top border gradient */}
            <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-cyan-500 to-transparent" />

            <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">
              Ready to demystify{" "}
              <span className="bg-gradient-to-r from-cyan-400 to-purple-500 bg-clip-text text-transparent">
                machine learning
              </span>
              ?
            </h2>

            <p className="text-slate-400 text-lg mb-10 max-w-xl mx-auto">
              No accounts. No credit cards. No data saved.
              <br />
              Just you, the algorithms, and understanding.
            </p>

            <motion.a
              href="#models"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="inline-flex items-center gap-2 px-10 py-5 rounded-xl font-semibold text-lg bg-gradient-to-r from-cyan-500 to-blue-600 text-white hover:shadow-lg hover:shadow-cyan-500/25 transition-shadow"
            >
              <Zap className="w-5 h-5" />
              Start Your First Model
            </motion.a>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

export default function Home() {
  return (
    <main className="relative">
      <AnimatedBackground />

      <div className="relative z-10">
        <HeroSection />
        <HowItWorksSection />
        <ModelGallerySection />
        <CTASection />
      </div>
    </main>
  );
}
