import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Interactive ML Playground | Train AI. Break it. Understand it.",
  description: "The world's first interactive Machine Learning playground. Learn by doing with real models, real data, and real code.",
  keywords: ["machine learning", "AI", "interactive learning", "sklearn", "python", "data science"],
  openGraph: {
    title: "Interactive ML Playground",
    description: "Train AI. Break it. Understand it.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} antialiased`}>
        <div className="min-h-screen flex flex-col">
          {/* Header */}
          <header className="fixed top-0 left-0 right-0 z-50">
            <div className="absolute inset-0 bg-slate-900/80 backdrop-blur-xl border-b border-slate-700/50" />
            <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex items-center justify-between h-16">
                <a href="/" className="flex items-center gap-3 group">
                  {/* Logo */}
                  <div className="relative">
                    <div className="absolute inset-0 bg-gradient-to-br from-cyan-400 to-blue-600 rounded-lg blur opacity-50 group-hover:opacity-75 transition-opacity" />
                    <div className="relative w-9 h-9 bg-gradient-to-br from-cyan-400 to-blue-600 rounded-lg flex items-center justify-center">
                      <svg
                        viewBox="0 0 24 24"
                        fill="none"
                        className="w-5 h-5 text-white"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <path d="M12 2L2 7l10 5 10-5-10-5z" />
                        <path d="M2 17l10 5 10-5" />
                        <path d="M2 12l10 5 10-5" />
                      </svg>
                    </div>
                  </div>
                  <span className="font-bold text-white text-lg hidden sm:block">
                    ML Playground
                  </span>
                </a>

                <nav className="flex items-center gap-6">
                  <a
                    href="#models"
                    className="text-sm text-slate-400 hover:text-white transition-colors hidden sm:block"
                  >
                    Models
                  </a>
                  <a
                    href="https://github.com"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-slate-400 hover:text-white transition-colors hidden sm:block"
                  >
                    GitHub
                  </a>
                  <a
                    href="#models"
                    className="px-4 py-2 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 text-white text-sm font-medium hover:shadow-lg hover:shadow-cyan-500/25 transition-shadow"
                  >
                    Get Started
                  </a>
                </nav>
              </div>
            </div>
          </header>

          {/* Main content */}
          <main className="flex-1 pt-16">{children}</main>

          {/* Footer */}
          <footer className="relative z-10 border-t border-slate-800">
            <div className="absolute inset-0 bg-slate-900/50 backdrop-blur-xl" />
            <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
                <div>
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-8 h-8 bg-gradient-to-br from-cyan-400 to-blue-600 rounded-lg flex items-center justify-center">
                      <svg
                        viewBox="0 0 24 24"
                        fill="none"
                        className="w-4 h-4 text-white"
                        stroke="currentColor"
                        strokeWidth="2"
                      >
                        <path d="M12 2L2 7l10 5 10-5-10-5z" />
                        <path d="M2 17l10 5 10-5" />
                        <path d="M2 12l10 5 10-5" />
                      </svg>
                    </div>
                    <span className="font-bold text-white">ML Playground</span>
                  </div>
                  <p className="text-slate-400 text-sm">
                    The interactive way to learn machine learning. No accounts, no data saved.
                  </p>
                </div>

                <div>
                  <h4 className="text-white font-semibold mb-4">Models</h4>
                  <ul className="space-y-2 text-sm">
                    <li>
                      <a href="/models/linear-regression" className="text-slate-400 hover:text-cyan-400 transition-colors">
                        Linear Regression
                      </a>
                    </li>
                    <li>
                      <span className="text-slate-500">Logistic Regression (Soon)</span>
                    </li>
                    <li>
                      <span className="text-slate-500">K-Nearest Neighbors (Soon)</span>
                    </li>
                    <li>
                      <span className="text-slate-500">K-Means Clustering (Soon)</span>
                    </li>
                  </ul>
                </div>

                <div>
                  <h4 className="text-white font-semibold mb-4">Built With</h4>
                  <ul className="space-y-2 text-sm text-slate-400">
                    <li>Next.js + React</li>
                    <li>FastAPI + Python</li>
                    <li>scikit-learn</li>
                    <li>Framer Motion</li>
                  </ul>
                </div>
              </div>

              <div className="pt-8 border-t border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-4">
                <p className="text-sm text-slate-500">
                  2024 Interactive ML Playground. Open source and free forever.
                </p>
                <div className="flex items-center gap-4">
                  <span className="text-xs text-slate-600 px-3 py-1 rounded-full bg-slate-800/50">
                    No tracking
                  </span>
                  <span className="text-xs text-slate-600 px-3 py-1 rounded-full bg-slate-800/50">
                    No cookies
                  </span>
                  <span className="text-xs text-slate-600 px-3 py-1 rounded-full bg-slate-800/50">
                    Privacy first
                  </span>
                </div>
              </div>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}
