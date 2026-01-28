import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import { AuthProvider } from "@/context/AuthContext";
import Header from "@/components/Header";

export const metadata: Metadata = {
  title: "ML TERMINAL | MACHINE LEARNING RESEARCH INTERFACE",
  description: "Government-grade machine learning research terminal. Train models. Analyze data. Generate code.",
  keywords: ["machine learning", "AI", "research terminal", "sklearn", "python", "data science"],
  openGraph: {
    title: "ML TERMINAL",
    description: "Machine Learning Research Interface",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">
        <AuthProvider>
          <div className="min-h-screen flex flex-col bg-terminal-bg grid-bg">
            {/* Header */}
            <Header />

            {/* Main content */}
            <main className="flex-1 pt-14">{children}</main>

          {/* Footer */}
          <footer className="bg-terminal-black text-terminal-mint border-t-2 border-terminal-black">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
                <div>
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-6 h-6 border-2 border-terminal-mint flex items-center justify-center">
                      <svg
                        viewBox="0 0 24 24"
                        fill="none"
                        className="w-3 h-3 text-terminal-mint"
                        stroke="currentColor"
                        strokeWidth="2"
                      >
                        <path d="M12 2L2 7l10 5 10-5-10-5z" />
                        <path d="M2 17l10 5 10-5" />
                        <path d="M2 12l10 5 10-5" />
                      </svg>
                    </div>
                    <span className="font-mono font-bold text-xs uppercase tracking-terminal">ML TERMINAL</span>
                  </div>
                  <p className="text-terminal-grid text-xs font-mono leading-relaxed">
                    SECURE RESEARCH INTERFACE // NO DATA RETENTION // LOCAL PROCESSING ONLY
                  </p>
                </div>

                <div>
                  <h4 className="font-mono font-bold text-xs uppercase tracking-terminal mb-4 text-terminal-mint">AVAILABLE MODELS</h4>
                  <ul className="space-y-2 text-xs font-mono">
                    <li>
                      <Link href="/models/linear-regression" className="text-terminal-grid hover:text-terminal-mint hover:underline transition-colors">
                        &gt; LINEAR_REGRESSION
                      </Link>
                    </li>
                    <li>
                      <Link href="/models/logistic-regression" className="text-terminal-grid hover:text-terminal-mint hover:underline transition-colors">
                        &gt; LOGISTIC_REGRESSION
                      </Link>
                    </li>
                    <li>
                      <Link href="/models/knn" className="text-terminal-grid hover:text-terminal-mint hover:underline transition-colors">
                        &gt; KNN
                      </Link>
                    </li>
                    <li>
                      <Link href="/models/kmeans" className="text-terminal-grid hover:text-terminal-mint hover:underline transition-colors">
                        &gt; K_MEANS
                      </Link>
                    </li>
                  </ul>
                </div>

                <div>
                  <h4 className="font-mono font-bold text-xs uppercase tracking-terminal mb-4 text-terminal-mint">RESOURCES</h4>
                  <ul className="space-y-2 text-xs font-mono">
                    <li>
                      <Link href="/research-papers" className="text-terminal-grid hover:text-terminal-mint hover:underline transition-colors">
                        &gt; RESEARCH_PAPERS
                      </Link>
                    </li>
                    <li>
                      <Link href="/practice" className="text-terminal-grid hover:text-terminal-mint hover:underline transition-colors">
                        &gt; PRACTICE
                      </Link>
                    </li>
                  </ul>
                </div>
              </div>

              <div className="pt-6 border-t border-terminal-grid/30 flex flex-col sm:flex-row items-center justify-between gap-4">
                <p className="text-xs font-mono text-terminal-grid">
                  [2026] ML TERMINAL // OPEN SOURCE // UNRESTRICTED ACCESS
                </p>
                <div className="flex items-center gap-3">
                  <span className="text-xs font-mono px-2 py-1 border border-terminal-accent text-terminal-accent uppercase tracking-wider">
                    NO TRACKING
                  </span>
                  <span className="text-xs font-mono px-2 py-1 border border-terminal-accent text-terminal-accent uppercase tracking-wider">
                    NO COOKIES
                  </span>
                  <span className="text-xs font-mono px-2 py-1 border border-terminal-warning text-terminal-warning uppercase tracking-wider">
                    SECURE
                  </span>
                </div>
              </div>
            </div>
          </footer>
          </div>
        </AuthProvider>
      </body>
    </html>
  );
}
