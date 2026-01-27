import type { Metadata } from "next";
import "./globals.css";

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
        <div className="min-h-screen flex flex-col bg-terminal-bg grid-bg">
          {/* Header */}
          <header className="fixed top-0 left-0 right-0 z-50 bg-terminal-panel border-b-2 border-terminal-black">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex items-center justify-between h-14">
                <a href="/" className="flex items-center gap-3 group">
                  {/* Logo */}
                  <div className="w-8 h-8 bg-terminal-black flex items-center justify-center">
                    <svg
                      viewBox="0 0 24 24"
                      fill="none"
                      className="w-5 h-5 text-terminal-mint"
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
                  <span className="font-mono font-bold text-terminal-black text-sm uppercase tracking-terminal hidden sm:block">
                    ML TERMINAL
                  </span>
                </a>

                <nav className="flex items-center gap-4">
                  <a
                    href="#models"
                    className="text-xs font-mono font-bold uppercase tracking-terminal text-terminal-black hover:bg-terminal-black hover:text-terminal-mint px-3 py-2 border-2 border-transparent hover:border-terminal-black transition-all hidden sm:block"
                  >
                    MODELS
                  </a>
                  <a
                    href="https://github.com"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs font-mono font-bold uppercase tracking-terminal text-terminal-black hover:bg-terminal-black hover:text-terminal-mint px-3 py-2 border-2 border-transparent hover:border-terminal-black transition-all hidden sm:block"
                  >
                    GITHUB
                  </a>
                  <a
                    href="#models"
                    className="btn-terminal"
                  >
                    ACCESS SYSTEM
                  </a>
                </nav>
              </div>
            </div>
          </header>

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
                      <a href="/models/linear-regression" className="text-terminal-grid hover:text-terminal-mint hover:underline transition-colors">
                        &gt; LINEAR_REGRESSION
                      </a>
                    </li>
                    <li>
                      <a href="/models/logistic-regression" className="text-terminal-grid hover:text-terminal-mint hover:underline transition-colors">
                        &gt; LOGISTIC_REGRESSION
                      </a>
                    </li>
                    <li>
                      <span className="text-terminal-grid opacity-50">&gt; KNN [PENDING]</span>
                    </li>
                    <li>
                      <span className="text-terminal-grid opacity-50">&gt; K_MEANS [PENDING]</span>
                    </li>
                  </ul>
                </div>

                <div>
                  <h4 className="font-mono font-bold text-xs uppercase tracking-terminal mb-4 text-terminal-mint">Paper & Coding</h4>
                  <ul className="space-y-2 text-xs font-mono text-terminal-grid">
                    <li>Reasearch Paper</li>
                    <li>Questions Practice</li>
                  </ul>
                </div>
              </div>

              <div className="pt-6 border-t border-terminal-grid/30 flex flex-col sm:flex-row items-center justify-between gap-4">
                <p className="text-xs font-mono text-terminal-grid">
                  [2024] ML TERMINAL // OPEN SOURCE // UNRESTRICTED ACCESS
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
      </body>
    </html>
  );
}
