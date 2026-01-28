"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ArrowLeft, LogIn, Eye, EyeOff, AlertCircle } from "lucide-react";
import { useAuth } from "@/context/AuthContext";

export default function LoginPage() {
  const router = useRouter();
  const { login } = useAuth();

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    try {
      await login({ username, password });
      router.push("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-terminal-bg flex items-center justify-center px-4">
      <div className="w-full max-w-md">
        <Link
          href="/"
          className="inline-flex items-center gap-2 font-mono text-xs uppercase tracking-terminal text-terminal-black hover:text-terminal-accent transition-colors mb-6"
        >
          <ArrowLeft className="w-4 h-4" />
          RETURN TO TERMINAL
        </Link>

        <div className="bg-terminal-panel border-2 border-terminal-black">
          <div className="border-b-2 border-terminal-black p-4 flex items-center gap-3">
            <div className="w-10 h-10 bg-terminal-black flex items-center justify-center">
              <LogIn className="w-5 h-5 text-terminal-mint" />
            </div>
            <div>
              <h1 className="heading-terminal text-xl text-terminal-black">
                SYSTEM_LOGIN
              </h1>
              <p className="font-mono text-xs text-terminal-black/60">
                AUTHENTICATE TO ACCESS TERMINAL
              </p>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="p-6 space-y-4">
            {error && (
              <div className="flex items-center gap-2 p-3 bg-red-100 border-2 border-red-500 text-red-700">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span className="font-mono text-xs">{error}</span>
              </div>
            )}

            <div>
              <label className="block font-mono text-xs uppercase tracking-terminal text-terminal-black mb-2">
                USERNAME
              </label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full px-3 py-2 font-mono text-sm bg-terminal-bg border-2 border-terminal-black focus:border-terminal-accent focus:outline-none"
                placeholder="Enter username"
                required
                disabled={isLoading}
              />
            </div>

            <div>
              <label className="block font-mono text-xs uppercase tracking-terminal text-terminal-black mb-2">
                PASSWORD
              </label>
              <div className="relative">
                <input
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-3 py-2 pr-10 font-mono text-sm bg-terminal-bg border-2 border-terminal-black focus:border-terminal-accent focus:outline-none"
                  placeholder="Enter password"
                  required
                  disabled={isLoading}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-terminal-black/50 hover:text-terminal-black"
                >
                  {showPassword ? (
                    <EyeOff className="w-4 h-4" />
                  ) : (
                    <Eye className="w-4 h-4" />
                  )}
                </button>
              </div>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full py-3 bg-terminal-black text-terminal-mint font-mono text-sm font-bold uppercase tracking-terminal hover:bg-terminal-black/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? "AUTHENTICATING..." : "LOGIN"}
            </button>

            <div className="text-center pt-4 border-t-2 border-terminal-black/20">
              <p className="font-mono text-xs text-terminal-black/60">
                NO ACCOUNT?{" "}
                <Link
                  href="/register"
                  className="text-terminal-accent hover:underline"
                >
                  REGISTER HERE
                </Link>
              </p>
            </div>
          </form>
        </div>

        <div className="mt-6 p-4 border-2 border-dashed border-terminal-black/30">
          <p className="font-mono text-xs text-terminal-black/50 text-center">
            SECURE AUTHENTICATION // ALL DATA ENCRYPTED
          </p>
        </div>
      </div>
    </div>
  );
}
