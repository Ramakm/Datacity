"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ArrowLeft, UserPlus, Eye, EyeOff, AlertCircle, Check } from "lucide-react";
import { useAuth } from "@/context/AuthContext";

export default function RegisterPage() {
  const router = useRouter();
  const { register } = useAuth();

  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (password !== confirmPassword) {
      setError("Passwords do not match");
      return;
    }

    if (password.length < 6) {
      setError("Password must be at least 6 characters");
      return;
    }

    setIsLoading(true);

    try {
      await register({ username, email, password });
      setSuccess(true);
      setTimeout(() => {
        router.push("/login");
      }, 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Registration failed");
    } finally {
      setIsLoading(false);
    }
  };

  if (success) {
    return (
      <div className="min-h-screen bg-terminal-bg flex items-center justify-center px-4">
        <div className="w-full max-w-md">
          <div className="bg-terminal-panel border-2 border-terminal-accent p-8 text-center">
            <div className="w-16 h-16 bg-terminal-accent mx-auto mb-4 flex items-center justify-center">
              <Check className="w-8 h-8 text-terminal-black" />
            </div>
            <h2 className="heading-terminal text-xl text-terminal-black mb-2">
              REGISTRATION_COMPLETE
            </h2>
            <p className="font-mono text-xs text-terminal-black/70">
              REDIRECTING TO LOGIN TERMINAL...
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-terminal-bg flex items-center justify-center px-4 py-8">
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
              <UserPlus className="w-5 h-5 text-terminal-mint" />
            </div>
            <div>
              <h1 className="heading-terminal text-xl text-terminal-black">
                CREATE_ACCOUNT
              </h1>
              <p className="font-mono text-xs text-terminal-black/60">
                REGISTER NEW TERMINAL ACCESS
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
                placeholder="Choose a username"
                required
                minLength={3}
                maxLength={50}
                disabled={isLoading}
              />
              <p className="font-mono text-[10px] text-terminal-black/50 mt-1">
                3-50 CHARACTERS
              </p>
            </div>

            <div>
              <label className="block font-mono text-xs uppercase tracking-terminal text-terminal-black mb-2">
                EMAIL
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-3 py-2 font-mono text-sm bg-terminal-bg border-2 border-terminal-black focus:border-terminal-accent focus:outline-none"
                placeholder="Enter email address"
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
                  placeholder="Create a password"
                  required
                  minLength={6}
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
              <p className="font-mono text-[10px] text-terminal-black/50 mt-1">
                MINIMUM 6 CHARACTERS
              </p>
            </div>

            <div>
              <label className="block font-mono text-xs uppercase tracking-terminal text-terminal-black mb-2">
                CONFIRM PASSWORD
              </label>
              <input
                type={showPassword ? "text" : "password"}
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                className="w-full px-3 py-2 font-mono text-sm bg-terminal-bg border-2 border-terminal-black focus:border-terminal-accent focus:outline-none"
                placeholder="Confirm your password"
                required
                disabled={isLoading}
              />
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full py-3 bg-terminal-black text-terminal-mint font-mono text-sm font-bold uppercase tracking-terminal hover:bg-terminal-black/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? "CREATING ACCOUNT..." : "REGISTER"}
            </button>

            <div className="text-center pt-4 border-t-2 border-terminal-black/20">
              <p className="font-mono text-xs text-terminal-black/60">
                ALREADY HAVE AN ACCOUNT?{" "}
                <Link
                  href="/login"
                  className="text-terminal-accent hover:underline"
                >
                  LOGIN HERE
                </Link>
              </p>
            </div>
          </form>
        </div>

        <div className="mt-6 p-4 border-2 border-dashed border-terminal-black/30">
          <p className="font-mono text-xs text-terminal-black/50 text-center">
            YOUR DATA IS STORED SECURELY // NO TRACKING
          </p>
        </div>
      </div>
    </div>
  );
}
