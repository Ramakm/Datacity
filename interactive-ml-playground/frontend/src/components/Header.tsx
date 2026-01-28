"use client";

import Link from "next/link";
import { useAuth } from "@/context/AuthContext";
import { LogIn, LogOut, User } from "lucide-react";
import { useState, useRef, useEffect } from "react";

export default function Header() {
  const { user, isLoading, logout } = useAuth();
  const [showDropdown, setShowDropdown] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowDropdown(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleLogout = async () => {
    await logout();
    setShowDropdown(false);
  };

  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-terminal-panel border-b-2 border-terminal-black">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-14">
          <Link href="/" className="flex items-center gap-3 group">
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
          </Link>

          <nav className="flex items-center gap-4">
            <Link
              href="/#models"
              className="text-xs font-mono font-bold uppercase tracking-terminal text-terminal-black hover:bg-terminal-black hover:text-terminal-mint px-3 py-2 border-2 border-transparent hover:border-terminal-black transition-all hidden sm:block"
            >
              MODELS
            </Link>
            <Link
              href="/research-papers"
              className="text-xs font-mono font-bold uppercase tracking-terminal text-terminal-black hover:bg-terminal-black hover:text-terminal-mint px-3 py-2 border-2 border-transparent hover:border-terminal-black transition-all hidden sm:block"
            >
              PAPERS
            </Link>
            <Link
              href="/practice"
              className="text-xs font-mono font-bold uppercase tracking-terminal text-terminal-black hover:bg-terminal-black hover:text-terminal-mint px-3 py-2 border-2 border-transparent hover:border-terminal-black transition-all hidden sm:block"
            >
              PRACTICE
            </Link>
            <a
              href="https://github.com/Ramakm"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs font-mono font-bold uppercase tracking-terminal text-terminal-black hover:bg-terminal-black hover:text-terminal-mint px-3 py-2 border-2 border-transparent hover:border-terminal-black transition-all hidden sm:block"
            >
              GITHUB
            </a>

            {isLoading ? (
              <div className="w-8 h-8 border-2 border-terminal-black/30 animate-pulse" />
            ) : user ? (
              <div className="relative" ref={dropdownRef}>
                <button
                  onClick={() => setShowDropdown(!showDropdown)}
                  className="flex items-center gap-2 px-3 py-2 bg-terminal-black text-terminal-mint font-mono text-xs font-bold uppercase tracking-terminal border-2 border-terminal-black hover:bg-terminal-black/90 transition-all"
                >
                  <User className="w-4 h-4" />
                  <span className="hidden sm:inline">{user.username}</span>
                </button>

                {showDropdown && (
                  <div className="absolute right-0 mt-2 w-48 bg-terminal-panel border-2 border-terminal-black shadow-lg">
                    <div className="p-3 border-b-2 border-terminal-black/20">
                      <p className="font-mono text-xs text-terminal-black font-bold truncate">
                        {user.username}
                      </p>
                      <p className="font-mono text-[10px] text-terminal-black/60 truncate">
                        {user.email}
                      </p>
                    </div>
                    <button
                      onClick={handleLogout}
                      className="w-full flex items-center gap-2 px-3 py-2 font-mono text-xs uppercase tracking-terminal text-terminal-black hover:bg-terminal-black hover:text-terminal-mint transition-all"
                    >
                      <LogOut className="w-4 h-4" />
                      LOGOUT
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <Link
                href="/login"
                className="flex items-center gap-2 px-3 py-2 bg-terminal-black text-terminal-mint font-mono text-xs font-bold uppercase tracking-terminal border-2 border-terminal-black hover:bg-terminal-black/90 transition-all"
              >
                <LogIn className="w-4 h-4" />
                <span className="hidden sm:inline">LOGIN</span>
              </Link>
            )}
          </nav>
        </div>
      </div>
    </header>
  );
}
