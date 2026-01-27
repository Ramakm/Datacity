import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        terminal: {
          bg: "#cfeee3",
          panel: "#f5f5f0",
          black: "#0a0a0a",
          grid: "#d4d4c8",
          accent: "#1a5c3a",
          warning: "#c4a000",
          mint: "#cfeee3",
        },
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', '"IBM Plex Mono"', '"Space Mono"', 'monospace'],
        condensed: ['"Oswald"', '"Barlow Condensed"', 'sans-serif'],
      },
      borderRadius: {
        terminal: '2px',
      },
      letterSpacing: {
        terminal: '0.12em',
      },
    },
  },
  plugins: [],
};

export default config;
