import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          900: "#0b0f19",
          800: "#111827",
          700: "#1f2937",
          500: "#4b5563",
          400: "#6b7280",
          200: "#e5e7eb",
        },
        mint: {
          500: "#6ee7b7",
        },
      },
      fontFamily: {
        sans: ["var(--font-inter)", "ui-sans-serif", "system-ui"],
      },
      boxShadow: {
        soft: "0 10px 25px -15px rgba(15, 23, 42, 0.3)",
      },
    },
  },
  plugins: [],
};

export default config;
