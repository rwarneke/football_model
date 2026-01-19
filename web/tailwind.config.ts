import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        "primary-dark": "var(--color-primary-dark)",
        "primary-light": "var(--color-primary-light)",
        "accent-dark": "var(--color-accent-dark)",
        "accent-light": "var(--color-accent-light)",
        ebony: "var(--color-primary-dark)",
        white: "var(--color-primary-light)",
        sand: "var(--color-accent-light)",
        celadon: "var(--color-accent-dark)",
        ink: {
          900: "var(--color-primary-dark)",
          800: "var(--color-accent-light)",
          700: "var(--color-accent-dark)",
          500: "var(--color-accent-dark)",
          400: "var(--color-accent-dark)",
          200: "var(--color-primary-dark)",
        },
      },
      fontFamily: {
        sans: ["var(--font-inter)", "ui-sans-serif", "system-ui"],
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular"],
      },
      boxShadow: {
        soft: "0 10px 25px -15px rgba(15, 23, 42, 0.3)",
      },
    },
  },
  plugins: [],
};

export default config;
