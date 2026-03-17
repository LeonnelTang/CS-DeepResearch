/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}", // 必须有这一行，确保它能扫到 App.tsx
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}