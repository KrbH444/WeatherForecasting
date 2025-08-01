/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./templates/**/*.html",
    "./static/**/*.js",
    "./app.py"
  ],
  theme: {
    extend: {
      colors: {
        'weather-blue': '#3B82F6',
        'weather-dark-blue': '#1E40AF',
        'weather-light-blue': '#DBEAFE',
        'weather-red': '#EF4444',
        'weather-green': '#10B981',
        'weather-yellow': '#F59E0B',
        'weather-purple': '#8B5CF6',
        'weather-gray': '#6B7280',
        'weather-light-gray': '#F3F4F6',
        'weather-dark': '#1F2937',
      },
      fontFamily: {
        'sans': ['Inter', 'Arial', 'sans-serif'],
      },
      backgroundImage: {
        'weather-bg': "url('/static/css/weather.jpg')",
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'pulse-slow': 'pulse 3s infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
} 