// tailwind.config.js
module.exports = {
    content: [
      './templates/**/*.html',  // Adjust the path based on your file structure
      './static/**/*.css',
      './app.py',  // Include Python files if you have inline styles
    ],
    purge: [
      // Paths to your templates
      './templates/**/*.html',
      // Paths to your Python files
      './app.py',
    ],
    darkMode: false,
    theme: {
      extend: {},
    },
    variants: {},
    plugins: [],
  }
  