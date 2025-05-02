const express = require('express');
const cors = require('cors');
const morgan = require('morgan');
const dotenv = require('dotenv');

// Import routes
const analysisRoutes = require('./api/routes/analysis');
const monitoringRoutes = require('./api/routes/monitoring');
const performanceRoutes = require('./api/routes/performance');

// Import middleware from local auth module (which uses common-js-lib)
const { apiKeyAuth } = require('./api/middleware/auth');

// Load environment variables
dotenv.config();

// Initialize Express app
const app = express();

// Setup middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(cors());
app.use(morgan('dev'));

// Apply security middleware to protected routes
const protectedRoutes = express.Router();
// Only protected routes need API key auth
protectedRoutes.use(apiKeyAuth);

// Mount routes
app.use('/api/analysis', analysisRoutes);
app.use('/api/monitoring', monitoringRoutes);
app.use('/api/performance', performanceRoutes);

// Protected API endpoints
protectedRoutes.get('/healthcheck', (req, res) => {
  res.json({ status: 'ok', service: req.serviceName });
});
app.use('/api/protected', protectedRoutes);

// Import custom error handler middleware
const errorHandler = require('./middleware/errorHandler');

// Error handling middleware
app.use(errorHandler);

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Trading Gateway Service running on port ${PORT}`);
});

module.exports = app; // Export for testing
