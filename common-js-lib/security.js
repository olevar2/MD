// Common security utilities (JWT, API Key)

const jwt = require('jsonwebtoken');

/**
 * Validates an API key against a dictionary of valid keys.
 * @param {string} apiKey - The API key to validate.
 * @param {Object.<string, string>} validKeys - Dictionary mapping service names to valid API keys.
 * @param {string} [serviceName] - Optional service name to validate against.
 * @returns {boolean} - True if valid, False otherwise.
 */
function validateApiKey(apiKey, validKeys, serviceName) {
  if (!apiKey || !validKeys) {
    return false;
  }
  if (serviceName) {
    return serviceName in validKeys && apiKey === validKeys[serviceName];
  } else {
    return Object.values(validKeys).includes(apiKey);
  }
}

/**
 * Creates a JWT token.
 * @param {string|object|Buffer} payload - Payload to sign.
 * @param {string} secretKey - Secret key for signing.
 * @param {object} [options] - Options for jwt.sign (e.g., expiresIn).
 * @returns {string} - Encoded JWT token string.
 */
function createJwtToken(payload, secretKey, options = { expiresIn: '15m' }) {
  if (!payload || !secretKey) {
    throw new Error('Payload and secretKey are required to create a JWT token.');
  }
  return jwt.sign(payload, secretKey, options);
}

/**
 * Validates a JWT token and verifies required scopes.
 * @param {string} token - JWT token to validate.
 * @param {string} secretKey - Secret key for verification.
 * @param {object} [options] - Options for jwt.verify (e.g., algorithms).
 * @param {string[]} [requiredScopes] - List of required scopes.
 * @returns {object} - Decoded token payload if valid.
 * @throws {Error} - If token is invalid or missing required scopes.
 */
function validateToken(token, secretKey, options = { algorithms: ['HS256'] }, requiredScopes = []) {
  if (!token || !secretKey) {
    throw new Error('Token and secretKey are required for validation.');
  }

  try {
    const decoded = jwt.verify(token, secretKey, options);

    if (requiredScopes && requiredScopes.length > 0) {
      const tokenScopes = new Set(decoded.scopes || []);
      const requiredScopeSet = new Set(requiredScopes);
      const missingScopes = [...requiredScopeSet].filter(scope => !tokenScopes.has(scope));

      if (missingScopes.length > 0) {
        throw new Error(`Token missing required scopes: ${missingScopes.join(', ')}`);
      }
    }
    return decoded;
  } catch (err) {
    // Re-throw JWT errors or scope validation errors
    throw new Error(`Token validation failed: ${err.message}`);
  }
}

/**
 * Express middleware for API Key authentication.
 * @param {Object.<string, string>} validKeys - Dictionary mapping service names to valid API keys.
 * @param {string} [apiKeyHeader='X-API-Key'] - Header name for the API key.
 * @param {string} [serviceNameHeader='X-Service-Name'] - Header name for the service name.
 * @returns {function} - Express middleware function.
 */
function apiKeyAuthMiddleware(validKeys, apiKeyHeader = 'X-API-Key', serviceNameHeader = 'X-Service-Name') {
  return (req, res, next) => {
    const apiKey = req.headers[apiKeyHeader.toLowerCase()];
    const serviceName = req.headers[serviceNameHeader.toLowerCase()];

    if (!validateApiKey(apiKey, validKeys, serviceName)) {
      console.warn(`API Key Auth Failed: Invalid key '${apiKey}' for service '${serviceName || 'any'}' from ${req.ip}`);
      return res.status(401).json({ error: 'Unauthorized: Invalid API Key' });
    }
    // Attach service name if provided, useful for downstream logic
    if (serviceName) {
        req.serviceName = serviceName;
    }
    next();
  };
}


/**
 * Express middleware for JWT authentication.
 * @param {string} secretKey - Secret key for JWT verification.
 * @param {string[]} [requiredScopes=[]] - List of required scopes.
 * @param {object} [options] - Options for jwt.verify.
 * @returns {function} - Express middleware function.
 */
function jwtAuthMiddleware(secretKey, requiredScopes = [], options = { algorithms: ['HS256'] }) {
  return (req, res, next) => {
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return res.status(401).json({ error: 'Unauthorized: Missing or invalid Bearer token' });
    }

    const token = authHeader.split(' ')[1];

    try {
      const payload = validateToken(token, secretKey, options, requiredScopes);
      req.user = payload; // Attach decoded payload to request object
      next();
    } catch (error) {
      console.warn(`JWT Auth Failed: ${error.message} from ${req.ip}`);
      // Differentiate between forbidden (scope issue) and unauthorized (invalid token)
      if (error.message.includes('scopes')) {
          return res.status(403).json({ error: `Forbidden: ${error.message}` });
      }
      return res.status(401).json({ error: `Unauthorized: ${error.message}` });
    }
  };
}


module.exports = {
  validateApiKey,
  createJwtToken,
  validateToken,
  apiKeyAuthMiddleware,
  jwtAuthMiddleware,
};

