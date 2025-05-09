/**
 * Standardized Express.js Endpoint Template
 * 
 * This template demonstrates how to create API endpoints that follow
 * the platform's standardized API design patterns.
 */

const express = require('express');
const router = express.Router();
const { body, param, query, validationResult } = require('express-validator');
const { logger } = require('common-js-lib/logging');
const { ResourceNotFoundError, ValidationError } = require('common-js-lib/errors');

/**
 * Resource schema for documentation and validation
 */
const resourceSchema = {
  id: { type: 'string', description: 'Unique identifier for the resource' },
  name: { type: 'string', description: 'Name of the resource' },
  description: { type: 'string', description: 'Description of the resource' },
  createdAt: { type: 'string', format: 'date-time', description: 'Creation timestamp' },
  updatedAt: { type: 'string', format: 'date-time', description: 'Last update timestamp' }
};

/**
 * Middleware to validate request parameters
 */
const validateRequest = (req, res, next) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    logger.warn('Validation error', { errors: errors.array() });
    return res.status(400).json({
      error: {
        message: 'Validation error',
        details: errors.array(),
        type: 'ValidationError',
        code: 'VALIDATION_ERROR'
      },
      success: false
    });
  }
  next();
};

/**
 * Error handler middleware
 */
const errorHandler = (err, req, res, next) => {
  logger.error('API error', { error: err.message, stack: err.stack });
  
  if (err instanceof ResourceNotFoundError) {
    return res.status(404).json({
      error: {
        message: err.message,
        type: err.constructor.name,
        code: err.code || 'RESOURCE_NOT_FOUND'
      },
      success: false
    });
  }
  
  if (err instanceof ValidationError) {
    return res.status(400).json({
      error: {
        message: err.message,
        type: err.constructor.name,
        code: err.code || 'VALIDATION_ERROR'
      },
      success: false
    });
  }
  
  return res.status(500).json({
    error: {
      message: err.message,
      type: err.constructor.name,
      code: err.code || 'INTERNAL_SERVER_ERROR'
    },
    success: false
  });
};

/**
 * @route GET /v1/service-name/resources
 * @description Get a paginated list of resources with optional filtering
 * @param {number} page - Page number (starts at 1)
 * @param {number} pageSize - Number of items per page (max 100)
 * @param {string} name - Optional filter by name
 * @returns {object} Paginated list of resources
 */
router.get('/v1/service-name/resources', [
  query('page').optional().isInt({ min: 1 }).toInt(),
  query('pageSize').optional().isInt({ min: 1, max: 100 }).toInt(),
  query('name').optional().isString(),
  validateRequest
], async (req, res, next) => {
  try {
    const { page = 1, pageSize = 10, name } = req.query;
    
    // Implementation logic here
    // This is just an example
    const items = [
      {
        id: 'resource-1',
        name: 'Example Resource 1',
        description: 'This is example resource 1',
        createdAt: '2023-01-01T00:00:00Z',
        updatedAt: '2023-01-02T00:00:00Z'
      },
      {
        id: 'resource-2',
        name: 'Example Resource 2',
        description: 'This is example resource 2',
        createdAt: '2023-01-03T00:00:00Z',
        updatedAt: '2023-01-04T00:00:00Z'
      }
    ];
    
    return res.status(200).json({
      items,
      total: items.length,
      page,
      pageSize
    });
  } catch (error) {
    next(error);
  }
});

/**
 * @route GET /v1/service-name/resources/:resourceId
 * @description Get a specific resource by ID
 * @param {string} resourceId - Resource ID
 * @returns {object} Resource details
 */
router.get('/v1/service-name/resources/:resourceId', [
  param('resourceId').isString(),
  validateRequest
], async (req, res, next) => {
  try {
    const { resourceId } = req.params;
    
    // Implementation logic here
    // This is just an example
    const resource = {
      id: resourceId,
      name: 'Example Resource',
      description: 'This is an example resource',
      createdAt: '2023-01-01T00:00:00Z',
      updatedAt: '2023-01-02T00:00:00Z'
    };
    
    return res.status(200).json(resource);
  } catch (error) {
    if (error.message.includes('not found')) {
      return next(new ResourceNotFoundError(`Resource ${req.params.resourceId} not found`));
    }
    next(error);
  }
});

/**
 * @route POST /v1/service-name/resources
 * @description Create a new resource
 * @param {object} request - Resource creation request
 * @returns {object} Created resource
 */
router.post('/v1/service-name/resources', [
  body('name').isString().notEmpty(),
  body('description').optional().isString(),
  validateRequest
], async (req, res, next) => {
  try {
    const { name, description } = req.body;
    
    // Implementation logic here
    // This is just an example
    const resource = {
      id: 'resource-new',
      name,
      description,
      createdAt: '2023-01-01T00:00:00Z',
      updatedAt: '2023-01-01T00:00:00Z'
    };
    
    return res.status(201).json(resource);
  } catch (error) {
    next(error);
  }
});

/**
 * @route PUT /v1/service-name/resources/:resourceId
 * @description Update an existing resource
 * @param {string} resourceId - Resource ID
 * @param {object} request - Resource update request
 * @returns {object} Updated resource
 */
router.put('/v1/service-name/resources/:resourceId', [
  param('resourceId').isString(),
  body('name').optional().isString(),
  body('description').optional().isString(),
  validateRequest
], async (req, res, next) => {
  try {
    const { resourceId } = req.params;
    const { name, description } = req.body;
    
    // Implementation logic here
    // This is just an example
    const resource = {
      id: resourceId,
      name: name || 'Example Resource',
      description: description || 'This is an updated resource',
      createdAt: '2023-01-01T00:00:00Z',
      updatedAt: '2023-01-02T00:00:00Z'
    };
    
    return res.status(200).json(resource);
  } catch (error) {
    if (error.message.includes('not found')) {
      return next(new ResourceNotFoundError(`Resource ${req.params.resourceId} not found`));
    }
    next(error);
  }
});

/**
 * @route DELETE /v1/service-name/resources/:resourceId
 * @description Delete an existing resource
 * @param {string} resourceId - Resource ID
 * @returns {null} No content
 */
router.delete('/v1/service-name/resources/:resourceId', [
  param('resourceId').isString(),
  validateRequest
], async (req, res, next) => {
  try {
    const { resourceId } = req.params;
    
    // Implementation logic here
    // This is just an example
    
    return res.status(204).send();
  } catch (error) {
    if (error.message.includes('not found')) {
      return next(new ResourceNotFoundError(`Resource ${req.params.resourceId} not found`));
    }
    next(error);
  }
});

/**
 * @route POST /v1/service-name/resources/:resourceId/actions/validate
 * @description Perform validation on a resource
 * @param {string} resourceId - Resource ID
 * @returns {object} Validation results
 */
router.post('/v1/service-name/resources/:resourceId/actions/validate', [
  param('resourceId').isString(),
  validateRequest
], async (req, res, next) => {
  try {
    const { resourceId } = req.params;
    
    // Implementation logic here
    // This is just an example
    const validationResults = {
      resourceId,
      isValid: true,
      validationTimestamp: '2023-01-01T00:00:00Z',
      validationResults: [
        { check: 'format', passed: true },
        { check: 'content', passed: true }
      ]
    };
    
    return res.status(200).json(validationResults);
  } catch (error) {
    if (error.message.includes('not found')) {
      return next(new ResourceNotFoundError(`Resource ${req.params.resourceId} not found`));
    }
    next(error);
  }
});

// Apply error handler
router.use(errorHandler);

module.exports = router;