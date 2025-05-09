/**
 * Tests for the error handler middleware.
 */

const { expect } = require('chai');
const sinon = require('sinon');
const httpMocks = require('node-mocks-http');

const errorHandler = require('../src/middleware/errorHandler');
const {
  ForexTradingPlatformError,
  DataValidationError,
  ServiceError,
  ServiceUnavailableError,
  TradingError,
  OrderExecutionError,
  AuthenticationError,
  AuthorizationError
} = require('../src/utils/errors');

describe('Error Handler Middleware', () => {
  let loggerStub;
  let req, res, next;
  
  beforeEach(() => {
    // Stub the logger
    loggerStub = {
      error: sinon.stub(),
      warn: sinon.stub(),
      info: sinon.stub()
    };
    
    // Replace the logger in the module
    const logger = require('../src/utils/logger');
    Object.keys(loggerStub).forEach(key => {
      logger[key] = loggerStub[key];
    });
    
    // Create mock request and response
    req = httpMocks.createRequest({
      method: 'GET',
      url: '/api/test',
      path: '/api/test'
    });
    
    res = httpMocks.createResponse();
    res.status = sinon.stub().returns(res);
    res.json = sinon.stub().returns(res);
    
    next = sinon.stub();
  });
  
  afterEach(() => {
    // Restore all stubs
    sinon.restore();
  });
  
  it('should handle ForexTradingPlatformError', () => {
    const error = new ForexTradingPlatformError(
      'Platform error',
      'PLATFORM_ERROR',
      { test: 'value' }
    );
    
    errorHandler(error, req, res, next);
    
    expect(res.status.calledWith(500)).to.be.true;
    expect(res.json.calledOnce).to.be.true;
    expect(res.json.firstCall.args[0]).to.deep.include({
      error_type: 'ForexTradingPlatformError',
      error_code: 'PLATFORM_ERROR',
      message: 'Platform error'
    });
    expect(loggerStub.error.calledOnce).to.be.true;
  });
  
  it('should handle DataValidationError with 400 status', () => {
    const error = new DataValidationError(
      'Validation error',
      'VALIDATION_ERROR',
      { field: 'price', error: 'Invalid price' }
    );
    
    errorHandler(error, req, res, next);
    
    expect(res.status.calledWith(400)).to.be.true;
    expect(res.json.calledOnce).to.be.true;
    expect(res.json.firstCall.args[0]).to.deep.include({
      error_type: 'DataValidationError',
      error_code: 'VALIDATION_ERROR',
      message: 'Validation error'
    });
    expect(loggerStub.warn.calledOnce).to.be.true;
  });
  
  it('should handle ServiceUnavailableError with 503 status', () => {
    const error = new ServiceUnavailableError(
      'Service unavailable',
      { service: 'broker' }
    );
    
    errorHandler(error, req, res, next);
    
    expect(res.status.calledWith(503)).to.be.true;
    expect(res.json.calledOnce).to.be.true;
    expect(res.json.firstCall.args[0]).to.deep.include({
      error_type: 'ServiceUnavailableError',
      message: 'Service unavailable'
    });
    expect(loggerStub.error.calledOnce).to.be.true;
  });
  
  it('should handle AuthenticationError with 401 status', () => {
    const error = new AuthenticationError(
      'Authentication failed',
      { reason: 'Invalid token' }
    );
    
    errorHandler(error, req, res, next);
    
    expect(res.status.calledWith(401)).to.be.true;
    expect(res.json.calledOnce).to.be.true;
    expect(res.json.firstCall.args[0]).to.deep.include({
      error_type: 'AuthenticationError',
      message: 'Authentication failed'
    });
    expect(loggerStub.warn.calledOnce).to.be.true;
  });
  
  it('should handle AuthorizationError with 403 status', () => {
    const error = new AuthorizationError(
      'Authorization failed',
      { resource: 'order', action: 'create' }
    );
    
    errorHandler(error, req, res, next);
    
    expect(res.status.calledWith(403)).to.be.true;
    expect(res.json.calledOnce).to.be.true;
    expect(res.json.firstCall.args[0]).to.deep.include({
      error_type: 'AuthorizationError',
      message: 'Authorization failed'
    });
    expect(loggerStub.warn.calledOnce).to.be.true;
  });
  
  it('should handle standard Error with 500 status', () => {
    const error = new Error('Standard error');
    
    errorHandler(error, req, res, next);
    
    expect(res.status.calledWith(500)).to.be.true;
    expect(res.json.calledOnce).to.be.true;
    expect(res.json.firstCall.args[0]).to.deep.include({
      error_type: 'InternalServerError',
      message: 'Standard error'
    });
    expect(loggerStub.error.calledOnce).to.be.true;
  });
  
  it('should handle errors with custom status code', () => {
    const error = new Error('Custom status error');
    error.statusCode = 422;
    
    errorHandler(error, req, res, next);
    
    expect(res.status.calledWith(422)).to.be.true;
    expect(res.json.calledOnce).to.be.true;
    expect(loggerStub.warn.calledOnce).to.be.true;
  });
});
