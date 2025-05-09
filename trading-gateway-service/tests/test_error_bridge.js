/**
 * Tests for the error bridge between JavaScript and Python components.
 */

const { expect } = require('chai');
const sinon = require('sinon');

const {
  convertPythonError,
  convertToPythonError,
  handleError,
  withErrorHandling,
  withAsyncErrorHandling
} = require('../src/utils/errorBridge');

const {
  ForexTradingPlatformError,
  DataValidationError,
  ServiceError
} = require('../src/utils/errors');

describe('Error Bridge', () => {
  let loggerStub;
  
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
  });
  
  afterEach(() => {
    // Restore all stubs
    sinon.restore();
  });
  
  describe('convertPythonError', () => {
    it('should convert a Python error object to a JavaScript error', () => {
      const pythonError = {
        error_type: 'DataValidationError',
        error_code: 'VALIDATION_ERROR',
        message: 'Validation failed',
        details: { field: 'price', error: 'Invalid price' }
      };
      
      const jsError = convertPythonError(pythonError);
      
      expect(jsError).to.be.instanceOf(DataValidationError);
      expect(jsError.message).to.equal('Validation failed');
      expect(jsError.errorCode).to.equal('VALIDATION_ERROR');
      expect(jsError.details).to.deep.equal({ field: 'price', error: 'Invalid price' });
    });
    
    it('should convert a Python error JSON string to a JavaScript error', () => {
      const pythonErrorJson = JSON.stringify({
        error_type: 'ServiceError',
        error_code: 'SERVICE_ERROR',
        message: 'Service unavailable',
        details: { service: 'broker' }
      });
      
      const jsError = convertPythonError(pythonErrorJson);
      
      expect(jsError).to.be.instanceOf(ServiceError);
      expect(jsError.message).to.equal('Service unavailable');
      expect(jsError.errorCode).to.equal('SERVICE_ERROR');
      expect(jsError.details).to.deep.equal({ service: 'broker' });
    });
    
    it('should handle unknown error types', () => {
      const pythonError = {
        error_type: 'UnknownError',
        error_code: 'UNKNOWN_ERROR',
        message: 'Unknown error',
        details: {}
      };
      
      const jsError = convertPythonError(pythonError);
      
      expect(jsError).to.be.instanceOf(ForexTradingPlatformError);
      expect(jsError.message).to.equal('Unknown error');
      expect(jsError.errorCode).to.equal('UNKNOWN_ERROR');
    });
  });
  
  describe('convertToPythonError', () => {
    it('should convert a JavaScript error to a Python-compatible format', () => {
      const jsError = new DataValidationError(
        'Validation failed',
        'VALIDATION_ERROR',
        { field: 'price', error: 'Invalid price' }
      );
      
      const pythonError = convertToPythonError(jsError);
      
      expect(pythonError).to.deep.equal({
        error_type: 'DataValidationError',
        error_code: 'VALIDATION_ERROR',
        message: 'Validation failed',
        details: { field: 'price', error: 'Invalid price' }
      });
    });
    
    it('should convert a standard JavaScript error to a Python-compatible format', () => {
      const jsError = new Error('Standard error');
      
      const pythonError = convertToPythonError(jsError);
      
      expect(pythonError.error_type).to.equal('ServiceError');
      expect(pythonError.error_code).to.equal('JS_EXCEPTION');
      expect(pythonError.message).to.equal('Standard error');
      expect(pythonError.details.original_error).to.equal('Error');
      expect(pythonError.details.stack).to.be.a('string');
    });
  });
  
  describe('handleError', () => {
    it('should log a ForexTradingPlatformError', () => {
      const error = new ServiceError('Service error', 'SERVICE_ERROR', { service: 'broker' });
      const context = { operation: 'test' };
      
      handleError(error, context);
      
      expect(loggerStub.error.calledOnce).to.be.true;
      expect(loggerStub.error.firstCall.args[0]).to.equal('ServiceError: Service error');
      expect(loggerStub.error.firstCall.args[1].errorType).to.equal('ServiceError');
      expect(loggerStub.error.firstCall.args[1].operation).to.equal('test');
    });
    
    it('should convert to Python format if requested', () => {
      const error = new ServiceError('Service error', 'SERVICE_ERROR', { service: 'broker' });
      const context = { operation: 'test' };
      
      const result = handleError(error, context, true);
      
      expect(result).to.deep.equal({
        error_type: 'ServiceError',
        error_code: 'SERVICE_ERROR',
        message: 'Service error',
        details: { service: 'broker' }
      });
    });
  });
  
  describe('withErrorHandling', () => {
    it('should wrap a function with error handling', () => {
      const originalFunc = () => {
        throw new Error('Test error');
      };
      
      const wrappedFunc = withErrorHandling(originalFunc, {
        context: { operation: 'test' },
        rethrow: false
      });
      
      const result = wrappedFunc();
      
      expect(loggerStub.error.calledOnce).to.be.true;
      expect(result).to.be.instanceOf(Error);
    });
    
    it('should rethrow errors as ForexTradingPlatformError', () => {
      const originalFunc = () => {
        throw new Error('Test error');
      };
      
      const wrappedFunc = withErrorHandling(originalFunc, {
        context: { operation: 'test' },
        rethrow: true,
        defaultErrorClass: ServiceError
      });
      
      expect(() => wrappedFunc()).to.throw(ServiceError);
    });
  });
  
  describe('withAsyncErrorHandling', () => {
    it('should wrap an async function with error handling', async () => {
      const originalFunc = async () => {
        throw new Error('Test error');
      };
      
      const wrappedFunc = withAsyncErrorHandling(originalFunc, {
        context: { operation: 'test' },
        rethrow: false
      });
      
      const result = await wrappedFunc();
      
      expect(loggerStub.error.calledOnce).to.be.true;
      expect(result).to.be.instanceOf(Error);
    });
    
    it('should rethrow errors as ForexTradingPlatformError', async () => {
      const originalFunc = async () => {
        throw new Error('Test error');
      };
      
      const wrappedFunc = withAsyncErrorHandling(originalFunc, {
        context: { operation: 'test' },
        rethrow: true,
        defaultErrorClass: ServiceError
      });
      
      try {
        await wrappedFunc();
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error).to.be.instanceOf(ServiceError);
      }
    });
  });
});
