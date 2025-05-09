/**
 * Standardized Service Client Template for TypeScript
 * 
 * This module provides a template for creating service clients that follow the
 * platform's standardized patterns for service communication, error handling,
 * resilience, and metrics collection.
 * 
 * Usage:
 * 1. Copy this template to your service's clients directory
 * 2. Rename the class to match your service (e.g., MarketDataClient)
 * 3. Implement service-specific methods using the base HTTP methods
 * 4. Create a factory function in your service's clientFactory.ts
 * 
 * Example:
 * ```typescript
 * import { BaseServiceClient, ClientConfig } from 'common-js-lib';
 * 
 * export class ExampleServiceClient extends BaseServiceClient {
 *   constructor(config: ClientConfig) {
 *     super(config);
 *   }
 *   
 *   async getResource(resourceId: string): Promise<any> {
 *     return this.get(`resources/${resourceId}`);
 *   }
 * }
 * ```
 */

import { 
  BaseServiceClient, 
  ClientConfig, 
  ClientError,
  ClientConnectionError,
  ClientTimeoutError,
  ClientValidationError,
  ClientAuthenticationError
} from '../index';

/**
 * Standardized service client template
 * 
 * This class extends BaseServiceClient with additional standardized methods
 * and patterns for service communication.
 * 
 * Features:
 * 1. Consistent error handling and logging
 * 2. Type safety with generics
 * 3. Standardized method signatures
 * 4. Built-in request/response logging
 * 5. Correlation ID propagation
 */
export class StandardServiceClient<T = any> extends BaseServiceClient {
  /**
   * Initialize the standardized service client
   * 
   * @param config Client configuration
   */
  constructor(config: ClientConfig) {
    super(config);
  }
  
  /**
   * Template method for getting a resource by ID
   * 
   * @param resourceId Resource identifier
   * @returns Resource data
   * @throws ClientError if the request fails
   * @throws ClientConnectionError if connection to the service fails
   * @throws ClientTimeoutError if the request times out
   * @throws ClientValidationError if the request is invalid
   * @throws ClientAuthenticationError if authentication fails
   */
  async getResource(resourceId: string): Promise<T> {
    this.logger.debug(`Getting resource ${resourceId}`);
    try {
      return await this.get(`resources/${resourceId}`);
    } catch (error) {
      this.logger.error(`Failed to get resource ${resourceId}: ${error}`);
      throw error;
    }
  }
  
  /**
   * Template method for creating a resource
   * 
   * @param data Resource data
   * @returns Created resource data
   * @throws ClientError if the request fails
   * @throws ClientConnectionError if connection to the service fails
   * @throws ClientTimeoutError if the request times out
   * @throws ClientValidationError if the request is invalid
   * @throws ClientAuthenticationError if authentication fails
   */
  async createResource(data: any): Promise<T> {
    this.logger.debug(`Creating resource`);
    try {
      return await this.post('resources', data);
    } catch (error) {
      this.logger.error(`Failed to create resource: ${error}`);
      throw error;
    }
  }
  
  /**
   * Template method for updating a resource
   * 
   * @param resourceId Resource identifier
   * @param data Updated resource data
   * @returns Updated resource data
   * @throws ClientError if the request fails
   * @throws ClientConnectionError if connection to the service fails
   * @throws ClientTimeoutError if the request times out
   * @throws ClientValidationError if the request is invalid
   * @throws ClientAuthenticationError if authentication fails
   */
  async updateResource(resourceId: string, data: any): Promise<T> {
    this.logger.debug(`Updating resource ${resourceId}`);
    try {
      return await this.put(`resources/${resourceId}`, data);
    } catch (error) {
      this.logger.error(`Failed to update resource ${resourceId}: ${error}`);
      throw error;
    }
  }
  
  /**
   * Template method for deleting a resource
   * 
   * @param resourceId Resource identifier
   * @returns Deletion confirmation
   * @throws ClientError if the request fails
   * @throws ClientConnectionError if connection to the service fails
   * @throws ClientTimeoutError if the request times out
   * @throws ClientValidationError if the request is invalid
   * @throws ClientAuthenticationError if authentication fails
   */
  async deleteResource(resourceId: string): Promise<any> {
    this.logger.debug(`Deleting resource ${resourceId}`);
    try {
      return await this.delete(`resources/${resourceId}`);
    } catch (error) {
      this.logger.error(`Failed to delete resource ${resourceId}: ${error}`);
      throw error;
    }
  }
  
  /**
   * Template method for listing resources
   * 
   * @param filters Optional filters to apply
   * @param page Page number
   * @param pageSize Number of items per page
   * @returns List of resources
   * @throws ClientError if the request fails
   * @throws ClientConnectionError if connection to the service fails
   * @throws ClientTimeoutError if the request times out
   * @throws ClientValidationError if the request is invalid
   * @throws ClientAuthenticationError if authentication fails
   */
  async listResources(filters?: Record<string, any>, page: number = 1, pageSize: number = 100): Promise<{ data: T[], meta: any }> {
    this.logger.debug(`Listing resources (page=${page}, pageSize=${pageSize})`);
    const params: Record<string, any> = { page, pageSize };
    
    if (filters) {
      Object.assign(params, filters);
    }
    
    try {
      return await this.get('resources', { params });
    } catch (error) {
      this.logger.error(`Failed to list resources: ${error}`);
      throw error;
    }
  }
  
  /**
   * Template method for executing a custom operation
   * 
   * @param operation Operation name
   * @param resourceId Optional resource identifier
   * @param data Optional operation data
   * @returns Operation result
   * @throws ClientError if the request fails
   * @throws ClientConnectionError if connection to the service fails
   * @throws ClientTimeoutError if the request times out
   * @throws ClientValidationError if the request is invalid
   * @throws ClientAuthenticationError if authentication fails
   */
  async executeOperation(operation: string, resourceId?: string, data?: any): Promise<any> {
    const endpoint = resourceId ? `resources/${resourceId}/${operation}` : `operations/${operation}`;
    
    this.logger.debug(`Executing operation ${operation}`);
    try {
      return await this.post(endpoint, data);
    } catch (error) {
      this.logger.error(`Failed to execute operation ${operation}: ${error}`);
      throw error;
    }
  }
  
  /**
   * Check the health of the service
   * 
   * @returns Health status
   * @throws ClientError if the request fails
   * @throws ClientConnectionError if connection to the service fails
   * @throws ClientTimeoutError if the request times out
   */
  async getHealth(): Promise<any> {
    this.logger.debug(`Checking service health`);
    try {
      return await this.get('health');
    } catch (error) {
      this.logger.error(`Failed to check service health: ${error}`);
      throw error;
    }
  }
  
  /**
   * Create a new client instance with the specified correlation ID
   * 
   * This method allows for easy propagation of correlation IDs across service calls.
   * 
   * @param correlationId Correlation ID to use for requests
   * @returns New client instance with the correlation ID set
   */
  withCorrelationId(correlationId: string): StandardServiceClient<T> {
    // Create a copy of the configuration
    const configCopy = { ...this.config };
    
    // Update headers with correlation ID
    configCopy.headers = {
      ...configCopy.headers,
      'X-Correlation-ID': correlationId
    };
    
    // Create new client with updated configuration
    return new (this.constructor as any)(configCopy);
  }
}