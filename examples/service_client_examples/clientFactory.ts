/**
 * Client Factory Example
 * 
 * This module demonstrates how to implement a client factory for standardized service clients.
 */

import { ClientConfig } from '../../common-js-lib/index';
import { MarketDataClient } from './MarketDataClient';

// Client registry for singleton instances
const clientRegistry: Record<string, any> = {};

// Default client configurations
const defaultConfigs: Record<string, ClientConfig> = {};

/**
 * Initialize service clients with proper configuration
 * 
 * This function should be called during service startup to register
 * client configurations and initialize clients.
 */
export function initializeClients(): void {
  console.log('Initializing service clients...');
  
  // Configure Market Data client
  const marketDataConfig: ClientConfig = {
    baseURL: 'http://market-data-service:8000/api/v1',
    serviceName: 'market-data-service',
    timeout: 30000,
    retryConfig: {
      baseDelay: 500,
      maxRetries: 3
    },
    circuitBreakerConfig: {
      failureThreshold: 5,
      resetTimeoutMs: 60000
    },
    bulkheadConfig: {
      maxConcurrent: 20
    }
  };
  
  // Register client configurations
  registerClientConfig('market-data-service', marketDataConfig);
  
  // Add configurations for other clients as needed
  
  console.log('Service clients initialized successfully');
}

/**
 * Register a client configuration
 * 
 * @param serviceName Name of the service
 * @param config Client configuration
 */
export function registerClientConfig(serviceName: string, config: ClientConfig): void {
  defaultConfigs[serviceName] = config;
}

/**
 * Get a client configuration
 * 
 * @param serviceName Name of the service
 * @returns Client configuration
 */
export function getClientConfig(serviceName: string): ClientConfig {
  if (defaultConfigs[serviceName]) {
    return defaultConfigs[serviceName];
  }
  
  // Create a default configuration
  const defaultConfig: ClientConfig = {
    baseURL: `http://${serviceName}:8000/api/v1`,
    serviceName: serviceName,
    timeout: 30000
  };
  
  // Register for future use
  defaultConfigs[serviceName] = defaultConfig;
  
  return defaultConfig;
}

/**
 * Create a service client with proper configuration
 * 
 * @param ClientClass Client class to instantiate
 * @param serviceName Name of the service
 * @param configOverride Optional configuration overrides
 * @returns Configured client instance
 */
export function createClient<T>(
  ClientClass: new (config: ClientConfig) => T,
  serviceName: string,
  configOverride?: Partial<ClientConfig>
): T {
  // Get base configuration
  const config = getClientConfig(serviceName);
  
  // Apply overrides
  const mergedConfig = configOverride ? { ...config, ...configOverride } : config;
  
  // Create client
  const client = new ClientClass(mergedConfig);
  
  console.log(`Created ${ClientClass.name} for ${serviceName}`);
  
  return client;
}

/**
 * Get a service client, creating it if necessary
 * 
 * @param ClientClass Client class to instantiate
 * @param serviceName Name of the service
 * @param configOverride Optional configuration overrides
 * @param singleton Whether to use a singleton instance
 * @returns Configured client instance
 */
export function getClient<T>(
  ClientClass: new (config: ClientConfig) => T,
  serviceName: string,
  configOverride?: Partial<ClientConfig>,
  singleton: boolean = true
): T {
  // For non-singleton clients, always create a new instance
  if (!singleton) {
    return createClient(ClientClass, serviceName, configOverride);
  }
  
  // For singleton clients, check the registry
  const clientKey = `${serviceName}:${ClientClass.name}`;
  
  if (clientRegistry[clientKey]) {
    return clientRegistry[clientKey] as T;
  }
  
  // Create and register the client
  const client = createClient(ClientClass, serviceName, configOverride);
  clientRegistry[clientKey] = client;
  
  return client;
}

/**
 * Get a configured Market Data client
 * 
 * @param configOverride Optional configuration overrides
 * @returns Configured Market Data client
 */
export function getMarketDataClient(configOverride?: Partial<ClientConfig>): MarketDataClient {
  return getClient(
    MarketDataClient,
    'market-data-service',
    configOverride
  );
}

// Add factory functions for other clients as needed