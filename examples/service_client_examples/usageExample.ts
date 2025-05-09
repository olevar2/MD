/**
 * Service Client Usage Example
 * 
 * This module demonstrates how to use the standardized service clients.
 */

import { 
  initializeClients, 
  getMarketDataClient 
} from './clientFactory';

import {
  ClientError,
  ClientConnectionError,
  ClientTimeoutError,
  ClientValidationError,
  ClientAuthenticationError
} from '../../common-js-lib/index';

/**
 * Example of using the Market Data client
 */
async function marketDataExample() {
  console.log('Running Market Data client example...');
  
  // Get the client
  const client = getMarketDataClient();
  
  // Example 1: Get OHLCV data
  try {
    const startTime = new Date();
    startTime.setDate(startTime.getDate() - 7);
    
    const result = await client.getOHLCVData(
      'EUR/USD',
      '1h',
      startTime,
      new Date(),
      100
    );
    
    console.log(`Got ${result.data.length} OHLCV data points`);
  } catch (error) {
    if (error instanceof ClientTimeoutError) {
      console.error(`Request timed out: ${error.message}`);
    } else if (error instanceof ClientConnectionError) {
      console.error(`Connection error: ${error.message}`);
    } else if (error instanceof ClientError) {
      console.error(`Client error: ${error.message}`);
    } else {
      console.error(`Unexpected error: ${error}`);
    }
  }
  
  // Example 2: Get instrument information
  try {
    const instrument = await client.getInstrument('EUR/USD');
    console.log(`Got instrument information: ${JSON.stringify(instrument)}`);
  } catch (error) {
    console.error(`Failed to get instrument information: ${error}`);
  }
  
  // Example 3: List instruments
  try {
    const instruments = await client.listInstruments('forex');
    console.log(`Got ${instruments.data.length} forex instruments`);
  } catch (error) {
    console.error(`Failed to list instruments: ${error}`);
  }
  
  // Example 4: Get latest price
  try {
    const price = await client.getLatestPrice('EUR/USD');
    console.log(`Latest EUR/USD price: ${JSON.stringify(price)}`);
  } catch (error) {
    console.error(`Failed to get latest price: ${error}`);
  }
  
  // Example 5: Using correlation ID
  try {
    // Create a client with correlation ID
    const correlationId = 'example-correlation-id';
    const clientWithCorrelationId = client.withCorrelationId(correlationId);
    
    // Make a request with the correlation ID
    const result = await clientWithCorrelationId.getOHLCVData(
      'EUR/USD',
      '1h',
      undefined,
      undefined,
      10
    );
    
    console.log(`Got ${result.data.length} OHLCV data points with correlation ID`);
  } catch (error) {
    console.error(`Failed to get OHLCV data with correlation ID: ${error}`);
  }
}

/**
 * Main function
 */
async function main() {
  console.log('Starting service client example...');
  
  // Initialize clients
  initializeClients();
  
  // Run examples
  await marketDataExample();
  
  console.log('Service client example completed');
}

// Run the example
main().catch(error => {
  console.error('Error running example:', error);
});