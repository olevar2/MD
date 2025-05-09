/**
 * Market Data Client Example
 * 
 * This module demonstrates how to implement a service client using the standardized template.
 */

import { StandardServiceClient } from '../../common-js-lib/templates/ServiceClientTemplate';
import { ClientConfig } from '../../common-js-lib/index';

/**
 * Interface for OHLCV data
 */
interface OHLCVData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

/**
 * Interface for instrument information
 */
interface Instrument {
  symbol: string;
  name: string;
  category: string;
  pipValue: number;
  minLotSize: number;
  maxLotSize: number;
  tradingHours: string;
  active: boolean;
}

/**
 * Client for interacting with the Market Data Service
 * 
 * This client provides methods for:
 * 1. Retrieving market data (OHLCV, ticks)
 * 2. Getting instrument information
 * 3. Subscribing to market data updates
 */
export class MarketDataClient extends StandardServiceClient<any> {
  /**
   * Initialize the market data client
   * 
   * @param config Client configuration
   */
  constructor(config: ClientConfig) {
    super(config);
  }
  
  /**
   * Get OHLCV (Open, High, Low, Close, Volume) data for a symbol
   * 
   * @param symbol Trading symbol (e.g., 'EUR/USD')
   * @param timeframe Timeframe for the data (e.g., '1m', '1h', '1d')
   * @param startTime Start time for the data
   * @param endTime End time for the data
   * @param limit Maximum number of data points to return
   * @returns OHLCV data
   */
  async getOHLCVData(
    symbol: string,
    timeframe: string,
    startTime?: Date,
    endTime?: Date,
    limit?: number
  ): Promise<{ data: OHLCVData[], meta: any }> {
    this.logger.debug(`Getting OHLCV data for ${symbol} (${timeframe})`);
    
    // Prepare parameters
    const params: Record<string, any> = {
      symbol,
      timeframe
    };
    
    if (startTime) {
      params.start_time = startTime.toISOString();
    }
    
    if (endTime) {
      params.end_time = endTime.toISOString();
    }
    
    if (limit) {
      params.limit = limit;
    }
    
    try {
      return await this.get('market-data/ohlcv', { params });
    } catch (error) {
      this.logger.error(`Failed to get OHLCV data for ${symbol}: ${error}`);
      throw error;
    }
  }
  
  /**
   * Get tick data for a symbol
   * 
   * @param symbol Trading symbol (e.g., 'EUR/USD')
   * @param startTime Start time for the data
   * @param endTime End time for the data
   * @param limit Maximum number of data points to return
   * @returns Tick data
   */
  async getTickData(
    symbol: string,
    startTime?: Date,
    endTime?: Date,
    limit?: number
  ): Promise<{ data: any[], meta: any }> {
    this.logger.debug(`Getting tick data for ${symbol}`);
    
    // Prepare parameters
    const params: Record<string, any> = {
      symbol
    };
    
    if (startTime) {
      params.start_time = startTime.toISOString();
    }
    
    if (endTime) {
      params.end_time = endTime.toISOString();
    }
    
    if (limit) {
      params.limit = limit;
    }
    
    try {
      return await this.get('market-data/ticks', { params });
    } catch (error) {
      this.logger.error(`Failed to get tick data for ${symbol}: ${error}`);
      throw error;
    }
  }
  
  /**
   * Get information about a trading instrument
   * 
   * @param symbol Trading symbol (e.g., 'EUR/USD')
   * @returns Instrument information
   */
  async getInstrument(symbol: string): Promise<Instrument> {
    this.logger.debug(`Getting instrument information for ${symbol}`);
    
    try {
      return await this.get(`market-data/instruments/${symbol}`);
    } catch (error) {
      this.logger.error(`Failed to get instrument information for ${symbol}: ${error}`);
      throw error;
    }
  }
  
  /**
   * List available trading instruments
   * 
   * @param category Filter by instrument category (e.g., 'forex', 'crypto')
   * @param activeOnly Whether to return only active instruments
   * @returns List of instruments
   */
  async listInstruments(
    category?: string,
    activeOnly: boolean = true
  ): Promise<{ data: Instrument[], meta: any }> {
    this.logger.debug(`Listing instruments (category=${category}, activeOnly=${activeOnly})`);
    
    // Prepare parameters
    const params: Record<string, any> = {
      active_only: activeOnly
    };
    
    if (category) {
      params.category = category;
    }
    
    try {
      return await this.get('market-data/instruments', { params });
    } catch (error) {
      this.logger.error(`Failed to list instruments: ${error}`);
      throw error;
    }
  }
  
  /**
   * Get the latest price for a symbol
   * 
   * @param symbol Trading symbol (e.g., 'EUR/USD')
   * @returns Latest price information
   */
  async getLatestPrice(symbol: string): Promise<any> {
    this.logger.debug(`Getting latest price for ${symbol}`);
    
    try {
      return await this.get(`market-data/prices/${symbol}/latest`);
    } catch (error) {
      this.logger.error(`Failed to get latest price for ${symbol}: ${error}`);
      throw error;
    }
  }
}