import axios, { AxiosInstance } from 'axios';

export interface TradingRequest {
  symbol: string;
  volume: number;
  price?: number;
  type?: 'MARKET' | 'LIMIT';
  stopLoss?: number;
  takeProfit?: number;
}

export interface TradingResponse {
  orderId: string;
  status: 'PENDING' | 'FILLED' | 'REJECTED';
  message?: string;
  filledPrice?: number;
}

export interface MarketData {
  symbol: string;
  bid: number;
  ask: number;
  timestamp: number;
}

export interface Position {
  id: string;
  symbol: string;
  direction: 'long' | 'short';
  size: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnl: number;
  status: 'OPEN' | 'CLOSED';
}

export class TradingService {
  private readonly api: AxiosInstance;
  
  constructor(baseUrl: string = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001/api') {
    this.api = axios.create({
      baseURL: baseUrl,
      timeout: 5000,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }

  async fetchMarketData(symbols?: string[]): Promise<MarketData[]> {
    try {
      const params = symbols ? { symbols: symbols.join(',') } : undefined;
      const response = await this.api.get<MarketData[]>('/trading/market-data', { params });
      return response.data;
    } catch (error) {
      if (error instanceof Error) {
        console.error('Error fetching market data:', error);
        throw new Error(`Failed to fetch market data: ${error.message}`);
      }
      throw error;
    }
  }

  async buy(request: TradingRequest): Promise<TradingResponse> {
    try {
      const response = await this.api.post<TradingResponse>('/trading/orders', {
        ...request,
        side: 'BUY'
      });
      return response.data;
    } catch (error) {
      if (error instanceof Error) {
        console.error('Error submitting buy order:', error);
        throw new Error(`Failed to submit buy order: ${error.message}`);
      }
      throw error;
    }
  }

  async sell(request: TradingRequest): Promise<TradingResponse> {
    try {
      const response = await this.api.post<TradingResponse>('/trading/orders', {
        ...request,
        side: 'SELL'
      });
      return response.data;
    } catch (error) {
      if (error instanceof Error) {
        console.error('Error submitting sell order:', error);
        throw new Error(`Failed to submit sell order: ${error.message}`);
      }
      throw error;
    }
  }

  async getOrderStatus(orderId: string): Promise<TradingResponse> {
    try {
      const response = await this.api.get<TradingResponse>(`/trading/orders/${orderId}`);
      return response.data;
    } catch (error) {
      if (error instanceof Error) {
        console.error('Error fetching order status:', error);
        throw new Error(`Failed to fetch order status: ${error.message}`);
      }
      throw error;
    }
  }

  async fetchPositions(): Promise<Position[]> {
    try {
      const response = await this.api.get<Position[]>('/trading/positions');
      return response.data;
    } catch (error) {
      if (error instanceof Error) {
        console.error('Error fetching positions:', error);
        throw new Error(`Failed to fetch positions: ${error.message}`);
      }
      throw error;
    }
  }

  async fetchTradeHistory(): Promise<Position[]> {
    try {
      const response = await this.api.get<Position[]>('/trading/history');
      return response.data;
    } catch (error) {
      if (error instanceof Error) {
        console.error('Error fetching trade history:', error);
        throw new Error(`Failed to fetch trade history: ${error.message}`);
      }
      throw error;
    }
  }

  async closePosition(positionId: string): Promise<void> {
    try {
      await this.api.post(`/trading/positions/${positionId}/close`);
    } catch (error) {
      if (error instanceof Error) {
        console.error('Error closing position:', error);
        throw new Error(`Failed to close position: ${error.message}`);
      }
      throw error;
    }
  }

  async updatePositionLevels(
    positionId: string,
    levels: { stopLoss?: number; takeProfit?: number }
  ): Promise<void> {
    try {
      await this.api.patch(`/trading/positions/${positionId}`, levels);
    } catch (error) {
      if (error instanceof Error) {
        console.error('Error updating position levels:', error);
        throw new Error(`Failed to update position levels: ${error.message}`);
      }
      throw error;
    }
  }
}
