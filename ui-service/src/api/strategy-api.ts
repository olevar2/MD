/**
 * API client for strategy management
 */
import axios from 'axios';
import { Strategy, BacktestResult, TradeRecord } from '@/types/strategy';

// Base API URL would typically come from environment variables
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000/api/v1';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const strategyApi = {
  // Strategy CRUD operations
  getStrategies: async (): Promise<Strategy[]> => {
    const response = await apiClient.get('/strategies');
    return response.data;
  },

  getStrategy: async (id: string): Promise<Strategy> => {
    const response = await apiClient.get(`/strategies/${id}`);
    return response.data;
  },

  createStrategy: async (strategy: Partial<Strategy>): Promise<Strategy> => {
    const response = await apiClient.post('/strategies', strategy);
    return response.data;
  },

  updateStrategy: async (id: string, strategy: Partial<Strategy>): Promise<Strategy> => {
    const response = await apiClient.put(`/strategies/${id}`, strategy);
    return response.data;
  },

  deleteStrategy: async (id: string): Promise<void> => {
    await apiClient.delete(`/strategies/${id}`);
  },

  // Strategy activation/deactivation
  activateStrategy: async (id: string): Promise<Strategy> => {
    const response = await apiClient.post(`/strategies/${id}/activate`);
    return response.data;
  },

  deactivateStrategy: async (id: string): Promise<Strategy> => {
    const response = await apiClient.post(`/strategies/${id}/deactivate`);
    return response.data;
  },

  // Backtesting
  runBacktest: async (id: string, params: {
    startDate: string;
    endDate: string;
    symbol: string;
    initialBalance?: number;
  }): Promise<BacktestResult> => {
    const response = await apiClient.post(`/strategies/${id}/backtest`, params);
    return response.data;
  },

  getBacktestResults: async (strategyId: string): Promise<BacktestResult[]> => {
    const response = await apiClient.get(`/strategies/${strategyId}/backtests`);
    return response.data;
  },

  getBacktestResult: async (backtestId: string): Promise<BacktestResult> => {
    const response = await apiClient.get(`/backtests/${backtestId}`);
    return response.data;
  },

  // Strategy templates
  getStrategyTemplates: async (): Promise<any[]> => {
    const response = await apiClient.get('/strategy-templates');
    return response.data;
  },

  // Performance analysis
  getStrategyPerformance: async (id: string, params?: {
    period?: string,
    symbol?: string,
    timeframe?: string
  }): Promise<any> => {
    const response = await apiClient.get(`/strategies/${id}/performance`, { params });
    return response.data;
  },

  // Trade history
  getStrategyTrades: async (id: string, params?: {
    limit?: number,
    offset?: number,
    status?: string
  }): Promise<TradeRecord[]> => {
    const response = await apiClient.get(`/strategies/${id}/trades`, { params });
    return response.data;
  },

  // Optimization
  optimizeStrategyParameters: async (id: string, params: {
    symbol: string,
    timeframe: string,
    targetMetric?: string,
    parameterRanges?: Record<string, any>
  }): Promise<any> => {
    const response = await apiClient.post(`/strategies/${id}/optimize`, params);
    return response.data;
  },

  // Market analysis
  getMarketRegime: async (params: {
    symbol: string,
    timeframe: string
  }): Promise<{
    regime: string,
    confidence: number,
    timestamp: string
  }> => {
    const response = await apiClient.get('/market/regime', { params });
    return response.data;
  }
};
