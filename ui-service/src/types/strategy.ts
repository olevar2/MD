/**
 * Type definitions for strategy management
 */

export enum MarketRegime {
  TRENDING = 'trending',
  RANGING = 'ranging',
  VOLATILE = 'volatile',
  BREAKOUT = 'breakout'
}

export enum TimeFrame {
  M1 = '1m',
  M5 = '5m',
  M15 = '15m',
  M30 = '30m',
  H1 = '1h',
  H4 = '4h',
  D1 = 'daily',
  W1 = 'weekly'
}

export interface StrategyParameter {
  id: string;
  name: string;
  description: string;
  type: 'number' | 'boolean' | 'string' | 'select';
  value: any;
  defaultValue: any;
  min?: number;
  max?: number;
  step?: number;
  options?: Array<{value: any, label: string}>;
  category: 'main' | 'advanced' | 'risk' | 'adaptive';
  regimeSpecific?: boolean;
}

export interface IndicatorConfig {
  id: string;
  type: string;
  name: string;
  parameters: Record<string, any>;
  visible: boolean;
  color?: string;
}

export interface StrategyPerformanceMetrics {
  winRate: number;
  profitFactor: number;
  totalTrades: number;
  avgProfit: number;
  avgLoss: number;
  expectancy: number;
  maxDrawdown: number;
  sharpeRatio?: number;
  regimePerformance?: Record<MarketRegime, {
    winRate: number;
    profitFactor: number;
    totalTrades: number;
  }>;
}

export interface BacktestResult {
  id: string;
  strategyId: string;
  startDate: string;
  endDate: string;
  symbol: string;
  initialBalance: number;
  finalBalance: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  metrics: StrategyPerformanceMetrics;
  trades: Array<TradeRecord>;
  equityCurve: Array<{timestamp: string, equity: number}>;
  parameters: Record<string, any>;
}

export interface TradeRecord {
  id: string;
  strategyId: string;
  symbol: string;
  direction: 'buy' | 'sell';
  entryPrice: number;
  entryTime: string;
  exitPrice?: number;
  exitTime?: string;
  stopLoss: number;
  takeProfit: number;
  quantity: number;
  profit?: number;
  pips?: number;
  status: 'open' | 'closed' | 'cancelled';
  result?: 'win' | 'loss';
  marketRegime: MarketRegime;
  timeframe: TimeFrame;
  reason?: string;
}

export interface Strategy {
  id: string;
  name: string;
  description: string;
  type: 'adaptive_ma' | 'elliott_wave' | 'multi_timeframe_confluence' | 'harmonic_pattern' | 'advanced_breakout' | 'custom';
  symbols: string[];
  timeframes: TimeFrame[];
  primaryTimeframe: TimeFrame;
  parameters: Record<string, any>;
  parameterTemplates: StrategyParameter[];
  indicators: IndicatorConfig[];
  isActive: boolean;
  createdAt: string;
  updatedAt: string;
  performance?: StrategyPerformanceMetrics;
  regimeParameters?: Record<MarketRegime, Record<string, any>>;
}

export interface StrategyTemplate {
  id: string;
  name: string;
  description: string;
  type: string;
  parameterTemplates: StrategyParameter[];
  defaultIndicators: IndicatorConfig[];
}
