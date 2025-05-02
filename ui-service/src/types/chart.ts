/**
 * Chart Types for Trading Platform
 * 
 * Type definitions for chart components, including support for confluence visualization.
 */
import { Time } from 'lightweight-charts';

/**
 * Types of confluence points that can be highlighted on charts
 */
export type ConfluenceType = 
  | 'support' 
  | 'resistance' 
  | 'harmonic_pattern'
  | 'ma_confluence'
  | 'fibonacci_confluence'
  | 'multi_timeframe'
  | 'indicator_confluence'
  | 'pattern_completion'
  | 'volume_profile'
  | 'order_flow'
  | 'volatility_contraction'
  | 'liquidity_zone'
  | 'smart_money_concept';

/**
 * A confluence point to be highlighted on the chart
 */
export interface ConfluencePoint {
  time: Time;             // Timestamp for the confluence point
  price: number;          // Price level of the confluence
  type: ConfluenceType;   // Type of confluence
  strength: number;       // Strength from 0.0 to 1.0
  description?: string;   // Optional description
  sources?: string[];     // Which indicators/timeframes contributed to this confluence
  id?: string;            // Optional unique ID
  importance?: number;    // Importance rating (0-10)
  duration?: number;      // Expected duration in bars 
  regime?: string;        // Market regime where this confluence is most valid  relatedPatterns?: PatternData[]; // Related pattern data
}

/**
 * Pattern types supported in visualization
 */
export type PatternType = 
  | 'gartley'
  | 'butterfly'
  | 'bat'
  | 'crab'
  | 'shark'
  | 'cypher'
  | 'abcd'
  | 'triangle'
  | 'elliott_wave'
  | 'head_shoulders'
  | 'double_top'
  | 'double_bottom'
  | 'flag'
  | 'wedge';

/**
 * Point in a pattern
 */
export interface PatternPoint {
  time: Time;
  price: number;
  label?: string;
}

/**
 * Label point for patterns
 */
export interface PatternLabelPoint {
  time: Time;
  price: number;
  text: string;
}

/**
 * Pattern data structure for visualization
 */
export interface PatternData {
  id: string;
  type: PatternType;
  points: PatternPoint[];
  completion: number;        // Pattern completion percentage: 0.0 to 1.0
  direction: 'bullish' | 'bearish' | 'neutral';
  projectionPoints?: PatternPoint[];  // Future projection points
  labelPoints?: PatternLabelPoint[];  // Label points (like A,B,C,D)
  confidence?: number;      // Confidence score: 0.0 to 1.0
  annotationText?: string;  // Text to show on pattern
  relatedConfluences?: string[];  // IDs of related confluence points
}

/**
 * Candle data structure compatible with lightweight-charts
 */
export interface Candle {
  time: Time;
  open: number;
  high: number;
  close: number;
  low: number;
  volume?: number;
}

/**
 * Data structure for chart visualization
 */
export interface ChartData {
  candles: Candle[];
  indicators?: Record<string, any[]>;
  patterns?: any[];
  confluences?: ConfluencePoint[];
}

/**
 * Chart configuration options
 */
export interface ChartOptions {
  darkMode: boolean;
  showVolume: boolean;
  showGrid: boolean;
  showConfluence: boolean;
  confluenceThreshold: number; // Minimum confluence strength to display (0.0-1.0)
  chartOptions: Record<string, any>; // Additional lightweight-charts options
}

/**
 * Market data timeframes
 */
export enum TimeFrame {
  M1 = '1m',
  M5 = '5m',
  M15 = '15m',
  M30 = '30m',
  H1 = '1h',
  H4 = '4h',
  D1 = '1d',
  W1 = '1w'
}
