/**
 * Types for the Chat Interface
 */

export interface Message {
  id: string;
  text: string;
  sender: 'user' | 'assistant';
  timestamp: Date;
  isLoading?: boolean;
  attachments?: Attachment[];
  tradingAction?: TradingAction;
  chartData?: ChartData;
}

export interface Attachment {
  type: string;
  url: string;
  name: string;
}

export interface TradingAction {
  type: 'buy' | 'sell' | 'close';
  symbol: string;
  amount?: number;
  price?: number;
  stopLoss?: number;
  takeProfit?: number;
  timeframe?: string;
  strategy?: string;
}

export interface ChartData {
  symbol: string;
  timeframe: string;
  data: any; // This would be the actual chart data structure
  annotations?: ChartAnnotation[];
}

export interface ChartAnnotation {
  type: 'support' | 'resistance' | 'trend' | 'pattern';
  startPoint: { time: number; price: number };
  endPoint?: { time: number; price: number };
  label?: string;
  color?: string;
}

export interface ChatContextData {
  currentSymbol?: string;
  currentTimeframe?: string;
  openPositions?: any[];
  accountBalance?: number;
  recentAnalysis?: any;
}

export interface ChatServiceConfig {
  baseUrl: string;
  apiKey?: string;
  defaultContext?: ChatContextData;
}

export interface ChatInterfaceProps {
  initialMessages?: Message[];
  onSendMessage?: (message: string) => Promise<void>;
  onExecuteTradingAction?: (action: TradingAction) => Promise<void>;
  height?: string | number;
  width?: string | number;
  serviceConfig?: ChatServiceConfig;
}
