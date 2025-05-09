import { Message, TradingAction } from './types';

/**
 * ChatService handles communication with the backend for the chat interface
 */
class ChatService {
  private baseUrl: string;
  private apiKey: string | null;
  
  constructor(baseUrl: string = '/api/v1/chat', apiKey: string | null = null) {
    this.baseUrl = baseUrl;
    this.apiKey = apiKey;
  }
  
  /**
   * Send a message to the chat service and get a response
   * 
   * @param message The message text to send
   * @param context Optional context information (e.g., current symbol, timeframe)
   * @returns Promise resolving to the assistant's response
   */
  async sendMessage(message: string, context?: Record<string, any>): Promise<Message> {
    try {
      // In a real implementation, this would call the backend API
      // For now, we'll simulate a response
      
      // Simulate network delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Simple pattern matching for demo purposes
      const lowerMessage = message.toLowerCase();
      
      if (lowerMessage.includes('buy') || lowerMessage.includes('sell')) {
        // Trading action response
        const isBuy = lowerMessage.includes('buy');
        const symbol = this.extractSymbol(lowerMessage) || 'EURUSD';
        
        return {
          id: Date.now().toString(),
          text: `I can help you ${isBuy ? 'buy' : 'sell'} ${symbol}. Would you like me to execute this trade for you?`,
          sender: 'assistant',
          timestamp: new Date(),
          tradingAction: {
            type: isBuy ? 'buy' : 'sell',
            symbol: symbol
          }
        };
      } else if (lowerMessage.includes('chart') || lowerMessage.includes('graph')) {
        // Chart data response
        const symbol = this.extractSymbol(lowerMessage) || 'EURUSD';
        const timeframe = this.extractTimeframe(lowerMessage) || '1h';
        
        return {
          id: Date.now().toString(),
          text: `Here's the ${timeframe} chart for ${symbol}. I've highlighted some key support and resistance levels.`,
          sender: 'assistant',
          timestamp: new Date(),
          chartData: {
            symbol,
            timeframe,
            data: {}, // This would contain actual chart data
            annotations: [
              {
                type: 'support',
                startPoint: { time: Date.now() - 86400000, price: 1.0750 },
                label: 'Support'
              },
              {
                type: 'resistance',
                startPoint: { time: Date.now() - 86400000, price: 1.0850 },
                label: 'Resistance'
              }
            ]
          }
        };
      } else if (lowerMessage.includes('analysis') || lowerMessage.includes('predict')) {
        // Analysis response
        const symbol = this.extractSymbol(lowerMessage) || 'EURUSD';
        
        return {
          id: Date.now().toString(),
          text: `Based on my analysis of recent market data, ${symbol} is showing a bullish trend on the 4-hour timeframe. The RSI indicator is at 65, suggesting moderate bullish momentum, while the MACD is showing a recent crossover. Key resistance levels are at 1.0850 and 1.0900.`,
          sender: 'assistant',
          timestamp: new Date()
        };
      } else {
        // Default response
        return {
          id: Date.now().toString(),
          text: `I'm your Forex Trading Assistant. I can help you analyze markets, execute trades, and monitor your portfolio. What would you like to do today?`,
          sender: 'assistant',
          timestamp: new Date()
        };
      }
    } catch (error) {
      console.error('Error in sendMessage:', error);
      throw error;
    }
  }
  
  /**
   * Execute a trading action
   * 
   * @param action The trading action to execute
   * @returns Promise resolving to the result of the action
   */
  async executeTradingAction(action: TradingAction): Promise<any> {
    try {
      // In a real implementation, this would call the backend API
      // For now, we'll simulate a successful execution
      
      // Simulate network delay
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      return {
        success: true,
        orderId: Math.random().toString(36).substring(2, 15),
        message: `Successfully executed ${action.type} order for ${action.symbol}`,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      console.error('Error in executeTradingAction:', error);
      throw error;
    }
  }
  
  /**
   * Get chat history
   * 
   * @param limit Maximum number of messages to retrieve
   * @param before Timestamp to get messages before
   * @returns Promise resolving to an array of messages
   */
  async getChatHistory(limit: number = 50, before?: Date): Promise<Message[]> {
    try {
      // In a real implementation, this would call the backend API
      // For now, we'll return an empty array
      return [];
    } catch (error) {
      console.error('Error in getChatHistory:', error);
      throw error;
    }
  }
  
  /**
   * Clear chat history
   * 
   * @returns Promise resolving when history is cleared
   */
  async clearChatHistory(): Promise<void> {
    try {
      // In a real implementation, this would call the backend API
      return;
    } catch (error) {
      console.error('Error in clearChatHistory:', error);
      throw error;
    }
  }
  
  /**
   * Extract symbol from message
   * 
   * @param message User message
   * @returns Symbol string or null
   */
  private extractSymbol(message: string): string | null {
    const commonPairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'];
    for (const pair of commonPairs) {
      if (message.toUpperCase().includes(pair)) {
        return pair;
      }
    }
    return null;
  }
  
  /**
   * Extract timeframe from message
   * 
   * @param message User message
   * @returns Timeframe string or null
   */
  private extractTimeframe(message: string): string | null {
    const timeframes: Record<string, string[]> = {
      '1m': ['1m', '1 minute', '1min'],
      '5m': ['5m', '5 minute', '5min'],
      '15m': ['15m', '15 minute', '15min'],
      '30m': ['30m', '30 minute', '30min'],
      '1h': ['1h', '1 hour', 'hourly'],
      '4h': ['4h', '4 hour'],
      '1d': ['1d', 'daily', 'day'],
      '1w': ['1w', 'weekly', 'week']
    };
    
    const messageLower = message.toLowerCase();
    for (const [tf, aliases] of Object.entries(timeframes)) {
      if (aliases.some(alias => messageLower.includes(alias))) {
        return tf;
      }
    }
    
    return null;
  }
}

export default ChatService;
