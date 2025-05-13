/**
 * Chat Service
 * 
 * This service handles communication with the backend for the chat interface.
 * It manages sending messages, receiving responses, and handling trading actions.
 */

import { Message, TradingAction } from './types';

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
      const response = await fetch(`${this.baseUrl}/message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this.apiKey ? { 'X-API-Key': this.apiKey } : {})
        },
        body: JSON.stringify({
          message,
          context,
          timestamp: new Date().toISOString()
        })
      });
      
      if (!response.ok) {
        throw new Error(`Error sending message: ${response.statusText}`);
      }
      
      const data = await response.json();
      return {
        id: data.id || Date.now().toString(),
        text: data.text,
        sender: 'assistant',
        timestamp: new Date(data.timestamp || Date.now()),
        tradingAction: data.tradingAction,
        chartData: data.chartData,
        attachments: data.attachments
      };
    } catch (error) {
      console.error('Error in sendMessage:', error);
      throw error;
    }
  }
  
  /**
   * Execute a trading action through the chat service
   * 
   * @param action The trading action to execute
   * @returns Promise resolving to the result of the action
   */
  async executeTradingAction(action: TradingAction): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/execute-action`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this.apiKey ? { 'X-API-Key': this.apiKey } : {})
        },
        body: JSON.stringify({
          action,
          timestamp: new Date().toISOString()
        })
      });
      
      if (!response.ok) {
        throw new Error(`Error executing trading action: ${response.statusText}`);
      }
      
      return await response.json();
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
      const url = new URL(`${this.baseUrl}/history`);
      url.searchParams.append('limit', limit.toString());
      if (before) {
        url.searchParams.append('before', before.toISOString());
      }
      
      const response = await fetch(url.toString(), {
        method: 'GET',
        headers: {
          ...(this.apiKey ? { 'X-API-Key': this.apiKey } : {})
        }
      });
      
      if (!response.ok) {
        throw new Error(`Error getting chat history: ${response.statusText}`);
      }
      
      const data = await response.json();
      return data.messages.map((msg: any) => ({
        id: msg.id,
        text: msg.text,
        sender: msg.sender,
        timestamp: new Date(msg.timestamp),
        tradingAction: msg.tradingAction,
        chartData: msg.chartData,
        attachments: msg.attachments
      }));
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
      const response = await fetch(`${this.baseUrl}/history`, {
        method: 'DELETE',
        headers: {
          ...(this.apiKey ? { 'X-API-Key': this.apiKey } : {})
        }
      });
      
      if (!response.ok) {
        throw new Error(`Error clearing chat history: ${response.statusText}`);
      }
    } catch (error) {
      console.error('Error in clearChatHistory:', error);
      throw error;
    }
  }
}

export default ChatService;
