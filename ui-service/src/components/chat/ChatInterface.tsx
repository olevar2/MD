import React, { useState, useEffect } from 'react';
import ChatWindow from './ChatWindow';
import { Message, TradingAction, ChatInterfaceProps } from './types';
import ChatService from './ChatService';

/**
 * Main ChatInterface component that manages state and communication with backend
 */
const ChatInterface: React.FC<ChatInterfaceProps> = ({
  initialMessages = [],
  onSendMessage: externalSendMessage,
  onExecuteTradingAction: externalExecuteTradingAction,
  height = '600px',
  width = '100%',
  serviceConfig
}) => {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [isLoading, setIsLoading] = useState(false);
  const [chatService] = useState(() => new ChatService(
    serviceConfig?.baseUrl || '/api/v1/chat',
    serviceConfig?.apiKey
  ));

  // Load initial messages if empty
  useEffect(() => {
    if (initialMessages.length === 0) {
      // Add welcome message
      const welcomeMessage: Message = {
        id: 'welcome',
        text: "Hello! I'm your Forex Trading Assistant. I can help you analyze markets, execute trades, and monitor your portfolio. What would you like to do today?",
        sender: 'assistant',
        timestamp: new Date()
      };
      setMessages([welcomeMessage]);
    }
  }, [initialMessages.length]);

  const handleSendMessage = async (text: string) => {
    if (text.trim() === '' || isLoading) return;

    // Create user message
    const userMessage: Message = {
      id: Date.now().toString(),
      text,
      sender: 'user',
      timestamp: new Date()
    };

    // Add user message to chat
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    // Add loading message
    const loadingMessage: Message = {
      id: `loading-${Date.now()}`,
      text: 'Thinking...',
      sender: 'assistant',
      timestamp: new Date(),
      isLoading: true
    };
    setMessages(prev => [...prev, loadingMessage]);

    try {
      // Use external handler if provided
      if (externalSendMessage) {
        await externalSendMessage(text);
      } else {
        // Use chat service
        const context = serviceConfig?.defaultContext || {};
        const response = await chatService.sendMessage(text, context);
        
        // Remove loading message and add response
        setMessages(prev => 
          prev.filter(msg => msg.id !== loadingMessage.id).concat(response)
        );
      }
    } catch (error) {
      console.error('Error sending message:', error);
      
      // Remove loading message and add error message
      setMessages(prev => 
        prev.filter(msg => msg.id !== loadingMessage.id).concat({
          id: Date.now().toString(),
          text: 'Sorry, I encountered an error processing your request. Please try again.',
          sender: 'assistant',
          timestamp: new Date()
        })
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleExecuteTradingAction = async (action: TradingAction) => {
    if (externalExecuteTradingAction) {
      try {
        await externalExecuteTradingAction(action);
      } catch (error) {
        console.error('Error executing trading action:', error);
        
        // Add error message
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          text: `Failed to execute ${action.type} order for ${action.symbol}. Please try again.`,
          sender: 'assistant',
          timestamp: new Date()
        }]);
      }
    } else {
      try {
        // Use chat service
        const result = await chatService.executeTradingAction(action);
        
        // Add confirmation message
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          text: `Successfully executed ${action.type} order for ${action.symbol}.`,
          sender: 'assistant',
          timestamp: new Date()
        }]);
      } catch (error) {
        console.error('Error executing trading action:', error);
        
        // Add error message
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          text: `Failed to execute ${action.type} order for ${action.symbol}. Please try again.`,
          sender: 'assistant',
          timestamp: new Date()
        }]);
      }
    }
  };

  return (
    <ChatWindow
      messages={messages}
      isLoading={isLoading}
      onSendMessage={handleSendMessage}
      onExecuteTradingAction={handleExecuteTradingAction}
      height={height}
      width={width}
    />
  );
};

export default ChatInterface;
