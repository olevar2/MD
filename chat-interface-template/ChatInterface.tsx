import React, { useState, useRef, useEffect } from 'react';
import { 
  Box, 
  Paper, 
  TextField, 
  IconButton, 
  Typography, 
  Avatar, 
  CircularProgress,
  Divider,
  Tooltip,
  useTheme
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import AttachFileIcon from '@mui/icons-material/AttachFile';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import PersonIcon from '@mui/icons-material/Person';

// Types
interface Message {
  id: string;
  text: string;
  sender: 'user' | 'assistant';
  timestamp: Date;
  isLoading?: boolean;
  attachments?: Array<{
    type: string;
    url: string;
    name: string;
  }>;
  tradingAction?: {
    type: 'buy' | 'sell' | 'close';
    symbol: string;
    amount?: number;
    price?: number;
  };
  chartData?: any;
}

interface ChatInterfaceProps {
  initialMessages?: Message[];
  onSendMessage?: (message: string) => Promise<void>;
  onExecuteTradingAction?: (action: any) => Promise<void>;
  height?: string | number;
  width?: string | number;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  initialMessages = [],
  onSendMessage,
  onExecuteTradingAction,
  height = '600px',
  width = '100%'
}) => {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const theme = useTheme();

  // Scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleSendMessage = async () => {
    if (inputValue.trim() === '' || isLoading) return;

    const newMessage: Message = {
      id: Date.now().toString(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, newMessage]);
    setInputValue('');
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
      if (onSendMessage) {
        await onSendMessage(inputValue);
      }

      // This would be replaced with the actual response from the backend
      // Simulating a response for demonstration purposes
      const response = await simulateResponse(inputValue);

      // Remove loading message and add response
      setMessages(prev => 
        prev.filter(msg => msg.id !== loadingMessage.id).concat(response)
      );
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

  // Simulate a response from the AI assistant
  const simulateResponse = async (userMessage: string): Promise<Message> => {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 1500));

    // Simple pattern matching for demo purposes
    const lowerMessage = userMessage.toLowerCase();
    
    if (lowerMessage.includes('buy') || lowerMessage.includes('sell')) {
      // Trading action response
      const isBuy = lowerMessage.includes('buy');
      const symbol = extractSymbol(lowerMessage) || 'EURUSD';
      
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
      return {
        id: Date.now().toString(),
        text: `Here's the chart you requested. I've highlighted some key support and resistance levels.`,
        sender: 'assistant',
        timestamp: new Date(),
        chartData: { /* This would contain chart data */ }
      };
    } else if (lowerMessage.includes('analysis') || lowerMessage.includes('predict')) {
      // Analysis response
      return {
        id: Date.now().toString(),
        text: `Based on my analysis of recent market data, EURUSD is showing a bullish trend on the 4-hour timeframe. The RSI indicator is at 65, suggesting moderate bullish momentum, while the MACD is showing a recent crossover. Key resistance levels are at 1.0850 and 1.0900.`,
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
  };

  // Extract symbol from message (simplified)
  const extractSymbol = (message: string): string | null => {
    const commonPairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'];
    for (const pair of commonPairs) {
      if (message.toUpperCase().includes(pair)) {
        return pair;
      }
    }
    return null;
  };

  // Handle trading action
  const handleTradingAction = async (action: any) => {
    if (onExecuteTradingAction) {
      try {
        await onExecuteTradingAction(action);
        
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

  // Render message content based on type
  const renderMessageContent = (message: Message) => {
    if (message.isLoading) {
      return (
        <Box display="flex" alignItems="center">
          <CircularProgress size={16} sx={{ mr: 1 }} />
          <Typography variant="body1">{message.text}</Typography>
        </Box>
      );
    }

    if (message.tradingAction) {
      return (
        <Box>
          <Typography variant="body1" gutterBottom>{message.text}</Typography>
          <Box 
            sx={{ 
              display: 'flex', 
              gap: 1, 
              mt: 1 
            }}
          >
            <Tooltip title="Execute Trade">
              <IconButton 
                color="primary" 
                size="small"
                onClick={() => handleTradingAction(message.tradingAction)}
              >
                <AutoGraphIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
      );
    }

    if (message.chartData) {
      return (
        <Box>
          <Typography variant="body1" gutterBottom>{message.text}</Typography>
          <Paper 
            sx={{ 
              height: 200, 
              width: '100%', 
              bgcolor: 'background.default',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              mt: 1
            }}
          >
            <Typography variant="body2" color="text.secondary">
              [Chart Visualization Would Appear Here]
            </Typography>
          </Paper>
        </Box>
      );
    }

    return <Typography variant="body1">{message.text}</Typography>;
  };

  return (
    <Paper 
      elevation={3} 
      sx={{ 
        height, 
        width, 
        display: 'flex', 
        flexDirection: 'column',
        overflow: 'hidden',
        borderRadius: 2
      }}
    >
      {/* Header */}
      <Box 
        sx={{ 
          p: 2, 
          borderBottom: 1, 
          borderColor: 'divider',
          bgcolor: theme.palette.primary.main,
          color: 'white',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}
      >
        <Box display="flex" alignItems="center">
          <SmartToyIcon sx={{ mr: 1 }} />
          <Typography variant="h6">Trading Assistant</Typography>
        </Box>
        <IconButton color="inherit" size="small">
          <MoreVertIcon />
        </IconButton>
      </Box>

      {/* Messages */}
      <Box 
        sx={{ 
          flexGrow: 1, 
          overflow: 'auto', 
          p: 2,
          bgcolor: theme.palette.background.default
        }}
      >
        {messages.map((message) => (
          <Box
            key={message.id}
            sx={{
              display: 'flex',
              flexDirection: message.sender === 'user' ? 'row-reverse' : 'row',
              mb: 2
            }}
          >
            <Avatar 
              sx={{ 
                bgcolor: message.sender === 'user' 
                  ? theme.palette.secondary.main 
                  : theme.palette.primary.main,
                width: 36,
                height: 36
              }}
            >
              {message.sender === 'user' ? <PersonIcon /> : <SmartToyIcon />}
            </Avatar>
            <Paper
              sx={{
                p: 2,
                ml: message.sender === 'user' ? 0 : 1,
                mr: message.sender === 'user' ? 1 : 0,
                maxWidth: '70%',
                bgcolor: message.sender === 'user' 
                  ? theme.palette.secondary.light 
                  : theme.palette.background.paper,
                borderRadius: 2
              }}
            >
              {renderMessageContent(message)}
              <Typography 
                variant="caption" 
                color="text.secondary"
                sx={{ 
                  display: 'block', 
                  mt: 1, 
                  textAlign: message.sender === 'user' ? 'right' : 'left' 
                }}
              >
                {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </Typography>
            </Paper>
          </Box>
        ))}
        <div ref={messagesEndRef} />
      </Box>

      {/* Input */}
      <Box 
        sx={{ 
          p: 2, 
          borderTop: 1, 
          borderColor: 'divider',
          bgcolor: theme.palette.background.paper
        }}
      >
        <Box 
          sx={{ 
            display: 'flex', 
            alignItems: 'center' 
          }}
        >
          <TextField
            fullWidth
            variant="outlined"
            placeholder="Type a message..."
            value={inputValue}
            onChange={handleInputChange}
            onKeyPress={handleKeyPress}
            size="small"
            disabled={isLoading}
            InputProps={{
              endAdornment: (
                <IconButton 
                  color="primary" 
                  onClick={handleSendMessage}
                  disabled={isLoading || inputValue.trim() === ''}
                >
                  <SendIcon />
                </IconButton>
              ),
              startAdornment: (
                <IconButton color="primary" disabled={isLoading}>
                  <AttachFileIcon />
                </IconButton>
              )
            }}
          />
        </Box>
      </Box>
    </Paper>
  );
};

export default ChatInterface;
