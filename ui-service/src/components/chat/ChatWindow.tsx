import React, { useRef, useEffect } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  IconButton,
  useTheme
} from '@mui/material';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import { ChatWindowProps, TradingAction } from './types';

/**
 * ChatWindow component that combines messages and input
 */
const ChatWindow: React.FC<ChatWindowProps> = ({
  messages,
  isLoading,
  onSendMessage,
  onExecuteTradingAction,
  height = '600px',
  width = '100%'
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const theme = useTheme();

  // Scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleActionClick = async (action: TradingAction) => {
    if (onExecuteTradingAction) {
      await onExecuteTradingAction(action);
    }
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
        {messages.length === 0 ? (
          <Box 
            sx={{ 
              height: '100%', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center' 
            }}
          >
            <Typography color="text.secondary">
              No messages yet. Start a conversation!
            </Typography>
          </Box>
        ) : (
          messages.map((message) => (
            <ChatMessage 
              key={message.id} 
              message={message} 
              onActionClick={handleActionClick}
            />
          ))
        )}
        <div ref={messagesEndRef} />
      </Box>

      {/* Input */}
      <ChatInput 
        onSendMessage={onSendMessage} 
        isLoading={isLoading} 
      />
    </Paper>
  );
};

export default ChatWindow;
