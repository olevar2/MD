import React from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  Avatar, 
  CircularProgress,
  IconButton,
  Tooltip,
  useTheme
} from '@mui/material';
import PersonIcon from '@mui/icons-material/Person';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';
import { ChatMessageProps } from './types';

/**
 * ChatMessage component displays a single message in the chat interface
 */
const ChatMessage: React.FC<ChatMessageProps> = ({ 
  message,
  onActionClick
}) => {
  const theme = useTheme();
  const isUser = message.sender === 'user';

  // Render message content based on type
  const renderMessageContent = () => {
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
                onClick={() => onActionClick && onActionClick(message.tradingAction!)}
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

    // Default text message
    return <Typography variant="body1">{message.text}</Typography>;
  };

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: isUser ? 'row-reverse' : 'row',
        mb: 2
      }}
    >
      <Avatar 
        sx={{ 
          bgcolor: isUser 
            ? theme.palette.secondary.main 
            : theme.palette.primary.main,
          width: 36,
          height: 36
        }}
      >
        {isUser ? <PersonIcon /> : <SmartToyIcon />}
      </Avatar>
      <Paper
        sx={{
          p: 2,
          ml: isUser ? 0 : 1,
          mr: isUser ? 1 : 0,
          maxWidth: '70%',
          bgcolor: isUser 
            ? theme.palette.secondary.light 
            : theme.palette.background.paper,
          borderRadius: 2
        }}
      >
        {renderMessageContent()}
        <Typography 
          variant="caption" 
          color="text.secondary"
          sx={{ 
            display: 'block', 
            mt: 1, 
            textAlign: isUser ? 'right' : 'left' 
          }}
        >
          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </Typography>
      </Paper>
    </Box>
  );
};

export default ChatMessage;
