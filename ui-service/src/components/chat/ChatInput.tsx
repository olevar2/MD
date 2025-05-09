import React, { useState, KeyboardEvent } from 'react';
import { 
  Box, 
  TextField, 
  IconButton,
  Tooltip
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import AttachFileIcon from '@mui/icons-material/AttachFile';
import { ChatInputProps } from './types';

/**
 * ChatInput component for user message entry
 */
const ChatInput: React.FC<ChatInputProps> = ({ 
  onSendMessage, 
  isLoading,
  placeholder = "Type a message..."
}) => {
  const [inputValue, setInputValue] = useState('');

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  const handleKeyPress = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleSendMessage = () => {
    if (inputValue.trim() === '' || isLoading) return;
    
    onSendMessage(inputValue);
    setInputValue('');
  };

  return (
    <Box 
      sx={{ 
        p: 2, 
        borderTop: 1, 
        borderColor: 'divider',
        bgcolor: 'background.paper'
      }}
    >
      <Box 
        sx={{ 
          display: 'flex', 
          alignItems: 'center' 
        }}
      >
        <Tooltip title="Attach file">
          <IconButton color="primary" disabled={isLoading}>
            <AttachFileIcon />
          </IconButton>
        </Tooltip>
        
        <TextField
          fullWidth
          variant="outlined"
          placeholder={placeholder}
          value={inputValue}
          onChange={handleInputChange}
          onKeyPress={handleKeyPress}
          size="small"
          disabled={isLoading}
          autoComplete="off"
          sx={{ mx: 1 }}
        />
        
        <Tooltip title="Send message">
          <span>
            <IconButton 
              color="primary" 
              onClick={handleSendMessage}
              disabled={isLoading || inputValue.trim() === ''}
            >
              <SendIcon />
            </IconButton>
          </span>
        </Tooltip>
      </Box>
    </Box>
  );
};

export default ChatInput;
