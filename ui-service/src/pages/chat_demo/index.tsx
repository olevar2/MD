import React from 'react';
import { Container, Box, Typography, Paper } from '@mui/material';
import { ChatInterface } from '../../components/chat';

/**
 * Demo page for the Chat Interface
 */
const ChatDemoPage: React.FC = () => {
  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Forex Trading Platform - Chat Interface Demo
        </Typography>
        
        <Paper sx={{ p: 2, mb: 2 }}>
          <Typography variant="body1" paragraph>
            This is a demonstration of the chat interface for the Forex Trading Platform.
            You can interact with the trading assistant by typing messages in the chat box below.
          </Typography>
          
          <Typography variant="body1" paragraph>
            Try asking about market analysis, requesting charts, or simulating trading actions.
          </Typography>
          
          <Typography variant="subtitle1" gutterBottom>
            Example commands:
          </Typography>
          
          <ul>
            <li>Show me a chart for EURUSD</li>
            <li>What's your analysis of GBPUSD?</li>
            <li>I want to buy USDJPY</li>
            <li>Explain RSI indicator</li>
          </ul>
        </Paper>
        
        <Box sx={{ height: '600px' }}>
          <ChatInterface 
            height="100%"
            serviceConfig={{
              baseUrl: '/api/v1/chat',
              defaultContext: {
                currentSymbol: 'EURUSD',
                currentTimeframe: '1h'
              }
            }}
          />
        </Box>
      </Box>
    </Container>
  );
};

export default ChatDemoPage;
