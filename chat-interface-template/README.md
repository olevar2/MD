# Chat Interface for Forex Trading Platform

This directory contains a template implementation for a chat interface that can be integrated into the Forex Trading Platform. The chat interface allows users to interact with the platform's AI capabilities through natural language.

## Components

- **ChatInterface.tsx**: The main React component for the chat interface
- **ChatService.ts**: Service for handling communication with the backend
- **types.ts**: TypeScript type definitions for the chat interface

## Features

- Interactive chat interface with user and assistant messages
- Support for trading actions through chat
- Chart visualization capabilities
- Message history management
- Responsive design

## Integration Guide

### 1. Add the Chat Interface to the UI Service

Copy the files in this directory to the UI service:

```
cp -r chat_interface_template/* ui-service/src/components/chat/
```

### 2. Create Backend API Endpoints

Create the necessary API endpoints in the appropriate service (e.g., ml-integration-service):

- `POST /api/v1/chat/message`: Process user messages and generate responses
- `POST /api/v1/chat/execute-action`: Execute trading actions
- `GET /api/v1/chat/history`: Retrieve chat history
- `DELETE /api/v1/chat/history`: Clear chat history

### 3. Add the Chat Interface to the Dashboard

Import and use the ChatInterface component in the dashboard:

```tsx
import ChatInterface from '../components/chat/ChatInterface';
import ChatService from '../components/chat/ChatService';

// In your dashboard component
const chatService = new ChatService('/api/v1/chat');

// In your render method
<ChatInterface 
  height="600px"
  width="100%"
  serviceConfig={{
    baseUrl: '/api/v1/chat',
    defaultContext: {
      currentSymbol: 'EURUSD',
      currentTimeframe: '1h'
    }
  }}
/>
```

### 4. Implement NLP Capabilities

Enhance the backend with NLP capabilities to understand user intents and entities:

- Intent recognition (e.g., trading, analysis, information)
- Entity extraction (e.g., currency pairs, timeframes, indicators)
- Context management for multi-turn conversations

### 5. Connect to ML Models

Integrate the chat interface with ML models to provide intelligent responses:

- Market analysis based on current conditions
- Trading suggestions based on user preferences
- Performance insights based on historical data

## Customization

The chat interface can be customized in several ways:

- **Theme**: Adjust the styling to match the platform's theme
- **Features**: Add or remove features based on requirements
- **Integration**: Connect to different backend services as needed

## Example Usage

```tsx
// Basic usage
<ChatInterface />

// With custom height and width
<ChatInterface height="800px" width="400px" />

// With custom message handlers
<ChatInterface 
  onSendMessage={handleSendMessage}
  onExecuteTradingAction={handleTradingAction}
/>

// With initial messages
<ChatInterface 
  initialMessages={[
    {
      id: '1',
      text: 'Welcome to the Forex Trading Platform!',
      sender: 'assistant',
      timestamp: new Date()
    }
  ]}
/>
```
