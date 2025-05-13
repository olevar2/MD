# Chat Interface Integration Guide

This guide provides detailed instructions for integrating the chat interface into the Forex Trading Platform.

## Overview

The chat interface consists of:

1. **Frontend Components**: React components for the UI
2. **Backend Service**: Python service for processing messages
3. **API Endpoints**: FastAPI endpoints for communication

## Integration Steps

### 1. Frontend Integration

#### 1.1. Add Components to UI Service

Copy the frontend components to the UI service:

```bash
mkdir -p ui-service/src/components/chat
cp chat_interface_template/ChatInterface.tsx ui-service/src/components/chat/
cp chat_interface_template/ChatService.ts ui-service/src/components/chat/
cp chat_interface_template/types.ts ui-service/src/components/chat/
```

#### 1.2. Update Package Dependencies

Ensure the UI service has the necessary dependencies:

```bash
cd ui-service
npm install @mui/icons-material @mui/material
```

#### 1.3. Add Chat Interface to Dashboard

Modify the dashboard component to include the chat interface:

```tsx
// ui-service/src/pages/dashboard/index.tsx
import React from 'react';
import dynamic from 'next/dynamic';
import { Grid, Paper, Box } from '@mui/material';
import { styled } from '@mui/material/styles';
import DashboardLayout from '../../components/layout/DashboardLayout';

// Dynamically import heavy components to improve initial load time
const ChartComponent = dynamic(() => import('../../components/trading/Chart'), { ssr: false });
const ActiveTradesPanel = dynamic(() => import('../../components/trading/ActiveTradesPanel'));
const MarketRegimeIndicator = dynamic(() => import('../../components/trading/MarketRegimeIndicator'));
const TradingSignals = dynamic(() => import('../../components/trading/TradingSignals'));
const ChatInterface = dynamic(() => import('../../components/chat/ChatInterface'), { ssr: false });

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
}));

const Dashboard = () => {
  return (
    <DashboardLayout>
      <Grid container spacing={2}>
        {/* Main Chart Section */}
        <Grid item xs={12} lg={8}>
          <StyledPaper>
            <ChartComponent />
          </StyledPaper>
        </Grid>

        {/* Active Trades and Performance Metrics */}
        <Grid item xs={12} lg={4}>
          <StyledPaper>
            <ActiveTradesPanel />
          </StyledPaper>
        </Grid>

        {/* Market Regime and Signal Analysis */}
        <Grid item xs={12} md={6}>
          <StyledPaper>
            <MarketRegimeIndicator />
          </StyledPaper>
        </Grid>

        {/* Trading Signals and Alerts */}
        <Grid item xs={12} md={6}>
          <StyledPaper>
            <TradingSignals />
          </StyledPaper>
        </Grid>
        
        {/* Chat Interface */}
        <Grid item xs={12}>
          <StyledPaper>
            <ChatInterface 
              height="400px"
              serviceConfig={{
                baseUrl: '/api/v1/chat',
                defaultContext: {
                  currentSymbol: 'EURUSD',
                  currentTimeframe: '1h'
                }
              }}
            />
          </StyledPaper>
        </Grid>
      </Grid>
    </DashboardLayout>
  );
};

export default Dashboard;
```

### 2. Backend Integration

#### 2.1. Add Backend Service to ML Integration Service

Copy the backend service to the ML integration service:

```bash
mkdir -p ml-integration-service/ml_integration_service/chat
cp chat_interface_template/chat_backend_service.py ml-integration-service/ml_integration_service/chat/
cp chat_interface_template/api_endpoints.py ml-integration-service/ml_integration_service/chat/
```

#### 2.2. Create `__init__.py` File

Create an `__init__.py` file to make the module importable:

```bash
echo "from .ChatBackendService import ChatBackendService" > ml-integration-service/ml_integration_service/chat/__init__.py
```

#### 2.3. Update Main Application

Modify the main application to include the chat API endpoints:

```python
# ml-integration-service/ml_integration_service/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import API routers
from ml_integration_service.api.v1 import (
    model_registry_router,
    model_training_router,
    model_serving_router,
    model_monitoring_router
)

# Import chat routes
from ml_integration_service.chat.api_endpoints import setup_chat_routes

# Create FastAPI app
app = FastAPI(
    title="ML Integration Service",
    description="Service for integrating ML models with the trading platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(model_registry_router)
app.include_router(model_training_router)
app.include_router(model_serving_router)
app.include_router(model_monitoring_router)

# Setup chat routes
setup_chat_routes(app)

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "service": "ml-integration-service"}
```

### 3. NLP Integration

#### 3.1. Enhance NLP Capabilities

If the platform already has NLP components, integrate them with the chat service:

```python
# ml-integration-service/ml_integration_service/chat/chat_backend_service.py

# Update the __init__ method to use existing NLP components
def __init__(self, config: Dict[str, Any] = None):
    """
    Initialize the chat backend service.
    
    Args:
        config: Configuration parameters
    """
    self.config = config or {}
    self.message_history = {}  # User ID -> List of messages
    
    # Initialize NLP components
    try:
        from analysis_engine.analysis.nlp import NewsAnalyzer
        self.nlp_processor = NewsAnalyzer()
        logger.info("NLP processor initialized with NewsAnalyzer")
    except ImportError:
        logger.warning("NewsAnalyzer not available, falling back to basic NLP")
        try:
            from analysis_engine.analysis.nlp import BaseNLPAnalyzer
            self.nlp_processor = BaseNLPAnalyzer()
            logger.info("NLP processor initialized with BaseNLPAnalyzer")
        except ImportError:
            logger.warning("NLP processor not available")
            self.nlp_processor = None
```

#### 3.2. Add Intent Recognition

Enhance the intent recognition capabilities:

```python
# ml-integration-service/ml_integration_service/chat/intent_recognition.py
"""
Intent Recognition for Chat Interface

This module provides intent recognition capabilities for the chat interface.
"""

from typing import Dict, List, Any, Tuple
import re

def recognize_intent(message: str, entities: List[Dict[str, Any]]) -> Tuple[str, float]:
    """
    Recognize the intent of a message.
    
    Args:
        message: User message
        entities: Extracted entities
        
    Returns:
        Tuple of (intent, confidence)
    """
    intents = {
        "trading": 0.0,
        "analysis": 0.0,
        "chart": 0.0,
        "information": 0.0,
        "general": 0.0
    }
    
    # Trading intent patterns
    trading_patterns = [
        r'\b(buy|sell|trade|order|position|entry|exit|stop loss|take profit)\b',
        r'\b(long|short|market order|limit order|open|close)\b'
    ]
    
    # Analysis intent patterns
    analysis_patterns = [
        r'\b(analyze|analysis|predict|forecast|trend|outlook|sentiment)\b',
        r'\b(technical|fundamental|indicator|oscillator|moving average|rsi|macd)\b'
    ]
    
    # Chart intent patterns
    chart_patterns = [
        r'\b(chart|graph|plot|visualization|candlestick|line chart|bar chart)\b',
        r'\b(timeframe|period|interval|daily|hourly|weekly|monthly)\b'
    ]
    
    # Information intent patterns
    information_patterns = [
        r'\b(what|how|explain|tell me|show me|info|information|details)\b',
        r'\b(why|when|where|who|which|describe|definition|meaning)\b'
    ]
    
    # Check trading patterns
    for pattern in trading_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            intents["trading"] += 0.3
    
    # Check analysis patterns
    for pattern in analysis_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            intents["analysis"] += 0.3
    
    # Check chart patterns
    for pattern in chart_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            intents["chart"] += 0.3
    
    # Check information patterns
    for pattern in information_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            intents["information"] += 0.3
    
    # Check entities
    for entity in entities:
        if entity["label"] in ["CURRENCY_PAIR", "PRICE", "AMOUNT"]:
            intents["trading"] += 0.1
        elif entity["label"] in ["INDICATOR", "TIMEFRAME", "PATTERN"]:
            intents["analysis"] += 0.1
            intents["chart"] += 0.1
    
    # Ensure general intent has a minimum value
    intents["general"] = 0.1
    
    # Get the intent with the highest confidence
    max_intent = max(intents.items(), key=lambda x: x[1])
    
    return max_intent
```

### 4. ML Model Integration

#### 4.1. Connect to ML Models

Enhance the chat service to use ML models for responses:

```python
# ml-integration-service/ml_integration_service/chat/ml_integration.py
"""
ML Integration for Chat Interface

This module provides integration with ML models for the chat interface.
"""

from typing import Dict, List, Any, Optional
import logging
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("chat-ml-integration")

class ChatMLIntegration:
    """Integration with ML models for the chat interface."""
    
    def __init__(self):
        """Initialize the ML integration."""
        self.ml_client = None
        try:
            from ml_integration_service.clients import get_ml_workbench_client
            self.ml_client = get_ml_workbench_client()
            logger.info("ML client initialized")
        except ImportError:
            logger.warning("ML client not available")
    
    async def get_market_analysis(
        self, 
        symbol: str, 
        timeframe: str
    ) -> Dict[str, Any]:
        """
        Get market analysis from ML models.
        
        Args:
            symbol: Currency pair symbol
            timeframe: Timeframe
            
        Returns:
            Analysis results
        """
        if not self.ml_client:
            logger.warning("ML client not available")
            return self._get_default_analysis(symbol, timeframe)
        
        try:
            # This would be replaced with actual ML client call
            # analysis = await self.ml_client.get_analysis(symbol, timeframe)
            
            # Simulate ML analysis
            await asyncio.sleep(1)  # Simulate processing time
            
            return self._get_default_analysis(symbol, timeframe)
        except Exception as e:
            logger.error(f"Error getting analysis from ML client: {str(e)}")
            return self._get_default_analysis(symbol, timeframe)
    
    def _get_default_analysis(
        self, 
        symbol: str, 
        timeframe: str
    ) -> Dict[str, Any]:
        """
        Get default analysis when ML client is not available.
        
        Args:
            symbol: Currency pair symbol
            timeframe: Timeframe
            
        Returns:
            Default analysis
        """
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "trend": "bullish",
            "confidence": 0.65,
            "indicators": {
                "rsi": 65,
                "macd": "bullish crossover",
                "moving_averages": "bullish"
            },
            "support_levels": [1.0800, 1.0750],
            "resistance_levels": [1.0850, 1.0900],
            "recommendation": "buy",
            "risk_reward_ratio": 2.5
        }
```

#### 4.2. Update Chat Backend Service

Update the chat backend service to use the ML integration:

```python
# ml-integration-service/ml_integration_service/chat/chat_backend_service.py

# Add import
from ml_integration_service.chat.ml_integration import ChatMLIntegration

# Update __init__ method
def __init__(self, config: Dict[str, Any] = None):
    # ... existing code ...
    
    # Initialize ML integration
    self.ml_integration = ChatMLIntegration()

# Update _handle_analysis_intent method
async def _handle_analysis_intent(
    self, 
    message: str, 
    entities: List[Dict[str, Any]], 
    context: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Handle analysis intent.
    
    Args:
        message: User message
        entities: Extracted entities
        context: Message context
        
    Returns:
        Response message
    """
    # Extract analysis parameters
    symbol = self._extract_symbol(message, entities, context)
    timeframe = self._extract_timeframe(message, entities, context)
    
    # Get analysis from ML integration
    analysis = await self.ml_integration.get_market_analysis(symbol, timeframe)
    
    # Generate response text
    response_text = f"Based on my analysis of recent market data, {symbol} is showing a {analysis['trend']} trend on the {timeframe} timeframe. "
    response_text += f"The RSI indicator is at {analysis['indicators']['rsi']}, suggesting "
    
    if analysis['indicators']['rsi'] > 70:
        response_text += "overbought conditions, "
    elif analysis['indicators']['rsi'] < 30:
        response_text += "oversold conditions, "
    elif analysis['indicators']['rsi'] > 50:
        response_text += "moderate bullish momentum, "
    else:
        response_text += "moderate bearish momentum, "
    
    response_text += f"while the MACD is showing a {analysis['indicators']['macd']}. "
    
    if analysis['support_levels'] and analysis['resistance_levels']:
        response_text += f"Key support levels are at {', '.join([str(level) for level in analysis['support_levels']])} "
        response_text += f"and resistance levels at {', '.join([str(level) for level in analysis['resistance_levels']])}."
    
    return {
        "text": response_text
    }
```

### 5. Testing

#### 5.1. Create Test Script

Create a test script to verify the chat interface:

```python
# ml-integration-service/tests/chat/test_chat_service.py
"""
Tests for the Chat Backend Service
"""

import pytest
import asyncio
from ml_integration_service.chat import ChatBackendService

@pytest.fixture
def chat_service():
    """Create a chat service for testing."""
    return ChatBackendService()

@pytest.mark.asyncio
async def test_process_message(chat_service):
    """Test processing a message."""
    user_id = "test_user"
    message = "Show me a chart for EURUSD"
    context = {"currentSymbol": "EURUSD", "currentTimeframe": "1h"}
    
    response = await chat_service.process_message(user_id, message, context)
    
    assert response is not None
    assert "text" in response
    assert "EURUSD" in response["text"]
    assert "chartData" in response

@pytest.mark.asyncio
async def test_trading_intent(chat_service):
    """Test processing a trading intent message."""
    user_id = "test_user"
    message = "Buy EURUSD at market price"
    
    response = await chat_service.process_message(user_id, message, {})
    
    assert response is not None
    assert "text" in response
    assert "buy" in response["text"].lower()
    assert "EURUSD" in response["text"]
    assert "tradingAction" in response
    assert response["tradingAction"]["type"] == "buy"
    assert response["tradingAction"]["symbol"] == "EURUSD"

@pytest.mark.asyncio
async def test_analysis_intent(chat_service):
    """Test processing an analysis intent message."""
    user_id = "test_user"
    message = "Analyze GBPUSD on the 4h timeframe"
    
    response = await chat_service.process_message(user_id, message, {})
    
    assert response is not None
    assert "text" in response
    assert "GBPUSD" in response["text"]
    assert "4h" in response["text"] or "4-hour" in response["text"]

@pytest.mark.asyncio
async def test_chat_history(chat_service):
    """Test chat history management."""
    user_id = "test_user"
    
    # Clear history
    chat_service.clear_chat_history(user_id)
    
    # Send a message
    await chat_service.process_message(user_id, "Hello", {})
    
    # Get history
    history = chat_service.get_chat_history(user_id)
    
    assert len(history) == 2  # User message and assistant response
    assert history[0]["sender"] == "user"
    assert history[1]["sender"] == "assistant"
```

#### 5.2. Run Tests

Run the tests to verify the chat interface:

```bash
cd ml-integration-service
pytest tests/chat/test_chat_service.py -v
```

### 6. Documentation

#### 6.1. Update Service Documentation

Update the ML integration service documentation to include the chat interface:

```markdown
# ML Integration Service

This service acts as a bridge between the core trading platform components and the machine learning model development and execution environments.

## Features

- Feature extraction for ML models
- Model interaction and prediction
- Chat interface for interactive communication
- ...

## Chat Interface

The chat interface allows users to interact with the platform's AI capabilities through natural language. It supports:

- Trading actions through chat
- Market analysis requests
- Chart visualization
- Information queries

### API Endpoints

- `POST /api/v1/chat/message`: Process user messages
- `POST /api/v1/chat/execute-action`: Execute trading actions
- `GET /api/v1/chat/history`: Retrieve chat history
- `DELETE /api/v1/chat/history`: Clear chat history
```

#### 6.2. Add User Guide

Create a user guide for the chat interface:

```markdown
# Chat Interface User Guide

The chat interface allows you to interact with the Forex Trading Platform using natural language. You can ask questions, request analysis, view charts, and execute trades through the chat.

## Getting Started

To start using the chat interface, simply type your message in the input field at the bottom of the chat window and press Enter or click the Send button.

## Example Commands

Here are some examples of what you can do with the chat interface:

### Trading

- "Buy EURUSD at market price"
- "Sell 0.1 lots of GBPUSD"
- "Close my USDJPY position"

### Analysis

- "Analyze EURUSD on the 4h timeframe"
- "What's your prediction for GBPUSD today?"
- "Show me the trend for USDJPY"

### Charts

- "Show me a chart for EURUSD"
- "Display the 1h chart for GBPUSD with RSI"
- "Can I see the daily chart for USDJPY?"

### Information

- "What is RSI?"
- "Explain the MACD indicator"
- "How does a moving average work?"

## Tips

- Be specific about the currency pair and timeframe when requesting analysis or charts
- You can ask follow-up questions to get more details
- The chat interface will remember the context of your conversation
```

## Conclusion

This integration guide provides a comprehensive approach to adding a chat interface to the Forex Trading Platform. By following these steps, you can enhance the platform with interactive communication capabilities, allowing users to interact with the platform's AI features through natural language.
