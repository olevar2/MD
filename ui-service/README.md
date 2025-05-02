# UI Service

## Overview
The UI Service provides the frontend interface for the Forex Trading Platform. It offers a modern, responsive web application for traders, analysts, and administrators to interact with the platform's features, monitor market data, execute trades, and analyze performance.

## Setup

### Prerequisites
- Node.js 18.x or higher
- npm or yarn (dependency management)
- Python 3.10 or higher (for backend components)
- Poetry (dependency management for Python components)

### Installation
1. Clone the repository
2. Navigate to the service directory:
```bash
cd ui-service
```
3. Install frontend dependencies:
```bash
npm install
```
4. Install backend dependencies using Poetry:
```bash
poetry install
```

### Environment Variables
The following environment variables are required:

| Variable | Description | Default |
|----------|-------------|---------|
| `NODE_ENV` | Environment (development, production) | development |
| `PORT` | Service port | 3000 |
| `NEXT_PUBLIC_API_URL` | Base URL for API services | http://localhost:8000/api |
| `ANALYSIS_ENGINE_URL` | URL to the Analysis Engine Service | http://localhost:8002 |
| `FEATURE_STORE_URL` | URL to the Feature Store Service | http://localhost:8001 |
| `TRADING_GATEWAY_URL` | URL to the Trading Gateway Service | http://localhost:8004 |
| `AUTH_SECRET` | Secret for authentication | - |
| `SESSION_TIMEOUT_MIN` | Session timeout in minutes | 60 |
| `ENABLE_ANALYTICS` | Enable usage analytics | false |

Example .env.local file:
```
NODE_ENV=development
PORT=3000
NEXT_PUBLIC_API_URL=http://localhost:8000/api
ANALYSIS_ENGINE_URL=http://localhost:8002
FEATURE_STORE_URL=http://localhost:8001
TRADING_GATEWAY_URL=http://localhost:8004
AUTH_SECRET=your-auth-secret
SESSION_TIMEOUT_MIN=60
ENABLE_ANALYTICS=false
```

### Running the Service
Run the development server:
```bash
npm run dev
```

For production build:
```bash
npm run build
npm start
```

## Application Structure

### Frontend
The UI Service is built using Next.js with TypeScript and follows a modular component architecture:

- **Pages**: Route-based page components
- **Components**: Reusable UI components
- **Hooks**: Custom React hooks for shared logic
- **Contexts**: Global state management
- **Services**: API service integrations
- **Utils**: Utility functions
- **Styles**: Global styles and themes

### Backend Components
The service includes Python-based backend components for:
- Data preprocessing for visualization
- User preference management
- Authentication services

## Features

### Dashboard
- Real-time market overview
- Portfolio summary
- Performance metrics
- Active alerts and notifications

### Market Analysis
- Interactive charts with technical indicators
- Pattern recognition visualization
- Market sentiment analysis
- Fundamental data display

### Trading Interface
- Order entry forms
- Position management
- Trade history
- Risk management controls

### Strategy Builder
- Visual strategy builder
- Backtesting interface
- Strategy optimization
- Performance analytics

### Administration
- User management
- System monitoring
- Configuration interface
- Audit logs

## Integration with Other Services
The UI Service integrates with:

- Analysis Engine Service for market analysis and visualization
- Feature Store Service for indicator data
- Trading Gateway Service for order execution
- Portfolio Management Service for position tracking
- Strategy Execution Engine for strategy management
- Monitoring & Alerting Service for alerts and notifications

## Security Features
- Token-based authentication
- Role-based access control
- CSRF protection
- Input validation
- Session management
- Security headers

## Responsive Design
The UI is fully responsive and optimized for:
- Desktop workstations (multi-monitor support)
- Laptops
- Tablets
- Mobile devices

## Error Handling
The service implements comprehensive error handling with user-friendly error messages and automatic error reporting.
