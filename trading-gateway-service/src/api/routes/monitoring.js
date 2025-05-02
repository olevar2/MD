const express = require('express');
const router = express.Router();

// System metrics endpoint
router.get('/system-metrics', async (req, res) => {
  const { timeRange } = req.query;
  try {
    // In production, fetch real metrics from your monitoring systems
    // For now, we'll return mock data similar to what the UI expects
    
    // Generate timestamps based on time range
    const now = new Date();
    const timestamps = [];
    let points = 0;
    let interval = 0;

    switch (timeRange) {
      case '15m':
        points = 15;
        interval = 60 * 1000; // 1 minute
        break;
      case '1h':
        points = 60;
        interval = 60 * 1000; // 1 minute
        break;
      case '4h':
        points = 48;
        interval = 5 * 60 * 1000; // 5 minutes
        break;
      case '1d':
        points = 24;
        interval = 60 * 60 * 1000; // 1 hour
        break;
      default:
        points = 60;
        interval = 60 * 1000; // 1 minute
    }

    for (let i = points - 1; i >= 0; i--) {
      const time = new Date(now.getTime() - i * interval);
      timestamps.push(time.toISOString().replace('T', ' ').substr(0, 19));
    }
    
    // Generate CPU and Memory usage data
    const resources = timestamps.map((timestamp, index) => ({
      timestamp,
      cpuUsage: 20 + Math.random() * 40 + (index / points) * 20,
      memoryUsage: 4 + Math.random() * 2 + (index / points)
    }));

    // Generate API response time data
    const apiResponse = [
      { endpoint: 'market/quotes', responseTime: 50 + Math.random() * 100, errorRate: Math.random() * 2 },
      { endpoint: 'orders/create', responseTime: 80 + Math.random() * 150, errorRate: Math.random() * 3 },
      { endpoint: 'positions/list', responseTime: 30 + Math.random() * 70, errorRate: Math.random() * 1.5 },
      { endpoint: 'analysis/indicators', responseTime: 120 + Math.random() * 200, errorRate: Math.random() * 4 },
      { endpoint: 'auth/validate', responseTime: 20 + Math.random() * 40, errorRate: Math.random() * 0.5 }
    ];

    // Generate database performance data
    const database = timestamps.map((timestamp, index) => ({
      timestamp,
      queryPerSecond: 100 + Math.random() * 150 + (index / points) * 50,
      latency: 5 + Math.random() * 30 + (index % 5 === 0 ? 15 : 0)
    }));

    // Generate service health data
    const serviceHealth = [
      { name: 'Trading Gateway', value: 98 },
      { name: 'Analysis Engine', value: 96 },
      { name: 'Authentication', value: 99 },
      { name: 'Data Pipeline', value: 97 },
      { name: 'User Interface', value: 100 }
    ];

    res.json({
      resources,
      apiResponse,
      database,
      serviceHealth
    });
  } catch (error) {
    console.error('Error fetching system metrics:', error);
    res.status(500).json({ error: 'Failed to fetch system metrics' });
  }
});

// Market activity endpoint
router.get('/market-activity', async (req, res) => {
  const { timeRange } = req.query;
  try {
    // Generate timestamps based on time range
    const now = new Date();
    const timestamps = [];
    let points = 0;
    let interval = 0;

    switch (timeRange) {
      case '15m':
        points = 15;
        interval = 60 * 1000; // 1 minute
        break;
      case '1h':
        points = 60;
        interval = 60 * 1000; // 1 minute
        break;
      case '4h':
        points = 48;
        interval = 5 * 60 * 1000; // 5 minutes
        break;
      case '1d':
        points = 24;
        interval = 60 * 60 * 1000; // 1 hour
        break;
      default:
        points = 60;
        interval = 60 * 1000; // 1 minute
    }

    for (let i = points - 1; i >= 0; i--) {
      const time = new Date(now.getTime() - i * interval);
      timestamps.push(time.toISOString().replace('T', ' ').substr(0, 19));
    }

    // Generate volume data
    const volume = timestamps.map((timestamp, index) => {
      const baseVolume = 1000 + Math.random() * 500;
      const trendFactor = Math.sin(index / 10) * 200;
      const buyVolume = baseVolume + trendFactor + Math.random() * 300;
      const sellVolume = baseVolume - trendFactor + Math.random() * 300;
      return {
        timestamp,
        buyVolume,
        sellVolume
      };
    });

    // Generate price movement data
    const prices = timestamps.map((timestamp, index) => {
      const baseEUR = 1.08 + Math.sin(index / 20) * 0.02 + (index / points) * 0.01;
      const baseGBP = 1.26 + Math.cos(index / 15) * 0.03 - (index / points) * 0.005;
      const baseJPY = 153.5 + Math.sin(index / 10) * 1.5 + (index / points) * 0.2;
      
      return {
        timestamp,
        'EUR/USD': baseEUR + Math.random() * 0.002,
        'GBP/USD': baseGBP + Math.random() * 0.003,
        'USD/JPY': baseJPY + Math.random() * 0.1
      };
    });

    // Generate spread data
    const spreads = timestamps.map((timestamp, index) => {
      return {
        timestamp,
        'EUR/USD': 0.0001 + Math.random() * 0.0003 + (index % 10 === 0 ? 0.0005 : 0),
        'GBP/USD': 0.0002 + Math.random() * 0.0004 + (index % 15 === 0 ? 0.0006 : 0),
        'USD/JPY': 0.02 + Math.random() * 0.03 + (index % 12 === 0 ? 0.05 : 0)
      };
    });

    // Generate liquidity heat map data
    const liquidityPairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 'EUR/GBP'];
    const liquidity = liquidityPairs.map(pair => ({
      pair,
      liquidity: 30 + Math.random() * 70,
      value: 30 + Math.random() * 70
    }));

    res.json({
      volume,
      prices,
      spreads,
      liquidity
    });
  } catch (error) {
    console.error('Error fetching market activity data:', error);
    res.status(500).json({ error: 'Failed to fetch market activity data' });
  }
});

// Alerts endpoint
router.get('/alerts', async (req, res) => {
  try {
    const mockActiveAlerts = [
      {
        id: 'alert-1',
        title: 'High CPU Usage Detected',
        message: 'System CPU usage has exceeded 80% for more than 5 minutes.',
        severity: 'warning',
        timestamp: '2025-04-24T08:32:15Z',
        source: 'System Monitoring'
      },
      {
        id: 'alert-2',
        title: 'Critical Market Volatility',
        message: 'EUR/USD volatility has increased by 200% in the last hour.',
        severity: 'critical',
        timestamp: '2025-04-24T08:45:22Z',
        source: 'Market Analysis'
      },
      {
        id: 'alert-3',
        title: 'New Strategy Signal',
        message: 'Momentum strategy has generated a buy signal for GBP/USD.',
        severity: 'info',
        timestamp: '2025-04-24T09:12:08Z',
        source: 'Strategy Engine'
      }
    ];

    const mockAlertHistory = [
      { date: '2025-04-17', info: 12, warning: 5, critical: 1 },
      { date: '2025-04-18', info: 8, warning: 3, critical: 0 },
      { date: '2025-04-19', info: 6, warning: 4, critical: 2 },
      { date: '2025-04-20', info: 10, warning: 2, critical: 0 },
      { date: '2025-04-21', info: 14, warning: 6, critical: 1 },
      { date: '2025-04-22', info: 9, warning: 4, critical: 2 },
      { date: '2025-04-23', info: 11, warning: 3, critical: 1 },
      { date: '2025-04-24', info: 7, warning: 5, critical: 1 }
    ];

    const mockAlertByType = [
      { name: 'System', value: 25 },
      { name: 'Market', value: 35 },
      { name: 'Security', value: 10 },
      { name: 'Strategy', value: 20 },
      { name: 'User', value: 10 }
    ];

    res.json({
      active: mockActiveAlerts,
      history: mockAlertHistory,
      byType: mockAlertByType
    });
  } catch (error) {
    console.error('Error fetching alerts:', error);
    res.status(500).json({ error: 'Failed to fetch alerts' });
  }
});

// Position risk endpoint
router.get('/positions-risk', async (req, res) => {
  try {
    const positions = [
      'EUR/USD Long',
      'GBP/USD Short',
      'USD/JPY Long',
      'AUD/USD Short',
      'USD/CAD Long'
    ];

    const mockPositionRisk = positions.map(position => ({
      position,
      var: 500 + Math.random() * 2000,
      maxLoss: 1000 + Math.random() * 3000
    }));

    const dates = [];
    for (let i = 0; i < 30; i++) {
      const date = new Date();
      date.setDate(date.getDate() - (29 - i));
      dates.push(date.toISOString().split('T')[0]);
    }

    const mockPortfolioRisk = dates.map((date, index) => {
      const trend = index / 30;
      return {
        date,
        var: 2000 + trend * 500 + Math.random() * 500,
        cvar: 3000 + trend * 700 + Math.random() * 700,
        sharpe: 1.5 + trend * 0.4 + Math.random() * 0.3 - 0.15
      };
    });

    const currencies = ['EUR', 'GBP', 'JPY', 'USD', 'CAD', 'AUD'];
    const pairs = [];
    
    for (let i = 0; i < currencies.length; i++) {
      for (let j = i + 1; j < currencies.length; j++) {
        pairs.push({
          volatility: 5 + Math.random() * 20,
          correlation: (Math.random() * 2 - 1),
          name: `${currencies[i]}/${currencies[j]}`
        });
      }
    }

    res.json({
      positionRisk: mockPositionRisk,
      portfolioRisk: mockPortfolioRisk,
      riskMap: pairs
    });
  } catch (error) {
    console.error('Error fetching position risk data:', error);
    res.status(500).json({ error: 'Failed to fetch position risk data' });
  }
});

// Portfolio allocation endpoint
router.get('/portfolio-allocation', async (req, res) => {
  try {
    const currencies = [
      { name: 'EUR', value: 30 },
      { name: 'USD', value: 25 },
      { name: 'GBP', value: 15 },
      { name: 'JPY', value: 10 },
      { name: 'CAD', value: 10 },
      { name: 'AUD', value: 10 }
    ];

    res.json({
      byCurrency: currencies
    });
  } catch (error) {
    console.error('Error fetching portfolio allocation:', error);
    res.status(500).json({ error: 'Failed to fetch portfolio allocation data' });
  }
});

module.exports = router;
