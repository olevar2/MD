const express = require('express');
const router = express.Router();
const {
  DataValidationError,
  ServiceError,
  withErrorHandling
} = require('../../utils/errors');
const { withAsyncErrorHandling, convertPythonError } = require('../../utils/errorBridge');

// Technical indicator analysis endpoint
router.get('/indicators', async (req, res) => {
  try {
    const { indicators } = req.query;
    const indicatorList = indicators ? indicators.split(',') : ['macd', 'rsi'];

    // Generate mock indicator data
    const result = {};
    const dates = generateDates(100);

    // Generate data for each requested indicator
    indicatorList.forEach(indicator => {
      switch (indicator) {
        case 'macd':
          result[indicator] = dates.map((date, i) => ({
            date,
            value: Math.sin(i / 10) * 0.5 + Math.random() * 0.2 - 0.1,
            signal: Math.sin(i / 10 - 1) * 0.5 + Math.random() * 0.1,
            histogram: Math.sin(i / 10) * 0.5 - Math.sin(i / 10 - 1) * 0.5 + Math.random() * 0.1
          }));
          break;
        case 'rsi':
          result[indicator] = dates.map((date, i) => ({
            date,
            value: 50 + Math.sin(i / 8) * 20 + Math.random() * 5,
          }));
          break;
        case 'bollinger':
          const basePrice = 100;
          result[indicator] = dates.map((date, i) => {
            const price = basePrice + Math.sin(i / 15) * 10 + i / 10 + Math.random() * 2;
            const volatility = 2 * (1 + Math.abs(Math.sin(i / 30))) * (1 + Math.random() * 0.5);

            return {
              date,
              value: price,
              upperBand: price + volatility * 2,
              lowerBand: price - volatility * 2
            };
          });
          break;
        case 'ema':
          result[indicator] = dates.map((date, i) => ({
            date,
            value: 100 + i / 5 + Math.sin(i / 20) * 10 + Math.random() * 1,
          }));
          break;
        case 'sma':
          result[indicator] = dates.map((date, i) => ({
            date,
            value: 100 + i / 5 + Math.sin(i / 20) * 8 + Math.random() * 0.5,
          }));
          break;
        case 'atr':
          result[indicator] = dates.map((date, i) => ({
            date,
            value: 0.5 + Math.abs(Math.sin(i / 25)) * 0.5 + Math.random() * 0.2,
          }));
          break;
        default:
          result[indicator] = dates.map((date, i) => ({
            date,
            value: Math.random() * 100,
          }));
      }
    });

    res.json(result);
  } catch (error) {
    console.error('Error generating indicator data:', error);
    res.status(500).json({ error: 'Failed to generate indicator data' });
  }
});

// Correlation analysis endpoint
router.get('/correlation', async (req, res) => {
  try {
    const { assets, period } = req.query;
    const assetList = assets ? assets.split(',') : ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD'];

    const results = [];

    // Create correlation data between each pair of assets
    for (let i = 0; i < assetList.length; i++) {
      for (let j = 0; j < assetList.length; j++) {
        if (i !== j) {
          // Generate a correlation coefficient between -1 and 1
          // Assets with similar names should have higher correlation
          const similarity = assetList[i].substring(0, 3) === assetList[j].substring(0, 3) ? 0.5 : 0;
          const correlation = similarity + (Math.random() * 1.2 - 0.6);
          const clampedCorr = Math.max(-0.95, Math.min(0.95, correlation));

          results.push({
            x: assetList[i],
            y: assetList[j],
            z: clampedCorr
          });
        }
      }
    }

    res.json(results);
  } catch (error) {
    console.error('Error generating correlation data:', error);
    res.status(500).json({ error: 'Failed to generate correlation data' });
  }
});

// Regression analysis endpoint
router.get('/regression', async (req, res) => {
  try {
    const { asset } = req.query;
    const dates = generateDates(60);
    const predictions = [];

    // Generate actual vs predicted values
    for (let i = 0; i < dates.length; i++) {
      const base = 1.1 + Math.sin(i / 15) * 0.05 + i * 0.001;
      const actual = base + Math.random() * 0.01 - 0.005;
      const predicted = base + Math.random() * 0.008 - 0.004;

      predictions.push({
        date: dates[i],
        actual,
        predicted
      });
    }

    // Generate model statistics
    const stats = {
      r2: 0.87 + Math.random() * 0.1,
      mse: 0.0002 + Math.random() * 0.0001,
      mae: 0.005 + Math.random() * 0.002
    };

    // Generate feature importance
    const featureImportance = [
      { name: 'Price_Momentum', importance: 0.35 + Math.random() * 0.1 },
      { name: 'Volume', importance: 0.25 + Math.random() * 0.1 },
      { name: 'Volatility', importance: 0.20 + Math.random() * 0.1 },
      { name: 'Sentiment', importance: 0.12 + Math.random() * 0.1 },
      { name: 'Market_Trend', importance: 0.08 + Math.random() * 0.05 }
    ];

    res.json({
      predictions,
      stats,
      featureImportance
    });
  } catch (error) {
    console.error('Error generating regression analysis:', error);
    res.status(500).json({ error: 'Failed to generate regression analysis' });
  }
});

// Backtest endpoint
router.post('/backtest', async (req, res) => {
  try {
    const { strategy, initialCapital, startDate, endDate } = req.body;
    const dates = generateDatesBetween(startDate, endDate);
    const equityCurve = [];

    // Different strategy types have different performance characteristics
    let performanceFactor = 1.0;
    let volatilityFactor = 1.0;

    switch (strategy) {
      case 'momentum':
        performanceFactor = 1.3;
        volatilityFactor = 1.2;
        break;
      case 'meanReversion':
        performanceFactor = 1.1;
        volatilityFactor = 0.8;
        break;
      case 'breakout':
        performanceFactor = 1.4;
        volatilityFactor = 1.5;
        break;
      case 'mlBased':
        performanceFactor = 1.6;
        volatilityFactor = 1.0;
        break;
    }

    // Generate equity curve
    let equity = initialCapital;
    let benchmark = initialCapital;

    for (let i = 0; i < dates.length; i++) {
      // Apply performance and volatility factors
      const dailyReturn = (Math.random() * 0.03 - 0.01) * performanceFactor;
      const benchmarkReturn = Math.random() * 0.02 - 0.008;

      equity = equity * (1 + dailyReturn);
      benchmark = benchmark * (1 + benchmarkReturn);

      equityCurve.push({
        date: dates[i],
        equity,
        benchmark
      });
    }

    // Calculate performance metrics
    const finalReturn = (equity / initialCapital) - 1;
    const benchmarkReturn = (benchmark / initialCapital) - 1;

    // Generate trade statistics
    const tradeCount = Math.floor(30 + Math.random() * 50);
    const winRate = 0.5 + Math.random() * 0.2 * performanceFactor;
    const winCount = Math.floor(tradeCount * winRate);
    const lossCount = tradeCount - winCount;

    const avgWin = initialCapital * 0.02 * performanceFactor;
    const avgLoss = initialCapital * 0.01 * volatilityFactor;

    const metrics = {
      totalReturn: finalReturn,
      annualizedReturn: finalReturn * (365 / dates.length),
      benchmarkReturn,
      alpha: finalReturn - benchmarkReturn,
      beta: 0.8 + Math.random() * 0.4,
      sharpeRatio: 1.2 + Math.random() * 1.0 * performanceFactor,
      maxDrawdown: 0.1 + Math.random() * 0.15 * volatilityFactor,
      volatility: 0.15 + Math.random() * 0.1 * volatilityFactor
    };

    const tradeStats = {
      totalTrades: tradeCount,
      winningTrades: winCount,
      losingTrades: lossCount,
      winRate,
      profitFactor: (winCount * avgWin) / (lossCount * avgLoss),
      averageWin: avgWin,
      averageLoss: avgLoss,
      largestWin: avgWin * (2 + Math.random() * 3),
      largestLoss: avgLoss * (2 + Math.random() * 3),
      averageTradeDuration: Math.floor(2 + Math.random() * 5)
    };

    res.json({
      strategy,
      equityCurve,
      metrics,
      tradeStats
    });
  } catch (error) {
    console.error('Error running backtest:', error);
    res.status(500).json({ error: 'Failed to run backtest' });
  }
});

// Volatility analysis endpoint
router.get('/volatility', async (req, res) => {
  try {
    const { assets } = req.query;
    const assetList = assets ? assets.split(',') : ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD'];

    const dates = generateDates(100);

    // Generate historical volatility
    const historical = dates.map(date => {
      const dataPoint = { date };

      assetList.forEach(asset => {
        const assetKey = asset.replace('/', '');
        dataPoint[assetKey] = 0.005 + Math.abs(Math.sin(dates.indexOf(date) / 20)) * 0.015 + Math.random() * 0.005;
      });

      return dataPoint;
    });

    // Generate current volatility regime
    const currentRegime = {};
    const stats = {};

    assetList.forEach(asset => {
      const volatilityLevel = Math.random();
      let regime = 'Medium';

      if (volatilityLevel < 0.3) regime = 'Low';
      else if (volatilityLevel > 0.7) regime = 'High';

      currentRegime[asset] = {
        regime,
        percentile: Math.round(volatilityLevel * 100)
      };

      stats[asset] = {
        mean: 0.01 + Math.random() * 0.005,
        min: 0.003 + Math.random() * 0.002,
        max: 0.02 + Math.random() * 0.01,
        current: 0.005 + Math.random() * 0.015
      };
    });

    res.json({
      historical,
      currentRegime,
      stats
    });
  } catch (error) {
    console.error('Error generating volatility analysis:', error);
    res.status(500).json({ error: 'Failed to generate volatility analysis' });
  }
});

// Error handling demonstration endpoints
// This endpoint uses our new error handling bridge
router.get('/error-demo', withAsyncErrorHandling(async (req, res) => {
  const { errorType } = req.query;

  switch (errorType) {
    case 'validation':
      throw new DataValidationError(
        'Invalid parameters provided',
        'VALIDATION_ERROR',
        { field: 'symbol', error: 'Symbol is required' }
      );
    case 'service':
      throw new ServiceError(
        'External service unavailable',
        'SERVICE_ERROR',
        { service: 'market-data-service' }
      );
    case 'python':
      // Simulate an error from Python component
      // In a real scenario, this would come from a Python API response
      // We're creating a mock response error object
      const pythonError = {
        error_type: 'MarketDataError',
        error_code: 'MARKET_DATA_ERROR',
        message: 'Failed to fetch market data',
        details: { symbol: 'EURUSD' }
      };
      // Convert Python error to JavaScript error
      throw convertPythonError(pythonError);
    case 'standard':
      // Standard JavaScript error
      throw new Error('Standard JavaScript error');
    default:
      // Return success if no error type specified
      res.json({ message: 'No error triggered', success: true });
  }
}));

// Helper function to generate dates
function generateDates(count) {
  const dates = [];
  const now = new Date();

  for (let i = count - 1; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(now.getDate() - i);
    dates.push(date.toISOString().split('T')[0]);
  }

  return dates;
}

// Helper function to generate dates between start and end
function generateDatesBetween(startDate, endDate) {
  const dates = [];
  const start = new Date(startDate);
  const end = new Date(endDate);
  const days = Math.round((end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24));

  // If period is too long, sample dates to keep the data points manageable
  const stride = days > 100 ? Math.floor(days / 100) : 1;

  for (let i = 0; i <= days; i += stride) {
    const date = new Date(start);
    date.setDate(start.getDate() + i);
    dates.push(date.toISOString().split('T')[0]);
  }

  return dates;
}

module.exports = router;
