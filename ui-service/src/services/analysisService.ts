import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001/api';

// Fetch technical indicator data
export const fetchIndicatorData = async (indicators: string[]) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/analysis/indicators`, {
      params: { indicators: indicators.join(',') }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching indicator data:', error);
    // Return mock data for development if API fails
    return generateMockIndicatorData(indicators);
  }
};

// Fetch correlation data
export const fetchCorrelationData = async (assets: string[], period: number) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/analysis/correlation`, {
      params: {
        assets: assets.join(','),
        period
      }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching correlation data:', error);
    // Return mock data for development if API fails
    return generateMockCorrelationData(assets);
  }
};

// Fetch regression analysis
export const fetchRegressionAnalysis = async (asset: string) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/analysis/regression`, {
      params: { asset }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching regression analysis:', error);
    // Return mock data for development if API fails
    return generateMockRegressionAnalysis();
  }
};

// Run backtest
export const runBacktest = async (params: any) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/analysis/backtest`, params);
    return response.data;
  } catch (error) {
    console.error('Error running backtest:', error);
    // Return mock data for development if API fails
    return generateMockBacktestResults(params);
  }
};

// Fetch volatility analysis
export const fetchVolatilityAnalysis = async (assets: string[], window: number) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/analysis/volatility`, {
      params: {
        assets: assets.join(','),
        window
      }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching volatility analysis:', error);
    // Return mock data for development if API fails
    return generateMockVolatilityAnalysis(assets);
  }
};

// Mock data generation functions

// Generate mock indicator data
const generateMockIndicatorData = (indicators: string[]) => {
  const result: Record<string, any[]> = {};
  const dates = generateDates(100);

  // Generate data for each requested indicator
  indicators.forEach(indicator => {
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

  return result;
};

// Generate mock correlation data
const generateMockCorrelationData = (assets: string[]) => {
  const results = [];
  
  // Create correlation data between each pair of assets
  for (let i = 0; i < assets.length; i++) {
    for (let j = 0; j < assets.length; j++) {
      if (i !== j) {
        // Generate a correlation coefficient between -1 and 1
        // Assets with similar names should have higher correlation
        const similarity = assets[i].substring(0, 3) === assets[j].substring(0, 3) ? 0.5 : 0;
        const correlation = similarity + (Math.random() * 1.2 - 0.6);
        const clampedCorr = Math.max(-0.95, Math.min(0.95, correlation));
        
        results.push({
          x: assets[i],
          y: assets[j],
          z: clampedCorr
        });
      }
    }
  }
  
  return results;
};

// Generate mock regression analysis data
const generateMockRegressionAnalysis = () => {
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
  
  return {
    predictions,
    stats,
    featureImportance
  };
};

// Generate mock backtest results
const generateMockBacktestResults = (params: any) => {
  const { strategy, initialCapital, startDate, endDate } = params;
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
  
  return {
    strategy,
    equityCurve,
    metrics,
    tradeStats
  };
};

// Generate mock volatility analysis
const generateMockVolatilityAnalysis = (assets: string[]) => {
  const dates = generateDates(100);
  
  // Generate historical volatility
  const historical = dates.map(date => {
    const dataPoint: any = { date };
    
    assets.forEach(asset => {
      const assetKey = asset.replace('/', '');
      dataPoint[assetKey] = 0.005 + Math.abs(Math.sin(dates.indexOf(date) / 20)) * 0.015 + Math.random() * 0.005;
    });
    
    return dataPoint;
  });
  
  // Generate current volatility regime
  const currentRegime: Record<string, any> = {};
  const stats: Record<string, any> = {};
  
  assets.forEach(asset => {
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
  
  return {
    historical,
    currentRegime,
    stats
  };
};

// Helper function to generate dates
const generateDates = (count: number) => {
  const dates = [];
  const now = new Date();
  
  for (let i = count - 1; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(now.getDate() - i);
    dates.push(date.toISOString().split('T')[0]);
  }
  
  return dates;
};

// Helper function to generate dates between start and end
const generateDatesBetween = (startDate: string, endDate: string) => {
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
};
