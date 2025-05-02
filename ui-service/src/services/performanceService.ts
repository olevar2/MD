import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001/api';

// Fetch performance data
export const fetchPerformanceData = async (timeRange: string) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/performance/returns`, {
      params: { timeRange }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching performance data:', error);
    // Return mock data for development if API fails
    return generateMockPerformanceData(timeRange);
  }
};

// Fetch drawdown analysis
export const fetchDrawdownAnalysis = async (timeRange: string) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/performance/drawdowns`, {
      params: { timeRange }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching drawdown analysis:', error);
    // Return mock data for development if API fails
    return generateMockDrawdownAnalysis(timeRange);
  }
};

// Fetch risk-adjusted returns
export const fetchRiskAdjustedReturns = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/performance/risk-adjusted`);
    return response.data;
  } catch (error) {
    console.error('Error fetching risk-adjusted returns:', error);
    // Return mock data for development if API fails
    return generateMockRiskAdjustedReturns();
  }
};

// Fetch trading statistics
export const fetchTradingStats = async (timeRange: string) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/performance/trading-stats`, {
      params: { timeRange }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching trading statistics:', error);
    // Return mock data for development if API fails
    return generateMockTradingStats();
  }
};

// Fetch portfolio comparison
export const fetchPortfolioComparison = async (benchmark: string, timeRange: string) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/performance/comparison`, {
      params: { benchmark, timeRange }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching portfolio comparison:', error);
    // Return mock data for development if API fails
    return generateMockPortfolioComparison(benchmark, timeRange);
  }
};

// Mock data generation functions

// Generate mock performance data
const generateMockPerformanceData = (timeRange: string) => {
  // Generate dates based on time range
  const dates = generateDatesByTimeRange(timeRange);
  const dataPoints = dates.length;
  
  // Generate cumulative returns
  const cumulativeReturns = [];
  let portfolioReturn = 0;
  let benchmarkReturn = 0;
  
  for (let i = 0; i < dataPoints; i++) {
    // Simulate daily returns with some correlation to each other
    const marketMove = (Math.random() * 0.02 - 0.01) * (1 + Math.sin(i / 20) * 0.5);
    const portfolioDailyReturn = marketMove + (Math.random() * 0.01 - 0.003);
    const benchmarkDailyReturn = marketMove * 0.8 + (Math.random() * 0.008 - 0.003);
    
    // Accumulate returns
    portfolioReturn += portfolioDailyReturn + (portfolioDailyReturn * portfolioReturn);
    benchmarkReturn += benchmarkDailyReturn + (benchmarkDailyReturn * benchmarkReturn);
    
    cumulativeReturns.push({
      date: dates[i],
      portfolio: portfolioReturn,
      benchmark: benchmarkReturn
    });
  }
  
  // Generate periodic returns (e.g., daily or monthly)
  const periodicReturns = dates.map((date, i) => {
    const marketMove = (Math.random() * 0.02 - 0.01) * (1 + Math.sin(i / 20) * 0.5);
    return {
      date,
      portfolio: marketMove + (Math.random() * 0.01 - 0.003),
      benchmark: marketMove * 0.8 + (Math.random() * 0.008 - 0.003)
    };
  });
  
  // Generate return distribution
  const returnRanges = [];
  for (let r = -0.03; r <= 0.03; r += 0.005) {
    returnRanges.push({
      range: `${(r * 100).toFixed(1)}%`,
      frequency: Math.floor(Math.exp(-Math.pow(r * 20, 2)) * 30 + Math.random() * 5)
    });
  }
  
  // Generate performance metrics
  const metrics = {
    totalReturn: portfolioReturn,
    annualReturn: (Math.pow(1 + portfolioReturn, 252 / dataPoints) - 1),
    volatility: 0.12 + Math.random() * 0.05,
    sharpeRatio: 1.2 + Math.random() * 0.8,
    sortinoRatio: 1.5 + Math.random() * 1.0,
    maxDrawdown: 0.12 + Math.random() * 0.08,
    alpha: 0.02 + Math.random() * 0.04,
    beta: 0.85 + Math.random() * 0.3,
    informationRatio: 0.5 + Math.random() * 0.6
  };
  
  // Generate monthly returns table
  const currentYear = new Date().getFullYear();
  const monthlyReturns = [];
  
  // Last 3 years
  for (let y = 0; y < 3; y++) {
    const year = {
      year: currentYear - 2 + y,
      jan: (Math.random() * 0.06 - 0.02),
      feb: (Math.random() * 0.06 - 0.02),
      mar: (Math.random() * 0.06 - 0.02),
      apr: (Math.random() * 0.06 - 0.02),
      may: (Math.random() * 0.06 - 0.02),
      jun: (Math.random() * 0.06 - 0.02),
      jul: (Math.random() * 0.06 - 0.02),
      aug: (Math.random() * 0.06 - 0.02),
      sep: (Math.random() * 0.06 - 0.02),
      oct: (Math.random() * 0.06 - 0.02),
      nov: (Math.random() * 0.06 - 0.02),
      dec: (Math.random() * 0.06 - 0.02)
    };
    
    // Calculate YTD
    const months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'];
    let ytd = 0;
    let ytdMonths = y < 2 ? 12 : new Date().getMonth() + 1; // Current year is partial
    
    for (let m = 0; m < ytdMonths; m++) {
      ytd = ytd + year[months[m]] + (ytd * year[months[m]]);
    }
    year.ytd = ytd;
    
    monthlyReturns.push(year);
  }
  
  return {
    cumulativeReturns,
    periodicReturns,
    returnDistribution: returnRanges,
    metrics,
    monthlyReturns
  };
};

// Generate mock drawdown analysis
const generateMockDrawdownAnalysis = (timeRange: string) => {
  // Generate dates based on time range
  const dates = generateDatesByTimeRange(timeRange);
  const dataPoints = dates.length;
  
  // Generate equity curve and drawdowns
  const drawdowns = [];
  let equity = 10000;
  let peak = equity;
  
  for (let i = 0; i < dataPoints; i++) {
    // Simulate daily returns
    const dailyReturn = (Math.random() * 0.02 - 0.008) * (1 + Math.sin(i / 30) * 0.8);
    
    // Update equity
    equity = equity * (1 + dailyReturn);
    
    // Update peak
    peak = Math.max(peak, equity);
    
    // Calculate drawdown
    const drawdown = (equity - peak) / peak;
    
    drawdowns.push({
      date: dates[i],
      equity,
      drawdown
    });
  }
  
  // Generate worst drawdowns
  const worstDrawdowns = [];
  let inDrawdown = false;
  let drawdownStart = '';
  let drawdownStartValue = 0;
  let drawdownEnd = '';
  let drawdownEndValue = 0;
  let maxDrawdownPct = 0;
  
  // Process drawdowns to find worst periods
  for (let i = 1; i < drawdowns.length; i++) {
    const curr = drawdowns[i];
    const prev = drawdowns[i - 1];
    
    if (prev.drawdown === 0 && curr.drawdown < 0) {
      // Start of a drawdown
      inDrawdown = true;
      drawdownStart = prev.date;
      drawdownStartValue = prev.equity;
    } else if (inDrawdown && curr.drawdown === 0) {
      // End of a drawdown
      inDrawdown = false;
      drawdownEnd = curr.date;
      drawdownEndValue = curr.equity;
      
      // Calculate drawdown metrics
      const drawdownPct = (drawdownStartValue - Math.min(...drawdowns.slice(drawdowns.findIndex(d => d.date === drawdownStart), i + 1).map(d => d.equity))) / drawdownStartValue;
      const startDate = new Date(drawdownStart);
      const endDate = new Date(drawdownEnd);
      const lengthDays = Math.round((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24));
      
      worstDrawdowns.push({
        start: drawdownStart,
        end: drawdownEnd,
        drawdown: drawdownPct,
        lengthDays,
        recoveryDays: Math.round(lengthDays * 0.8 + Math.random() * lengthDays * 0.4)
      });
      
      maxDrawdownPct = Math.max(maxDrawdownPct, drawdownPct);
    }
  }
  
  // If still in a drawdown at the end, record it as ongoing
  if (inDrawdown) {
    const lastDrawdown = drawdowns[drawdowns.length - 1];
    const drawdownPct = (drawdownStartValue - Math.min(...drawdowns.slice(drawdowns.findIndex(d => d.date === drawdownStart)).map(d => d.equity))) / drawdownStartValue;
    const startDate = new Date(drawdownStart);
    const endDate = new Date(lastDrawdown.date);
    const lengthDays = Math.round((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24));
    
    worstDrawdowns.push({
      start: drawdownStart,
      end: null, // Ongoing
      drawdown: drawdownPct,
      lengthDays,
      recoveryDays: null // Unknown since still ongoing
    });
    
    maxDrawdownPct = Math.max(maxDrawdownPct, drawdownPct);
  }
  
  // Sort worst drawdowns by magnitude
  worstDrawdowns.sort((a, b) => b.drawdown - a.drawdown);
  
  // Generate drawdown distribution
  const drawdownDistribution = [];
  const drawdownRanges = [0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15];
  
  for (let i = 0; i < drawdownRanges.length - 1; i++) {
    drawdownDistribution.push({
      range: `${(drawdownRanges[i] * 100).toFixed(0)}%-${(drawdownRanges[i+1] * 100).toFixed(0)}%`,
      frequency: Math.floor(Math.exp(-Math.pow((drawdownRanges[i] + drawdownRanges[i+1]) / 2 * 10, 2)) * 30 + Math.random() * 5)
    });
  }
  
  return {
    drawdowns,
    worstDrawdowns: worstDrawdowns.slice(0, 5), // Return only top 5 worst drawdowns
    drawdownDistribution,
    maxDrawdown: maxDrawdownPct
  };
};

// Generate mock risk-adjusted returns
const generateMockRiskAdjustedReturns = () => {
  // Generate metrics
  const metrics = [
    {
      name: 'Sharpe Ratio',
      value: 1.2 + Math.random() * 0.8,
      benchmark: 0.9 + Math.random() * 0.6,
      percentile: 0.7 + Math.random() * 0.25,
      benchmarkPercentile: 0.4 + Math.random() * 0.3,
      description: 'Risk-adjusted return using standard deviation of returns'
    },
    {
      name: 'Sortino Ratio',
      value: 1.5 + Math.random() * 1.0,
      benchmark: 1.1 + Math.random() * 0.7,
      percentile: 0.75 + Math.random() * 0.2,
      benchmarkPercentile: 0.5 + Math.random() * 0.2,
      description: 'Risk-adjusted return using downside deviation'
    },
    {
      name: 'Calmar Ratio',
      value: 0.8 + Math.random() * 0.7,
      benchmark: 0.5 + Math.random() * 0.5,
      percentile: 0.65 + Math.random() * 0.3,
      benchmarkPercentile: 0.45 + Math.random() * 0.3,
      description: 'Return divided by maximum drawdown'
    },
    {
      name: 'Information Ratio',
      value: 0.5 + Math.random() * 0.6,
      benchmark: 0.3 + Math.random() * 0.4,
      percentile: 0.6 + Math.random() * 0.3,
      benchmarkPercentile: 0.4 + Math.random() * 0.3,
      description: 'Active return divided by tracking error'
    }
  ];
  
  // Generate risk-return scatter plot data
  const assets = [
    { name: 'Portfolio', risk: 0.12 + Math.random() * 0.04, return: 0.14 + Math.random() * 0.06 },
    { name: 'S&P500', risk: 0.15 + Math.random() * 0.03, return: 0.10 + Math.random() * 0.04 },
    { name: 'USD Index', risk: 0.08 + Math.random() * 0.03, return: 0.06 + Math.random() * 0.04 },
    { name: 'EUR/USD', risk: 0.10 + Math.random() * 0.04, return: 0.08 + Math.random() * 0.06 },
    { name: 'GBP/USD', risk: 0.11 + Math.random() * 0.05, return: 0.09 + Math.random() * 0.05 },
    { name: 'USD/JPY', risk: 0.09 + Math.random() * 0.04, return: 0.07 + Math.random() * 0.05 },
    { name: 'Gold', risk: 0.14 + Math.random() * 0.05, return: 0.12 + Math.random() * 0.05 },
    { name: 'Bonds', risk: 0.05 + Math.random() * 0.03, return: 0.04 + Math.random() * 0.03 }
  ];
  
  // Generate efficient frontier
  const efficientFrontier = [];
  for (let risk = 0.04; risk <= 0.2; risk += 0.01) {
    efficientFrontier.push({
      risk,
      return: 0.03 + Math.sqrt(risk) * 0.3 + Math.random() * 0.01
    });
  }
  
  // Generate rolling metrics
  const dates = generateDates(90);
  const rollingMetrics = dates.map((date, i) => {
    return {
      date,
      sharpe: 1.0 + Math.sin(i / 30) * 0.5 + Math.random() * 0.3,
      sortino: 1.3 + Math.sin(i / 35) * 0.6 + Math.random() * 0.4,
      calmar: 0.6 + Math.sin(i / 40) * 0.4 + Math.random() * 0.3,
    };
  });
  
  return {
    metrics,
    riskReturn: assets,
    efficientFrontier,
    rollingMetrics
  };
};

// Generate mock trading statistics
const generateMockTradingStats = () => {
  // Generate trade outcome distribution
  const totalTrades = 100 + Math.floor(Math.random() * 150);
  const winRate = 0.52 + Math.random() * 0.1;
  const winCount = Math.round(totalTrades * winRate);
  const lossCount = totalTrades - winCount;
  
  const outcomes = [
    { name: 'Win', value: winCount },
    { name: 'Loss', value: lossCount }
  ];
  
  // Generate trading metrics
  const avgWin = 150 + Math.random() * 100;
  const avgLoss = 100 + Math.random() * 50;
  const profitFactor = (winCount * avgWin) / (lossCount * avgLoss);
  
  const metrics = {
    totalTrades,
    winCount,
    lossCount,
    winRate,
    profitFactor,
    expectancy: winRate * avgWin - (1 - winRate) * avgLoss,
    averageWin: avgWin,
    averageLoss: avgLoss,
    largestWin: avgWin * (2 + Math.random() * 3),
    largestLoss: avgLoss * (2 + Math.random() * 3),
    averageTradeDuration: 3 + Math.random() * 4,
    tradesPerMonth: 20 + Math.random() * 15
  };
  
  // Generate profit distribution
  const profitDistribution = [];
  for (let p = -500; p <= 500; p += 50) {
    // More winning trades than losing trades, and winning trades have higher profits
    const frequency = Math.floor(Math.exp(-Math.pow((p + 50) / 200, 2)) * 15);
    profitDistribution.push({
      range: `$${p}`,
      frequency: p < 0 ? frequency : frequency * 1.3
    });
  }
  
  // Generate duration analysis
  const durationAnalysis = [
    { duration: '< 1 hour', count: 20 + Math.floor(Math.random() * 30), winRate: 0.4 + Math.random() * 0.2 },
    { duration: '1-4 hours', count: 35 + Math.floor(Math.random() * 25), winRate: 0.45 + Math.random() * 0.2 },
    { duration: '4-24 hours', count: 25 + Math.floor(Math.random() * 20), winRate: 0.5 + Math.random() * 0.2 },
    { duration: '1-3 days', count: 15 + Math.floor(Math.random() * 15), winRate: 0.55 + Math.random() * 0.2 },
    { duration: '3-7 days', count: 10 + Math.floor(Math.random() * 10), winRate: 0.6 + Math.random() * 0.2 },
    { duration: '> 7 days', count: 5 + Math.floor(Math.random() * 5), winRate: 0.65 + Math.random() * 0.2 }
  ];
  
  return {
    outcomes,
    metrics,
    profitDistribution,
    durationAnalysis
  };
};

// Generate mock portfolio comparison
const generateMockPortfolioComparison = (benchmark: string, timeRange: string) => {
  // Generate dates based on time range
  const dates = generateDatesByTimeRange(timeRange);
  const dataPoints = dates.length;
  
  // Set up benchmark characteristics
  let benchmarkVolatility = 0.15;
  let benchmarkReturn = 0.10;
  let correlation = 0.7;
  
  switch (benchmark) {
    case 'S&P500':
      benchmarkVolatility = 0.15;
      benchmarkReturn = 0.10;
      correlation = 0.7;
      break;
    case 'USD_Index':
      benchmarkVolatility = 0.08;
      benchmarkReturn = 0.05;
      correlation = 0.3;
      break;
    case 'EUR_Index':
      benchmarkVolatility = 0.10;
      benchmarkReturn = 0.07;
      correlation = 0.5;
      break;
    case 'ForexAvg':
      benchmarkVolatility = 0.12;
      benchmarkReturn = 0.08;
      correlation = 0.8;
      break;
  }
  
  // Generate comparison data
  const comparison = [];
  let portfolioCumulative = 0;
  let benchmarkCumulative = 0;
  
  for (let i = 0; i < dataPoints; i++) {
    // Generate correlated returns
    const marketFactor = Math.random() * 0.02 - 0.01;
    const specificFactor = Math.random() * 0.015 - 0.0075;
    
    // Portfolio return has higher alpha and slightly higher volatility
    const portfolioDaily = (correlation * marketFactor + (1 - correlation) * specificFactor) * 1.2 + 0.0002;
    const benchmarkDaily = marketFactor * benchmarkVolatility / 0.15;
    
    // Accumulate returns
    portfolioCumulative += portfolioDaily + (portfolioDaily * portfolioCumulative);
    benchmarkCumulative += benchmarkDaily + (benchmarkDaily * benchmarkCumulative);
    
    // Calculate excess return
    const excess = portfolioDaily - benchmarkDaily;
    
    comparison.push({
      date: dates[i],
      portfolio: portfolioCumulative,
      benchmark: benchmarkCumulative,
      excess
    });
  }
  
  // Generate rolling beta
  const rollingBeta = [];
  const windowSize = Math.min(20, Math.floor(dataPoints / 5));
  
  for (let i = windowSize; i < dataPoints; i++) {
    const window = comparison.slice(i - windowSize, i);
    const portfolioReturns = window.map((d, j) => j === 0 ? 0 : (window[j].portfolio - window[j-1].portfolio) / (1 + window[j-1].portfolio));
    const benchmarkReturns = window.map((d, j) => j === 0 ? 0 : (window[j].benchmark - window[j-1].benchmark) / (1 + window[j-1].benchmark));
    
    // Calculate beta (simplified)
    const beta = 0.8 + Math.sin(i / 30) * 0.3 + Math.random() * 0.2;
    
    rollingBeta.push({
      date: dates[i],
      beta
    });
  }
  
  // Generate rolling alpha
  const rollingAlpha = [];
  
  for (let i = windowSize; i < dataPoints; i++) {
    const window = comparison.slice(i - windowSize, i);
    const portfolioReturn = window[windowSize-1].portfolio - window[0].portfolio;
    const benchmarkReturn = window[windowSize-1].benchmark - window[0].benchmark;
    
    // Calculate alpha (simplified)
    const alpha = 0.01 + Math.sin(i / 40) * 0.01 + Math.random() * 0.005;
    
    rollingAlpha.push({
      date: dates[i],
      alpha
    });
  }
  
  // Generate performance comparison metrics
  const metrics = [
    {
      name: 'Total Return',
      portfolio: portfolioCumulative,
      benchmark: benchmarkCumulative,
      difference: portfolioCumulative - benchmarkCumulative
    },
    {
      name: 'Annualized Return',
      portfolio: Math.pow(1 + portfolioCumulative, 252 / dataPoints) - 1,
      benchmark: Math.pow(1 + benchmarkCumulative, 252 / dataPoints) - 1,
      difference: Math.pow(1 + portfolioCumulative, 252 / dataPoints) - Math.pow(1 + benchmarkCumulative, 252 / dataPoints)
    },
    {
      name: 'Volatility',
      portfolio: 0.12 + Math.random() * 0.04,
      benchmark: benchmarkVolatility * (0.9 + Math.random() * 0.2),
      difference: 0.12 + Math.random() * 0.04 - benchmarkVolatility * (0.9 + Math.random() * 0.2)
    },
    {
      name: 'Sharpe Ratio',
      portfolio: 1.2 + Math.random() * 0.6,
      benchmark: 0.9 + Math.random() * 0.4,
      difference: 0.3 + Math.random() * 0.2
    },
    {
      name: 'Max Drawdown',
      portfolio: 0.15 + Math.random() * 0.05,
      benchmark: 0.18 + Math.random() * 0.07,
      difference: -0.03 - Math.random() * 0.02
    },
    {
      name: 'Beta',
      portfolio: 0.85 + Math.random() * 0.2,
      benchmark: 1.0,
      difference: -0.15 + Math.random() * 0.2
    },
    {
      name: 'Alpha',
      portfolio: 0.03 + Math.random() * 0.02,
      benchmark: 0,
      difference: 0.03 + Math.random() * 0.02
    },
    {
      name: 'Information Ratio',
      portfolio: 0.6 + Math.random() * 0.4,
      benchmark: 0,
      difference: 0.6 + Math.random() * 0.4
    }
  ];
  
  return {
    comparison,
    rollingBeta,
    rollingAlpha,
    metrics
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

// Helper function to generate dates based on time range
const generateDatesByTimeRange = (timeRange: string) => {
  const now = new Date();
  const dates = [];
  let days = 30; // default to 1 month
  
  switch (timeRange) {
    case '1w':
      days = 7;
      break;
    case '1m':
      days = 30;
      break;
    case '3m':
      days = 90;
      break;
    case '6m':
      days = 180;
      break;
    case '1y':
      days = 365;
      break;
    case 'ytd':
      const startOfYear = new Date(now.getFullYear(), 0, 1);
      days = Math.floor((now.getTime() - startOfYear.getTime()) / (1000 * 60 * 60 * 24));
      break;
    case 'all':
      days = 1095; // 3 years
      break;
  }
  
  // If period is too long, sample dates to keep the chart readable
  const stride = days > 365 ? Math.floor(days / 365) : 1;
  
  for (let i = days; i >= 0; i -= stride) {
    const date = new Date(now);
    date.setDate(now.getDate() - i);
    dates.push(date.toISOString().split('T')[0]);
  }
  
  return dates;
};
