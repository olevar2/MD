import React from 'react';
import { 
  Box, 
  Grid, 
  Typography, 
  Paper, 
  Tabs, 
  Tab,
  ButtonGroup,
  Button as MuiButton,
  IconButton,
  TextField,
  Menu,
  MenuItem,
  Divider,
  Chip,
  Tooltip,
  FormControl,
  InputLabel,
  Select
} from '@mui/material';
import { Card } from '../ui_library/Card';
import { Chart } from '../ui_library/Chart';
import { StatusIndicator } from '../ui_library/StatusIndicator';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`chart-tabpanel-${index}`}
      aria-labelledby={`chart-tab-${index}`}
      {...other}
      style={{ height: '100%' }}
    >
      {value === index && (
        <Box sx={{ height: '100%' }}>
          {children}
        </Box>
      )}
    </div>
  );
}

// Mock chart data for price
const generateMockPriceData = (length: number, basePrice: number, volatility: number) => {
  const startTime = new Date(Date.now() - length * 15 * 60 * 1000).getTime();
  const data = [];

  let lastPrice = basePrice;

  for (let i = 0; i < length; i++) {
    const time = new Date(startTime + i * 15 * 60 * 1000).toISOString().slice(0, -5);
    // Generate a random price movement with some trend
    const change = (Math.random() - 0.5) * 2 * volatility;
    lastPrice = Math.max(0, lastPrice + change);
    data.push({
      time,
      value: lastPrice
    });
  }

  return data;
};

// Mock chart data
const mockEurUsdData = generateMockPriceData(500, 1.0824, 0.0005);
const mockGbpUsdData = generateMockPriceData(500, 1.2534, 0.0008);
const mockUsdJpyData = generateMockPriceData(500, 154.35, 0.05);

// Mock indicator data
const mockMovingAverageData = mockEurUsdData.map((item, index) => {
  if (index < 20) {
    return {
      time: item.time,
      value: null
    };
  }
  
  let sum = 0;
  for (let i = index - 20; i < index; i++) {
    sum += mockEurUsdData[i].value;
  }
  
  return {
    time: item.time,
    value: sum / 20
  };
});

const mockRsiData = mockEurUsdData.map((item, index) => {
  const randomRsi = 30 + Math.random() * 40; // Random RSI between 30 and 70
  return {
    time: item.time,
    value: randomRsi
  };
});

// Mock predictions data
const mockPredictionData = mockEurUsdData.slice(-30).map((item, index) => {
  return {
    time: new Date(new Date(item.time).getTime() + (index + 1) * 15 * 60 * 1000).toISOString().slice(0, -5),
    value: mockEurUsdData[mockEurUsdData.length - 1].value * (1 + (Math.random() - 0.3) * 0.002 * index)
  };
});

// Available technical indicators
const availableIndicators = [
  { name: 'Moving Average', key: 'ma', periods: [20, 50, 200] },
  { name: 'Bollinger Bands', key: 'bb', period: 20, stdDev: 2 },
  { name: 'RSI', key: 'rsi', period: 14 },
  { name: 'MACD', key: 'macd', fastPeriod: 12, slowPeriod: 26, signalPeriod: 9 },
  { name: 'Stochastic', key: 'stoch', kPeriod: 14, dPeriod: 3 },
  { name: 'ATR', key: 'atr', period: 14 },
  { name: 'Ichimoku Cloud', key: 'ichimoku' },
  { name: 'Volume', key: 'volume' },
  { name: 'Fibonacci Retracement', key: 'fib' },
];

// Pattern recognition options
const patternOptions = [
  { name: 'Double Top/Bottom', key: 'doubleTopBottom' },
  { name: 'Head and Shoulders', key: 'headAndShoulders' },
  { name: 'Triangle Patterns', key: 'triangles' },
  { name: 'Flag Patterns', key: 'flags' },
  { name: 'Engulfing Patterns', key: 'engulfing' },
  { name: 'Doji', key: 'doji' },
  { name: 'Morning/Evening Star', key: 'star' },
  { name: 'Elliott Wave Patterns', key: 'elliottWave' },
  { name: 'Harmonic Patterns', key: 'harmonic' },
];

export const AnalysisCharting: React.FC = () => {
  const [tabValue, setTabValue] = React.useState(0);
  const [currentSymbol, setCurrentSymbol] = React.useState('EURUSD');
  const [timeframe, setTimeframe] = React.useState('1H');
  const [chartType, setChartType] = React.useState('candlestick');
  const [showML, setShowML] = React.useState(false);
  const [activeIndicators, setActiveIndicators] = React.useState(['ma']);
  const [indicatorMenuAnchorEl, setIndicatorMenuAnchorEl] = React.useState<null | HTMLElement>(null);
  const [patternMenuAnchorEl, setPatternMenuAnchorEl] = React.useState<null | HTMLElement>(null);
  const [annotationMode, setAnnotationMode] = React.useState(false);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleTimeframeChange = (newTimeframe: string) => {
    setTimeframe(newTimeframe);
  };

  const handleChartTypeChange = (newChartType: string) => {
    setChartType(newChartType);
  };

  const handleSymbolChange = (event) => {
    setCurrentSymbol(event.target.value);
  };

  const handleIndicatorMenuOpen = (event: React.MouseEvent<HTMLButtonElement>) => {
    setIndicatorMenuAnchorEl(event.currentTarget);
  };

  const handleIndicatorMenuClose = () => {
    setIndicatorMenuAnchorEl(null);
  };

  const handlePatternMenuOpen = (event: React.MouseEvent<HTMLButtonElement>) => {
    setPatternMenuAnchorEl(event.currentTarget);
  };

  const handlePatternMenuClose = () => {
    setPatternMenuAnchorEl(null);
  };

  const handleToggleIndicator = (indicator: string) => {
    if (activeIndicators.includes(indicator)) {
      setActiveIndicators(activeIndicators.filter(i => i !== indicator));
    } else {
      setActiveIndicators([...activeIndicators, indicator]);
    }
  };

  const toggleML = () => {
    setShowML(!showML);
  };

  const toggleAnnotationMode = () => {
    setAnnotationMode(!annotationMode);
  };

  // Select the appropriate data based on the selected symbol
  const getCurrentChartData = () => {
    switch(currentSymbol) {
      case 'EURUSD':
        return mockEurUsdData;
      case 'GBPUSD':
        return mockGbpUsdData;
      case 'USDJPY':
        return mockUsdJpyData;
      default:
        return mockEurUsdData;
    }
  };

  return (
    <Box sx={{ flexGrow: 1, height: '100vh', overflow: 'hidden', p: 2 }}>
      <Grid container spacing={2} sx={{ height: '100%' }}>
        {/* Chart Header */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2, mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <FormControl sx={{ minWidth: 120 }}>
                <InputLabel id="symbol-select-label">Symbol</InputLabel>
                <Select
                  labelId="symbol-select-label"
                  id="symbol-select"
                  value={currentSymbol}
                  label="Symbol"
                  onChange={handleSymbolChange}
                  size="small"
                >
                  <MenuItem value="EURUSD">EUR/USD</MenuItem>
                  <MenuItem value="GBPUSD">GBP/USD</MenuItem>
                  <MenuItem value="USDJPY">USD/JPY</MenuItem>
                  <MenuItem value="AUDUSD">AUD/USD</MenuItem>
                  <MenuItem value="USDCAD">USD/CAD</MenuItem>
                </Select>
              </FormControl>

              <Box>
                <ButtonGroup variant="outlined" size="small">
                  <MuiButton 
                    variant={timeframe === '5m' ? 'contained' : 'outlined'} 
                    onClick={() => handleTimeframeChange('5m')}
                  >
                    5m
                  </MuiButton>
                  <MuiButton 
                    variant={timeframe === '15m' ? 'contained' : 'outlined'} 
                    onClick={() => handleTimeframeChange('15m')}
                  >
                    15m
                  </MuiButton>
                  <MuiButton 
                    variant={timeframe === '1H' ? 'contained' : 'outlined'} 
                    onClick={() => handleTimeframeChange('1H')}
                  >
                    1H
                  </MuiButton>
                  <MuiButton 
                    variant={timeframe === '4H' ? 'contained' : 'outlined'} 
                    onClick={() => handleTimeframeChange('4H')}
                  >
                    4H
                  </MuiButton>
                  <MuiButton 
                    variant={timeframe === '1D' ? 'contained' : 'outlined'} 
                    onClick={() => handleTimeframeChange('1D')}
                  >
                    1D
                  </MuiButton>
                  <MuiButton 
                    variant={timeframe === '1W' ? 'contained' : 'outlined'} 
                    onClick={() => handleTimeframeChange('1W')}
                  >
                    1W
                  </MuiButton>
                </ButtonGroup>
              </Box>

              <Box>
                <ButtonGroup variant="outlined" size="small">
                  <MuiButton 
                    variant={chartType === 'line' ? 'contained' : 'outlined'} 
                    onClick={() => handleChartTypeChange('line')}
                  >
                    Line
                  </MuiButton>
                  <MuiButton 
                    variant={chartType === 'candlestick' ? 'contained' : 'outlined'} 
                    onClick={() => handleChartTypeChange('candlestick')}
                  >
                    Candles
                  </MuiButton>
                  <MuiButton 
                    variant={chartType === 'bar' ? 'contained' : 'outlined'} 
                    onClick={() => handleChartTypeChange('bar')}
                  >
                    Bars
                  </MuiButton>
                  <MuiButton 
                    variant={chartType === 'area' ? 'contained' : 'outlined'} 
                    onClick={() => handleChartTypeChange('area')}
                  >
                    Area
                  </MuiButton>
                </ButtonGroup>
              </Box>
            </Box>

            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <MuiButton 
                variant="outlined" 
                color="primary"
                onClick={handleIndicatorMenuOpen}
                endIcon={<span>+</span>}
              >
                Indicators
              </MuiButton>

              <MuiButton 
                variant="outlined" 
                color="secondary"
                onClick={handlePatternMenuOpen}
                endIcon={<span>+</span>}
              >
                Patterns
              </MuiButton>

              <MuiButton
                variant={showML ? 'contained' : 'outlined'}
                color="secondary"
                onClick={toggleML}
              >
                ML Predictions
              </MuiButton>

              <MuiButton
                variant={annotationMode ? 'contained' : 'outlined'}
                color="primary"
                onClick={toggleAnnotationMode}
              >
                Draw
              </MuiButton>
            </Box>
          </Paper>
        </Grid>

        {/* Main Chart Area */}
        <Grid item xs={12} md={9} sx={{ height: 'calc(70% - 48px)' }}>
          <Paper sx={{ height: '100%', p: 2, position: 'relative' }}>
            <Box sx={{ position: 'absolute', top: 10, left: 16, zIndex: 10 }}>
              <Typography variant="h6">
                {currentSymbol === 'EURUSD' ? 'EUR/USD' : currentSymbol === 'GBPUSD' ? 'GBP/USD' : 'USD/JPY'} - {timeframe}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {new Date().toLocaleString()}
              </Typography>
            </Box>

            <Box sx={{ position: 'absolute', top: 10, right: 16, zIndex: 10, display: 'flex', gap: 1 }}>
              {activeIndicators.map((indicator) => (
                <Chip 
                  key={indicator} 
                  label={availableIndicators.find(i => i.key === indicator)?.name || indicator} 
                  size="small" 
                  onDelete={() => handleToggleIndicator(indicator)}
                  color="primary"
                  variant="outlined"
                />
              ))}
            </Box>

            <Box sx={{ height: '100%', pt: 5 }}>
              <Chart
                data={getCurrentChartData()}
                height="100%"
                chartType="area"
              />
            </Box>
          </Paper>
        </Grid>

        {/* Right Sidebar - Analysis, Signals, Correlations */}
        <Grid item xs={12} md={3} sx={{ height: 'calc(70% - 48px)' }}>
          <Paper sx={{ height: '100%' }}>
            <Tabs value={tabValue} onChange={handleTabChange} aria-label="analysis tabs" variant="fullWidth">
              <Tab label="Analysis" />
              <Tab label="Signals" />
              <Tab label="Correlations" />
            </Tabs>
            
            <TabPanel value={tabValue} index={0}>
              <Box sx={{ p: 2, height: 'calc(100% - 48px)', overflow: 'auto' }}>
                <Typography variant="subtitle1" gutterBottom>Technical Analysis</Typography>
                
                <Card sx={{ mb: 2 }}>
                  <Typography variant="subtitle2">Moving Averages</Typography>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                    <Typography variant="body2">MA (20)</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600, color: 'success.main' }}>BULLISH</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                    <Typography variant="body2">MA (50)</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600, color: 'success.main' }}>BULLISH</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                    <Typography variant="body2">MA (200)</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600, color: 'error.main' }}>BEARISH</Typography>
                  </Box>
                </Card>
                
                <Card sx={{ mb: 2 }}>
                  <Typography variant="subtitle2">Oscillators</Typography>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                    <Typography variant="body2">RSI (14)</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600, color: 'warning.main' }}>NEUTRAL (52.3)</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                    <Typography variant="body2">MACD</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600, color: 'success.main' }}>BULLISH</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                    <Typography variant="body2">Stoch (14,3,3)</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600, color: 'success.main' }}>BULLISH</Typography>
                  </Box>
                </Card>
                
                <Card sx={{ mb: 2 }}>
                  <Typography variant="subtitle2">Pivot Points (Daily)</Typography>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                    <Typography variant="body2">R3</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>1.0912</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                    <Typography variant="body2">R2</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>1.0878</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                    <Typography variant="body2">R1</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>1.0856</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                    <Typography variant="body2">PP</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>1.0834</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                    <Typography variant="body2">S1</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>1.0812</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                    <Typography variant="body2">S2</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>1.0790</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                    <Typography variant="body2">S3</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>1.0768</Typography>
                  </Box>
                </Card>
                
                <Card sx={{ mb: 2 }}>
                  <Typography variant="subtitle2">Summary</Typography>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1, alignItems: 'center' }}>
                    <Typography variant="body1" sx={{ fontWeight: 600 }}>Overall Signal:</Typography>
                    <Chip label="BULLISH" color="success" size="small" />
                  </Box>
                </Card>
              </Box>
            </TabPanel>
            
            <TabPanel value={tabValue} index={1}>
              <Box sx={{ p: 2, height: 'calc(100% - 48px)', overflow: 'auto' }}>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>Recent Signals</Typography>
                  
                  <Paper elevation={0} sx={{ p: 2, bgcolor: 'background.default', borderRadius: 2, mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="subtitle2">Bullish Engulfing</Typography>
                      <Typography variant="caption" color="text.secondary">15 min ago</Typography>
                    </Box>
                    <Typography variant="body2" sx={{ mt: 1 }}>Strong bullish engulfing pattern detected at support level</Typography>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                      <Typography variant="caption" color="text.secondary">Confidence: 78%</Typography>
                      <StatusIndicator status="success" size="small" />
                    </Box>
                  </Paper>
                  
                  <Paper elevation={0} sx={{ p: 2, bgcolor: 'background.default', borderRadius: 2, mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="subtitle2">RSI Divergence</Typography>
                      <Typography variant="caption" color="text.secondary">45 min ago</Typography>
                    </Box>
                    <Typography variant="body2" sx={{ mt: 1 }}>Bullish divergence between price and RSI indicator</Typography>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                      <Typography variant="caption" color="text.secondary">Confidence: 65%</Typography>
                      <StatusIndicator status="warning" size="small" />
                    </Box>
                  </Paper>
                  
                  <Paper elevation={0} sx={{ p: 2, bgcolor: 'background.default', borderRadius: 2, mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="subtitle2">Support Test</Typography>
                      <Typography variant="caption" color="text.secondary">1h 20m ago</Typography>
                    </Box>
                    <Typography variant="body2" sx={{ mt: 1 }}>Price testing key support level at 1.0812</Typography>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                      <Typography variant="caption" color="text.secondary">Confidence: 82%</Typography>
                      <StatusIndicator status="success" size="small" />
                    </Box>
                  </Paper>
                </Box>
              </Box>
            </TabPanel>
            
            <TabPanel value={tabValue} index={2}>
              <Box sx={{ p: 2, height: 'calc(100% - 48px)', overflow: 'auto' }}>
                <Typography variant="subtitle1" gutterBottom>Currency Correlations</Typography>
                
                <Card sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>EUR/USD Correlations</Typography>
                  
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                    <Typography variant="body2">GBP/USD</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>+0.92</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                    <Typography variant="body2">USD/CHF</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>-0.95</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                    <Typography variant="body2">AUD/USD</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>+0.75</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                    <Typography variant="body2">USD/JPY</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>-0.52</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                    <Typography variant="body2">USD/CAD</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>-0.81</Typography>
                  </Box>
                </Card>
                
                <Card>
                  <Typography variant="subtitle2" gutterBottom>Correlation Heatmap</Typography>
                  <Typography variant="caption" color="text.secondary">
                    Hover over cells to see exact correlation values between currency pairs
                  </Typography>
                </Card>
              </Box>
            </TabPanel>
          </Paper>
        </Grid>

        {/* Bottom Panels - Multi-timeframe and Secondary Charts */}
        <Grid item xs={12} sx={{ height: 'calc(30%)' }}>
          <Paper sx={{ height: '100%', p: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="subtitle1">Multi-Timeframe Analysis</Typography>
              <ButtonGroup variant="outlined" size="small">
                <MuiButton>15m</MuiButton>
                <MuiButton>1H</MuiButton>
                <MuiButton>4H</MuiButton>
                <MuiButton>1D</MuiButton>
              </ButtonGroup>
            </Box>
            
            <Grid container spacing={2} sx={{ height: 'calc(100% - 40px)' }}>
              <Grid item xs={3} sx={{ height: '100%' }}>
                <Box sx={{ height: '100%', position: 'relative', bgcolor: 'background.default', borderRadius: 1 }}>
                  <Box sx={{ position: 'absolute', top: 8, left: 8, zIndex: 5 }}>
                    <Typography variant="caption">EUR/USD - 15m</Typography>
                  </Box>
                  <Chart
                    data={mockEurUsdData}
                    height="100%"
                    chartType="candlestick"
                  />
                </Box>
              </Grid>
              <Grid item xs={3} sx={{ height: '100%' }}>
                <Box sx={{ height: '100%', position: 'relative', bgcolor: 'background.default', borderRadius: 1 }}>
                  <Box sx={{ position: 'absolute', top: 8, left: 8, zIndex: 5 }}>
                    <Typography variant="caption">EUR/USD - 1H</Typography>
                  </Box>
                  <Chart
                    data={mockEurUsdData}
                    height="100%"
                    chartType="candlestick"
                  />
                </Box>
              </Grid>
              <Grid item xs={3} sx={{ height: '100%' }}>
                <Box sx={{ height: '100%', position: 'relative', bgcolor: 'background.default', borderRadius: 1 }}>
                  <Box sx={{ position: 'absolute', top: 8, left: 8, zIndex: 5 }}>
                    <Typography variant="caption">EUR/USD - 4H</Typography>
                  </Box>
                  <Chart
                    data={mockEurUsdData}
                    height="100%"
                    chartType="candlestick"
                  />
                </Box>
              </Grid>
              <Grid item xs={3} sx={{ height: '100%' }}>
                <Box sx={{ height: '100%', position: 'relative', bgcolor: 'background.default', borderRadius: 1 }}>
                  <Box sx={{ position: 'absolute', top: 8, left: 8, zIndex: 5 }}>
                    <Typography variant="caption">EUR/USD - 1D</Typography>
                  </Box>
                  <Chart
                    data={mockEurUsdData}
                    height="100%"
                    chartType="candlestick"
                  />
                </Box>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>

      {/* Indicators Menu */}
      <Menu
        anchorEl={indicatorMenuAnchorEl}
        open={Boolean(indicatorMenuAnchorEl)}
        onClose={handleIndicatorMenuClose}
      >
        {availableIndicators.map((indicator) => (
          <MenuItem 
            key={indicator.key} 
            onClick={() => {
              handleToggleIndicator(indicator.key);
              handleIndicatorMenuClose();
            }}
            sx={{ 
              backgroundColor: activeIndicators.includes(indicator.key) ? 'action.selected' : 'inherit'
            }}
          >
            {indicator.name}
          </MenuItem>
        ))}
      </Menu>

      {/* Pattern Recognition Menu */}
      <Menu
        anchorEl={patternMenuAnchorEl}
        open={Boolean(patternMenuAnchorEl)}
        onClose={handlePatternMenuClose}
      >
        {patternOptions.map((pattern) => (
          <MenuItem 
            key={pattern.key} 
            onClick={handlePatternMenuClose}
          >
            {pattern.name}
          </MenuItem>
        ))}
      </Menu>
    </Box>
  );
};

export default AnalysisCharting;
