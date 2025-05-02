import React, { useState, useCallback } from 'react';
import { Box, Paper, Grid, Typography, Tabs, Tab, Button, Slider, FormControl, InputLabel, Select, MenuItem, TextField } from '@mui/material';
import { styled } from '@mui/material/styles';
import { useQuery, useMutation } from 'react-query';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, ZAxis } from 'recharts';

// Services
import { 
  fetchIndicatorData, 
  fetchCorrelationData, 
  fetchRegressionAnalysis,
  runBacktest,
  fetchVolatilityAnalysis
} from '../../services/analysisService';

const AnalysisContainer = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  height: '100%',
  overflow: 'auto',
}));

const ControlPanel = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  marginBottom: theme.spacing(2),
}));

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
      id={`analysis-tabpanel-${index}`}
      aria-labelledby={`analysis-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

export const InteractiveAnalysisTools: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [selectedIndicators, setSelectedIndicators] = useState<string[]>(['macd', 'rsi']);
  const [correlationPeriod, setCorrelationPeriod] = useState(30);
  const [selectedAssets, setSelectedAssets] = useState<string[]>(['EUR/USD', 'GBP/USD']);
  const [backtestParams, setBacktestParams] = useState({
    strategy: 'momentum',
    startDate: '2023-01-01',
    endDate: '2023-12-31',
    initialCapital: 10000,
  });
  const [volatilityWindow, setVolatilityWindow] = useState(14);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // Fetch indicator data
  const { data: indicatorData, isLoading: isIndicatorsLoading } = useQuery(
    ['indicators', selectedIndicators], 
    () => fetchIndicatorData(selectedIndicators)
  );

  // Fetch correlation data
  const { data: correlationData, isLoading: isCorrelationLoading } = useQuery(
    ['correlation', correlationPeriod, selectedAssets], 
    () => fetchCorrelationData(selectedAssets, correlationPeriod)
  );

  // Fetch regression analysis
  const { data: regressionData, isLoading: isRegressionLoading } = useQuery(
    ['regression', selectedAssets[0]], 
    () => fetchRegressionAnalysis(selectedAssets[0])
  );

  // Fetch volatility analysis
  const { data: volatilityData, isLoading: isVolatilityLoading } = useQuery(
    ['volatility', volatilityWindow, selectedAssets], 
    () => fetchVolatilityAnalysis(selectedAssets, volatilityWindow)
  );

  // Backtest mutation
  const backtestMutation = useMutation(runBacktest, {
    onSuccess: (data) => {
      console.log('Backtest completed successfully', data);
    },
  });

  const handleBacktest = () => {
    backtestMutation.mutate(backtestParams);
  };

  const handleIndicatorChange = (event: React.ChangeEvent<{ value: unknown }>) => {
    setSelectedIndicators(event.target.value as string[]);
  };

  const handleAssetChange = (event: React.ChangeEvent<{ value: unknown }>) => {
    setSelectedAssets(event.target.value as string[]);
  };

  const handleBacktestParamChange = (param: string) => (event: React.ChangeEvent<HTMLInputElement>) => {
    setBacktestParams({
      ...backtestParams,
      [param]: event.target.value,
    });
  };

  return (
    <Box sx={{ height: '100%' }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Interactive Analysis Tools
      </Typography>

      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={tabValue} onChange={handleTabChange} aria-label="analysis tabs">
          <Tab label="Technical Indicators" id="analysis-tab-0" />
          <Tab label="Correlation Analysis" id="analysis-tab-1" />
          <Tab label="Regression Analysis" id="analysis-tab-2" />
          <Tab label="Backtesting" id="analysis-tab-3" />
          <Tab label="Volatility Analysis" id="analysis-tab-4" />
        </Tabs>
      </Box>

      {/* Technical Indicators Tab */}
      <TabPanel value={tabValue} index={0}>
        <ControlPanel>
          <FormControl fullWidth>
            <InputLabel id="indicator-select-label">Indicators</InputLabel>
            <Select
              labelId="indicator-select-label"
              multiple
              value={selectedIndicators}
              onChange={handleIndicatorChange}
              renderValue={(selected) => (selected as string[]).join(', ')}
            >
              <MenuItem value="macd">MACD</MenuItem>
              <MenuItem value="rsi">RSI</MenuItem>
              <MenuItem value="bollinger">Bollinger Bands</MenuItem>
              <MenuItem value="ema">EMA</MenuItem>
              <MenuItem value="sma">SMA</MenuItem>
              <MenuItem value="atr">ATR</MenuItem>
            </Select>
          </FormControl>
          <FormControl fullWidth sx={{ mt: 2 }}>
            <InputLabel id="asset-select-label">Assets</InputLabel>
            <Select
              labelId="asset-select-label"
              value={selectedAssets[0]}
              onChange={(e) => setSelectedAssets([e.target.value as string])}
            >
              <MenuItem value="EUR/USD">EUR/USD</MenuItem>
              <MenuItem value="GBP/USD">GBP/USD</MenuItem>
              <MenuItem value="USD/JPY">USD/JPY</MenuItem>
              <MenuItem value="AUD/USD">AUD/USD</MenuItem>
            </Select>
          </FormControl>
        </ControlPanel>

        <AnalysisContainer>
          <Typography variant="h6" gutterBottom>
            Technical Indicator Analysis
          </Typography>
          {isIndicatorsLoading ? (
            <Typography>Loading indicator data...</Typography>
          ) : (
            <Grid container spacing={2}>
              {selectedIndicators.map((indicator) => (
                <Grid item xs={12} md={6} key={indicator}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      {indicator.toUpperCase()}
                    </Typography>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart
                        data={indicatorData?.[indicator] || []}
                        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="value" stroke="#8884d8" />
                        {indicator === 'bollinger' && (
                          <>
                            <Line type="monotone" dataKey="upperBand" stroke="#82ca9d" />
                            <Line type="monotone" dataKey="lowerBand" stroke="#ff8042" />
                          </>
                        )}
                      </LineChart>
                    </ResponsiveContainer>
                  </Paper>
                </Grid>
              ))}
            </Grid>
          )}
        </AnalysisContainer>
      </TabPanel>

      {/* Correlation Analysis Tab */}
      <TabPanel value={tabValue} index={1}>
        <ControlPanel>
          <Typography gutterBottom>Time Period (days)</Typography>
          <Slider
            value={correlationPeriod}
            onChange={(_event, newValue) => setCorrelationPeriod(newValue as number)}
            aria-labelledby="correlation-period-slider"
            valueLabelDisplay="auto"
            step={5}
            marks
            min={5}
            max={90}
          />
          <FormControl fullWidth sx={{ mt: 2 }}>
            <InputLabel id="correlation-assets-label">Assets</InputLabel>
            <Select
              labelId="correlation-assets-label"
              multiple
              value={selectedAssets}
              onChange={handleAssetChange}
              renderValue={(selected) => (selected as string[]).join(', ')}
            >
              <MenuItem value="EUR/USD">EUR/USD</MenuItem>
              <MenuItem value="GBP/USD">GBP/USD</MenuItem>
              <MenuItem value="USD/JPY">USD/JPY</MenuItem>
              <MenuItem value="AUD/USD">AUD/USD</MenuItem>
              <MenuItem value="USD/CAD">USD/CAD</MenuItem>
              <MenuItem value="EUR/GBP">EUR/GBP</MenuItem>
            </Select>
          </FormControl>
        </ControlPanel>

        <AnalysisContainer>
          <Typography variant="h6" gutterBottom>
            Asset Correlation Matrix
          </Typography>
          {isCorrelationLoading ? (
            <Typography>Loading correlation data...</Typography>
          ) : (
            <ResponsiveContainer width="100%" height={500}>
              <ScatterChart
                margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
              >
                <CartesianGrid />
                <XAxis 
                  type="category" 
                  dataKey="x" 
                  name="Asset 1" 
                  allowDuplicatedCategory={false} 
                />
                <YAxis 
                  type="category" 
                  dataKey="y" 
                  name="Asset 2" 
                  allowDuplicatedCategory={false}
                />
                <ZAxis 
                  type="number" 
                  dataKey="z" 
                  range={[50, 500]} 
                  name="Correlation" 
                />
                <Tooltip 
                  cursor={{ strokeDasharray: '3 3' }} 
                  formatter={(value, name) => [
                    name === 'z' ? `${(Number(value) * 100).toFixed(2)}%` : value, 
                    name === 'z' ? 'Correlation' : name
                  ]}
                />
                <Scatter 
                  name="Correlation" 
                  data={correlationData || []} 
                  fill="#8884d8" 
                  shape="circle"
                />
              </ScatterChart>
            </ResponsiveContainer>
          )}
        </AnalysisContainer>
      </TabPanel>

      {/* Regression Analysis Tab */}
      <TabPanel value={tabValue} index={2}>
        <ControlPanel>
          <FormControl fullWidth>
            <InputLabel id="regression-asset-label">Asset</InputLabel>
            <Select
              labelId="regression-asset-label"
              value={selectedAssets[0]}
              onChange={(e) => setSelectedAssets([e.target.value as string])}
            >
              <MenuItem value="EUR/USD">EUR/USD</MenuItem>
              <MenuItem value="GBP/USD">GBP/USD</MenuItem>
              <MenuItem value="USD/JPY">USD/JPY</MenuItem>
              <MenuItem value="AUD/USD">AUD/USD</MenuItem>
            </Select>
          </FormControl>
        </ControlPanel>

        <AnalysisContainer>
          <Typography variant="h6" gutterBottom>
            Regression Analysis Results
          </Typography>
          {isRegressionLoading ? (
            <Typography>Loading regression data...</Typography>
          ) : (
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Price vs Predicted
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart
                      data={regressionData?.predictions || []}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="actual" stroke="#8884d8" name="Actual" />
                      <Line type="monotone" dataKey="predicted" stroke="#82ca9d" name="Predicted" />
                    </LineChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Model Statistics
                  </Typography>
                  <Typography variant="body2">
                    <strong>R-Squared:</strong> {regressionData?.stats?.r2.toFixed(4)}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Mean Squared Error:</strong> {regressionData?.stats?.mse.toFixed(6)}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Mean Absolute Error:</strong> {regressionData?.stats?.mae.toFixed(6)}
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Feature Importance
                  </Typography>
                  {regressionData?.featureImportance?.map((feature) => (
                    <Box key={feature.name} sx={{ mb: 1 }}>
                      <Typography variant="body2">
                        {feature.name}: {feature.importance.toFixed(4)}
                      </Typography>
                      <Box
                        sx={{
                          width: `${feature.importance * 100}%`,
                          height: 8,
                          backgroundColor: '#8884d8',
                          borderRadius: 1,
                        }}
                      />
                    </Box>
                  ))}
                </Paper>
              </Grid>
            </Grid>
          )}
        </AnalysisContainer>
      </TabPanel>

      {/* Backtesting Tab */}
      <TabPanel value={tabValue} index={3}>
        <ControlPanel>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel id="strategy-label">Strategy</InputLabel>
                <Select
                  labelId="strategy-label"
                  value={backtestParams.strategy}
                  onChange={(e) => setBacktestParams({...backtestParams, strategy: e.target.value})}
                >
                  <MenuItem value="momentum">Momentum</MenuItem>
                  <MenuItem value="meanReversion">Mean Reversion</MenuItem>
                  <MenuItem value="breakout">Breakout</MenuItem>
                  <MenuItem value="mlBased">ML-Based</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                label="Initial Capital"
                type="number"
                fullWidth
                value={backtestParams.initialCapital}
                onChange={handleBacktestParamChange('initialCapital')}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                label="Start Date"
                type="date"
                fullWidth
                InputLabelProps={{ shrink: true }}
                value={backtestParams.startDate}
                onChange={handleBacktestParamChange('startDate')}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                label="End Date"
                type="date"
                fullWidth
                InputLabelProps={{ shrink: true }}
                value={backtestParams.endDate}
                onChange={handleBacktestParamChange('endDate')}
              />
            </Grid>
            <Grid item xs={12}>
              <Button 
                variant="contained" 
                color="primary" 
                fullWidth
                onClick={handleBacktest}
                disabled={backtestMutation.isLoading}
              >
                {backtestMutation.isLoading ? 'Running Backtest...' : 'Run Backtest'}
              </Button>
            </Grid>
          </Grid>
        </ControlPanel>

        <AnalysisContainer>
          <Typography variant="h6" gutterBottom>
            Backtest Results
          </Typography>
          {backtestMutation.isLoading ? (
            <Typography>Running backtest...</Typography>
          ) : backtestMutation.isSuccess ? (
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Equity Curve
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart
                      data={backtestMutation.data?.equityCurve || []}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="equity" stroke="#8884d8" name="Portfolio Value" />
                      <Line type="monotone" dataKey="benchmark" stroke="#82ca9d" name="Benchmark" />
                    </LineChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Performance Metrics
                  </Typography>
                  <Grid container spacing={1}>
                    {backtestMutation.data?.metrics && Object.entries(backtestMutation.data.metrics).map(([key, value]) => (
                      <Grid item xs={6} key={key}>
                        <Typography variant="body2">
                          <strong>{key.replace(/([A-Z])/g, ' $1').replace(/^./, (str) => str.toUpperCase())}:</strong> {typeof value === 'number' ? value.toFixed(4) : value}
                        </Typography>
                      </Grid>
                    ))}
                  </Grid>
                </Paper>
              </Grid>
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Trade Statistics
                  </Typography>
                  <Grid container spacing={1}>
                    {backtestMutation.data?.tradeStats && Object.entries(backtestMutation.data.tradeStats).map(([key, value]) => (
                      <Grid item xs={6} key={key}>
                        <Typography variant="body2">
                          <strong>{key.replace(/([A-Z])/g, ' $1').replace(/^./, (str) => str.toUpperCase())}:</strong> {typeof value === 'number' ? value.toFixed(2) : value}
                        </Typography>
                      </Grid>
                    ))}
                  </Grid>
                </Paper>
              </Grid>
            </Grid>
          ) : null}
        </AnalysisContainer>
      </TabPanel>

      {/* Volatility Analysis Tab */}
      <TabPanel value={tabValue} index={4}>
        <ControlPanel>
          <Typography gutterBottom>Volatility Window (days)</Typography>
          <Slider
            value={volatilityWindow}
            onChange={(_event, newValue) => setVolatilityWindow(newValue as number)}
            aria-labelledby="volatility-window-slider"
            valueLabelDisplay="auto"
            step={1}
            marks
            min={5}
            max={30}
          />
          <FormControl fullWidth sx={{ mt: 2 }}>
            <InputLabel id="volatility-assets-label">Assets</InputLabel>
            <Select
              labelId="volatility-assets-label"
              multiple
              value={selectedAssets}
              onChange={handleAssetChange}
              renderValue={(selected) => (selected as string[]).join(', ')}
            >
              <MenuItem value="EUR/USD">EUR/USD</MenuItem>
              <MenuItem value="GBP/USD">GBP/USD</MenuItem>
              <MenuItem value="USD/JPY">USD/JPY</MenuItem>
              <MenuItem value="AUD/USD">AUD/USD</MenuItem>
            </Select>
          </FormControl>
        </ControlPanel>

        <AnalysisContainer>
          <Typography variant="h6" gutterBottom>
            Volatility Analysis
          </Typography>
          {isVolatilityLoading ? (
            <Typography>Loading volatility data...</Typography>
          ) : (
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Historical Volatility
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart
                      data={volatilityData?.historical || []}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      {selectedAssets.map((asset, index) => (
                        <Line 
                          key={asset} 
                          type="monotone" 
                          dataKey={asset.replace('/', '')} 
                          stroke={['#8884d8', '#82ca9d', '#ffc658', '#ff8042'][index % 4]} 
                          name={asset}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Volatility Regime
                  </Typography>
                  {volatilityData?.currentRegime && Object.entries(volatilityData.currentRegime).map(([asset, regime]) => (
                    <Box key={asset} sx={{ mb: 2 }}>
                      <Typography variant="body1">
                        <strong>{asset}:</strong> {regime.regime}
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                        <Box sx={{ width: '100%', mr: 1 }}>
                          <Box
                            sx={{
                              height: 10,
                              background: `linear-gradient(to right, #4caf50, #ffeb3b, #f44336)`,
                              borderRadius: 5,
                              position: 'relative',
                            }}
                          >
                            <Box
                              sx={{
                                position: 'absolute',
                                height: 16,
                                width: 4,
                                backgroundColor: '#000',
                                borderRadius: 1,
                                left: `${regime.percentile}%`,
                                top: -3,
                              }}
                            />
                          </Box>
                        </Box>
                        <Box sx={{ minWidth: 35 }}>
                          <Typography variant="body2" color="text.secondary">
                            {`${regime.percentile}%`}
                          </Typography>
                        </Box>
                      </Box>
                    </Box>
                  ))}
                </Paper>
              </Grid>
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Summary Statistics
                  </Typography>
                  {volatilityData?.stats && Object.entries(volatilityData.stats).map(([asset, stats]) => (
                    <Box key={asset} sx={{ mb: 2 }}>
                      <Typography variant="body1">
                        <strong>{asset}</strong>
                      </Typography>
                      <Typography variant="body2">
                        Mean: {stats.mean.toFixed(5)}
                      </Typography>
                      <Typography variant="body2">
                        Min: {stats.min.toFixed(5)}
                      </Typography>
                      <Typography variant="body2">
                        Max: {stats.max.toFixed(5)}
                      </Typography>
                      <Typography variant="body2">
                        Current: {stats.current.toFixed(5)}
                      </Typography>
                    </Box>
                  ))}
                </Paper>
              </Grid>
            </Grid>
          )}
        </AnalysisContainer>
      </TabPanel>
    </Box>
  );
};

export default InteractiveAnalysisTools;
