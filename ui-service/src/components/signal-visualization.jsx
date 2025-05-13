import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Container, 
  Paper, 
  Typography, 
  Grid, 
  Card, 
  CardContent, 
  Button, 
  CircularProgress, 
  Chip,
  Slider,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Divider,
  IconButton,
  Tooltip,
  useTheme
} from '@mui/material';
import { 
  Refresh as RefreshIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  TrendingFlat as TrendingFlatIcon,
  Timeline as TimelineIcon,
  Assessment as AssessmentIcon,
  Info as InfoIcon,
  PriorityHigh as AlertIcon,
  CheckCircle as CheckCircleIcon
} from '@mui/icons-material';
import { 
  AreaChart, 
  Area, 
  BarChart, 
  Bar, 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as ChartTooltip, 
  Legend, 
  ResponsiveContainer,
  ReferenceLine,
  ComposedChart,
  Scatter
} from 'recharts';
import { useSignalApi } from '../hooks/useSignalApi';
import { formatDateTime, timeAgo } from '../utils/formatters';
import SignalDetailModal from './SignalDetailModal';
import SignalConfidenceGauge from './SignalConfidenceGauge';
import ModelMetricsCard from './ModelMetricsCard';
import MarketRegimePanel from './MarketRegimePanel';
import CorrelationHeatmap from './CorrelationHeatmap';

/**
 * Component for visualizing trading signals with confidence metrics
 */
const SignalVisualization = ({ instrumentId, modelIds }) => {
  const theme = useTheme();
  const [signals, setSignals] = useState([]);
  const [selectedSignal, setSelectedSignal] = useState(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [timeFrame, setTimeFrame] = useState('1h');
  const [confidenceThreshold, setConfidenceThreshold] = useState(60);
  const [selectedModels, setSelectedModels] = useState(modelIds || []);
  const [marketContext, setMarketContext] = useState(null);
  const [signalMetrics, setSignalMetrics] = useState(null);
  const [refreshKey, setRefreshKey] = useState(0);
  
  const signalApi = useSignalApi();
  
  // Available time frames
  const timeFrames = [
    { value: '5m', label: '5 min' },
    { value: '15m', label: '15 min' },
    { value: '1h', label: '1 hour' },
    { value: '4h', label: '4 hours' },
    { value: '1d', label: 'Daily' }
  ];
  
  // Fetch signals
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        
        // Fetch signals, market context, and metrics in parallel
        const [signalsData, contextData, metricsData] = await Promise.all([
          signalApi.getSignalsForInstrument(instrumentId, {
            timeFrame,
            models: selectedModels,
            minimumConfidence: confidenceThreshold / 100
          }),
          signalApi.getMarketContext(instrumentId),
          signalApi.getSignalMetrics(instrumentId, selectedModels)
        ]);
        
        setSignals(signalsData);
        setMarketContext(contextData);
        setSignalMetrics(metricsData);
        setError(null);
      } catch (err) {
        console.error('Error fetching signal data:', err);
        setError('Failed to load signal data. Please try again.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
    
    // Set up auto-refresh
    const intervalId = setInterval(() => {
      setRefreshKey(prevKey => prevKey + 1);
    }, 60000); // Refresh every minute
    
    return () => clearInterval(intervalId);
  }, [instrumentId, timeFrame, selectedModels, confidenceThreshold, refreshKey]);
  
  // Handle signal selection
  const handleSignalClick = (signal) => {
    setSelectedSignal(signal);
    setModalOpen(true);
  };
  
  // Handle modal close
  const handleModalClose = () => {
    setModalOpen(false);
  };
  
  // Handle manual refresh
  const handleRefresh = () => {
    setRefreshKey(prevKey => prevKey + 1);
  };
  
  // Handle time frame change
  const handleTimeFrameChange = (event) => {
    setTimeFrame(event.target.value);
  };
  
  // Handle confidence threshold change
  const handleConfidenceChange = (event, newValue) => {
    setConfidenceThreshold(newValue);
  };
  
  // Handle model selection change
  const handleModelSelectionChange = (event) => {
    setSelectedModels(event.target.value);
  };
  
  // Get color for signal based on direction and confidence
  const getSignalColor = (signal) => {
    if (signal.direction === 'BUY') {
      return signal.confidence > 75 ? theme.palette.success.main : theme.palette.success.light;
    } else if (signal.direction === 'SELL') {
      return signal.confidence > 75 ? theme.palette.error.main : theme.palette.error.light;
    } else {
      return theme.palette.text.secondary;
    }
  };
  
  // Get icon for signal direction
  const getDirectionIcon = (direction) => {
    if (direction === 'BUY') {
      return <TrendingUpIcon color="success" />;
    } else if (direction === 'SELL') {
      return <TrendingDownIcon color="error" />;
    } else {
      return <TrendingFlatIcon color="disabled" />;
    }
  };
  
  // Group signals by model for comparison
  const signalsByModel = selectedModels.reduce((acc, modelId) => {
    acc[modelId] = signals.filter(s => s.modelId === modelId);
    return acc;
  }, {});
  
  // Prepare chart data
  const prepareChartData = () => {
    // Create a map of time points to organize signals
    const timeMap = new Map();
    
    signals.forEach(signal => {
      const timeKey = new Date(signal.timestamp).getTime();
      if (!timeMap.has(timeKey)) {
        timeMap.set(timeKey, {
          time: signal.timestamp,
          price: signal.price
        });
      }
      
      // Add model-specific confidence
      timeMap.get(timeKey)[`${signal.modelId}_conf`] = signal.confidence;
      
      // Add model-specific direction (1=BUY, -1=SELL, 0=NEUTRAL)
      timeMap.get(timeKey)[`${signal.modelId}_dir`] = 
        signal.direction === 'BUY' ? 1 : signal.direction === 'SELL' ? -1 : 0;
    });
    
    // Convert map to array and sort by time
    return Array.from(timeMap.values()).sort((a, b) => 
      new Date(a.time) - new Date(b.time)
    );
  };
  
  const chartData = prepareChartData();
  
  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h4" component="h1">
            Signal Visualization: {instrumentId}
          </Typography>
          <Button 
            variant="outlined" 
            startIcon={<RefreshIcon />} 
            onClick={handleRefresh}
            disabled={loading}
          >
            Refresh
          </Button>
        </Box>
        
        {/* Controls */}
        <Paper sx={{ p: 2, mb: 3 }}>
          <Grid container spacing={3} alignItems="center">
            <Grid item xs={12} md={3}>
              <FormControl fullWidth>
                <InputLabel>Time Frame</InputLabel>
                <Select
                  value={timeFrame}
                  label="Time Frame"
                  onChange={handleTimeFrameChange}
                >
                  {timeFrames.map(tf => (
                    <MenuItem key={tf.value} value={tf.value}>{tf.label}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={3}>
              <FormControl fullWidth>
                <InputLabel>Models</InputLabel>
                <Select
                  multiple
                  value={selectedModels}
                  label="Models"
                  onChange={handleModelSelectionChange}
                  renderValue={(selected) => selected.join(', ')}
                >
                  {modelIds.map(modelId => (
                    <MenuItem key={modelId} value={modelId}>{modelId}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Confidence Threshold: {confidenceThreshold}%</Typography>
              <Slider
                value={confidenceThreshold}
                onChange={handleConfidenceChange}
                min={0}
                max={100}
                step={5}
                valueLabelDisplay="auto"
              />
            </Grid>
          </Grid>
        </Paper>
        
        {loading && !signals.length ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
            <CircularProgress />
          </Box>
        ) : error ? (
          <Paper sx={{ p: 3, bgcolor: 'error.lighter' }}>
            <Typography color="error">{error}</Typography>
          </Paper>
        ) : (
          <>
            {/* Market Context and Signal Metrics */}
            <Grid container spacing={2} sx={{ mb: 3 }}>
              <Grid item xs={12} md={8}>
                <MarketRegimePanel marketContext={marketContext} />
              </Grid>
              <Grid item xs={12} md={4}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>Signal Summary</Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">Total Signals (24h)</Typography>
                        <Typography variant="h6">{signalMetrics?.totalSignals24h || 0}</Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">Avg. Confidence</Typography>
                        <Typography variant="h6">{signalMetrics?.averageConfidence?.toFixed(1) || 0}%</Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">Signal Agreement</Typography>
                        <Typography variant="h6">{signalMetrics?.modelAgreementPercent?.toFixed(1) || 0}%</Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">Direction</Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          {getDirectionIcon(signalMetrics?.dominantDirection || 'NEUTRAL')}
                          <Typography variant="h6" sx={{ ml: 1 }}>
                            {signalMetrics?.dominantDirection || 'NEUTRAL'}
                          </Typography>
                        </Box>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
            
            {/* Signal Confidence Chart */}
            <Paper sx={{ p: 2, mb: 3 }}>
              <Typography variant="h6" gutterBottom>Signal Confidence Timeline</Typography>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" tickFormatter={(time) => formatDateTime(time, 'short')} />
                    <YAxis yAxisId="price" domain={['auto', 'auto']} />
                    <YAxis yAxisId="confidence" orientation="right" domain={[0, 100]} />
                    <ChartTooltip formatter={(value, name) => {
                      if (name.includes('_conf')) return [`${value}%`, 'Confidence'];
                      if (name.includes('_dir')) {
                        return [value === 1 ? 'BUY' : value === -1 ? 'SELL' : 'NEUTRAL', 'Direction'];
                      }
                      return [value, name];
                    }} />
                    <Legend />
                    
                    {/* Price line */}
                    <Line 
                      yAxisId="price" 
                      type="monotone" 
                      dataKey="price" 
                      name="Price"
                      stroke={theme.palette.info.main} 
                      dot={false}
                      strokeWidth={2}
                    />
                    
                    {/* Signal confidence by model */}
                    {selectedModels.map((modelId, index) => (
                      <React.Fragment key={modelId}>
                        <Bar 
                          yAxisId="confidence"
                          dataKey={`${modelId}_conf`}
                          name={`${modelId} Confidence`}
                          fill={theme.palette.primary[index % 5 * 100 + 300]}
                          barSize={20}
                        />
                        
                        <Scatter 
                          yAxisId="price"
                          dataKey="price"
                          name={`${modelId} Signals`}
                          fill={(entry) => {
                            const dir = entry[`${modelId}_dir`];
                            if (dir === 1) return theme.palette.success.main;
                            if (dir === -1) return theme.palette.error.main;
                            return theme.palette.grey[500];
                          }}
                          shape={(props) => {
                            const dir = props.payload[`${modelId}_dir`];
                            if (!dir) return null;
                            
                            return (
                              <svg x={props.cx - 10} y={props.cy - 10} width={20} height={20} fill="none" viewBox="0 0 24 24">
                                {dir === 1 ? (
                                  <path d="M7 14l5-5 5 5" fill="none" stroke={theme.palette.success.main} strokeWidth={2} />
                                ) : dir === -1 ? (
                                  <path d="M7 10l5 5 5-5" fill="none" stroke={theme.palette.error.main} strokeWidth={2} />
                                ) : null}
                              </svg>
                            );
                          }}
                        />
                      </React.Fragment>
                    ))}
                    
                    {/* Reference line for confidence threshold */}
                    <ReferenceLine y={confidenceThreshold} yAxisId="confidence" stroke={theme.palette.warning.main} strokeDasharray="3 3" />
                  </ComposedChart>
                </ResponsiveContainer>
              </Box>
            </Paper>
            
            {/* Models Performance */}
            <Typography variant="h6" gutterBottom>Model Performance</Typography>
            <Grid container spacing={2} sx={{ mb: 3 }}>
              {selectedModels.map((modelId, index) => (
                <Grid item xs={12} md={6} lg={4} key={modelId}>
                  <ModelMetricsCard
                    modelId={modelId}
                    metrics={signalMetrics?.modelMetrics?.[modelId] || {}}
                    signalCount={signalsByModel[modelId]?.length || 0}
                    recentSignals={signalsByModel[modelId]?.slice(0, 5) || []}
                  />
                </Grid>
              ))}
            </Grid>
            
            {/* Latest Signals */}
            <Paper sx={{ p: 2, mb: 3 }}>
              <Typography variant="h6" gutterBottom>Latest Signals</Typography>
              <Grid container spacing={2}>
                {signals.slice(0, 6).map((signal) => (
                  <Grid item xs={12} sm={6} md={4} key={signal.signalId}>
                    <Card 
                      sx={{ 
                        cursor: 'pointer', 
                        '&:hover': { boxShadow: 6 },
                        borderLeft: 5,
                        borderColor: getSignalColor(signal)
                      }}
                      onClick={() => handleSignalClick(signal)}
                    >
                      <CardContent>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                          <Typography variant="body2" color="text.secondary">
                            {signal.modelId}
                          </Typography>
                          <Chip 
                            size="small" 
                            label={signal.direction}
                            color={signal.direction === 'BUY' ? 'success' : signal.direction === 'SELL' ? 'error' : 'default'}
                          />
                        </Box>
                        
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                          <SignalConfidenceGauge confidence={signal.confidence} size={40} />
                          <Box sx={{ ml: 2 }}>
                            <Typography variant="h6" component="div">
                              {signal.confidence}% Confidence
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              {timeAgo(signal.timestamp)}
                            </Typography>
                          </Box>
                        </Box>
                        
                        <Box>
                          <Typography variant="body2" sx={{ mb: 0.5 }}>
                            Price: {signal.price}
                          </Typography>
                          <Typography variant="body2" sx={{ mb: 0.5 }}>
                            Timeframe: {signal.timeFrame}
                          </Typography>
                          {signal.tags && signal.tags.length > 0 && (
                            <Box sx={{ mt: 1 }}>
                              {signal.tags.map(tag => (
                                <Chip key={tag} size="small" label={tag} sx={{ mr: 0.5, mb: 0.5 }} />
                              ))}
                            </Box>
                          )}
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
              {signals.length > 6 && (
                <Box sx={{ textAlign: 'center', mt: 2 }}>
                  <Button onClick={() => setModalOpen(true)}>
                    View All {signals.length} Signals
                  </Button>
                </Box>
              )}
            </Paper>
            
            {/* Feature Correlation */}
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>Feature Correlation</Typography>
              <CorrelationHeatmap instrumentId={instrumentId} timeFrame={timeFrame} />
            </Paper>
          </>
        )}
      </Box>
      
      {/* Signal Detail Modal */}
      <SignalDetailModal
        signal={selectedSignal}
        open={modalOpen}
        onClose={handleModalClose}
        allSignals={signals}
      />
    </Container>
  );
};

export default SignalVisualization;
