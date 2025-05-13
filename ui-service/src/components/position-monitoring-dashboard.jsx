import React, { useState, useEffect } from 'react';
import { 
  Box,
  Container, 
  Paper, 
  Typography, 
  Grid, 
  Table, 
  TableBody, 
  TableCell, 
  TableContainer, 
  TableHead, 
  TableRow,
  Button,
  CircularProgress,
  Tabs,
  Tab,
  Card,
  CardContent,
  IconButton,
  Tooltip,
  useTheme
} from '@mui/material';
import { 
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  ArrowUpward as ArrowUpIcon,
  ArrowDownward as ArrowDownIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import { formatCurrency, formatPercent, formatPips } from '../utils/formatters';
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as ChartTooltip, Legend, ResponsiveContainer } from 'recharts';
import { useTradingApi } from '../hooks/useTradingApi';
import PositionDetailDrawer from './PositionDetailDrawer';
import ProfitLossCard from './ProfitLossCard';
import MarginUtilizationGauge from './MarginUtilizationGauge';
import CurrencyExposureChart from './CurrencyExposureChart';
import TradingRegimeIndicator from './TradingRegimeIndicator';

/**
 * Position monitoring dashboard component with real-time P&L visualization
 */
const PositionMonitoringDashboard = ({ accountId }) => {
  const theme = useTheme();
  const [positions, setPositions] = useState([]);
  const [accountSummary, setAccountSummary] = useState(null);
  const [selectedPosition, setSelectedPosition] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [timeRange, setTimeRange] = useState('1d');
  const [plHistory, setPlHistory] = useState([]);
  const [refreshKey, setRefreshKey] = useState(0);
  const [sortField, setSortField] = useState('unrealizedPl');
  const [sortDirection, setSortDirection] = useState('desc');
  const [tabValue, setTabValue] = useState(0);
  
  const tradingApi = useTradingApi();
  
  // Fetch positions and account data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        
        // Fetch positions and account summary in parallel
        const [positionsData, summaryData, plHistoryData] = await Promise.all([
          tradingApi.getOpenPositions(accountId),
          tradingApi.getAccountSummary(accountId),
          tradingApi.getProfitLossHistory(accountId, timeRange)
        ]);
        
        setPositions(positionsData);
        setAccountSummary(summaryData);
        setPlHistory(plHistoryData);
        setError(null);
      } catch (err) {
        console.error('Error fetching position data:', err);
        setError('Failed to load position data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
    
    // Set up real-time updates
    const intervalId = setInterval(() => {
      setRefreshKey(prevKey => prevKey + 1);
    }, 10000); // Update every 10 seconds
    
    return () => clearInterval(intervalId);
  }, [accountId, timeRange, refreshKey]);
  
  // Handle manual refresh
  const handleRefresh = () => {
    setRefreshKey(prevKey => prevKey + 1);
  };
  
  // Handle position selection
  const handlePositionClick = (position) => {
    setSelectedPosition(position);
    setDrawerOpen(true);
  };
  
  // Handle drawer close
  const handleDrawerClose = () => {
    setDrawerOpen(false);
  };
  
  // Handle sort change
  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };
  
  // Sort positions
  const sortedPositions = [...positions].sort((a, b) => {
    let comparison = 0;
    
    if (sortField === 'instrument') {
      comparison = a.instrumentId.localeCompare(b.instrumentId);
    } else if (sortField === 'volume') {
      comparison = a.volume - b.volume;
    } else if (sortField === 'unrealizedPl') {
      comparison = a.unrealizedPl - b.unrealizedPl;
    } else if (sortField === 'unrealizedPlPercent') {
      comparison = a.unrealizedPlPercent - b.unrealizedPlPercent;
    } else if (sortField === 'duration') {
      comparison = new Date(a.openTime) - new Date(b.openTime);
    }
    
    return sortDirection === 'asc' ? comparison : -comparison;
  });
  
  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };
  
  // Change time range for P/L history
  const handleTimeRangeChange = (range) => {
    setTimeRange(range);
  };
  
  // Calculate summary statistics
  const totalUnrealizedPL = positions.reduce((sum, pos) => sum + pos.unrealizedPl, 0);
  const totalPositionCount = positions.length;
  const winningPositions = positions.filter(pos => pos.unrealizedPl > 0);
  const losingPositions = positions.filter(pos => pos.unrealizedPl < 0);
  const winningPercentage = positions.length > 0 ? (winningPositions.length / positions.length) * 100 : 0;
  
  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h4" component="h1">
            Position Monitor
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
        
        {loading && !accountSummary ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
            <CircularProgress />
          </Box>
        ) : error ? (
          <Paper sx={{ p: 3, bgcolor: 'error.light', color: 'error.contrastText' }}>
            <Typography>{error}</Typography>
          </Paper>
        ) : (
          <>
            {/* Account Summary Cards */}
            <Grid container spacing={2} sx={{ mb: 4 }}>
              <Grid item xs={12} md={3}>
                <ProfitLossCard 
                  balance={accountSummary?.balance || 0}
                  equity={accountSummary?.equity || 0}
                  todayPL={accountSummary?.todayPL || 0}
                  todayPLPercent={accountSummary?.todayPLPercent || 0}
                />
              </Grid>
              <Grid item xs={12} md={3}>
                <MarginUtilizationGauge 
                  marginUsed={accountSummary?.marginUsed || 0}
                  marginAvailable={accountSummary?.marginAvailable || 0}
                  marginLevel={accountSummary?.marginLevel || 0}
                />
              </Grid>
              <Grid item xs={12} md={3}>
                <Card sx={{ height: '100%' }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>Position Summary</Typography>
                    <Grid container spacing={1}>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">Open Positions</Typography>
                        <Typography variant="h6">{totalPositionCount}</Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">Winning %</Typography>
                        <Typography variant="h6" color={winningPercentage >= 50 ? 'success.main' : 'error.main'}>
                          {formatPercent(winningPercentage)}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">Winning</Typography>
                        <Typography variant="h6" color="success.main">{winningPositions.length}</Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">Losing</Typography>
                        <Typography variant="h6" color="error.main">{losingPositions.length}</Typography>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={3}>
                <TradingRegimeIndicator />
              </Grid>
            </Grid>
            
            {/* P/L Chart and Currency Exposure */}
            <Grid container spacing={2} sx={{ mb: 4 }}>
              <Grid item xs={12} md={8}>
                <Paper sx={{ p: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="h6">Profit/Loss History</Typography>
                    <Box>
                      {['1d', '1w', '1m', '3m'].map(range => (
                        <Button 
                          key={range}
                          size="small"
                          variant={timeRange === range ? 'contained' : 'text'}
                          onClick={() => handleTimeRangeChange(range)}
                          sx={{ minWidth: 40 }}
                        >
                          {range}
                        </Button>
                      ))}
                    </Box>
                  </Box>
                  <Box sx={{ height: 250 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart
                        data={plHistory}
                        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="time" />
                        <YAxis />
                        <ChartTooltip formatter={(value) => formatCurrency(value)} />
                        <Area 
                          type="monotone" 
                          dataKey="equity" 
                          stroke={theme.palette.primary.main}
                          fill={theme.palette.primary.light}
                          strokeWidth={2}
                        />
                        <Area 
                          type="monotone" 
                          dataKey="balance"
                          stroke={theme.palette.secondary.main}
                          fill={theme.palette.secondary.light}
                          strokeWidth={2}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </Box>
                </Paper>
              </Grid>
              <Grid item xs={12} md={4}>
                <Paper sx={{ p: 2, height: '100%' }}>
                  <Typography variant="h6" gutterBottom>Currency Exposure</Typography>
                  <Box sx={{ height: 250 }}>
                    <CurrencyExposureChart positions={positions} />
                  </Box>
                </Paper>
              </Grid>
            </Grid>
            
            {/* Positions Table */}
            <Paper>
              <Tabs
                value={tabValue}
                onChange={handleTabChange}
                indicatorColor="primary"
                textColor="primary"
              >
                <Tab label={`All Positions (${positions.length})`} />
                <Tab label={`Winning (${winningPositions.length})`} />
                <Tab label={`Losing (${losingPositions.length})`} />
              </Tabs>
              
              <TableContainer component={Box} sx={{ maxHeight: 400 }}>
                <Table stickyHeader>
                  <TableHead>
                    <TableRow>
                      <TableCell 
                        onClick={() => handleSort('instrument')}
                        sx={{ cursor: 'pointer' }}
                      >
                        Instrument
                        {sortField === 'instrument' && (sortDirection === 'asc' ? <ArrowUpIcon fontSize="small" /> : <ArrowDownIcon fontSize="small" />)}
                      </TableCell>
                      <TableCell align="right">Direction</TableCell>
                      <TableCell 
                        align="right"
                        onClick={() => handleSort('volume')}
                        sx={{ cursor: 'pointer' }}
                      >
                        Volume
                        {sortField === 'volume' && (sortDirection === 'asc' ? <ArrowUpIcon fontSize="small" /> : <ArrowDownIcon fontSize="small" />)}
                      </TableCell>
                      <TableCell align="right">Entry Price</TableCell>
                      <TableCell align="right">Current Price</TableCell>
                      <TableCell 
                        align="right"
                        onClick={() => handleSort('unrealizedPl')}
                        sx={{ cursor: 'pointer' }}
                      >
                        Unrealized P/L
                        {sortField === 'unrealizedPl' && (sortDirection === 'asc' ? <ArrowUpIcon fontSize="small" /> : <ArrowDownIcon fontSize="small" />)}
                      </TableCell>
                      <TableCell 
                        align="right"
                        onClick={() => handleSort('unrealizedPlPercent')}
                        sx={{ cursor: 'pointer' }}
                      >
                        P/L %
                        {sortField === 'unrealizedPlPercent' && (sortDirection === 'asc' ? <ArrowUpIcon fontSize="small" /> : <ArrowDownIcon fontSize="small" />)}
                      </TableCell>
                      <TableCell 
                        align="right"
                        onClick={() => handleSort('duration')}
                        sx={{ cursor: 'pointer' }}
                      >
                        Duration
                        {sortField === 'duration' && (sortDirection === 'asc' ? <ArrowUpIcon fontSize="small" /> : <ArrowDownIcon fontSize="small" />)}
                      </TableCell>
                      <TableCell align="right">Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {(tabValue === 0 ? sortedPositions : 
                      tabValue === 1 ? sortedPositions.filter(p => p.unrealizedPl > 0) : 
                      sortedPositions.filter(p => p.unrealizedPl <= 0)
                    ).map((position) => (
                      <TableRow 
                        key={position.positionId}
                        onClick={() => handlePositionClick(position)}
                        sx={{ 
                          cursor: 'pointer',
                          '&:hover': { backgroundColor: 'action.hover' },
                          backgroundColor: position.unrealizedPl > 0 ? 'success.lighter' : position.unrealizedPl < 0 ? 'error.lighter' : 'inherit'
                        }}
                      >
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            {position.instrumentId}
                          </Box>
                        </TableCell>
                        <TableCell align="right">
                          <Box sx={{ 
                            display: 'inline-block', 
                            bgcolor: position.direction === 'BUY' ? 'success.main' : 'error.main',
                            color: 'white',
                            px: 1,
                            borderRadius: 1
                          }}>
                            {position.direction}
                          </Box>
                        </TableCell>
                        <TableCell align="right">{position.volume}</TableCell>
                        <TableCell align="right">{position.entryPrice}</TableCell>
                        <TableCell align="right">{position.currentPrice}</TableCell>
                        <TableCell 
                          align="right"
                          sx={{ 
                            color: position.unrealizedPl > 0 ? 'success.main' : position.unrealizedPl < 0 ? 'error.main' : 'text.primary',
                            fontWeight: 'bold'
                          }}
                        >
                          {formatCurrency(position.unrealizedPl)}
                        </TableCell>
                        <TableCell 
                          align="right"
                          sx={{ 
                            color: position.unrealizedPLPercent > 0 ? 'success.main' : position.unrealizedPLPercent < 0 ? 'error.main' : 'text.primary'
                          }}
                        >
                          {formatPercent(position.unrealizedPLPercent)}
                        </TableCell>
                        <TableCell align="right">{position.durationFormatted}</TableCell>
                        <TableCell align="right">
                          <Box>
                            <IconButton size="small" onClick={(e) => {
                              e.stopPropagation();
                              handlePositionClick(position);
                            }}>
                              <InfoIcon fontSize="small" />
                            </IconButton>
                          </Box>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
          </>
        )}
        
        {/* Position Details Drawer */}
        {selectedPosition && (
          <PositionDetailDrawer
            position={selectedPosition}
            open={drawerOpen}
            onClose={handleDrawerClose}
          />
        )}
      </Box>
    </Container>
  );
};

export default PositionMonitoringDashboard;
