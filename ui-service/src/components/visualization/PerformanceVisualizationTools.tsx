import React, { useState, useEffect } from 'react';
import { Box, Grid, Paper, Typography, Tabs, Tab, FormControl, InputLabel, Select, MenuItem, ToggleButtonGroup, ToggleButton } from '@mui/material';
import { styled } from '@mui/material/styles';
import { useQuery } from 'react-query';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ComposedChart, Area, Scatter, ReferenceLine
} from 'recharts';

// Import services for data fetching
import { fetchPerformanceData, fetchDrawdownAnalysis, fetchRiskAdjustedReturns, fetchTradingStats, fetchPortfolioComparison } from '../../services/performanceService';

const PerformanceContainer = styled(Box)(({ theme }) => ({
  height: '100%',
  padding: theme.spacing(2),
}));

const PerformancePanel = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
}));

const ChartContainer = styled(Box)({
  flexGrow: 1,
  minHeight: 250,
});

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
      id={`performance-tabpanel-${index}`}
      aria-labelledby={`performance-tab-${index}`}
      {...other}
      style={{ height: 'calc(100vh - 150px)', overflow: 'auto' }}
    >
      {value === index && <Box sx={{ pt: 2 }}>{children}</Box>}
    </div>
  );
}

export const PerformanceVisualizationTools: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [timeRange, setTimeRange] = useState('1m');
  const [chartType, setChartType] = useState('cumulative');
  const [benchmark, setBenchmark] = useState('S&P500');

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleChartTypeChange = (_event: React.SyntheticEvent, newType: string) => {
    if (newType) setChartType(newType);
  };

  // Fetch performance data
  const { data: performanceData, isLoading: isPerformanceLoading } = useQuery(
    ['performanceData', timeRange],
    () => fetchPerformanceData(timeRange)
  );

  // Fetch drawdown analysis
  const { data: drawdownData, isLoading: isDrawdownLoading } = useQuery(
    ['drawdownAnalysis', timeRange],
    () => fetchDrawdownAnalysis(timeRange)
  );

  // Fetch risk-adjusted returns
  const { data: riskAdjustedData, isLoading: isRiskAdjustedLoading } = useQuery(
    'riskAdjustedReturns',
    fetchRiskAdjustedReturns
  );

  // Fetch trading statistics
  const { data: tradingStats, isLoading: isStatsLoading } = useQuery(
    ['tradingStats', timeRange],
    () => fetchTradingStats(timeRange)
  );

  // Fetch portfolio comparison
  const { data: comparisonData, isLoading: isComparisonLoading } = useQuery(
    ['portfolioComparison', benchmark, timeRange],
    () => fetchPortfolioComparison(benchmark, timeRange)
  );

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

  return (
    <PerformanceContainer>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h4" component="h1">
          Performance Visualization Tools
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel id="time-range-label">Time Range</InputLabel>
            <Select
              labelId="time-range-label"
              value={timeRange}
              label="Time Range"
              onChange={(e) => setTimeRange(e.target.value)}
            >
              <MenuItem value="1w">1 Week</MenuItem>
              <MenuItem value="1m">1 Month</MenuItem>
              <MenuItem value="3m">3 Months</MenuItem>
              <MenuItem value="6m">6 Months</MenuItem>
              <MenuItem value="1y">1 Year</MenuItem>
              <MenuItem value="ytd">Year to Date</MenuItem>
              <MenuItem value="all">All Time</MenuItem>
            </Select>
          </FormControl>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel id="benchmark-label">Benchmark</InputLabel>
            <Select
              labelId="benchmark-label"
              value={benchmark}
              label="Benchmark"
              onChange={(e) => setBenchmark(e.target.value)}
            >
              <MenuItem value="S&P500">S&P 500</MenuItem>
              <MenuItem value="USD_Index">USD Index</MenuItem>
              <MenuItem value="EUR_Index">EUR Index</MenuItem>
              <MenuItem value="ForexAvg">Forex Avg</MenuItem>
            </Select>
          </FormControl>
        </Box>
      </Box>

      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={tabValue} onChange={handleTabChange} aria-label="performance tabs">
          <Tab label="Returns Analysis" id="performance-tab-0" />
          <Tab label="Drawdown Analysis" id="performance-tab-1" />
          <Tab label="Risk-Adjusted Returns" id="performance-tab-2" />
          <Tab label="Trading Statistics" id="performance-tab-3" />
          <Tab label="Portfolio Comparison" id="performance-tab-4" />
        </Tabs>
      </Box>

      {/* Returns Analysis Tab */}
      <TabPanel value={tabValue} index={0}>
        <Box sx={{ mb: 2 }}>
          <ToggleButtonGroup
            value={chartType}
            exclusive
            onChange={handleChartTypeChange}
            aria-label="chart type"
            size="small"
          >
            <ToggleButton value="cumulative" aria-label="cumulative">
              Cumulative Returns
            </ToggleButton>
            <ToggleButton value="periodic" aria-label="periodic">
              Periodic Returns
            </ToggleButton>
            <ToggleButton value="distribution" aria-label="distribution">
              Return Distribution
            </ToggleButton>
          </ToggleButtonGroup>
        </Box>

        <Grid container spacing={2}>
          <Grid item xs={12}>
            <PerformancePanel>
              <Typography variant="h6" gutterBottom>
                {chartType === 'cumulative' 
                  ? 'Cumulative Returns' 
                  : chartType === 'periodic' 
                    ? 'Periodic Returns' 
                    : 'Return Distribution'}
              </Typography>
              <ChartContainer>
                {isPerformanceLoading ? (
                  <Typography>Loading performance data...</Typography>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    {chartType === 'cumulative' ? (
                      <LineChart
                        data={performanceData?.cumulativeReturns || []}
                        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis />
                        <Tooltip formatter={(value) => [`${(value * 100).toFixed(2)}%`, 'Return']} />
                        <Legend />
                        <Line 
                          type="monotone" 
                          dataKey="portfolio" 
                          name="Portfolio" 
                          stroke="#8884d8" 
                          dot={false}
                          activeDot={{ r: 8 }} 
                        />
                        <Line 
                          type="monotone" 
                          dataKey="benchmark" 
                          name={benchmark} 
                          stroke="#82ca9d" 
                          dot={false} 
                        />
                      </LineChart>
                    ) : chartType === 'periodic' ? (
                      <BarChart
                        data={performanceData?.periodicReturns || []}
                        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                        <Tooltip formatter={(value) => [`${(value * 100).toFixed(2)}%`, 'Return']} />
                        <Legend />
                        <Bar dataKey="portfolio" name="Portfolio" fill="#8884d8" />
                        <Bar dataKey="benchmark" name={benchmark} fill="#82ca9d" />
                      </BarChart>
                    ) : (
                      <BarChart
                        data={performanceData?.returnDistribution || []}
                        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="range" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="frequency" name="Frequency" fill="#8884d8" />
                        <ReferenceLine x="0%" stroke="#ff7300" />
                      </BarChart>
                    )}
                  </ResponsiveContainer>
                )}
              </ChartContainer>
            </PerformancePanel>
          </Grid>
          <Grid item xs={12} md={6}>
            <PerformancePanel>
              <Typography variant="h6" gutterBottom>
                Performance Metrics
              </Typography>
              {isPerformanceLoading ? (
                <Typography>Loading metrics...</Typography>
              ) : (
                <Grid container spacing={2}>
                  {performanceData?.metrics && Object.entries(performanceData.metrics).map(([key, value]) => (
                    <Grid item xs={6} key={key}>
                      <Paper sx={{ p: 1.5 }}>
                        <Typography variant="body2" color="text.secondary">
                          {key.replace(/([A-Z])/g, ' $1').replace(/^./, (str) => str.toUpperCase())}
                        </Typography>
                        <Typography variant="h6">
                          {typeof value === 'number' 
                            ? ['Return', 'Volatility', 'Alpha', 'Beta'].includes(key.replace(/([A-Z])/g, ' $1').replace(/^./, (str) => str.toUpperCase()))
                              ? `${(value * 100).toFixed(2)}%` 
                              : value.toFixed(2) 
                            : value}
                        </Typography>
                      </Paper>
                    </Grid>
                  ))}
                </Grid>
              )}
            </PerformancePanel>
          </Grid>
          <Grid item xs={12} md={6}>
            <PerformancePanel>
              <Typography variant="h6" gutterBottom>
                Monthly Returns (%)
              </Typography>
              {isPerformanceLoading ? (
                <Typography>Loading monthly returns...</Typography>
              ) : (
                <Box sx={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr>
                        <th style={{ padding: '8px', textAlign: 'left', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                          Year
                        </th>
                        {['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'YTD'].map((month) => (
                          <th key={month} style={{ padding: '8px', textAlign: 'right', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                            {month}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {(performanceData?.monthlyReturns || []).map((year) => (
                        <tr key={year.year}>
                          <td style={{ padding: '8px', textAlign: 'left', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                            {year.year}
                          </td>
                          {['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'ytd'].map((month) => (
                            <td 
                              key={month} 
                              style={{ 
                                padding: '8px', 
                                textAlign: 'right', 
                                borderBottom: '1px solid rgba(224, 224, 224, 1)',
                                backgroundColor: year[month] > 0 ? 'rgba(76, 175, 80, 0.1)' : year[month] < 0 ? 'rgba(244, 67, 54, 0.1)' : 'transparent',
                                color: year[month] > 0 ? '#4caf50' : year[month] < 0 ? '#f44336' : 'inherit'
                              }}
                            >
                              {year[month] ? `${(year[month] * 100).toFixed(2)}%` : '-'}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Box>
              )}
            </PerformancePanel>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Drawdown Analysis Tab */}
      <TabPanel value={tabValue} index={1}>
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <PerformancePanel>
              <Typography variant="h6" gutterBottom>
                Drawdown Analysis
              </Typography>
              <ChartContainer>
                {isDrawdownLoading ? (
                  <Typography>Loading drawdown data...</Typography>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart
                      data={drawdownData?.drawdowns || []}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis yAxisId="left" />
                      <YAxis yAxisId="right" orientation="right" domain={[0, 'dataMax + 5']} />
                      <Tooltip formatter={(value, name) => {
                        if (name === 'drawdown') return [`${(value * 100).toFixed(2)}%`, 'Drawdown'];
                        if (name === 'equity') return [`$${value.toFixed(2)}`, 'Equity'];
                        return [value, name];
                      }} />
                      <Legend />
                      <Area
                        yAxisId="left"
                        type="monotone"
                        dataKey="drawdown"
                        name="Drawdown"
                        fill="#f44336"
                        fillOpacity={0.3}
                        stroke="#f44336"
                      />
                      <Line
                        yAxisId="right"
                        type="monotone"
                        dataKey="equity"
                        name="Equity"
                        stroke="#8884d8"
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                )}
              </ChartContainer>
            </PerformancePanel>
          </Grid>
          <Grid item xs={12} md={6}>
            <PerformancePanel>
              <Typography variant="h6" gutterBottom>
                Worst Drawdowns
              </Typography>
              {isDrawdownLoading ? (
                <Typography>Loading drawdown data...</Typography>
              ) : (
                <Box sx={{ overflowY: 'auto', maxHeight: '400px' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr>
                        <th style={{ padding: '8px', textAlign: 'left', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                          #
                        </th>
                        <th style={{ padding: '8px', textAlign: 'left', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                          Start
                        </th>
                        <th style={{ padding: '8px', textAlign: 'left', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                          End
                        </th>
                        <th style={{ padding: '8px', textAlign: 'right', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                          Length
                        </th>
                        <th style={{ padding: '8px', textAlign: 'right', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                          Drawdown
                        </th>
                        <th style={{ padding: '8px', textAlign: 'right', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                          Recovery
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {(drawdownData?.worstDrawdowns || []).map((drawdown, index) => (
                        <tr key={index}>
                          <td style={{ padding: '8px', textAlign: 'left', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                            {index + 1}
                          </td>
                          <td style={{ padding: '8px', textAlign: 'left', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                            {drawdown.start}
                          </td>
                          <td style={{ padding: '8px', textAlign: 'left', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                            {drawdown.end || 'Ongoing'}
                          </td>
                          <td style={{ padding: '8px', textAlign: 'right', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                            {drawdown.lengthDays} days
                          </td>
                          <td style={{ 
                            padding: '8px', 
                            textAlign: 'right', 
                            borderBottom: '1px solid rgba(224, 224, 224, 1)',
                            color: '#f44336',
                            fontWeight: 'bold'
                          }}>
                            {`${(drawdown.drawdown * 100).toFixed(2)}%`}
                          </td>
                          <td style={{ padding: '8px', textAlign: 'right', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                            {drawdown.recoveryDays ? `${drawdown.recoveryDays} days` : '-'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Box>
              )}
            </PerformancePanel>
          </Grid>
          <Grid item xs={12} md={6}>
            <PerformancePanel>
              <Typography variant="h6" gutterBottom>
                Drawdown Distribution
              </Typography>
              <ChartContainer>
                {isDrawdownLoading ? (
                  <Typography>Loading drawdown data...</Typography>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={drawdownData?.drawdownDistribution || []}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="range" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="frequency" name="Frequency" fill="#f44336" />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </ChartContainer>
            </PerformancePanel>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Risk-Adjusted Returns Tab */}
      <TabPanel value={tabValue} index={2}>
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <PerformancePanel>
              <Typography variant="h6" gutterBottom>
                Risk-Adjusted Metrics
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                {isRiskAdjustedLoading ? (
                  <Typography>Loading risk-adjusted metrics...</Typography>
                ) : (
                  (riskAdjustedData?.metrics || []).map((metric) => (
                    <Paper key={metric.name} sx={{ p: 2 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="subtitle1">{metric.name}</Typography>
                        <Typography 
                          variant="h6" 
                          sx={{ 
                            color: metric.value > metric.benchmark 
                              ? '#4caf50' 
                              : metric.value < metric.benchmark 
                                ? '#f44336' 
                                : 'inherit'
                          }}
                        >
                          {typeof metric.value === 'number' ? metric.value.toFixed(2) : metric.value}
                        </Typography>
                      </Box>
                      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                        {metric.description}
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                        <Box sx={{ width: '100%', mr: 1 }}>
                          <Box
                            sx={{
                              height: 8,
                              borderRadius: 5,
                              background: `linear-gradient(to right, #f44336, #ffeb3b, #4caf50)`,
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
                                left: `${Math.min(Math.max(metric.percentile * 100, 0), 100)}%`,
                                top: -4,
                              }}
                            />
                            <Box
                              sx={{
                                position: 'absolute',
                                height: 16,
                                width: 4,
                                backgroundColor: '#555',
                                borderRadius: 1,
                                left: `${Math.min(Math.max(metric.benchmarkPercentile * 100, 0), 100)}%`,
                                top: -4,
                              }}
                            />
                          </Box>
                        </Box>
                        <Box sx={{ minWidth: 35 }}>
                          <Typography variant="body2" color="text.secondary">
                            {`${(metric.percentile * 100).toFixed(0)}%`}
                          </Typography>
                        </Box>
                      </Box>
                    </Paper>
                  ))
                )}
              </Box>
            </PerformancePanel>
          </Grid>
          <Grid item xs={12} md={6}>
            <PerformancePanel>
              <Typography variant="h6" gutterBottom>
                Risk-Return Analysis
              </Typography>
              <ChartContainer>
                {isRiskAdjustedLoading ? (
                  <Typography>Loading risk-return data...</Typography>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart
                      margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                    >
                      <CartesianGrid />
                      <XAxis 
                        type="number" 
                        dataKey="risk" 
                        name="Risk (Volatility)" 
                        domain={[0, 'dataMax + 0.02']}
                        tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                      />
                      <YAxis 
                        type="number" 
                        dataKey="return" 
                        name="Return" 
                        domain={['dataMin - 0.02', 'dataMax + 0.02']}
                        tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                      />
                      <Tooltip 
                        formatter={(value, name) => {
                          if (name === 'risk') return [`${(value * 100).toFixed(2)}%`, 'Risk (Volatility)'];
                          if (name === 'return') return [`${(value * 100).toFixed(2)}%`, 'Return'];
                          return [value, name];
                        }}
                      />
                      <Legend />
                      <Scatter name="Assets" data={riskAdjustedData?.riskReturn || []} fill="#8884d8">
                        {(riskAdjustedData?.riskReturn || []).map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Scatter>
                      {riskAdjustedData?.efficientFrontier && (
                        <Line 
                          type="monotone" 
                          dataKey="return" 
                          data={riskAdjustedData.efficientFrontier} 
                          stroke="#ff7300" 
                          dot={false} 
                          activeDot={false}
                          name="Efficient Frontier"
                        />
                      )}
                    </ScatterChart>
                  </ResponsiveContainer>
                )}
              </ChartContainer>
            </PerformancePanel>
          </Grid>
          <Grid item xs={12}>
            <PerformancePanel>
              <Typography variant="h6" gutterBottom>
                Rolling Risk-Adjusted Metrics
              </Typography>
              <ChartContainer>
                {isRiskAdjustedLoading ? (
                  <Typography>Loading rolling metrics data...</Typography>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={riskAdjustedData?.rollingMetrics || []}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="sharpe" name="Sharpe Ratio" stroke="#8884d8" dot={false} />
                      <Line type="monotone" dataKey="sortino" name="Sortino Ratio" stroke="#82ca9d" dot={false} />
                      <Line type="monotone" dataKey="calmar" name="Calmar Ratio" stroke="#ff7300" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </ChartContainer>
            </PerformancePanel>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Trading Statistics Tab */}
      <TabPanel value={tabValue} index={3}>
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <PerformancePanel>
              <Typography variant="h6" gutterBottom>
                Trade Outcomes
              </Typography>
              <ChartContainer>
                {isStatsLoading ? (
                  <Typography>Loading trade outcome data...</Typography>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={tradingStats?.outcomes || []}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {tradingStats?.outcomes?.map((entry, index) => (
                          <Cell 
                            key={`cell-${index}`} 
                            fill={entry.name === 'Win' ? '#4caf50' : entry.name === 'Loss' ? '#f44336' : '#ffeb3b'} 
                          />
                        ))}
                      </Pie>
                      <Tooltip formatter={(value, name, props) => [`${value} trades (${(props.percent * 100).toFixed(2)}%)`, props.payload.name]} />
                    </PieChart>
                  </ResponsiveContainer>
                )}
              </ChartContainer>
            </PerformancePanel>
          </Grid>
          <Grid item xs={12} md={6}>
            <PerformancePanel>
              <Typography variant="h6" gutterBottom>
                Key Trading Metrics
              </Typography>
              {isStatsLoading ? (
                <Typography>Loading trading metrics...</Typography>
              ) : (
                <Grid container spacing={2}>
                  {tradingStats?.metrics && Object.entries(tradingStats.metrics).map(([key, value]) => (
                    <Grid item xs={6} md={4} key={key}>
                      <Paper sx={{ p: 1.5 }}>
                        <Typography variant="body2" color="text.secondary">
                          {key.replace(/([A-Z])/g, ' $1').replace(/^./, (str) => str.toUpperCase())}
                        </Typography>
                        <Typography variant="h6">
                          {typeof value === 'number' 
                            ? ['Win Rate', 'Profit Factor', 'Average Win', 'Average Loss', 'Largest Win', 'Largest Loss'].includes(key.replace(/([A-Z])/g, ' $1').replace(/^./, (str) => str.toUpperCase()))
                              ? key.toLowerCase().includes('rate') ? `${(value * 100).toFixed(2)}%` : `$${value.toFixed(2)}` 
                              : value.toFixed(2) 
                            : value}
                        </Typography>
                      </Paper>
                    </Grid>
                  ))}
                </Grid>
              )}
            </PerformancePanel>
          </Grid>
          <Grid item xs={12} md={6}>
            <PerformancePanel>
              <Typography variant="h6" gutterBottom>
                Profit Distribution
              </Typography>
              <ChartContainer>
                {isStatsLoading ? (
                  <Typography>Loading profit distribution data...</Typography>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={tradingStats?.profitDistribution || []}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="range" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar 
                        dataKey="frequency" 
                        name="Frequency" 
                        fill="#8884d8" 
                        shape={(props) => {
                          // Determine if this bar represents profit or loss
                          const value = props.x + props.width / 2;
                          const centerX = props.width * tradingStats.profitDistribution.length / 2;
                          return <rect {...props} fill={value < centerX ? '#f44336' : '#4caf50'} />;
                        }}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </ChartContainer>
            </PerformancePanel>
          </Grid>
          <Grid item xs={12} md={6}>
            <PerformancePanel>
              <Typography variant="h6" gutterBottom>
                Trade Duration Analysis
              </Typography>
              <ChartContainer>
                {isStatsLoading ? (
                  <Typography>Loading trade duration data...</Typography>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={tradingStats?.durationAnalysis || []}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="duration" />
                      <YAxis yAxisId="left" orientation="left" />
                      <YAxis yAxisId="right" orientation="right" />
                      <Tooltip />
                      <Legend />
                      <Bar yAxisId="left" dataKey="count" name="Trade Count" fill="#8884d8" />
                      <Bar yAxisId="right" dataKey="winRate" name="Win Rate" fill="#82ca9d" />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </ChartContainer>
            </PerformancePanel>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Portfolio Comparison Tab */}
      <TabPanel value={tabValue} index={4}>
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <PerformancePanel>
              <Typography variant="h6" gutterBottom>
                Portfolio vs Benchmark
              </Typography>
              <ChartContainer>
                {isComparisonLoading ? (
                  <Typography>Loading comparison data...</Typography>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart
                      data={comparisonData?.comparison || []}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis yAxisId="left" />
                      <YAxis yAxisId="right" orientation="right" />
                      <Tooltip formatter={(value, name) => {
                        if (name === 'portfolio' || name === 'benchmark') return [`${(value * 100).toFixed(2)}%`, name === 'portfolio' ? 'Portfolio' : benchmark];
                        if (name === 'excess') return [`${(value * 100).toFixed(2)}%`, 'Excess Return'];
                        return [value, name];
                      }} />
                      <Legend />
                      <Line 
                        yAxisId="left" 
                        type="monotone" 
                        dataKey="portfolio" 
                        name="Portfolio" 
                        stroke="#8884d8" 
                        dot={false}
                      />
                      <Line 
                        yAxisId="left" 
                        type="monotone" 
                        dataKey="benchmark" 
                        name={benchmark} 
                        stroke="#82ca9d" 
                        dot={false}
                      />
                      <Bar 
                        yAxisId="right" 
                        dataKey="excess" 
                        name="Excess Return" 
                        fill="#ff7300" 
                        opacity={0.5}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                )}
              </ChartContainer>
            </PerformancePanel>
          </Grid>
          <Grid item xs={12} md={6}>
            <PerformancePanel>
              <Typography variant="h6" gutterBottom>
                Rolling Beta
              </Typography>
              <ChartContainer>
                {isComparisonLoading ? (
                  <Typography>Loading beta data...</Typography>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={comparisonData?.rollingBeta || []}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis domain={['dataMin', 'dataMax']} />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="beta" name="Beta" stroke="#8884d8" dot={false} />
                      <ReferenceLine y={1} stroke="#ff7300" strokeDasharray="3 3" />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </ChartContainer>
            </PerformancePanel>
          </Grid>
          <Grid item xs={12} md={6}>
            <PerformancePanel>
              <Typography variant="h6" gutterBottom>
                Rolling Alpha
              </Typography>
              <ChartContainer>
                {isComparisonLoading ? (
                  <Typography>Loading alpha data...</Typography>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={comparisonData?.rollingAlpha || []}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                      <Tooltip formatter={(value) => [`${(value * 100).toFixed(2)}%`, 'Alpha']} />
                      <Legend />
                      <Line type="monotone" dataKey="alpha" name="Alpha" stroke="#82ca9d" dot={false} />
                      <ReferenceLine y={0} stroke="#ff7300" strokeDasharray="3 3" />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </ChartContainer>
            </PerformancePanel>
          </Grid>
          <Grid item xs={12}>
            <PerformancePanel>
              <Typography variant="h6" gutterBottom>
                Performance Comparison
              </Typography>
              {isComparisonLoading ? (
                <Typography>Loading comparison metrics...</Typography>
              ) : (
                <Grid container spacing={2}>
                  {comparisonData?.metrics?.map((metric, index) => (
                    <Grid item xs={12} md={6} key={index}>
                      <Paper sx={{ p: 2 }}>
                        <Typography variant="subtitle1">{metric.name}</Typography>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                          <Typography variant="body1" sx={{ fontWeight: 'bold', color: '#8884d8' }}>
                            Portfolio: {typeof metric.portfolio === 'number' ? 
                              metric.name.toLowerCase().includes('return') || metric.name.toLowerCase().includes('alpha') || metric.name.toLowerCase().includes('volatility') ? 
                                `${(metric.portfolio * 100).toFixed(2)}%` : 
                                metric.portfolio.toFixed(2) : 
                              metric.portfolio}
                          </Typography>
                          <Typography variant="body1" sx={{ fontWeight: 'bold', color: '#82ca9d' }}>
                            {benchmark}: {typeof metric.benchmark === 'number' ? 
                              metric.name.toLowerCase().includes('return') || metric.name.toLowerCase().includes('volatility') ? 
                                `${(metric.benchmark * 100).toFixed(2)}%` : 
                                metric.benchmark.toFixed(2) : 
                              metric.benchmark}
                          </Typography>
                        </Box>
                        {metric.difference !== undefined && (
                          <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 0.5 }}>
                            <Typography 
                              variant="body2" 
                              sx={{ 
                                fontWeight: 'bold', 
                                color: metric.difference > 0 ? '#4caf50' : metric.difference < 0 ? '#f44336' : 'inherit'
                              }}
                            >
                              Difference: {typeof metric.difference === 'number' ? 
                                (metric.name.toLowerCase().includes('return') || metric.name.toLowerCase().includes('alpha') || metric.name.toLowerCase().includes('volatility') ? 
                                  `${(metric.difference * 100).toFixed(2)}%` : 
                                  metric.difference.toFixed(2)) : 
                                metric.difference}
                            </Typography>
                          </Box>
                        )}
                      </Paper>
                    </Grid>
                  ))}
                </Grid>
              )}
            </PerformancePanel>
          </Grid>
        </Grid>
      </TabPanel>
    </PerformanceContainer>
  );
};

export default PerformanceVisualizationTools;
