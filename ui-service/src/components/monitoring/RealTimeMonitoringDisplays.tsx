import React, { useState, useEffect } from 'react';
import { Box, Grid, Paper, Typography, Tabs, Tab, Select, MenuItem, FormControl, InputLabel } from '@mui/material';
import { styled } from '@mui/material/styles';
import { useQuery } from 'react-query';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter
} from 'recharts';

// Services
import { 
  fetchSystemMetrics,
  fetchMarketActivity,
  fetchAlerts,
  fetchPositionsRisk,
  fetchPortfolioAllocation
} from '../../services/monitoringService';

const MonitoringContainer = styled(Box)(({ theme }) => ({
  height: '100%',
  padding: theme.spacing(2),
}));

const MonitoringPanel = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
}));

const ChartContainer = styled(Box)({
  flexGrow: 1,
  minHeight: 250,
});

interface MonitoringDisplayProps {
  refreshInterval?: number;
}

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
      id={`monitoring-tabpanel-${index}`}
      aria-labelledby={`monitoring-tab-${index}`}
      {...other}
      style={{ height: 'calc(100vh - 150px)', overflow: 'auto' }}
    >
      {value === index && <Box sx={{ pt: 2 }}>{children}</Box>}
    </div>
  );
}

export const RealTimeMonitoringDisplays: React.FC<MonitoringDisplayProps> = ({
  refreshInterval = 5000,
}) => {
  const [tabValue, setTabValue] = useState(0);
  const [timeRange, setTimeRange] = useState('1h');

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // System metrics
  const { data: systemMetrics, isLoading: isSystemLoading } = useQuery(
    ['systemMetrics', timeRange],
    () => fetchSystemMetrics(timeRange),
    { refetchInterval: refreshInterval }
  );

  // Market activity
  const { data: marketActivity, isLoading: isMarketLoading } = useQuery(
    ['marketActivity', timeRange],
    () => fetchMarketActivity(timeRange),
    { refetchInterval: refreshInterval }
  );

  // Alerts
  const { data: alerts, isLoading: isAlertsLoading } = useQuery(
    'alerts',
    fetchAlerts,
    { refetchInterval: refreshInterval }
  );

  // Risk metrics
  const { data: riskMetrics, isLoading: isRiskLoading } = useQuery(
    'positionsRisk',
    fetchPositionsRisk,
    { refetchInterval: refreshInterval }
  );

  // Portfolio allocation
  const { data: portfolioAllocation, isLoading: isAllocationLoading } = useQuery(
    'portfolioAllocation',
    fetchPortfolioAllocation,
    { refetchInterval: refreshInterval }
  );

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

  return (
    <MonitoringContainer>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h4" component="h1">
          Real-Time Monitoring
        </Typography>
        <FormControl sx={{ width: 120 }}>
          <InputLabel id="time-range-label">Time Range</InputLabel>
          <Select
            labelId="time-range-label"
            value={timeRange}
            label="Time Range"
            onChange={(e) => setTimeRange(e.target.value)}
            size="small"
          >
            <MenuItem value="15m">15 Minutes</MenuItem>
            <MenuItem value="1h">1 Hour</MenuItem>
            <MenuItem value="4h">4 Hours</MenuItem>
            <MenuItem value="1d">1 Day</MenuItem>
          </Select>
        </FormControl>
      </Box>

      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={tabValue} onChange={handleTabChange} aria-label="monitoring tabs">
          <Tab label="System Performance" id="monitoring-tab-0" />
          <Tab label="Market Activity" id="monitoring-tab-1" />
          <Tab label="Alerts & Notifications" id="monitoring-tab-2" />
          <Tab label="Risk Dashboard" id="monitoring-tab-3" />
        </Tabs>
      </Box>

      {/* System Performance Tab */}
      <TabPanel value={tabValue} index={0}>
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <MonitoringPanel>
              <Typography variant="h6" gutterBottom>
                CPU & Memory Usage
              </Typography>
              <ChartContainer>
                {isSystemLoading ? (
                  <Typography>Loading system metrics...</Typography>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={systemMetrics?.resources || []}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp" />
                      <YAxis yAxisId="left" />
                      <YAxis yAxisId="right" orientation="right" />
                      <Tooltip />
                      <Legend />
                      <Line
                        yAxisId="left"
                        type="monotone"
                        dataKey="cpuUsage"
                        name="CPU (%)"
                        stroke="#8884d8"
                        activeDot={{ r: 8 }}
                      />
                      <Line
                        yAxisId="right"
                        type="monotone"
                        dataKey="memoryUsage"
                        name="Memory (GB)"
                        stroke="#82ca9d"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </ChartContainer>
            </MonitoringPanel>
          </Grid>
          <Grid item xs={12} md={6}>
            <MonitoringPanel>
              <Typography variant="h6" gutterBottom>
                API Response Times
              </Typography>
              <ChartContainer>
                {isSystemLoading ? (
                  <Typography>Loading API metrics...</Typography>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={systemMetrics?.apiResponse || []}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="endpoint" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="responseTime" name="Response Time (ms)" fill="#8884d8" />
                      <Bar dataKey="errorRate" name="Error Rate (%)" fill="#ff8042" />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </ChartContainer>
            </MonitoringPanel>
          </Grid>
          <Grid item xs={12} md={6}>
            <MonitoringPanel>
              <Typography variant="h6" gutterBottom>
                Database Performance
              </Typography>
              <ChartContainer>
                {isSystemLoading ? (
                  <Typography>Loading database metrics...</Typography>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart
                      data={systemMetrics?.database || []}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Area
                        type="monotone"
                        dataKey="queryPerSecond"
                        name="Queries/sec"
                        stroke="#8884d8"
                        fill="#8884d8"
                        fillOpacity={0.3}
                      />
                      <Area
                        type="monotone"
                        dataKey="latency"
                        name="Latency (ms)"
                        stroke="#82ca9d"
                        fill="#82ca9d"
                        fillOpacity={0.3}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                )}
              </ChartContainer>
            </MonitoringPanel>
          </Grid>
          <Grid item xs={12} md={6}>
            <MonitoringPanel>
              <Typography variant="h6" gutterBottom>
                Service Health
              </Typography>
              <ChartContainer>
                {isSystemLoading ? (
                  <Typography>Loading service health metrics...</Typography>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={systemMetrics?.serviceHealth || []}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {systemMetrics?.serviceHealth?.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                )}
              </ChartContainer>
            </MonitoringPanel>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Market Activity Tab */}
      <TabPanel value={tabValue} index={1}>
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <MonitoringPanel>
              <Typography variant="h6" gutterBottom>
                Trading Volume
              </Typography>
              <ChartContainer>
                {isMarketLoading ? (
                  <Typography>Loading trading volume data...</Typography>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={marketActivity?.volume || []}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="buyVolume" name="Buy Volume" fill="#4caf50" stackId="a" />
                      <Bar dataKey="sellVolume" name="Sell Volume" fill="#f44336" stackId="a" />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </ChartContainer>
            </MonitoringPanel>
          </Grid>
          <Grid item xs={12} md={6}>
            <MonitoringPanel>
              <Typography variant="h6" gutterBottom>
                Price Movements
              </Typography>
              <ChartContainer>
                {isMarketLoading ? (
                  <Typography>Loading price movement data...</Typography>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={marketActivity?.prices || []}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      {marketActivity?.prices && marketActivity.prices[0] && 
                        Object.keys(marketActivity.prices[0])
                          .filter(key => key !== 'timestamp')
                          .map((currency, index) => (
                            <Line
                              key={currency}
                              type="monotone"
                              dataKey={currency}
                              stroke={COLORS[index % COLORS.length]}
                            />
                          ))}
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </ChartContainer>
            </MonitoringPanel>
          </Grid>
          <Grid item xs={12} md={6}>
            <MonitoringPanel>
              <Typography variant="h6" gutterBottom>
                Spread Analysis
              </Typography>
              <ChartContainer>
                {isMarketLoading ? (
                  <Typography>Loading spread data...</Typography>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart
                      data={marketActivity?.spreads || []}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      {marketActivity?.spreads && marketActivity.spreads[0] &&
                        Object.keys(marketActivity.spreads[0])
                          .filter(key => key !== 'timestamp')
                          .map((pair, index) => (
                            <Area
                              key={pair}
                              type="monotone"
                              dataKey={pair}
                              stroke={COLORS[index % COLORS.length]}
                              fill={COLORS[index % COLORS.length]}
                              fillOpacity={0.3}
                            />
                          ))}
                    </AreaChart>
                  </ResponsiveContainer>
                )}
              </ChartContainer>
            </MonitoringPanel>
          </Grid>
          <Grid item xs={12} md={12}>
            <MonitoringPanel>
              <Typography variant="h6" gutterBottom>
                Liquidity Heat Map
              </Typography>
              <ChartContainer>
                {isMarketLoading ? (
                  <Typography>Loading liquidity data...</Typography>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart
                      margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                    >
                      <CartesianGrid />
                      <XAxis 
                        type="category" 
                        dataKey="pair" 
                        name="Currency Pair" 
                        allowDuplicatedCategory={false} 
                      />
                      <YAxis 
                        type="number" 
                        dataKey="liquidity" 
                        name="Liquidity" 
                      />
                      <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                      <Scatter 
                        name="Liquidity" 
                        data={marketActivity?.liquidity || []} 
                        fill="#8884d8"
                      >
                        {(marketActivity?.liquidity || []).map((entry, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={entry.value > 75 ? '#4caf50' : entry.value > 50 ? '#ffeb3b' : '#f44336'}
                          />
                        ))}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                )}
              </ChartContainer>
            </MonitoringPanel>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Alerts & Notifications Tab */}
      <TabPanel value={tabValue} index={2}>
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <MonitoringPanel>
              <Typography variant="h6" gutterBottom>
                Active Alerts
              </Typography>
              <Box sx={{ overflowY: 'auto', maxHeight: '500px' }}>
                {isAlertsLoading ? (
                  <Typography>Loading alerts...</Typography>
                ) : (
                  <Box>
                    {(alerts?.active || []).map((alert, index) => (
                      <Paper 
                        key={alert.id || index}
                        sx={{ 
                          p: 2, 
                          mb: 2, 
                          borderLeft: `4px solid ${
                            alert.severity === 'critical' ? '#f44336' : 
                            alert.severity === 'warning' ? '#ff9800' : 
                            alert.severity === 'info' ? '#2196f3' : '#4caf50'
                          }`
                        }}
                      >
                        <Typography 
                          variant="subtitle1" 
                          sx={{ fontWeight: 'bold', color: alert.severity === 'critical' ? '#f44336' : 'inherit' }}
                        >
                          {alert.title}
                        </Typography>
                        <Typography variant="body2" sx={{ mt: 1 }}>
                          {alert.message}
                        </Typography>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                          <Typography variant="caption" color="text.secondary">
                            {alert.timestamp}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {alert.source}
                          </Typography>
                        </Box>
                      </Paper>
                    ))}

                    {(alerts?.active || []).length === 0 && (
                      <Typography>No active alerts</Typography>
                    )}
                  </Box>
                )}
              </Box>
            </MonitoringPanel>
          </Grid>
          <Grid item xs={12} md={6}>
            <MonitoringPanel>
              <Typography variant="h6" gutterBottom>
                Alert History
              </Typography>
              <ChartContainer>
                {isAlertsLoading ? (
                  <Typography>Loading alert history...</Typography>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={alerts?.history || []}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="info" name="Info" fill="#2196f3" stackId="a" />
                      <Bar dataKey="warning" name="Warning" fill="#ff9800" stackId="a" />
                      <Bar dataKey="critical" name="Critical" fill="#f44336" stackId="a" />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </ChartContainer>
            </MonitoringPanel>
          </Grid>
          <Grid item xs={12} md={6}>
            <MonitoringPanel>
              <Typography variant="h6" gutterBottom>
                Alert Distribution by Type
              </Typography>
              <ChartContainer>
                {isAlertsLoading ? (
                  <Typography>Loading alert distribution data...</Typography>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={alerts?.byType || []}
                        cx="50%"
                        cy="50%"
                        labelLine={true}
                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {(alerts?.byType || []).map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                )}
              </ChartContainer>
            </MonitoringPanel>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Risk Dashboard Tab */}
      <TabPanel value={tabValue} index={3}>
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <MonitoringPanel>
              <Typography variant="h6" gutterBottom>
                Position Risk
              </Typography>
              <ChartContainer>
                {isRiskLoading ? (
                  <Typography>Loading position risk data...</Typography>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={riskMetrics?.positionRisk || []}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="position" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="var" name="Value at Risk ($)" fill="#f44336" />
                      <Bar dataKey="maxLoss" name="Max Potential Loss ($)" fill="#ff9800" />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </ChartContainer>
            </MonitoringPanel>
          </Grid>
          <Grid item xs={12} md={6}>
            <MonitoringPanel>
              <Typography variant="h6" gutterBottom>
                Portfolio Risk Metrics
              </Typography>
              <ChartContainer>
                {isRiskLoading ? (
                  <Typography>Loading portfolio risk data...</Typography>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={riskMetrics?.portfolioRisk || []}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="var" name="VaR" stroke="#8884d8" />
                      <Line type="monotone" dataKey="cvar" name="CVaR" stroke="#f44336" />
                      <Line type="monotone" dataKey="sharpe" name="Sharpe Ratio" stroke="#4caf50" />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </ChartContainer>
            </MonitoringPanel>
          </Grid>
          <Grid item xs={12} md={6}>
            <MonitoringPanel>
              <Typography variant="h6" gutterBottom>
                Exposure by Currency
              </Typography>
              <ChartContainer>
                {isRiskLoading || isAllocationLoading ? (
                  <Typography>Loading exposure data...</Typography>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={portfolioAllocation?.byCurrency || []}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {(portfolioAllocation?.byCurrency || []).map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                )}
              </ChartContainer>
            </MonitoringPanel>
          </Grid>
          <Grid item xs={12} md={6}>
            <MonitoringPanel>
              <Typography variant="h6" gutterBottom>
                Risk Heat Map
              </Typography>
              <ChartContainer>
                {isRiskLoading ? (
                  <Typography>Loading risk heat map data...</Typography>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart
                      margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                    >
                      <CartesianGrid />
                      <XAxis 
                        type="number" 
                        dataKey="volatility" 
                        name="Volatility (%)" 
                        domain={[0, 'dataMax']}
                      />
                      <YAxis 
                        type="number" 
                        dataKey="correlation" 
                        name="Correlation" 
                        domain={[-1, 1]}
                      />
                      <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                      <Legend />
                      <Scatter 
                        name="Risk Profile" 
                        data={riskMetrics?.riskMap || []} 
                        fill="#8884d8"
                      >
                        {(riskMetrics?.riskMap || []).map((entry, index) => {
                          // Generate color based on risk score
                          const riskScore = entry.volatility * Math.abs(entry.correlation);
                          const color = riskScore > 15 ? '#f44336' : 
                                       riskScore > 10 ? '#ff9800' : 
                                       riskScore > 5 ? '#ffeb3b' : '#4caf50';
                          return (
                            <Cell key={`cell-${index}`} fill={color} />
                          );
                        })}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                )}
              </ChartContainer>
            </MonitoringPanel>
          </Grid>
        </Grid>
      </TabPanel>
    </MonitoringContainer>
  );
};

export default RealTimeMonitoringDisplays;
