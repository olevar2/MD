import React from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  CircularProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Card,
  CardContent,
  Tooltip
} from '@mui/material';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip as RechartsTooltip } from 'recharts';
import {
  HeatMap,
  HeatMapProps,
  HeatMapRow,
  HeatMapCell
} from '@nivo/heatmap';

interface Position {
  symbol: string;
  direction: 'LONG' | 'SHORT';
  size: number;
  entryPrice: number;
  currentPrice: number;
  pnl: number;
  margin: number;
  risk: number;
}

interface AssetAllocation {
  asset: string;
  percentage: number;
  value: number;
  risk: number;
}

interface CorrelationData {
  symbol: string;
  correlations: { [key: string]: number };
}

interface PortfolioBreakdownProps {
  accountId: string;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

const PortfolioBreakdown: React.FC<PortfolioBreakdownProps> = ({
  accountId
}) => {
  const [positions, setPositions] = React.useState<Position[]>([]);
  const [allocations, setAllocations] = React.useState<AssetAllocation[]>([]);
  const [correlations, setCorrelations] = React.useState<CorrelationData[]>([]);
  const [isLoading, setIsLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    const fetchPortfolioData = async () => {
      setIsLoading(true);
      setError(null);
      try {
        // In a real implementation, this would be an API call
        // const response = await api.get(`/accounts/${accountId}/portfolio`);
        await new Promise(resolve => setTimeout(resolve, 500)); // Simulate API call

        // Mock data
        const mockPositions: Position[] = [
          {
            symbol: 'EUR/USD',
            direction: 'LONG',
            size: 100000,
            entryPrice: 1.1850,
            currentPrice: 1.1900,
            pnl: 500,
            margin: 1000,
            risk: 0.8
          },
          {
            symbol: 'GBP/USD',
            direction: 'SHORT',
            size: 50000,
            entryPrice: 1.3800,
            currentPrice: 1.3750,
            pnl: 250,
            margin: 500,
            risk: 0.6
          }
        ];

        const mockAllocations: AssetAllocation[] = [
          { asset: 'EUR/USD', percentage: 40, value: 100000, risk: 0.8 },
          { asset: 'GBP/USD', percentage: 30, value: 75000, risk: 0.6 },
          { asset: 'USD/JPY', percentage: 20, value: 50000, risk: 0.4 },
          { asset: 'AUD/USD', percentage: 10, value: 25000, risk: 0.3 }
        ];

        const mockCorrelations: CorrelationData[] = [
          {
            symbol: 'EUR/USD',
            correlations: {
              'GBP/USD': 0.75,
              'USD/JPY': -0.3,
              'AUD/USD': 0.45
            }
          },
          {
            symbol: 'GBP/USD',
            correlations: {
              'EUR/USD': 0.75,
              'USD/JPY': -0.25,
              'AUD/USD': 0.5
            }
          },
          {
            symbol: 'USD/JPY',
            correlations: {
              'EUR/USD': -0.3,
              'GBP/USD': -0.25,
              'AUD/USD': -0.15
            }
          },
          {
            symbol: 'AUD/USD',
            correlations: {
              'EUR/USD': 0.45,
              'GBP/USD': 0.5,
              'USD/JPY': -0.15
            }
          }
        ];

        setPositions(mockPositions);
        setAllocations(mockAllocations);
        setCorrelations(mockCorrelations);
      } catch (err) {
        setError('Failed to load portfolio data');
        console.error('Portfolio data fetch error:', err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchPortfolioData();
    
    // Set up WebSocket or polling for updates if needed
    const interval = setInterval(fetchPortfolioData, 60000); // Update every minute
    
    return () => clearInterval(interval);
  }, [accountId]);

  const renderCorrelationMatrix = () => {
    if (!correlations.length) return null;

    const symbols = correlations.map(c => c.symbol);
    const data = symbols.map(symbol => {
      const correlationData = correlations.find(c => c.symbol === symbol)!;
      return {
        id: symbol,
        data: symbols.map(s => ({
          x: s,
          y: s === symbol ? 1 : correlationData.correlations[s] || 0
        }))
      };
    });

    return (
      <Box height={300}>
        <ResponsiveContainer width="100%" height="100%">
          <HeatMap
            data={data}
            margin={{ top: 30, right: 60, bottom: 60, left: 60 }}
            axisTop={{
              tickSize: 5,
              tickPadding: 5,
              tickRotation: -45,
              legend: '',
              legendOffset: 46
            }}
            axisRight={null}
            axisBottom={{
              tickSize: 5,
              tickPadding: 5,
              tickRotation: -45,
              legend: '',
              legendPosition: 'middle',
              legendOffset: 46
            }}
            axisLeft={{
              tickSize: 5,
              tickPadding: 5,
              tickRotation: 0,
              legend: '',
              legendPosition: 'middle',
              legendOffset: -40
            }}
            colors={{
              type: 'diverging',
              scheme: 'red_yellow_blue',
              divergeAt: 0.5,
              minValue: -1,
              maxValue: 1
            }}
            emptyColor="#555555"
            legends={[
              {
                anchor: 'bottom',
                translateX: 0,
                translateY: 30,
                length: 400,
                thickness: 8,
                direction: 'row',
                tickPosition: 'after',
                tickSize: 3,
                tickSpacing: 4,
                tickOverlap: false,
                title: 'Correlation →',
                titleAlign: 'start',
                titleOffset: 4
              }
            ]}
          />
        </ResponsiveContainer>
      </Box>
    );
  };

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" p={3}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error">
        {error}
      </Alert>
    );
  }

  return (
    <Grid container spacing={3}>
      {/* Positions Summary */}
      <Grid item xs={12}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Open Positions
          </Typography>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Symbol</TableCell>
                  <TableCell>Direction</TableCell>
                  <TableCell align="right">Size</TableCell>
                  <TableCell align="right">Entry Price</TableCell>
                  <TableCell align="right">Current Price</TableCell>
                  <TableCell align="right">P&L</TableCell>
                  <TableCell align="right">Risk</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {positions.map((position) => (
                  <TableRow key={position.symbol}>
                    <TableCell>{position.symbol}</TableCell>
                    <TableCell>{position.direction}</TableCell>
                    <TableCell align="right">{position.size.toLocaleString()}</TableCell>
                    <TableCell align="right">{position.entryPrice.toFixed(5)}</TableCell>
                    <TableCell align="right">{position.currentPrice.toFixed(5)}</TableCell>
                    <TableCell
                      align="right"
                      sx={{
                        color: position.pnl >= 0 ? 'success.main' : 'error.main',
                        fontWeight: 'bold'
                      }}
                    >
                      ${position.pnl.toLocaleString()}
                    </TableCell>
                    <TableCell
                      align="right"
                      sx={{
                        color: position.risk > 0.7 ? 'error.main' : 
                               position.risk > 0.4 ? 'warning.main' : 
                               'success.main'
                      }}
                    >
                      {(position.risk * 100).toFixed(1)}%
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      </Grid>

      {/* Asset Allocation */}
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Asset Allocation
          </Typography>
          <Box height={300}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={allocations}
                  dataKey="percentage"
                  nameKey="asset"
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  label={(entry) => `${entry.asset} (${entry.percentage}%)`}
                >
                  {allocations.map((entry, index) => (
                    <Cell key={entry.asset} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <RechartsTooltip
                  formatter={(value: number, name: string, props: any) => [
                    `${value}% ($${props.payload.value.toLocaleString()})`,
                    name
                  ]}
                />
              </PieChart>
            </ResponsiveContainer>
          </Box>
        </Paper>
      </Grid>

      {/* Risk Metrics */}
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Risk Metrics
          </Typography>
          <Grid container spacing={2}>
            {allocations.map((allocation) => (
              <Grid item xs={12} key={allocation.asset}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" gutterBottom>
                      {allocation.asset}
                    </Typography>
                    <Box
                      sx={{
                        width: '100%',
                        height: 4,
                        bgcolor: 'grey.200',
                        borderRadius: 2,
                        position: 'relative'
                      }}
                    >
                      <Box
                        sx={{
                          width: `${allocation.risk * 100}%`,
                          height: '100%',
                          bgcolor: allocation.risk > 0.7 ? 'error.main' : 
                                  allocation.risk > 0.4 ? 'warning.main' : 
                                  'success.main',
                          borderRadius: 2
                        }}
                      />
                    </Box>
                    <Typography variant="caption" color="text.secondary">
                      Risk Score: {(allocation.risk * 100).toFixed(1)}%
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Paper>
      </Grid>

      {/* Correlation Matrix */}
      <Grid item xs={12}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Asset Correlation Matrix
          </Typography>
          {renderCorrelationMatrix()}
        </Paper>
      </Grid>
    </Grid>
  );
};

export default PortfolioBreakdown;
