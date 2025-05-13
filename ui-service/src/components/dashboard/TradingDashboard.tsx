import React from 'react';
import { 
  Box, 
  Grid, 
  Typography, 
  Paper, 
  Divider,
  Tab,
  Tabs
} from '@mui/material';
import { Card } from '../ui_library/Card';
import { Chart } from '../ui_library/Chart';
import { DataTable } from '../ui_library/DataTable';
import { StatusIndicator } from '../ui_library/StatusIndicator';
import { GridColDef } from '@mui/x-data-grid';

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
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
      style={{ height: '100%' }}
    >
      {value === index && (
        <Box sx={{ p: 3, height: '100%' }}>
          {children}
        </Box>
      )}
    </div>
  );
}

const positionsColumns: GridColDef[] = [
  { field: 'id', headerName: 'ID', width: 70 },
  { field: 'symbol', headerName: 'Symbol', width: 100 },
  { field: 'direction', headerName: 'Direction', width: 100 },
  { field: 'entryPrice', headerName: 'Entry Price', width: 110, type: 'number' },
  { field: 'currentPrice', headerName: 'Current Price', width: 120, type: 'number' },
  { field: 'size', headerName: 'Size', width: 100, type: 'number' },
  { 
    field: 'pnl', 
    headerName: 'P&L', 
    width: 120, 
    type: 'number',
    renderCell: (params) => {
      const value = params.value as number;
      return (
        <Typography 
          variant="body2" 
          sx={{ 
            color: value >= 0 ? 'tradingProfit.main' : 'tradingLoss.main',
            fontWeight: 600
          }}
        >
          {value >= 0 ? '+' : ''}{value.toFixed(2)}
        </Typography>
      );
    }
  },
  { 
    field: 'pnlPercent', 
    headerName: 'P&L %', 
    width: 100, 
    type: 'number',
    renderCell: (params) => {
      const value = params.value as number;
      return (
        <Typography 
          variant="body2" 
          sx={{ 
            color: value >= 0 ? 'tradingProfit.main' : 'tradingLoss.main',
            fontWeight: 600
          }}
        >
          {value >= 0 ? '+' : ''}{value.toFixed(2)}%
        </Typography>
      );
    }
  },
  { field: 'openTime', headerName: 'Open Time', width: 180 },
  { 
    field: 'status', 
    headerName: 'Status', 
    width: 120,
    renderCell: (params) => {
      const status = params.value as string;
      let statusType: 'success' | 'warning' | 'error' | 'info';
      
      switch(status) {
        case 'Open':
          statusType = 'success';
          break;
        case 'Pending':
          statusType = 'warning';
          break;
        case 'Closed':
          statusType = 'info';
          break;
        case 'Error':
          statusType = 'error';
          break;
        default:
          statusType = 'info';
      }
      
      return <StatusIndicator status={statusType} label={status} />;
    }
  },
];

// Mock data for positions
const mockPositionsData = [
  { id: 1, symbol: 'EUR/USD', direction: 'Buy', entryPrice: 1.0823, currentPrice: 1.0834, size: 10000, pnl: 11.0, pnlPercent: 0.10, openTime: '2025-04-17 09:23:45', status: 'Open' },
  { id: 2, symbol: 'GBP/JPY', direction: 'Sell', entryPrice: 171.43, currentPrice: 171.53, size: 5000, pnl: -5.0, pnlPercent: -0.06, openTime: '2025-04-17 10:15:22', status: 'Open' },
  { id: 3, symbol: 'USD/CAD', direction: 'Buy', entryPrice: 1.3676, currentPrice: 1.3695, size: 15000, pnl: 28.5, pnlPercent: 0.14, openTime: '2025-04-16 14:45:30', status: 'Open' },
  { id: 4, symbol: 'AUD/USD', direction: 'Sell', entryPrice: 0.6423, currentPrice: 0.6410, size: 20000, pnl: 26.0, pnlPercent: 0.20, openTime: '2025-04-17 08:12:15', status: 'Open' },
  { id: 5, symbol: 'NZD/USD', direction: 'Buy', entryPrice: 0.5932, currentPrice: 0.5925, size: 8000, pnl: -5.6, pnlPercent: -0.12, openTime: '2025-04-17 11:05:40', status: 'Open' },
];

// Mock chart data
const mockChartData = Array.from({ length: 100 }, (_, i) => ({
  time: new Date(Date.now() - (100 - i) * 15 * 60 * 1000).toISOString().slice(0, -5),
  value: 1.0800 + Math.random() * 0.005,
}));

export const TradingDashboard: React.FC = () => {
  const [tabValue, setTabValue] = React.useState(0);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  return (
    <Box sx={{ flexGrow: 1, height: '100vh', overflow: 'hidden' }}>
      <Grid container spacing={2} sx={{ height: '100%', p: 2 }}>
        {/* Top stats panel */}
        <Grid item xs={12}>
          <Grid container spacing={2}>
            <Grid item xs={12} md={3}>
              <Card title="Account Balance">
                <Typography variant="h4">$54,321.00</Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                  <Typography 
                    variant="body2" 
                    sx={{ 
                      color: 'tradingProfit.main', 
                      fontWeight: 600, 
                      display: 'flex', 
                      alignItems: 'center' 
                    }}
                  >
                    +$1,243.50 (2.3%)
                  </Typography>
                </Box>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card title="Open Positions">
                <Typography variant="h4">5</Typography>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    Long: 3
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Short: 2
                  </Typography>
                </Box>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card title="Daily P&L">
                <Typography variant="h4" sx={{ color: 'tradingProfit.main' }}>+$328.75</Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    Realized: +$175.25 | Unrealized: +$153.50
                  </Typography>
                </Box>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card title="System Status">
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <StatusIndicator status="success" label="Trading Engine" />
                    <Typography variant="body2">100%</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <StatusIndicator status="success" label="Data Pipeline" />
                    <Typography variant="body2">99.8%</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <StatusIndicator status="warning" label="ML Integration" />
                    <Typography variant="body2">94.5%</Typography>
                  </Box>
                </Box>
              </Card>
            </Grid>
          </Grid>
        </Grid>

        {/* Chart panel */}
        <Grid item xs={12} md={8} sx={{ height: 'calc(50% - 96px)' }}>
          <Card title="EUR/USD" subtitle="1H Chart" sx={{ height: '100%' }}>
            <Box sx={{ height: 'calc(100% - 32px)' }}>
              <Chart 
                data={mockChartData} 
                height="100%"
                chartType="area"
                title="EUR/USD" 
              />
            </Box>
          </Card>
        </Grid>

        {/* Signals panel */}
        <Grid item xs={12} md={4} sx={{ height: 'calc(50% - 96px)' }}>
          <Card title="Recent Signals" sx={{ height: '100%' }}>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, height: 'calc(100% - 32px)', overflow: 'auto' }}>
              <Paper elevation={0} sx={{ p: 2, bgcolor: 'background.default', borderRadius: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="subtitle2">EUR/USD Buy Signal</Typography>
                  <Typography variant="caption" color="text.secondary">10:15 AM</Typography>
                </Box>
                <Typography variant="body2" sx={{ mt: 1 }}>Support level identified at 1.0820 with RSI divergence</Typography>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                  <Typography variant="caption" color="text.secondary">Confidence: 85%</Typography>
                  <StatusIndicator status="success" size="small" />
                </Box>
              </Paper>
              
              <Paper elevation={0} sx={{ p: 2, bgcolor: 'background.default', borderRadius: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="subtitle2">GBP/JPY Sell Signal</Typography>
                  <Typography variant="caption" color="text.secondary">9:45 AM</Typography>
                </Box>
                <Typography variant="body2" sx={{ mt: 1 }}>Resistance rejection at 171.50 with overbought conditions</Typography>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                  <Typography variant="caption" color="text.secondary">Confidence: 72%</Typography>
                  <StatusIndicator status="warning" size="small" />
                </Box>
              </Paper>
              
              <Paper elevation={0} sx={{ p: 2, bgcolor: 'background.default', borderRadius: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="subtitle2">USD/CAD Buy Signal</Typography>
                  <Typography variant="caption" color="text.secondary">Yesterday</Typography>
                </Box>
                <Typography variant="body2" sx={{ mt: 1 }}>Bullish engulfing pattern with MACD crossover</Typography>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                  <Typography variant="caption" color="text.secondary">Confidence: 91%</Typography>
                  <StatusIndicator status="success" size="small" />
                </Box>
              </Paper>
            </Box>
          </Card>
        </Grid>

        {/* Positions and orders panel */}
        <Grid item xs={12} sx={{ height: 'calc(50% - 16px)' }}>
          <Paper sx={{ height: '100%', borderRadius: 2, overflow: 'hidden' }}>
            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
              <Tabs value={tabValue} onChange={handleTabChange} aria-label="trading tabs">
                <Tab label="Open Positions" id="tab-0" />
                <Tab label="Order History" id="tab-1" />
                <Tab label="Trade History" id="tab-2" />
                <Tab label="Performance Analytics" id="tab-3" />
              </Tabs>
            </Box>
            <TabPanel value={tabValue} index={0}>
              <DataTable
                columns={positionsColumns}
                data={mockPositionsData}
                height="100%"
                stickyHeader
                toolbarEnabled
              />
            </TabPanel>
            <TabPanel value={tabValue} index={1}>
              <Typography>Order history content</Typography>
            </TabPanel>
            <TabPanel value={tabValue} index={2}>
              <Typography>Trade history content</Typography>
            </TabPanel>
            <TabPanel value={tabValue} index={3}>
              <Typography>Performance analytics content</Typography>
            </TabPanel>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default TradingDashboard;
