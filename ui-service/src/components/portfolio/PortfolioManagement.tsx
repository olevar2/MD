import React from 'react';
import { 
  Box, 
  Grid, 
  Typography, 
  Paper, 
  Tabs, 
  Tab,
  Button as MuiButton,
  ButtonGroup,
  MenuItem,
  Select,
  FormControl,
  InputLabel
} from '@mui/material';
import { Card } from '../ui_library/Card';
import { Chart } from '../ui_library/Chart';
import { DataTable } from '../ui_library/DataTable';
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
        <Box sx={{ p: 2, height: '100%' }}>
          {children}
        </Box>
      )}
    </div>
  );
}

// Mock chart data for portfolio performance
const mockPortfolioPerformanceData = Array.from({ length: 90 }, (_, i) => ({
  time: new Date(Date.now() - (90 - i) * 24 * 60 * 60 * 1000).toISOString().slice(0, 10),
  value: 50000 + (Math.random() * 2000 - 500) * i,
}));

// Mock data for currency exposure
const mockCurrencyExposureData = [
  { id: 1, currency: 'USD', longExposure: 25000, shortExposure: 12000, netExposure: 13000, percentOfPortfolio: 24 },
  { id: 2, currency: 'EUR', longExposure: 18000, shortExposure: 22000, netExposure: -4000, percentOfPortfolio: -7.4 },
  { id: 3, currency: 'GBP', longExposure: 15000, shortExposure: 5000, netExposure: 10000, percentOfPortfolio: 18.4 },
  { id: 4, currency: 'JPY', longExposure: 8000, shortExposure: 16000, netExposure: -8000, percentOfPortfolio: -14.7 },
  { id: 5, currency: 'CAD', longExposure: 12000, shortExposure: 3000, netExposure: 9000, percentOfPortfolio: 16.6 },
  { id: 6, currency: 'AUD', longExposure: 7000, shortExposure: 9000, netExposure: -2000, percentOfPortfolio: -3.7 },
  { id: 7, currency: 'CHF', longExposure: 6000, shortExposure: 1000, netExposure: 5000, percentOfPortfolio: 9.2 },
  { id: 8, currency: 'NZD', longExposure: 3000, shortExposure: 0, netExposure: 3000, percentOfPortfolio: 5.5 },
];

// Mock data for portfolio holdings
const mockPositionsData = [
  { id: 1, symbol: 'EUR/USD', direction: 'Long', entryPrice: 1.0823, currentPrice: 1.0834, quantity: 100000, marketValue: 108340, unrealizedPnl: 110, pnlPercent: 0.10, holdingPeriod: '3d 5h' },
  { id: 2, symbol: 'GBP/JPY', direction: 'Short', entryPrice: 171.43, currentPrice: 171.53, quantity: 50000, marketValue: 85765, unrealizedPnl: -50, pnlPercent: -0.06, holdingPeriod: '1d 2h' },
  { id: 3, symbol: 'USD/CAD', direction: 'Long', entryPrice: 1.3676, currentPrice: 1.3695, quantity: 75000, marketValue: 102712.5, unrealizedPnl: 142.5, pnlPercent: 0.14, holdingPeriod: '5d 7h' },
  { id: 4, symbol: 'AUD/USD', direction: 'Short', entryPrice: 0.6423, currentPrice: 0.6410, quantity: 60000, marketValue: 38460, unrealizedPnl: 78, pnlPercent: 0.20, holdingPeriod: '2d 4h' },
  { id: 5, symbol: 'NZD/USD', direction: 'Long', entryPrice: 0.5932, currentPrice: 0.5925, quantity: 40000, marketValue: 23700, unrealizedPnl: -28, pnlPercent: -0.12, holdingPeriod: '6h 15m' },
];

// Mock data for realized trades
const mockTradesData = [
  { id: 1, symbol: 'EUR/USD', direction: 'Long', entryPrice: 1.0750, exitPrice: 1.0830, quantity: 100000, realizedPnl: 800, pnlPercent: 0.74, holdingPeriod: '2d 3h', exitDate: '2025-04-15' },
  { id: 2, symbol: 'GBP/USD', direction: 'Short', entryPrice: 1.2650, exitPrice: 1.2580, quantity: 75000, realizedPnl: 525, pnlPercent: 0.55, holdingPeriod: '1d 5h', exitDate: '2025-04-14' },
  { id: 3, symbol: 'USD/JPY', direction: 'Long', entryPrice: 152.30, exitPrice: 151.85, quantity: 60000, realizedPnl: -270, pnlPercent: -0.30, holdingPeriod: '4h 20m', exitDate: '2025-04-16' },
  { id: 4, symbol: 'AUD/NZD', direction: 'Long', entryPrice: 1.0830, exitPrice: 1.0910, quantity: 50000, realizedPnl: 400, pnlPercent: 0.74, holdingPeriod: '3d 2h', exitDate: '2025-04-13' },
  { id: 5, symbol: 'USD/CAD', direction: 'Short', entryPrice: 1.3720, exitPrice: 1.3780, quantity: 80000, realizedPnl: -480, pnlPercent: -0.44, holdingPeriod: '1d 7h', exitDate: '2025-04-12' },
];

const currencyExposureColumns: GridColDef[] = [
  { field: 'currency', headerName: 'Currency', width: 100 },
  { 
    field: 'longExposure',
    headerName: 'Long Exposure', 
    width: 150,
    type: 'number',
    valueFormatter: (params) => {
      return `$${params.value.toLocaleString('en-US')}`;
    }
  },
  { 
    field: 'shortExposure', 
    headerName: 'Short Exposure', 
    width: 150,
    type: 'number',
    valueFormatter: (params) => {
      return `$${params.value.toLocaleString('en-US')}`;
    }
  },
  { 
    field: 'netExposure', 
    headerName: 'Net Exposure', 
    width: 150,
    type: 'number',
    valueFormatter: (params) => {
      return `$${params.value.toLocaleString('en-US')}`;
    },
    cellClassName: (params) => {
      return params.value >= 0 ? 'positive-value' : 'negative-value';
    }
  },
  { 
    field: 'percentOfPortfolio', 
    headerName: '% of Portfolio', 
    width: 150,
    type: 'number',
    valueFormatter: (params) => {
      return `${params.value}%`;
    },
    cellClassName: (params) => {
      return params.value >= 0 ? 'positive-value' : 'negative-value';
    }
  },
];

const positionsColumns: GridColDef[] = [
  { field: 'symbol', headerName: 'Symbol', width: 100 },
  { field: 'direction', headerName: 'Direction', width: 100 },
  { 
    field: 'entryPrice', 
    headerName: 'Entry Price', 
    width: 120,
    type: 'number',
    valueFormatter: (params) => {
      return params.value.toFixed(4);
    }
  },
  { 
    field: 'currentPrice', 
    headerName: 'Current Price', 
    width: 120,
    type: 'number',
    valueFormatter: (params) => {
      return params.value.toFixed(4);
    }
  },
  { 
    field: 'quantity', 
    headerName: 'Quantity', 
    width: 120,
    type: 'number',
    valueFormatter: (params) => {
      return params.value.toLocaleString('en-US');
    }
  },
  { 
    field: 'marketValue', 
    headerName: 'Market Value', 
    width: 140,
    type: 'number',
    valueFormatter: (params) => {
      return `$${params.value.toLocaleString('en-US')}`;
    }
  },
  { 
    field: 'unrealizedPnl', 
    headerName: 'Unrealized P&L', 
    width: 150,
    type: 'number',
    valueFormatter: (params) => {
      return `$${params.value.toLocaleString('en-US')}`;
    },
    cellClassName: (params) => {
      return params.value >= 0 ? 'positive-value' : 'negative-value';
    }
  },
  { 
    field: 'pnlPercent', 
    headerName: 'P&L %', 
    width: 100,
    type: 'number',
    valueFormatter: (params) => {
      return `${params.value.toFixed(2)}%`;
    },
    cellClassName: (params) => {
      return params.value >= 0 ? 'positive-value' : 'negative-value';
    }
  },
  { field: 'holdingPeriod', headerName: 'Holding Period', width: 150 },
];

const tradesColumns: GridColDef[] = [
  { field: 'symbol', headerName: 'Symbol', width: 100 },
  { field: 'direction', headerName: 'Direction', width: 100 },
  { 
    field: 'entryPrice', 
    headerName: 'Entry Price', 
    width: 120,
    type: 'number',
    valueFormatter: (params) => {
      return params.value.toFixed(4);
    }
  },
  { 
    field: 'exitPrice', 
    headerName: 'Exit Price', 
    width: 120,
    type: 'number',
    valueFormatter: (params) => {
      return params.value.toFixed(4);
    }
  },
  { 
    field: 'quantity', 
    headerName: 'Quantity', 
    width: 120,
    type: 'number',
    valueFormatter: (params) => {
      return params.value.toLocaleString('en-US');
    }
  },
  { 
    field: 'realizedPnl', 
    headerName: 'Realized P&L', 
    width: 140,
    type: 'number',
    valueFormatter: (params) => {
      return `$${params.value.toLocaleString('en-US')}`;
    },
    cellClassName: (params) => {
      return params.value >= 0 ? 'positive-value' : 'negative-value';
    }
  },
  { 
    field: 'pnlPercent', 
    headerName: 'P&L %', 
    width: 100,
    type: 'number',
    valueFormatter: (params) => {
      return `${params.value.toFixed(2)}%`;
    },
    cellClassName: (params) => {
      return params.value >= 0 ? 'positive-value' : 'negative-value';
    }
  },
  { field: 'holdingPeriod', headerName: 'Holding Period', width: 120 },
  { field: 'exitDate', headerName: 'Exit Date', width: 120 },
];

export const PortfolioManagement: React.FC = () => {
  const [tabValue, setTabValue] = React.useState(0);
  const [timeframe, setTimeframe] = React.useState('3m');
  const [exportFormat, setExportFormat] = React.useState('');

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleTimeframeChange = (event: React.MouseEvent<HTMLElement>, newTimeframe: string) => {
    if (newTimeframe !== null) {
      setTimeframe(newTimeframe);
    }
  };

  const handleExportFormatChange = (event) => {
    setExportFormat(event.target.value);
  };

  const handleExport = () => {
    if (exportFormat) {
      console.log(`Exporting portfolio data in ${exportFormat} format`);
      // Actual export implementation would go here
      setExportFormat('');
    }
  };

  return (
    <Box sx={{ flexGrow: 1, height: '100vh', overflow: 'hidden', p: 2 }}>
      <Grid container spacing={2} sx={{ height: '100%' }}>
        {/* Header with title and actions */}
        <Grid item xs={12}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h4">Portfolio Management</Typography>
            <Box sx={{ display: 'flex', gap: 2 }}>
              <FormControl sx={{ minWidth: 120 }}>
                <InputLabel id="export-format-label">Export</InputLabel>
                <Select
                  labelId="export-format-label"
                  value={exportFormat}
                  label="Export"
                  onChange={handleExportFormatChange}
                  size="small"
                >
                  <MenuItem value="csv">CSV</MenuItem>
                  <MenuItem value="excel">Excel</MenuItem>
                  <MenuItem value="json">JSON</MenuItem>
                  <MenuItem value="pdf">PDF</MenuItem>
                </Select>
              </FormControl>
              <MuiButton 
                variant="contained" 
                color="primary" 
                onClick={handleExport}
                disabled={!exportFormat}
              >
                Export
              </MuiButton>
            </Box>
          </Box>
        </Grid>

        {/* Portfolio Summary */}
        <Grid item xs={12}>
          <Grid container spacing={2}>
            <Grid item xs={12} md={3}>
              <Card title="Total Portfolio Value">
                <Typography variant="h4">$54,321.00</Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                  <Typography 
                    variant="body2" 
                    sx={{ 
                      color: 'tradingProfit.main', 
                      fontWeight: 600 
                    }}
                  >
                    +$1,243.50 (2.3%)
                  </Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                    this month
                  </Typography>
                </Box>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card title="Unrealized P&L">
                <Typography variant="h4" sx={{ color: 'tradingProfit.main' }}>+$252.50</Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    0.46% of portfolio value
                  </Typography>
                </Box>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card title="MTD Performance">
                <Typography variant="h4" sx={{ color: 'tradingProfit.main' }}>+3.2%</Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    vs. +1.8% benchmark
                  </Typography>
                </Box>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card title="YTD Performance">
                <Typography variant="h4" sx={{ color: 'tradingProfit.main' }}>+11.7%</Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    vs. +7.5% benchmark
                  </Typography>
                </Box>
              </Card>
            </Grid>
          </Grid>
        </Grid>

        {/* Performance Chart */}
        <Grid item xs={12} lg={8} sx={{ height: '300px' }}>
          <Card title="Portfolio Performance" sx={{ height: '100%' }}>
            <Box sx={{ mb: 2 }}>
              <ButtonGroup size="small">
                <MuiButton 
                  variant={timeframe === '1m' ? 'contained' : 'outlined'} 
                  onClick={(e) => handleTimeframeChange(e, '1m')}
                >
                  1M
                </MuiButton>
                <MuiButton 
                  variant={timeframe === '3m' ? 'contained' : 'outlined'} 
                  onClick={(e) => handleTimeframeChange(e, '3m')}
                >
                  3M
                </MuiButton>
                <MuiButton 
                  variant={timeframe === '6m' ? 'contained' : 'outlined'} 
                  onClick={(e) => handleTimeframeChange(e, '6m')}
                >
                  6M
                </MuiButton>
                <MuiButton 
                  variant={timeframe === '1y' ? 'contained' : 'outlined'} 
                  onClick={(e) => handleTimeframeChange(e, '1y')}
                >
                  1Y
                </MuiButton>
                <MuiButton 
                  variant={timeframe === 'ytd' ? 'contained' : 'outlined'} 
                  onClick={(e) => handleTimeframeChange(e, 'ytd')}
                >
                  YTD
                </MuiButton>
                <MuiButton 
                  variant={timeframe === 'all' ? 'contained' : 'outlined'} 
                  onClick={(e) => handleTimeframeChange(e, 'all')}
                >
                  ALL
                </MuiButton>
              </ButtonGroup>
            </Box>
            <Chart
              data={mockPortfolioPerformanceData}
              height="200px"
              chartType="area"
            />
          </Card>
        </Grid>

        {/* Currency Exposure */}
        <Grid item xs={12} lg={4} sx={{ height: '300px' }}>
          <Card title="Currency Exposure" sx={{ height: '100%' }}>
            <DataTable
              columns={currencyExposureColumns}
              data={mockCurrencyExposureData}
              height="100%"
              pageSize={5}
              stickyHeader
            />
          </Card>
        </Grid>

        {/* Detailed Portfolio Tabs */}
        <Grid item xs={12} sx={{ height: 'calc(100% - 460px)' }}>
          <Paper sx={{ height: '100%', borderRadius: 2, overflow: 'hidden' }}>
            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
              <Tabs value={tabValue} onChange={handleTabChange} aria-label="portfolio tabs">
                <Tab label="Current Holdings" id="tab-0" />
                <Tab label="Realized Trades" id="tab-1" />
                <Tab label="Performance Analytics" id="tab-2" />
                <Tab label="Tax Reporting" id="tab-3" />
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
              <DataTable
                columns={tradesColumns}
                data={mockTradesData}
                height="100%"
                stickyHeader
                toolbarEnabled
              />
            </TabPanel>
            <TabPanel value={tabValue} index={2}>
              <Typography>Performance analytics content</Typography>
            </TabPanel>
            <TabPanel value={tabValue} index={3}>
              <Typography>Tax reporting content</Typography>
            </TabPanel>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default PortfolioManagement;
