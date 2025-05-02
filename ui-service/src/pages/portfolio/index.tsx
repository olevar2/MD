import React from 'react';
import { Grid, Box, Paper, Typography } from '@mui/material';
import { ResponsivePie } from '@nivo/pie';
import { ResponsiveLine } from '@nivo/line';
import DashboardLayout from '../../components/layout/DashboardLayout';

interface PortfolioStats {
  totalValue: number;
  dailyPnL: number;
  dailyPnLPercent: number;
  exposureByPair: {
    pair: string;
    value: number;
  }[];
  riskMetrics: {
    var: number;
    sharpeRatio: number;
    maxDrawdown: number;
  };
  performance: {
    date: string;
    value: number;
  }[];
}

const Portfolio = () => {
  const [stats, setStats] = React.useState<PortfolioStats>({
    totalValue: 100000,
    dailyPnL: 1250,
    dailyPnLPercent: 1.25,
    exposureByPair: [
      { pair: 'EUR/USD', value: 35000 },
      { pair: 'GBP/USD', value: 25000 },
      { pair: 'USD/JPY', value: 20000 },
      { pair: 'AUD/USD', value: 15000 },
      { pair: 'USD/CAD', value: 5000 },
    ],
    riskMetrics: {
      var: 2.5,
      sharpeRatio: 1.8,
      maxDrawdown: 5.2,
    },
    performance: Array.from({ length: 30 }, (_, i) => ({
      date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
      value: 100000 * (1 + Math.sin(i / 10) * 0.1),
    })),
  });

  return (
    <DashboardLayout>
      <Grid container spacing={3}>
        {/* Portfolio Overview */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="h6" gutterBottom>Portfolio Overview</Typography>
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" color="text.secondary">Total Value</Typography>
              <Typography variant="h4">${stats.totalValue.toLocaleString()}</Typography>
            </Box>
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" color="text.secondary">Daily P&L</Typography>
              <Typography 
                variant="h5" 
                color={stats.dailyPnL >= 0 ? 'success.main' : 'error.main'}
              >
                ${stats.dailyPnL.toLocaleString()} ({stats.dailyPnLPercent}%)
              </Typography>
            </Box>
          </Paper>
        </Grid>

        {/* Risk Metrics */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="h6" gutterBottom>Risk Metrics</Typography>
            <Grid container spacing={2}>
              <Grid item xs={4}>
                <Typography variant="subtitle2" color="text.secondary">Value at Risk (95%)</Typography>
                <Typography variant="h6">{stats.riskMetrics.var}%</Typography>
              </Grid>
              <Grid item xs={4}>
                <Typography variant="subtitle2" color="text.secondary">Sharpe Ratio</Typography>
                <Typography variant="h6">{stats.riskMetrics.sharpeRatio}</Typography>
              </Grid>
              <Grid item xs={4}>
                <Typography variant="subtitle2" color="text.secondary">Max Drawdown</Typography>
                <Typography variant="h6">{stats.riskMetrics.maxDrawdown}%</Typography>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Currency Pair Exposure */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: 400 }}>
            <Typography variant="h6" gutterBottom>Currency Pair Exposure</Typography>
            <Box sx={{ height: 300 }}>
              <ResponsivePie
                data={stats.exposureByPair.map(({ pair, value }) => ({
                  id: pair,
                  label: pair,
                  value: value,
                }))}
                margin={{ top: 40, right: 80, bottom: 80, left: 80 }}
                innerRadius={0.5}
                padAngle={0.7}
                cornerRadius={3}
                activeOuterRadiusOffset={8}
                borderWidth={1}
                arcLinkLabelsSkipAngle={10}
                arcLinkLabelsTextColor="#333333"
              />
            </Box>
          </Paper>
        </Grid>

        {/* Performance Chart */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: 400 }}>
            <Typography variant="h6" gutterBottom>Performance History</Typography>
            <Box sx={{ height: 300 }}>
              <ResponsiveLine
                data={[{
                  id: 'portfolio-value',
                  data: stats.performance.map(({ date, value }) => ({
                    x: date,
                    y: value,
                  })),
                }]}
                margin={{ top: 20, right: 20, bottom: 50, left: 60 }}
                xScale={{ type: 'point' }}
                yScale={{ type: 'linear', min: 'auto', max: 'auto' }}
                curve="monotoneX"
                axisBottom={{
                  tickRotation: -45,
                }}
                enablePoints={false}
                enableArea={true}
                areaBaselineValue={stats.performance[0].value}
                colors={['#2196f3']}
              />
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </DashboardLayout>
  );
};

export default Portfolio;
