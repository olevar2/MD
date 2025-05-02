import React from 'react';
import dynamic from 'next/dynamic';
import { Grid, Paper, Box } from '@mui/material';
import { styled } from '@mui/material/styles';
import DashboardLayout from '../../components/layout/DashboardLayout';

// Dynamically import heavy components to improve initial load time
const ChartComponent = dynamic(() => import('../../components/trading/Chart'), { ssr: false });
const ActiveTradesPanel = dynamic(() => import('../../components/trading/ActiveTradesPanel'));
const MarketRegimeIndicator = dynamic(() => import('../../components/trading/MarketRegimeIndicator'));
const TradingSignals = dynamic(() => import('../../components/trading/TradingSignals'));

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
}));

const Dashboard = () => {
  return (
    <DashboardLayout>
      <Grid container spacing={2}>
        {/* Main Chart Section */}
        <Grid item xs={12} lg={8}>
          <StyledPaper>
            <ChartComponent />
          </StyledPaper>
        </Grid>

        {/* Active Trades and Performance Metrics */}
        <Grid item xs={12} lg={4}>
          <StyledPaper>
            <ActiveTradesPanel />
          </StyledPaper>
        </Grid>

        {/* Market Regime and Signal Analysis */}
        <Grid item xs={12} md={6}>
          <StyledPaper>
            <MarketRegimeIndicator />
          </StyledPaper>
        </Grid>

        {/* Trading Signals and Alerts */}
        <Grid item xs={12} md={6}>
          <StyledPaper>
            <TradingSignals />
          </StyledPaper>
        </Grid>
      </Grid>
    </DashboardLayout>
  );
};

export default Dashboard;
