import React, { useState, useEffect } from 'react';
import { Box, Grid, Paper, Typography, Tabs, Tab, CircularProgress } from '@mui/material';
import { styled } from '@mui/material/styles';
import { useQuery } from 'react-query';

// Component imports
import { TradingChart } from '../chart/TradingChart';
import { MarketOverview } from '../trading/MarketOverview';
import { OrderPanel } from '../trading/OrderPanel';
import { PositionsList } from '../trading/PositionsList';
import { AlertsWidget } from '../trading/AlertsWidget';
import { TradeHistory } from '../trading/TradeHistory';
import { PerformanceMetrics } from '../visualization/PerformanceMetrics';
import { CustomizableLayout } from '../layout/CustomizableLayout';

// Service imports
import { fetchMarketData, fetchPositions, fetchTradeHistory } from '../../services/tradingService';
import { fetchPerformanceMetrics } from '../../services/analysisService';

const DashboardContainer = styled(Box)(({ theme }) => ({
  padding: theme.spacing(2),
  minHeight: '100vh',
  backgroundColor: theme.palette.background.default,
}));

const LoadingContainer = styled(Box)({
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  height: '100vh',
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
      id={`dashboard-tabpanel-${index}`}
      aria-labelledby={`dashboard-tab-${index}`}
      {...other}
      style={{ height: 'calc(100vh - 112px)', overflowY: 'auto' }}
    >
      {value === index && children}
    </div>
  );
}

export const AdvancedTradingDashboard: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [layoutConfig, setLayoutConfig] = useState(() => {
    try {
      const saved = localStorage.getItem('dashboardLayout');
      return saved ? JSON.parse(saved) : null;
    } catch (e) {
      console.error('Failed to load layout configuration', e);
      return null;
    }
  });

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const { data: marketData, isLoading: isMarketLoading } = useQuery(
    'marketData', 
    fetchMarketData, 
    { refetchInterval: 5000 }
  );
  
  const { data: positions, isLoading: isPositionsLoading } = useQuery(
    'positions', 
    fetchPositions, 
    { refetchInterval: 10000 }
  );
  
  const { data: tradeHistory, isLoading: isHistoryLoading } = useQuery(
    'tradeHistory', 
    fetchTradeHistory
  );
  
  const { data: performanceData, isLoading: isPerformanceLoading } = useQuery(
    'performanceMetrics', 
    fetchPerformanceMetrics
  );

  const isLoading = isMarketLoading || isPositionsLoading || isHistoryLoading || isPerformanceLoading;

  const saveLayout = (newLayout: any) => {
    try {
      localStorage.setItem('dashboardLayout', JSON.stringify(newLayout));
      setLayoutConfig(newLayout);
    } catch (e) {
      console.error('Failed to save layout configuration', e);
    }
  };

  if (isLoading) {
    return (
      <LoadingContainer>
        <CircularProgress />
      </LoadingContainer>
    );
  }

  const defaultLayout = [
    { i: 'chart', x: 0, y: 0, w: 8, h: 4 },
    { i: 'overview', x: 8, y: 0, w: 4, h: 2 },
    { i: 'order', x: 8, y: 2, w: 4, h: 2 },
    { i: 'positions', x: 0, y: 4, w: 6, h: 2 },
    { i: 'alerts', x: 6, y: 4, w: 6, h: 2 },
    { i: 'history', x: 0, y: 6, w: 6, h: 2 },
    { i: 'performance', x: 6, y: 6, w: 6, h: 2 },
  ];

  const dashboard = (
    <>
      <Typography variant="h4" component="h1" gutterBottom>
        Advanced Trading Dashboard
      </Typography>
      
      <Tabs value={tabValue} onChange={handleTabChange} aria-label="dashboard tabs">
        <Tab label="Trading" id="dashboard-tab-0" aria-controls="dashboard-tabpanel-0" />
        <Tab label="Analysis" id="dashboard-tab-1" aria-controls="dashboard-tabpanel-1" />
        <Tab label="Performance" id="dashboard-tab-2" aria-controls="dashboard-tabpanel-2" />
      </Tabs>

      <TabPanel value={tabValue} index={0}>
        <CustomizableLayout 
          initialLayout={layoutConfig || defaultLayout}
          onLayoutChange={saveLayout}
          components={{
            chart: <TradingChart data={marketData} />,
            overview: <MarketOverview data={marketData} />,
            order: <OrderPanel />,
            positions: <PositionsList positions={positions} />,
            alerts: <AlertsWidget />,
            history: <TradeHistory history={tradeHistory} />,
            performance: <PerformanceMetrics data={performanceData} />
          }}
        />
      </TabPanel>
      
      <TabPanel value={tabValue} index={1}>
        <Typography variant="h6">Analysis Content</Typography>
        {/* Analysis components will be added here */}
      </TabPanel>
      
      <TabPanel value={tabValue} index={2}>
        <Typography variant="h6">Performance Content</Typography>
        {/* Performance components will be added here */}
      </TabPanel>
    </>
  );

  return <DashboardContainer>{dashboard}</DashboardContainer>;
};

export default AdvancedTradingDashboard;
