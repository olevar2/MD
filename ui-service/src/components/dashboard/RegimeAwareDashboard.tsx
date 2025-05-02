import React, { useState, useEffect } from 'react';
import { Grid, Card, Typography, CircularProgress, Alert } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { SignalConfidenceChart } from '../visualization/SignalConfidenceChart';
import { MarketRegimeIndicator } from './MarketRegimeIndicator';
import { PositionTable } from './PositionTable';
import { OrderEntry } from './OrderEntry';
import { useMarketData } from '../../hooks/useMarketData';
import { useMarketRegime } from '../../hooks/useMarketRegime';
import type { MarketRegime, Position } from '../../types';

export interface RegimeAwareDashboardProps {
  accountId: string;
  symbol: string;
  onRegimeChange?: (regime: MarketRegime) => void;
}

const RegimeAwareDashboard: React.FC<RegimeAwareDashboardProps> = ({
  accountId,
  symbol,
  onRegimeChange
}) => {
  const theme = useTheme();
  const { marketData, isLoading: isLoadingMarketData, error: marketDataError } = useMarketData(symbol);
  const { regime, isLoading: isLoadingRegime, error: regimeError } = useMarketRegime(symbol);

  useEffect(() => {
    if (regime && onRegimeChange) {
      onRegimeChange(regime);
    }
  }, [regime, onRegimeChange]);

  if (isLoadingMarketData || isLoadingRegime) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', padding: theme.spacing(4) }}>
        <CircularProgress />
      </div>
    );
  }

  if (marketDataError || regimeError) {
    return (
      <Alert severity="error">
        {marketDataError?.message || regimeError?.message || 'Failed to load dashboard data'}
      </Alert>
    );
  }

  const renderRegimeSpecificComponents = () => {
    switch (regime?.type) {
      case 'HighVolatility':
        return (
          <>
            <Grid item xs={12} md={8}>
              <Card>
                <SignalConfidenceChart
                  signalData={regime.signals || []}
                  timeRange="day"
                  confidenceThreshold={0.8}
                />
              </Card>
            </Grid>
            <Grid item xs={12} md={4}>
              <Card>
                <MarketRegimeIndicator
                  regime={regime}
                  volatility={regime.volatility}
                  trend={regime.trend}
                />
              </Card>
            </Grid>
          </>
        );

      case 'LowVolatility':
        return (
          <>
            <Grid item xs={12} md={6}>
              <Card>
                <SignalConfidenceChart
                  signalData={regime.signals || []}
                  timeRange="week"
                  confidenceThreshold={0.6}
                />
              </Card>
            </Grid>
            <Grid item xs={12} md={6}>
              <Card>
                <OrderEntry
                  symbol={symbol}
                  accountId={accountId}
                  riskLevel="low"
                />
              </Card>
            </Grid>
          </>
        );

      case 'Trending':
        return (
          <>
            <Grid item xs={12}>
              <Card>
                <SignalConfidenceChart
                  signalData={regime.signals || []}
                  timeRange="day"
                  confidenceThreshold={0.7}
                />
              </Card>
            </Grid>
            <Grid item xs={12}>
              <Card>
                <PositionTable
                  accountId={accountId}
                  symbol={symbol}
                />
              </Card>
            </Grid>
          </>
        );

      default:
        return (
          <Grid item xs={12}>
            <Card>
              <Typography variant="h6" gutterBottom>
                Normal Market Conditions
              </Typography>
              <SignalConfidenceChart
                signalData={regime?.signals || []}
                timeRange="day"
              />
            </Card>
          </Grid>
        );
    }
  };

  return (
    <div className="regime-aware-dashboard" role="region" aria-label={`Market Regime Dashboard - ${regime?.type || 'Normal'}`}>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Typography variant="h4" gutterBottom>
            {symbol} - {regime?.type || 'Normal'} Market
          </Typography>
        </Grid>
        {renderRegimeSpecificComponents()}
      </Grid>
    </div>
  );
};

export default RegimeAwareDashboard;
