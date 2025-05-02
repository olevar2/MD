import React, { useEffect, useState } from 'react';
import {
  Paper,
  Grid,
  Box,
  Typography,
  LinearProgress,
  Button,
  useTheme,
  Divider
} from '@mui/material';
import { formatDistance } from 'date-fns';
import { SignalConfidenceChart } from '../visualization/SignalConfidenceChart';

export interface TradingSignal {
  id: string;
  symbol: string;
  direction: 'BUY' | 'SELL' | 'NEUTRAL';
  confidence: number;
  timestamp: number;
  source: string;
  explanation: string;
  indicators: {
    name: string;
    value: number;
    contribution: number;
  }[];
}

export interface NewsItem {
  id: string;
  title: string;
  summary: string;
  source: string;
  impact: 'HIGH' | 'MEDIUM' | 'LOW';
  timestamp: number;
  url: string;
  sentiment?: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL';
}

interface SignalVisualizerProps {
  signals: TradingSignal[];
  news: NewsItem[];
  onSignalSelect: (signal: TradingSignal) => void;
  onNewsSelect: (news: NewsItem) => void;
}

const SignalVisualizer: React.FC<SignalVisualizerProps> = ({
  signals,
  news,
  onSignalSelect,
  onNewsSelect
}) => {
  const theme = useTheme();
  const [activeSignal, setActiveSignal] = useState<TradingSignal | null>(null);

  useEffect(() => {
    if (signals.length > 0 && !activeSignal) {
      setActiveSignal(signals[0]);
    }
  }, [signals]);

  const getSignalColor = (direction: string) => {
    switch (direction) {
      case 'BUY': return theme.palette.success.main;
      case 'SELL': return theme.palette.error.main;
      default: return theme.palette.grey[500];
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'HIGH': return theme.palette.error.main;
      case 'MEDIUM': return theme.palette.warning.main;
      default: return theme.palette.info.main;
    }
  };

  return (
    <Grid container spacing={2}>
      {/* Signals Section */}
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2, height: '100%' }}>
          <Typography variant="h6" gutterBottom>
            Trading Signals
          </Typography>
          
          {activeSignal && (
            <>
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle1" fontWeight="bold">
                  {activeSignal.symbol} - {activeSignal.direction}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {formatDistance(activeSignal.timestamp, new Date(), { addSuffix: true })}
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={activeSignal.confidence * 100}
                  sx={{
                    mt: 1,
                    mb: 1,
                    backgroundColor: theme.palette.grey[200],
                    '& .MuiLinearProgress-bar': {
                      backgroundColor: getSignalColor(activeSignal.direction)
                    }
                  }}
                />
                <Typography variant="body2">
                  {activeSignal.explanation}
                </Typography>
              </Box>

              <Divider sx={{ my: 2 }} />

              <Box sx={{ height: 200 }}>
                <SignalConfidenceChart
                  signalData={signals.map(s => ({
                    timestamp: s.timestamp,
                    signal: s.direction,
                    confidence: s.confidence,
                    direction: s.direction.toLowerCase() as 'buy' | 'sell' | 'neutral'
                  }))}
                  height={200}
                />
              </Box>
            </>
          )}
        </Paper>
      </Grid>

      {/* News Section */}
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2, height: '100%' }}>
          <Typography variant="h6" gutterBottom>
            Market News
          </Typography>
          
          <Box sx={{ mt: 2 }}>
            {news.map((item) => (
              <Box
                key={item.id}
                sx={{
                  mb: 2,
                  p: 1,
                  borderRadius: 1,
                  '&:hover': {
                    backgroundColor: theme.palette.action.hover,
                    cursor: 'pointer'
                  }
                }}
                onClick={() => onNewsSelect(item)}
              >
                <Grid container spacing={1}>
                  <Grid item xs={12}>
                    <Typography variant="subtitle2" fontWeight="bold">
                      {item.title}
                    </Typography>
                  </Grid>
                  <Grid item xs={8}>
                    <Typography variant="caption" color="text.secondary">
                      {item.source} - {formatDistance(item.timestamp, new Date(), { addSuffix: true })}
                    </Typography>
                  </Grid>
                  <Grid item xs={4}>
                    <Typography
                      variant="caption"
                      sx={{ color: getImpactColor(item.impact) }}
                      align="right"
                    >
                      {item.impact} IMPACT
                    </Typography>
                  </Grid>
                </Grid>
              </Box>
            ))}
          </Box>

          {news.length === 0 && (
            <Typography variant="body2" color="text.secondary" align="center">
              No recent news
            </Typography>
          )}
        </Paper>
      </Grid>
    </Grid>
  );
};

export default SignalVisualizer;
