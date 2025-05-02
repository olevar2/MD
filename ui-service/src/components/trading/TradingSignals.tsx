import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  CircularProgress,
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import TimelineIcon from '@mui/icons-material/Timeline';

interface Signal {
  id: string;
  symbol: string;
  type: 'BUY' | 'SELL';
  confidence: number;
  timestamp: string;
  strategy: string;
  indicators: {
    name: string;
    value: number;
    contribution: number;
  }[];
}

const TradingSignals: React.FC = () => {
  const [signals, setSignals] = React.useState<Signal[]>([
    {
      id: '1',
      symbol: 'EUR/USD',
      type: 'BUY',
      confidence: 0.85,
      timestamp: new Date().toISOString(),
      strategy: 'Trend Following',
      indicators: [
        { name: 'RSI', value: 32, contribution: 0.4 },
        { name: 'MA Cross', value: 1, contribution: 0.35 },
        { name: 'Volume', value: 1.5, contribution: 0.25 },
      ],
    },
    // Add more mock signals as needed
  ]);

  const renderConfidenceIndicator = (confidence: number) => (
    <Box sx={{ position: 'relative', display: 'inline-flex' }}>
      <CircularProgress
        variant="determinate"
        value={confidence * 100}
        color={confidence > 0.7 ? 'success' : confidence > 0.5 ? 'warning' : 'error'}
      />
      <Box
        sx={{
          top: 0,
          left: 0,
          bottom: 0,
          right: 0,
          position: 'absolute',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Typography variant="caption" component="div" color="text.secondary">
          {`${Math.round(confidence * 100)}%`}
        </Typography>
      </Box>
    </Box>
  );

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Trading Signals
      </Typography>

      <List>
        {signals.map((signal) => (
          <Paper key={signal.id} sx={{ mb: 2 }}>
            <ListItem>
              <ListItemIcon>
                {signal.type === 'BUY' ? (
                  <TrendingUpIcon color="success" />
                ) : (
                  <TrendingDownIcon color="error" />
                )}
              </ListItemIcon>
              <ListItemText
                primary={
                  <Grid container alignItems="center" spacing={2}>
                    <Grid item>
                      <Typography variant="subtitle1">
                        {signal.symbol} - {signal.type}
                      </Typography>
                    </Grid>
                    <Grid item>
                      {renderConfidenceIndicator(signal.confidence)}
                    </Grid>
                  </Grid>
                }
                secondary={
                  <Box sx={{ mt: 1 }}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Strategy: {signal.strategy}
                    </Typography>
                    <Grid container spacing={2}>
                      {signal.indicators.map((indicator) => (
                        <Grid item xs={4} key={indicator.name}>
                          <Paper
                            sx={{
                              p: 1,
                              backgroundColor: 'action.hover',
                            }}
                          >
                            <Typography variant="caption" display="block">
                              {indicator.name}
                            </Typography>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              <TimelineIcon sx={{ mr: 0.5, fontSize: '1rem' }} />
                              <Typography variant="body2">
                                {indicator.value.toFixed(2)}
                              </Typography>
                            </Box>
                            <Typography
                              variant="caption"
                              color="text.secondary"
                              display="block"
                            >
                              Impact: {(indicator.contribution * 100).toFixed(0)}%
                            </Typography>
                          </Paper>
                        </Grid>
                      ))}
                    </Grid>
                  </Box>
                }
              />
            </ListItem>
          </Paper>
        ))}
      </List>
    </Box>
  );
};

export default TradingSignals;
