import React from 'react';
import { Box, Typography, Paper, Grid } from '@mui/material';
import { ResponsiveLine } from '@nivo/line';

interface RegimeData {
  regime: string;
  confidence: number;
  indicators: {
    name: string;
    value: number;
    threshold: number;
  }[];
}

const MarketRegimeIndicator: React.FC = () => {
  const [regimeHistory, setRegimeHistory] = React.useState([
    {
      id: 'regime-confidence',
      data: [
        { x: '00:00', y: 0.8 },
        { x: '04:00', y: 0.75 },
        { x: '08:00', y: 0.9 },
        { x: '12:00', y: 0.85 },
        { x: '16:00', y: 0.7 },
        { x: '20:00', y: 0.95 },
      ],
    },
  ]);

  const [currentRegime, setCurrentRegime] = React.useState<RegimeData>({
    regime: 'Trending',
    confidence: 0.85,
    indicators: [
      { name: 'ADX', value: 28, threshold: 25 },
      { name: 'Volatility', value: 0.12, threshold: 0.15 },
      { name: 'Momentum', value: 0.65, threshold: 0.5 },
    ],
  });

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Market Regime Analysis
      </Typography>
      
      <Grid container spacing={2}>
        <Grid item xs={12}>
          <Paper sx={{ p: 2, height: '200px' }}>
            <Typography variant="subtitle2" gutterBottom>
              Regime Confidence Over Time
            </Typography>
            <Box sx={{ height: '150px' }}>
              <ResponsiveLine
                data={regimeHistory}
                margin={{ top: 10, right: 30, bottom: 30, left: 40 }}
                xScale={{ type: 'point' }}
                yScale={{ type: 'linear', min: 0, max: 1 }}
                curve="monotoneX"
                enablePoints={false}
                enableGridX={false}
                enableArea={true}
                areaBaselineValue={0}
                axisBottom={{
                  tickSize: 5,
                  tickPadding: 5,
                  tickRotation: 0,
                }}
                axisLeft={{
                  tickSize: 5,
                  tickPadding: 5,
                  tickRotation: 0,
                }}
                colors={['#2196f3']}
              />
            </Box>
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Current Regime: {currentRegime.regime}
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Confidence: {(currentRegime.confidence * 100).toFixed(1)}%
            </Typography>
            
            <Box sx={{ mt: 2 }}>
              {currentRegime.indicators.map((indicator) => (
                <Box key={indicator.name} sx={{ mb: 1 }}>
                  <Grid container justifyContent="space-between">
                    <Grid item>
                      <Typography variant="body2">{indicator.name}</Typography>
                    </Grid>
                    <Grid item>
                      <Typography
                        variant="body2"
                        color={indicator.value >= indicator.threshold ? 'success.main' : 'text.secondary'}
                      >
                        {indicator.value.toFixed(2)} / {indicator.threshold.toFixed(2)}
                      </Typography>
                    </Grid>
                  </Grid>
                  <Box
                    sx={{
                      width: '100%',
                      height: '4px',
                      bgcolor: 'grey.100',
                      mt: 0.5,
                      borderRadius: 1,
                      overflow: 'hidden',
                    }}
                  >
                    <Box
                      sx={{
                        width: `${(indicator.value / indicator.threshold) * 100}%`,
                        height: '100%',
                        bgcolor: indicator.value >= indicator.threshold ? 'success.main' : 'primary.main',
                      }}
                    />
                  </Box>
                </Box>
              ))}
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default MarketRegimeIndicator;
