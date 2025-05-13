/**
 * Divergence Analysis Widget
 * 
 * This component provides a user interface for analyzing divergences
 * between correlated currency pairs.
 */

import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Button, 
  Card, 
  CardContent, 
  CircularProgress, 
  Divider, 
  FormControl, 
  FormControlLabel, 
  Grid, 
  InputLabel, 
  MenuItem, 
  Paper, 
  Select, 
  Switch, 
  Typography 
} from '@mui/material';
import { 
  TrendingUp, 
  TrendingDown, 
  CompareArrows, 
  Timeline 
} from '@mui/icons-material';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer, 
  AreaChart, 
  Area, 
  ReferenceLine 
} from 'recharts';

import { analyzeDivergence, analyzeDivergenceML } from '../api/divergenceApi';
import { formatNumber, formatPercentage } from '../utils/formatters';
import { SYMBOLS, TIMEFRAMES } from '../constants';
import LoadingOverlay from './LoadingOverlay';
import ErrorMessage from './ErrorMessage';
import DivergenceCard from './DivergenceCard';
import ScoreGauge from './ScoreGauge';

const DivergenceAnalysisWidget = ({ onResultsChange }) => {
  // State
  const [symbol, setSymbol] = useState('EURUSD');
  const [timeframe, setTimeframe] = useState('H1');
  const [useML, setUseML] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      // Call API based on selected mode
      const response = useML 
        ? await analyzeDivergenceML({
            symbol,
            timeframe
          })
        : await analyzeDivergence({
            symbol,
            timeframe
          });
      
      setResults(response);
      
      // Notify parent component
      if (onResultsChange) {
        onResultsChange(response);
      }
    } catch (err) {
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  // Format prediction data for chart
  const formatPredictionData = (predictions) => {
    if (!predictions || !predictions.values) return [];
    
    return predictions.values.map((value, index) => ({
      index,
      value,
      lowerBound: predictions.lower_bound[index],
      upperBound: predictions.upper_bound[index]
    }));
  };

  // Format divergence data for chart
  const formatDivergenceData = (divergence) => {
    if (!divergence) return [];
    
    // Create data points for primary and related momentum
    const data = [];
    const numPoints = 10; // Number of data points to show
    
    for (let i = 0; i < numPoints; i++) {
      const primaryValue = divergence.primary_momentum * (1 + 0.1 * i);
      const relatedValue = divergence.related_momentum * (1 + 0.1 * i);
      const expectedValue = divergence.expected_momentum * (1 + 0.1 * i);
      
      data.push({
        index: i,
        primary: primaryValue,
        related: relatedValue,
        expected: expectedValue
      });
    }
    
    return data;
  };

  return (
    <Box sx={{ position: 'relative' }}>
      <LoadingOverlay open={loading} />
      
      <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Divergence Analysis
        </Typography>
        
        <form onSubmit={handleSubmit}>
          <Grid container spacing={3}>
            <Grid item xs={12} sm={6} md={4}>
              <FormControl fullWidth>
                <InputLabel>Symbol</InputLabel>
                <Select
                  value={symbol}
                  onChange={(e) => setSymbol(e.target.value)}
                  label="Symbol"
                >
                  {SYMBOLS.map((s) => (
                    <MenuItem key={s} value={s}>{s}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} sm={6} md={4}>
              <FormControl fullWidth>
                <InputLabel>Timeframe</InputLabel>
                <Select
                  value={timeframe}
                  onChange={(e) => setTimeframe(e.target.value)}
                  label="Timeframe"
                >
                  {TIMEFRAMES.map((t) => (
                    <MenuItem key={t} value={t}>{t}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} sm={6} md={4}>
              <FormControlLabel
                control={
                  <Switch
                    checked={useML}
                    onChange={(e) => setUseML(e.target.checked)}
                  />
                }
                label="Use Machine Learning"
              />
            </Grid>
            
            <Grid item xs={12}>
              <Button
                type="submit"
                variant="contained"
                color="primary"
                size="large"
                disabled={loading}
                startIcon={loading ? <CircularProgress size={20} /> : null}
              >
                Analyze Divergence
              </Button>
            </Grid>
          </Grid>
        </form>
      </Paper>
      
      {error && (
        <ErrorMessage message={error} onClose={() => setError(null)} />
      )}
      
      {results && (
        <Box>
          <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
            <Typography variant="h5" gutterBottom>
              Results for {results.symbol} - {results.timeframe}
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12} md={4}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Divergence Score
                    </Typography>
                    <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
                      <ScoreGauge value={results.divergence_score} />
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                      {results.divergences_found} divergences found
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              {results.price_prediction && (
                <Grid item xs={12} md={8}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Price Prediction
                      </Typography>
                      <ResponsiveContainer width="100%" height={200}>
                        <AreaChart data={formatPredictionData(results.price_prediction)}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="index" />
                          <YAxis domain={['auto', 'auto']} />
                          <Tooltip />
                          <Legend />
                          <Area type="monotone" dataKey="value" stroke="#8884d8" fill="#8884d8" fillOpacity={0.3} />
                          <Area type="monotone" dataKey="lowerBound" stroke="#82ca9d" fill="#82ca9d" fillOpacity={0.1} />
                          <Area type="monotone" dataKey="upperBound" stroke="#ffc658" fill="#ffc658" fillOpacity={0.1} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                </Grid>
              )}
            </Grid>
            
            <Divider sx={{ my: 3 }} />
            
            <Typography variant="h6" gutterBottom>
              Divergences ({results.divergences_found})
            </Typography>
            
            {results.divergences.length > 0 ? (
              <Grid container spacing={3}>
                {results.divergences.map((divergence, index) => (
                  <Grid item xs={12} key={index}>
                    <DivergenceCard divergence={divergence} />
                    
                    <Box sx={{ mt: 2 }}>
                      <ResponsiveContainer width="100%" height={200}>
                        <LineChart data={formatDivergenceData(divergence)}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="index" />
                          <YAxis domain={['auto', 'auto']} />
                          <Tooltip />
                          <Legend />
                          <Line type="monotone" dataKey="primary" name={`${results.symbol} Momentum`} stroke="#8884d8" />
                          <Line type="monotone" dataKey="related" name={`${divergence.pair} Momentum`} stroke="#82ca9d" />
                          <Line type="monotone" dataKey="expected" name="Expected Momentum" stroke="#ffc658" strokeDasharray="5 5" />
                          <ReferenceLine y={0} stroke="red" strokeDasharray="3 3" />
                        </LineChart>
                      </ResponsiveContainer>
                    </Box>
                    
                    <Divider sx={{ my: 2 }} />
                  </Grid>
                ))}
              </Grid>
            ) : (
              <Typography variant="body2" color="text.secondary">
                No divergences found
              </Typography>
            )}
            
            <Box sx={{ mt: 3, textAlign: 'right' }}>
              <Typography variant="caption" color="text.secondary">
                Execution time: {formatNumber(results.execution_time, 3)}s | Request ID: {results.request_id}
              </Typography>
            </Box>
          </Paper>
        </Box>
      )}
    </Box>
  );
};

export default DivergenceAnalysisWidget;
