/**
 * Confluence Detection Widget
 * 
 * This component provides a user interface for detecting confluence signals
 * across multiple currency pairs.
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
  Slider, 
  Switch, 
  Typography 
} from '@mui/material';
import { 
  TrendingUp, 
  TrendingDown, 
  CheckCircle, 
  Cancel, 
  Help, 
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
  Area 
} from 'recharts';

import { detectConfluence, detectConfluenceML } from '../api/confluenceApi';
import { formatNumber, formatPercentage } from '../utils/formatters';
import { SYMBOLS, TIMEFRAMES, SIGNAL_TYPES } from '../constants';
import LoadingOverlay from './LoadingOverlay';
import ErrorMessage from './ErrorMessage';
import ConfirmationCard from './ConfirmationCard';
import ContradictionCard from './ContradictionCard';
import ScoreGauge from './ScoreGauge';

const ConfluenceDetectionWidget = ({ onResultsChange }) => {
  // State
  const [symbol, setSymbol] = useState('EURUSD');
  const [timeframe, setTimeframe] = useState('H1');
  const [signalType, setSignalType] = useState('trend');
  const [signalDirection, setSignalDirection] = useState('bullish');
  const [useCurrencyStrength, setUseCurrencyStrength] = useState(true);
  const [minConfirmationStrength, setMinConfirmationStrength] = useState(0.3);
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
        ? await detectConfluenceML({
            symbol,
            timeframe,
            signalType,
            signalDirection,
            useCurrencyStrength,
            minConfirmationStrength
          })
        : await detectConfluence({
            symbol,
            timeframe,
            signalType,
            signalDirection,
            useCurrencyStrength,
            minConfirmationStrength
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

  return (
    <Box sx={{ position: 'relative' }}>
      <LoadingOverlay open={loading} />
      
      <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Confluence Detection
        </Typography>
        
        <form onSubmit={handleSubmit}>
          <Grid container spacing={3}>
            <Grid item xs={12} sm={6} md={3}>
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
            
            <Grid item xs={12} sm={6} md={3}>
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
            
            <Grid item xs={12} sm={6} md={3}>
              <FormControl fullWidth>
                <InputLabel>Signal Type</InputLabel>
                <Select
                  value={signalType}
                  onChange={(e) => setSignalType(e.target.value)}
                  label="Signal Type"
                >
                  {SIGNAL_TYPES.map((t) => (
                    <MenuItem key={t} value={t}>{t.charAt(0).toUpperCase() + t.slice(1)}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <FormControl fullWidth>
                <InputLabel>Signal Direction</InputLabel>
                <Select
                  value={signalDirection}
                  onChange={(e) => setSignalDirection(e.target.value)}
                  label="Signal Direction"
                >
                  <MenuItem value="bullish">Bullish</MenuItem>
                  <MenuItem value="bearish">Bearish</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <Typography gutterBottom>
                Minimum Confirmation Strength: {minConfirmationStrength}
              </Typography>
              <Slider
                value={minConfirmationStrength}
                onChange={(e, newValue) => setMinConfirmationStrength(newValue)}
                step={0.05}
                min={0}
                max={1}
                valueLabelDisplay="auto"
              />
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={useCurrencyStrength}
                    onChange={(e) => setUseCurrencyStrength(e.target.checked)}
                  />
                }
                label="Use Currency Strength"
              />
              
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
                Detect Confluence
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
                      Confluence Score
                    </Typography>
                    <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
                      <ScoreGauge value={results.confluence_score} />
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                      {results.confirmation_count} confirmations, {results.contradiction_count} contradictions
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              {useML && results.pattern_score !== undefined && (
                <>
                  <Grid item xs={12} sm={6} md={4}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Pattern Recognition
                        </Typography>
                        <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
                          <ScoreGauge value={results.pattern_score} />
                        </Box>
                        <Typography variant="body2" color="text.secondary">
                          {Object.entries(results.patterns || {})
                            .sort(([, a], [, b]) => b - a)
                            .slice(0, 3)
                            .map(([pattern, score]) => (
                              <Box key={pattern} sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>{pattern.replace('_', ' ')}</span>
                                <span>{formatPercentage(score)}</span>
                              </Box>
                            ))}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  
                  <Grid item xs={12} sm={6} md={4}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Price Prediction
                        </Typography>
                        <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
                          <ScoreGauge value={results.prediction_score} />
                        </Box>
                        <Typography variant="body2" color="text.secondary">
                          Predicted direction: {results.price_prediction?.values[0] < results.price_prediction?.values[results.price_prediction?.values.length - 1] ? 'Bullish' : 'Bearish'}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                </>
              )}
            </Grid>
            
            {results.price_prediction && (
              <Box sx={{ mt: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Price Prediction
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
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
              </Box>
            )}
            
            <Divider sx={{ my: 3 }} />
            
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  Confirmations ({results.confirmations.length})
                </Typography>
                {results.confirmations.length > 0 ? (
                  results.confirmations.map((confirmation, index) => (
                    <ConfirmationCard key={index} confirmation={confirmation} />
                  ))
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    No confirmations found
                  </Typography>
                )}
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  Contradictions ({results.contradictions.length})
                </Typography>
                {results.contradictions.length > 0 ? (
                  results.contradictions.map((contradiction, index) => (
                    <ContradictionCard key={index} contradiction={contradiction} />
                  ))
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    No contradictions found
                  </Typography>
                )}
              </Grid>
            </Grid>
            
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

export default ConfluenceDetectionWidget;
