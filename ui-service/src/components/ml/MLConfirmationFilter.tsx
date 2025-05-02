/**
 * ML Confirmation Filter Component - Integrates machine learning models to validate trading signals
 */
import { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  CircularProgress,
  Switch,
  FormControlLabel,
  Card,
  CardContent,
  Divider,
  Grid,
  Chip,
  Tooltip,
  IconButton,
  Slider,
  Alert,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import InfoIcon from '@mui/icons-material/Info';
import SettingsIcon from '@mui/icons-material/Settings';
import TuneIcon from '@mui/icons-material/Tune';
import PsychologyIcon from '@mui/icons-material/Psychology';

interface MLConfirmationFilterProps {
  symbol: string;
  timeframe: string;
  signalType: 'buy' | 'sell' | null;
  signalStrength?: number;
  patternType?: string;
  onConfirmationResult?: (result: MLConfirmationResult) => void;
  isEnabled: boolean;
  onToggle: (enabled: boolean) => void;
}

export interface MLModel {
  id: string;
  name: string;
  type: 'classification' | 'regression';
  version: string;
  accuracy: number; // 0.0 to 1.0
  enabled: boolean;
  weight: number; // 0.0 to 1.0, used in ensemble decision
}

export interface MLPrediction {
  modelId: string;
  confidence: number; // 0.0 to 1.0
  prediction: 'buy' | 'sell' | 'neutral';
  timestamp: string;
  features?: Record<string, number>;
  explanation?: string[];
}

export interface MLConfirmationResult {
  confirmed: boolean;
  confidence: number;
  predictions: MLPrediction[];
  ensembleDecision: 'buy' | 'sell' | 'neutral';
  insights: string[];
}

export default function MLConfirmationFilter({
  symbol,
  timeframe,
  signalType,
  signalStrength = 0.5,
  patternType,
  onConfirmationResult,
  isEnabled,
  onToggle
}: MLConfirmationFilterProps) {
  const [loading, setLoading] = useState<boolean>(false);
  const [models, setModels] = useState<MLModel[]>([]);
  const [predictions, setPredictions] = useState<MLPrediction[]>([]);
  const [result, setResult] = useState<MLConfirmationResult | null>(null);
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(70);
  const [showDetails, setShowDetails] = useState<boolean>(false);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  // Load available models on component mount
  useEffect(() => {
    if (isEnabled) {
      loadModels();
    }
  }, [isEnabled]);

  // Fetch predictions whenever signal changes or models are toggled
  useEffect(() => {
    if (isEnabled && signalType && models.length > 0) {
      fetchPredictions();
    }
  }, [isEnabled, signalType, symbol, timeframe, models.map(m => m.enabled).join(',')]);

  // Process the results whenever predictions change
  useEffect(() => {
    if (predictions.length > 0) {
      const result = processMLPredictions();
      setResult(result);
      
      // Notify parent component about the result
      if (onConfirmationResult) {
        onConfirmationResult(result);
      }
    }
  }, [predictions, confidenceThreshold]);

  // Load available ML models
  const loadModels = async () => {
    // This would be an API call in a real implementation
    // For demo purposes, we'll create mock models
    setLoading(true);
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 500));
    
    const mockModels: MLModel[] = [
      {
        id: 'trend-classifier-v2',
        name: 'Trend Classifier',
        type: 'classification',
        version: '2.1.0',
        accuracy: 0.78,
        enabled: true,
        weight: 0.9
      },
      {
        id: 'pattern-recognition-cnn',
        name: 'Pattern Recognition CNN',
        type: 'classification',
        version: '1.3.2',
        accuracy: 0.82,
        enabled: true,
        weight: 0.85
      },
      {
        id: 'volatility-predictor',
        name: 'Volatility Predictor',
        type: 'regression',
        version: '1.1.5',
        accuracy: 0.73,
        enabled: true,
        weight: 0.7
      },
      {
        id: 'regime-classifier',
        name: 'Market Regime Classifier',
        type: 'classification',
        version: '2.0.1',
        accuracy: 0.81,
        enabled: true,
        weight: 0.8
      },
      {
        id: 'sentiment-analyzer',
        name: 'News Sentiment Analyzer',
        type: 'classification',
        version: '1.4.0',
        accuracy: 0.67,
        enabled: false,
        weight: 0.6
      }
    ];
    
    setModels(mockModels);
    setLoading(false);
  };

  // Fetch ML predictions for current market conditions
  const fetchPredictions = async () => {
    setLoading(true);
    
    // This would be an API call in a real implementation
    // For demo purposes, we'll generate mock predictions
    await new Promise(resolve => setTimeout(resolve, 800));
    
    const enabledModels = models.filter(model => model.enabled);
    const mockPredictions: MLPrediction[] = enabledModels.map(model => {
      // Generate a prediction that's biased toward the signal type
      // but with some randomness to simulate different model opinions
      const rand = Math.random();
      let prediction: 'buy' | 'sell' | 'neutral';
      let confidence: number;
      
      if (signalType === 'buy') {
        prediction = rand < 0.7 ? 'buy' : (rand < 0.9 ? 'neutral' : 'sell');
        confidence = prediction === 'buy' ? 0.7 + Math.random() * 0.3 :
                   prediction === 'neutral' ? 0.4 + Math.random() * 0.3 :
                   0.3 + Math.random() * 0.3;
      } else if (signalType === 'sell') {
        prediction = rand < 0.7 ? 'sell' : (rand < 0.9 ? 'neutral' : 'buy');
        confidence = prediction === 'sell' ? 0.7 + Math.random() * 0.3 :
                   prediction === 'neutral' ? 0.4 + Math.random() * 0.3 :
                   0.3 + Math.random() * 0.3;
      } else {
        prediction = ['buy', 'sell', 'neutral'][Math.floor(Math.random() * 3)] as any;
        confidence = 0.4 + Math.random() * 0.4;
      }
      
      // Adjust confidence based on model accuracy
      confidence = confidence * model.accuracy;
      
      // Generate mock features that influenced the decision
      const features: Record<string, number> = {
        rsi: 30 + Math.random() * 40,
        macd_histogram: -0.5 + Math.random(),
        volatility: 0.2 + Math.random() * 0.5,
        ma_distance: -2 + Math.random() * 4
      };
      
      // Create feature-based explanations
      const explanations = [];
      if (features.rsi < 30) {
        explanations.push("RSI indicates oversold conditions");
      } else if (features.rsi > 70) {
        explanations.push("RSI indicates overbought conditions");
      }
      
      if (features.macd_histogram > 0) {
        explanations.push("MACD histogram is positive, suggesting bullish momentum");
      } else {
        explanations.push("MACD histogram is negative, suggesting bearish momentum");
      }
      
      if (features.volatility > 0.5) {
        explanations.push("High market volatility detected");
      }
      
      if (Math.abs(features.ma_distance) > 2) {
        explanations.push(`Price is ${features.ma_distance > 0 ? 'above' : 'below'} moving average by significant distance`);
      }
      
      return {
        modelId: model.id,
        confidence,
        prediction,
        timestamp: new Date().toISOString(),
        features,
        explanation: explanations
      };
    });
    
    setPredictions(mockPredictions);
    setLoading(false);
  };

  // Process ML predictions to generate an ensemble decision
  const processMLPredictions = (): MLConfirmationResult => {
    // Only use enabled models
    const enabledModels = models.filter(model => model.enabled);
    const enabledPredictions = predictions.filter(p => 
      enabledModels.some(m => m.id === p.modelId)
    );
    
    if (enabledPredictions.length === 0) {
      return {
        confirmed: false,
        confidence: 0,
        predictions: [],
        ensembleDecision: 'neutral',
        insights: ["No enabled models to make predictions"]
      };
    }
    
    // Calculate weighted votes for each class
    let buyVotes = 0;
    let sellVotes = 0;
    let neutralVotes = 0;
    
    enabledPredictions.forEach(prediction => {
      const model = enabledModels.find(m => m.id === prediction.modelId);
      if (!model) return;
      
      const weight = model.weight * prediction.confidence;
      
      if (prediction.prediction === 'buy') {
        buyVotes += weight;
      } else if (prediction.prediction === 'sell') {
        sellVotes += weight;
      } else {
        neutralVotes += weight;
      }
    });
    
    // Determine the ensemble decision
    const totalVotes = buyVotes + sellVotes + neutralVotes;
    const buyPercentage = (buyVotes / totalVotes) * 100;
    const sellPercentage = (sellVotes / totalVotes) * 100;
    const neutralPercentage = (neutralVotes / totalVotes) * 100;
    
    let ensembleDecision: 'buy' | 'sell' | 'neutral';
    let confidence: number;
    
    if (buyPercentage >= sellPercentage && buyPercentage >= neutralPercentage) {
      ensembleDecision = 'buy';
      confidence = buyPercentage;
    } else if (sellPercentage >= buyPercentage && sellPercentage >= neutralPercentage) {
      ensembleDecision = 'sell';
      confidence = sellPercentage;
    } else {
      ensembleDecision = 'neutral';
      confidence = neutralPercentage;
    }
    
    // Determine if the signal is confirmed
    const confirmed = (ensembleDecision === signalType) && (confidence >= confidenceThreshold);
    
    // Generate insights
    const insights = generateInsights(enabledPredictions, ensembleDecision, confidence, signalType);
    
    return {
      confirmed,
      confidence,
      predictions: enabledPredictions,
      ensembleDecision,
      insights
    };
  };

  // Generate human-readable insights from ML predictions
  const generateInsights = (
    predictions: MLPrediction[], 
    ensembleDecision: 'buy' | 'sell' | 'neutral',
    confidence: number,
    signalType: 'buy' | 'sell' | null
  ): string[] => {
    const insights: string[] = [];
    
    // Add ensemble decision insight
    if (signalType) {
      if (ensembleDecision === signalType) {
        insights.push(`ML models confirm the ${signalType} signal with ${confidence.toFixed(1)}% confidence`);
      } else {
        insights.push(`ML models suggest ${ensembleDecision} instead of ${signalType} (${confidence.toFixed(1)}% confidence)`);
      }
    } else {
      insights.push(`ML models suggest ${ensembleDecision} with ${confidence.toFixed(1)}% confidence`);
    }
    
    // Check if models agree
    const uniquePredictions = new Set(predictions.map(p => p.prediction));
    if (uniquePredictions.size === 1) {
      insights.push("All models are in agreement about market direction");
    } else if (uniquePredictions.size === 2 && !uniquePredictions.has('neutral')) {
      insights.push("Models have conflicting predictions - use caution");
    } else if (uniquePredictions.size > 1) {
      insights.push("Mixed signals from different models - consider market context");
    }
    
    // Add insights about specific feature values
    const highConfidenceModels = predictions.filter(p => p.confidence > 0.7);
    if (highConfidenceModels.length > 0) {
      const topModel = highConfidenceModels.reduce((prev, current) => 
        (current.confidence > prev.confidence) ? current : prev, highConfidenceModels[0]);
      
      if (topModel.explanation && topModel.explanation.length > 0) {
        insights.push(`${topModel.explanation[0]} (${models.find(m => m.id === topModel.modelId)?.name || 'Model'})`);
      }
    }
    
    return insights;
  };

  // Toggle model enabled status
  const handleToggleModel = (modelId: string) => {
    setModels(models.map(model => 
      model.id === modelId ? { ...model, enabled: !model.enabled } : model
    ));
  };

  // Handle confidence threshold change
  const handleThresholdChange = (_event: Event, newValue: number | number[]) => {
    setConfidenceThreshold(Array.isArray(newValue) ? newValue[0] : newValue);
  };

  // Handle component toggle
  const handleToggle = (event: React.ChangeEvent<HTMLInputElement>) => {
    onToggle(event.target.checked);
  };

  // Handle model selection for detailed view
  const handleModelSelect = (modelId: string) => {
    setSelectedModel(selectedModel === modelId ? null : modelId);
  };

  return (
    <Paper sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="subtitle1" fontWeight="bold">
          ML Confirmation Filter
        </Typography>
        <FormControlLabel
          control={<Switch checked={isEnabled} onChange={handleToggle} />}
          label="Enable ML Models"
          sx={{ m: 0 }}
        />
      </Box>
      
      {isEnabled ? (
        <>
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 3, alignItems: 'center' }}>
              <CircularProgress size={24} sx={{ mr: 1 }} />
              <Typography variant="body2">Processing market data...</Typography>
            </Box>
          ) : (
            <>
              {/* Result Summary */}
              {result && (
                <Card variant="outlined" sx={{ mb: 2 }}>
                  <CardContent>
                    <Typography variant="body2" fontWeight="medium" gutterBottom>
                      Signal Evaluation: {signalType?.toUpperCase() || 'None'}
                    </Typography>
                    
                    <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                      {result.confirmed ? (
                        <CheckCircleIcon color="success" sx={{ mr: 1 }} />
                      ) : (
                        <CancelIcon color="error" sx={{ mr: 1 }} />
                      )}
                      
                      <Typography variant={result.confirmed ? 'body1' : 'body2'} fontWeight={result.confirmed ? 'bold' : 'normal'}>
                        {result.confirmed
                          ? `Signal confirmed with ${result.confidence.toFixed(1)}% confidence`
                          : `Signal not confirmed (${result.confidence.toFixed(1)}% confidence)`}
                      </Typography>
                    </Box>
                    
                    <Divider sx={{ my: 1.5 }} />
                    
                    <Typography variant="caption" color="text.secondary" display="block">
                      Insights:
                    </Typography>
                    
                    <List dense disablePadding>
                      {result.insights.map((insight, index) => (
                        <ListItem key={index} disableGutters>
                          <ListItemIcon sx={{ minWidth: 30 }}>
                            <InfoIcon color="primary" fontSize="small" />
                          </ListItemIcon>
                          <ListItemText
                            primary={insight}
                            primaryTypographyProps={{ variant: 'body2' }}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </CardContent>
                </Card>
              )}
              
              {/* Confidence Threshold Slider */}
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" display="flex" justifyContent="space-between">
                  <span>Confidence Threshold: {confidenceThreshold}%</span>
                  <span>
                    <Tooltip title="Adjust how confident ML models must be to confirm a signal">
                      <InfoIcon fontSize="small" color="action" />
                    </Tooltip>
                  </span>
                </Typography>
                <Slider
                  value={confidenceThreshold}
                  onChange={handleThresholdChange}
                  aria-label="Confidence Threshold"
                  valueLabelDisplay="auto"
                />
              </Box>
              
              {/* Model List */}
              <Box>
                <Typography variant="body2" sx={{ mb: 1, display: 'flex', justifyContent: 'space-between' }}>
                  <span>ML Models ({models.filter(m => m.enabled).length}/{models.length} enabled)</span>
                  <IconButton 
                    size="small" 
                    onClick={() => setShowDetails(!showDetails)}
                    color={showDetails ? "primary" : "default"}
                  >
                    <TuneIcon fontSize="small" />
                  </IconButton>
                </Typography>
                
                <Box sx={{ maxHeight: '200px', overflowY: 'auto' }}>
                  <Grid container spacing={1}>
                    {models.map((model) => {
                      const prediction = predictions.find(p => p.modelId === model.id);
                      
                      return (
                        <Grid item xs={12} key={model.id}>
                          <Card 
                            variant="outlined"
                            sx={{
                              opacity: model.enabled ? 1 : 0.6,
                              mb: 0.5
                            }}
                          >
                            <CardContent sx={{ p: 1, '&:last-child': { pb: 1 } }}>
                              <Grid container alignItems="center">
                                <Grid item xs>
                                  <Typography variant="body2" fontWeight="medium">
                                    {model.name}
                                  </Typography>
                                </Grid>
                                
                                <Grid item>
                                  <Switch
                                    checked={model.enabled}
                                    onChange={() => handleToggleModel(model.id)}
                                    size="small"
                                  />
                                </Grid>
                              </Grid>
                              
                              {showDetails && (
                                <>
                                  <Divider sx={{ my: 1 }} />
                                  
                                  <Grid container spacing={1} alignItems="center">
                                    <Grid item xs={6}>
                                      <Typography variant="caption" color="text.secondary">
                                        Accuracy: {(model.accuracy * 100).toFixed(0)}%
                                      </Typography>
                                    </Grid>
                                    
                                    <Grid item xs={6}>
                                      <Typography variant="caption" color="text.secondary">
                                        Version: {model.version}
                                      </Typography>
                                    </Grid>
                                  </Grid>
                                  
                                  {model.enabled && prediction && (
                                    <>
                                      <Box sx={{ mt: 1 }}>
                                        <Typography variant="caption" color="text.secondary">
                                          Prediction: 
                                        </Typography>
                                        <Chip
                                          size="small"
                                          label={`${prediction.prediction} (${(prediction.confidence * 100).toFixed(0)}%)`}
                                          color={
                                            prediction.prediction === 'buy' ? 'success' :
                                            prediction.prediction === 'sell' ? 'error' : 'default'
                                          }
                                          sx={{ ml: 1 }}
                                        />
                                      </Box>
                                      
                                      <IconButton 
                                        size="small" 
                                        onClick={() => handleModelSelect(model.id)}
                                        sx={{ mt: 0.5 }}
                                      >
                                        <PsychologyIcon fontSize="small" />
                                      </IconButton>
                                      
                                      {selectedModel === model.id && prediction.explanation && (
                                        <Box sx={{ mt: 1, backgroundColor: 'action.hover', p: 1, borderRadius: 1 }}>
                                          <Typography variant="caption" color="text.secondary">
                                            Model reasoning:
                                          </Typography>
                                          <List dense disablePadding>
                                            {prediction.explanation.map((exp, idx) => (
                                              <ListItem key={idx} disableGutters>
                                                <Typography variant="caption">• {exp}</Typography>
                                              </ListItem>
                                            ))}
                                          </List>
                                        </Box>
                                      )}
                                    </>
                                  )}
                                </>
                              )}
                            </CardContent>
                          </Card>
                        </Grid>
                      );
                    })}
                  </Grid>
                </Box>
              </Box>
            </>
          )}
        </>
      ) : (
        <Alert severity="info" sx={{ mt: 2 }}>
          ML confirmation is currently disabled. Enable it to validate trading signals with machine learning models.
        </Alert>
      )}
    </Paper>
  );
}
