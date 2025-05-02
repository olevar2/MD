/**
 * AdaptiveStrategySettings Component - Integrates ML insights with strategy parameters
 */
import { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Typography,
  Divider,
  FormControlLabel,
  Switch,
  Tabs,
  Tab,
  Grid,
  Button,
  CircularProgress,
  Alert,
  TextField,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip
} from '@mui/material';
import ParameterConfigurator from './ParameterConfigurator';
import { Strategy, StrategyParameter, MarketRegime } from '@/types/strategy';
import { MLConfirmationResult } from '../ml/MLConfirmationFilter';

interface AdaptiveStrategySettingsProps {
  strategy: Strategy;
  mlInsights?: MLConfirmationResult | null;
  onParameterChange: (parameterKey: string, value: any, regime?: MarketRegime) => void;
  onRegimeParameterChange?: (regime: MarketRegime, parameterKey: string, value: any) => void;
  onAdaptiveToggle: (enabled: boolean) => void;
}

// User-friendly names for market regimes
const regimeLabels: Record<MarketRegime, string> = {
  [MarketRegime.TRENDING]: 'Trending Market',
  [MarketRegime.RANGING]: 'Range-bound Market',
  [MarketRegime.VOLATILE]: 'Volatile Market',
  [MarketRegime.BREAKOUT]: 'Breakout Market',
};

// Color schemes for different regimes
const regimeColors: Record<MarketRegime, string> = {
  [MarketRegime.TRENDING]: 'primary',
  [MarketRegime.RANGING]: 'success',
  [MarketRegime.VOLATILE]: 'error',
  [MarketRegime.BREAKOUT]: 'warning',
};

export default function AdaptiveStrategySettings({
  strategy,
  mlInsights,
  onParameterChange,
  onRegimeParameterChange,
  onAdaptiveToggle
}: AdaptiveStrategySettingsProps) {
  const [activeTab, setActiveTab] = useState<MarketRegime>(MarketRegime.TRENDING);
  const [adaptiveEnabled, setAdaptiveEnabled] = useState<boolean>(!!strategy.regimeParameters);
  const [currentRegime, setCurrentRegime] = useState<MarketRegime | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [recommendations, setRecommendations] = useState<Record<string, any> | null>(null);
  const [performanceData, setPerformanceData] = useState<any>(null);
  
  // Detect market regime on component mount and when ML insights change
  useEffect(() => {
    if (adaptiveEnabled) {
      detectMarketRegime();
      
      // If ML insights are available, use them to generate parameter recommendations
      if (mlInsights) {
        generateParameterRecommendations();
      }
    }
  }, [adaptiveEnabled, mlInsights]);
  
  // Simulate market regime detection
  const detectMarketRegime = async () => {
    setLoading(true);
    try {
      // This would call an API in a real implementation
      // For now, we'll simulate a delay and random regime
      await new Promise(resolve => setTimeout(resolve, 800));
      
      // Simulate detection based on a random choice
      // In a real system, this would use actual market data and ML models
      const regimes = Object.values(MarketRegime);
      const detectedRegime = regimes[Math.floor(Math.random() * regimes.length)];
      
      setCurrentRegime(detectedRegime);
      
      // If we have a new regime and it's different from the active tab,
      // we might want to switch to it automatically
      if (detectedRegime !== activeTab) {
        setActiveTab(detectedRegime);
      }
      
      // Simulate fetching performance data for this strategy in the detected regime
      fetchRegimePerformance(detectedRegime);
      
    } catch (error) {
      console.error('Error detecting market regime:', error);
    } finally {
      setLoading(false);
    }
  };
  
  // Generate parameter recommendations based on ML insights
  const generateParameterRecommendations = () => {
    if (!mlInsights || !currentRegime) return;
    
    // This would be more sophisticated in a real implementation
    // Here we'll just make mock recommendations based on the ML confidence
    
    let recommendedParams: Record<string, any> = {};
    
    // Get default parameters for the strategy
    const defaultParams = { ...strategy.parameters };
    
    // Adjust parameters based on ML insights and current regime
    if (currentRegime === MarketRegime.VOLATILE) {
      // For volatile markets, we might want to be more conservative
      recommendedParams = {
        stopLoss: defaultParams.stopLoss * 1.5,  // Wider stop loss
        takeProfit: defaultParams.takeProfit * 0.8,  // More conservative take profit
        trailingStop: true,  // Enable trailing stop
      };
    } else if (currentRegime === MarketRegime.TRENDING) {
      // For trending markets, optimize for trend following
      recommendedParams = {
        stopLoss: defaultParams.stopLoss * 1.2,
        takeProfit: defaultParams.takeProfit * 1.3,  // More aggressive take profit
        maFast: Math.max(5, defaultParams.maFast * 0.8),  // Faster MAs
        maSlow: Math.max(15, defaultParams.maSlow * 0.8),
      };
    } else if (currentRegime === MarketRegime.RANGING) {
      // For ranging markets, optimize for mean reversion
      recommendedParams = {
        stopLoss: defaultParams.stopLoss * 0.9,
        takeProfit: defaultParams.takeProfit * 0.9,
        oversold: Math.min(30, defaultParams.oversold + 5),
        overbought: Math.max(70, defaultParams.overbought - 5),
      };
    } else if (currentRegime === MarketRegime.BREAKOUT) {
      // For breakout markets, optimize for quick entries and exits
      recommendedParams = {
        stopLoss: defaultParams.stopLoss * 0.8,
        takeProfit: defaultParams.takeProfit * 1.5,
        breakoutThreshold: defaultParams.breakoutThreshold * 1.2,
      };
    }
    
    // If ML model has high confidence, make more aggressive adjustments
    if (mlInsights.confidence > 80) {
      // Amplify the changes for high confidence
      Object.keys(recommendedParams).forEach(key => {
        if (typeof recommendedParams[key] === 'number') {
          const diff = recommendedParams[key] - (defaultParams[key] || 0);
          recommendedParams[key] = defaultParams[key] + (diff * 1.2);
        }
      });
    }
    
    setRecommendations(recommendedParams);
  };
  
  // Simulate fetching performance data for the strategy in different regimes
  const fetchRegimePerformance = async (regime: MarketRegime) => {
    // This would be an API call in a real implementation
    // For now, we'll generate some mock data
    
    // Random but realistic performance metrics
    const generateMetrics = () => {
      const winRate = 0.4 + Math.random() * 0.3;  // 40-70%
      const profitFactor = 1 + Math.random();     // 1.0-2.0
      const avgWin = 20 + Math.random() * 30;     // 20-50 pips
      const avgLoss = 10 + Math.random() * 20;    // 10-30 pips
      const totalTrades = 50 + Math.floor(Math.random() * 100);  // 50-150 trades
      
      return { winRate, profitFactor, avgWin, avgLoss, totalTrades };
    };
    
    // Generate data with a bias for the strategy's strength in each regime
    // In a real system, this would be based on actual backtest results
    const baseMetrics = generateMetrics();
    
    // Adjust metrics based on regime and strategy type
    const adjustedMetrics = { ...baseMetrics };
    
    if (strategy.type === 'adaptive_ma') {
      // Adaptive MA might perform better in trending markets
      if (regime === MarketRegime.TRENDING) {
        adjustedMetrics.winRate += 0.1;
        adjustedMetrics.profitFactor += 0.5;
      }
    } else if (strategy.type === 'elliott_wave') {
      // Elliott Wave might perform better in volatile markets
      if (regime === MarketRegime.VOLATILE) {
        adjustedMetrics.winRate += 0.08;
        adjustedMetrics.profitFactor += 0.4;
      }
    }
    
    setPerformanceData({
      metrics: adjustedMetrics,
      regime
    });
  };
  
  // Handle tab changes
  const handleTabChange = (_event: React.SyntheticEvent, newValue: MarketRegime) => {
    setActiveTab(newValue);
  };
  
  // Handle adaptive mode toggle
  const handleAdaptiveToggle = (event: React.ChangeEvent<HTMLInputElement>) => {
    const isEnabled = event.target.checked;
    setAdaptiveEnabled(isEnabled);
    onAdaptiveToggle(isEnabled);
    
    if (isEnabled) {
      detectMarketRegime();
    } else {
      setCurrentRegime(null);
    }
  };
  
  // Handle parameter change for a specific regime
  const handleRegimeParameterChange = (paramKey: string, value: any) => {
    if (onRegimeParameterChange) {
      onRegimeParameterChange(activeTab, paramKey, value);
    }
  };
  
  // Apply ML recommendations to the current regime
  const applyRecommendations = () => {
    if (!recommendations || !onRegimeParameterChange) return;
    
    Object.entries(recommendations).forEach(([key, value]) => {
      onRegimeParameterChange(activeTab, key, value);
    });
    
    // Clear recommendations after applying
    setRecommendations(null);
  };

  return (
    <Card>
      <CardHeader 
        title="Adaptive Strategy Settings" 
        titleTypographyProps={{ variant: 'h6' }}
        action={
          <FormControlLabel
            control={
              <Switch 
                checked={adaptiveEnabled} 
                onChange={handleAdaptiveToggle}
                color="primary"
              />
            }
            label="Enable Adaptive Mode"
          />
        }
      />
      
      <Divider />
      
      {adaptiveEnabled ? (
        <>
          {currentRegime && (
            <Box sx={{ p: 2, bgcolor: 'background.paper' }}>
              <Typography variant="body2" gutterBottom>
                Current Detected Market Regime:
              </Typography>
              <Chip 
                label={regimeLabels[currentRegime] || currentRegime}
                color={regimeColors[currentRegime] as any || "default"}
                sx={{ fontWeight: 'bold' }}
              />
              
              {mlInsights && (
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  ML Confidence: {mlInsights.confidence.toFixed(1)}%
                </Typography>
              )}
            </Box>
          )}
          
          <Box>
            <Tabs
              value={activeTab}
              onChange={handleTabChange}
              variant="scrollable"
              scrollButtons="auto"
            >
              {Object.values(MarketRegime).map((regime) => (
                <Tab
                  key={regime}
                  value={regime}
                  label={regimeLabels[regime] || regime}
                  icon={currentRegime === regime ? <Chip size="small" label="Current" /> : undefined}
                  iconPosition="end"
                />
              ))}
            </Tabs>
            
            <Box sx={{ p: 2 }}>
              {loading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                  <CircularProgress size={24} sx={{ mr: 1 }} />
                  <Typography>Analyzing market conditions...</Typography>
                </Box>
              ) : (
                <>
                  <Typography variant="body2" gutterBottom>
                    Configure strategy parameters for {regimeLabels[activeTab] || activeTab} conditions:
                  </Typography>
                  
                  {recommendations && (
                    <Alert 
                      severity="info" 
                      sx={{ mb: 2 }}
                      action={
                        <Button 
                          color="inherit" 
                          size="small"
                          onClick={applyRecommendations}
                        >
                          Apply
                        </Button>
                      }
                    >
                      ML model recommendations available for {regimeLabels[activeTab]}
                    </Alert>
                  )}
                  
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={8}>
                      {/* Parameter configuration */}
                      {strategy.parameterTemplates.map((param) => {
                        // Get the current value from the regime-specific parameters, or fall back to default
                        const regimeParams = strategy.regimeParameters?.[activeTab] || {};
                        const currentValue = regimeParams[param.key] !== undefined ? 
                          regimeParams[param.key] : strategy.parameters[param.key];
                          
                        // Check if there's a recommendation for this parameter
                        const hasRecommendation = recommendations && recommendations[param.key] !== undefined;
                        const recommendedValue = hasRecommendation ? recommendations[param.key] : undefined;
                        
                        return (
                          <Box key={param.key} sx={{ mb: 2, position: 'relative' }}>
                            <ParameterConfigurator 
                              parameter={param}
                              value={currentValue}
                              onChange={(value) => handleRegimeParameterChange(param.key, value)}
                            />
                            
                            {hasRecommendation && (
                              <Box sx={{ mt: -1, mb: 1 }}>
                                <Typography variant="caption" color="primary">
                                  Recommended: {recommendedValue}
                                </Typography>
                              </Box>
                            )}
                          </Box>
                        );
                      })}
                    </Grid>
                    
                    <Grid item xs={12} md={4}>
                      {performanceData && performanceData.regime === activeTab && (
                        <Card variant="outlined" sx={{ mb: 2 }}>
                          <CardHeader 
                            title="Performance in this Regime" 
                            titleTypographyProps={{ variant: 'subtitle2' }}
                            sx={{ pb: 0 }}
                          />
                          <CardContent sx={{ pt: 1 }}>
                            <TableContainer>
                              <Table size="small">
                                <TableBody>
                                  <TableRow>
                                    <TableCell>Win Rate</TableCell>
                                    <TableCell align="right">
                                      {(performanceData.metrics.winRate * 100).toFixed(1)}%
                                    </TableCell>
                                  </TableRow>
                                  <TableRow>
                                    <TableCell>Profit Factor</TableCell>
                                    <TableCell align="right">
                                      {performanceData.metrics.profitFactor.toFixed(2)}
                                    </TableCell>
                                  </TableRow>
                                  <TableRow>
                                    <TableCell>Avg Win</TableCell>
                                    <TableCell align="right">
                                      {performanceData.metrics.avgWin.toFixed(1)} pips
                                    </TableCell>
                                  </TableRow>
                                  <TableRow>
                                    <TableCell>Avg Loss</TableCell>
                                    <TableCell align="right">
                                      {performanceData.metrics.avgLoss.toFixed(1)} pips
                                    </TableCell>
                                  </TableRow>
                                  <TableRow>
                                    <TableCell>Total Trades</TableCell>
                                    <TableCell align="right">
                                      {performanceData.metrics.totalTrades}
                                    </TableCell>
                                  </TableRow>
                                </TableBody>
                              </Table>
                            </TableContainer>
                          </CardContent>
                        </Card>
                      )}
                      
                      <Typography variant="caption" color="text.secondary">
                        Parameters configured for this market regime will be automatically applied when the system 
                        detects these market conditions.
                      </Typography>
                    </Grid>
                  </Grid>
                </>
              )}
            </Box>
          </Box>
        </>
      ) : (
        <Box sx={{ p: 2 }}>
          <Alert severity="info">
            Adaptive mode allows your strategy to automatically adjust parameters based on market conditions.
            Enable it to configure parameters for different market regimes.
          </Alert>
        </Box>
      )}
    </Card>
  );
}
