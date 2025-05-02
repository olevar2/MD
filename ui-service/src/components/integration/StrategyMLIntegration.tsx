/**
 * StrategyMLIntegration Component - Connects ML models with strategy execution
 * 
 * This component serves as the integration layer between the ML models and trading strategies,
 * providing a unified interface for strategy execution with ML-enhanced decisions.
 */
import { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  CircularProgress,
  Divider,
  Grid,
  Chip,
  Alert,
  AlertTitle,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  IconButton,
  Collapse
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import WarningIcon from '@mui/icons-material/Warning';
import InfoIcon from '@mui/icons-material/Info';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import TuneIcon from '@mui/icons-material/Tune';
import TimelineIcon from '@mui/icons-material/Timeline';
import PsychologyIcon from '@mui/icons-material/Psychology';

import MLConfirmationFilter, { MLConfirmationResult } from '../ml/MLConfirmationFilter';
import { Strategy, MarketRegime } from '@/types/strategy';
import AdaptiveStrategySettings from '../strategy/AdaptiveStrategySettings';

interface StrategyMLIntegrationProps {
  strategy: Strategy;
  symbol: string;
  timeframe: string;
  isActive: boolean;
  onActivate: () => void;
  onDeactivate: () => void;
  onParameterChange: (parameterKey: string, value: any) => void;
  onRegimeParameterChange: (regime: MarketRegime, parameterKey: string, value: any) => void;
  onAdaptiveToggle: (enabled: boolean) => void;
}

export default function StrategyMLIntegration({
  strategy,
  symbol,
  timeframe,
  isActive,
  onActivate,
  onDeactivate,
  onParameterChange,
  onRegimeParameterChange,
  onAdaptiveToggle
}: StrategyMLIntegrationProps) {
  const [mlEnabled, setMlEnabled] = useState<boolean>(true);
  const [mlConfirmationResult, setMlConfirmationResult] = useState<MLConfirmationResult | null>(null);
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    mlInsights: true,
    adaptiveSettings: false
  });
  const [lastSignal, setLastSignal] = useState<'buy' | 'sell' | null>(null);
  const [statusMessages, setStatusMessages] = useState<{message: string, type: 'info' | 'warning' | 'error' | 'success'}[]>([]);
  const [executionStatus, setExecutionStatus] = useState<'standby' | 'analyzing' | 'executing' | 'completed'>('standby');

  // Simulate signal detection on component mount
  useEffect(() => {
    if (isActive) {
      simulateSignalDetection();
    }
  }, [isActive]);
  
  // Clear status messages periodically
  useEffect(() => {
    const timer = setTimeout(() => {
      if (statusMessages.length > 5) {
        setStatusMessages(prev => prev.slice(-5));
      }
    }, 60000);
    
    return () => clearTimeout(timer);
  }, [statusMessages]);
  
  // Handle ML confirmation results
  const handleMlConfirmation = (result: MLConfirmationResult) => {
    setMlConfirmationResult(result);
    
    // Add status message based on confirmation result
    if (lastSignal) {
      if (result.confirmed) {
        addStatusMessage(`ML models confirmed ${lastSignal} signal with ${result.confidence.toFixed(1)}% confidence`, 'success');
        executeSignal(lastSignal, result);
      } else {
        addStatusMessage(`ML models rejected ${lastSignal} signal (${result.confidence.toFixed(1)}% confidence)`, 'warning');
        setExecutionStatus('standby');
      }
    }
  };
  
  // Toggle ML enablement
  const handleMlToggle = (enabled: boolean) => {
    setMlEnabled(enabled);
    addStatusMessage(`ML confirmation ${enabled ? 'enabled' : 'disabled'}`, 'info');
  };
  
  // Toggle expanded sections
  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };
  
  // Add status message
  const addStatusMessage = (message: string, type: 'info' | 'warning' | 'error' | 'success') => {
    setStatusMessages(prev => [...prev, { message, type }]);
  };
  
  // Simulate signal detection (in real implementation, this would come from strategy)
  const simulateSignalDetection = () => {
    const timer = setTimeout(() => {
      // Only generate signals when the strategy is active
      if (!isActive) return;
      
      setExecutionStatus('analyzing');
      addStatusMessage('Analyzing market conditions...', 'info');
      
      // Simulate analysis time
      setTimeout(() => {
        // 30% chance of a signal
        if (Math.random() > 0.7) {
          const signal = Math.random() > 0.5 ? 'buy' : 'sell';
          setLastSignal(signal as 'buy' | 'sell');
          addStatusMessage(`Strategy detected ${signal} signal`, 'info');
          
          // If ML is enabled, the confirmation handler will process the signal
          // Otherwise, execute directly
          if (!mlEnabled) {
            executeSignal(signal as 'buy' | 'sell', null);
          }
        } else {
          setLastSignal(null);
          setExecutionStatus('standby');
          addStatusMessage('No trading signals detected', 'info');
        }
        
        // Set up next signal check with random interval (3-8 minutes)
        const nextInterval = 3 * 60 * 1000 + Math.random() * 5 * 60 * 1000;
        simulateSignalDetection();
      }, 1500);
    }, 5000);
    
    return () => clearTimeout(timer);
  };
  
  // Simulate signal execution
  const executeSignal = (signal: 'buy' | 'sell', mlResult: MLConfirmationResult | null) => {
    setExecutionStatus('executing');
    addStatusMessage(`Executing ${signal} signal...`, 'info');
    
    // Simulate execution time
    setTimeout(() => {
      setExecutionStatus('completed');
      addStatusMessage(`${signal.toUpperCase()} order executed successfully`, 'success');
      
      // After some time, go back to standby
      setTimeout(() => {
        setExecutionStatus('standby');
      }, 3000);
    }, 2000);
  };
  
  // Status indicator color mapping
  const getStatusColor = () => {
    switch (executionStatus) {
      case 'analyzing': return 'info';
      case 'executing': return 'warning';
      case 'completed': return 'success';
      default: return 'default';
    }
  };
  
  // Status indicator text mapping
  const getStatusText = () => {
    switch (executionStatus) {
      case 'analyzing': return 'Analyzing Market';
      case 'executing': return 'Executing Trade';
      case 'completed': return 'Trade Completed';
      default: return 'Standing By';
    }
  };

  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">
            Strategy + ML Integration
          </Typography>
          
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Chip 
              label={getStatusText()}
              color={getStatusColor() as any}
              sx={{ mr: 2 }}
            />
            
            <Button
              variant="contained"
              color={isActive ? "error" : "success"}
              startIcon={isActive ? <PauseIcon /> : <PlayArrowIcon />}
              onClick={isActive ? onDeactivate : onActivate}
            >
              {isActive ? "Deactivate" : "Activate"} Strategy
            </Button>
          </Box>
        </Box>
        
        <Grid container spacing={2}>
          <Grid item xs={12} lg={6}>
            <Card variant="outlined">
              <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <PsychologyIcon color="primary" sx={{ mr: 1 }} />
                  <Typography variant="subtitle1">ML Confirmation</Typography>
                </Box>
                <IconButton size="small" onClick={() => toggleSection('mlInsights')}>
                  {expandedSections.mlInsights ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                </IconButton>
              </Box>
              
              <Collapse in={expandedSections.mlInsights}>
                <CardContent sx={{ pt: 0 }}>
                  <MLConfirmationFilter
                    symbol={symbol}
                    timeframe={timeframe}
                    signalType={lastSignal}
                    isEnabled={mlEnabled}
                    onToggle={handleMlToggle}
                    onConfirmationResult={handleMlConfirmation}
                  />
                </CardContent>
              </Collapse>
            </Card>
          </Grid>
          
          <Grid item xs={12} lg={6}>
            <Card variant="outlined">
              <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <TuneIcon color="primary" sx={{ mr: 1 }} />
                  <Typography variant="subtitle1">Adaptive Strategy Settings</Typography>
                </Box>
                <IconButton size="small" onClick={() => toggleSection('adaptiveSettings')}>
                  {expandedSections.adaptiveSettings ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                </IconButton>
              </Box>
              
              <Collapse in={expandedSections.adaptiveSettings}>
                <CardContent sx={{ pt: 0 }}>
                  <AdaptiveStrategySettings
                    strategy={strategy}
                    mlInsights={mlConfirmationResult}
                    onParameterChange={onParameterChange}
                    onRegimeParameterChange={onRegimeParameterChange}
                    onAdaptiveToggle={onAdaptiveToggle}
                  />
                </CardContent>
              </Collapse>
            </Card>
          </Grid>
        </Grid>
        
        {/* Execution Status Log */}
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Execution Log
          </Typography>
          
          <Card variant="outlined">
            <Box sx={{ maxHeight: '200px', overflowY: 'auto', p: 1 }}>
              {statusMessages.length === 0 ? (
                <Typography variant="body2" color="text.secondary" sx={{ p: 1 }}>
                  No activity yet. Status messages will appear here.
                </Typography>
              ) : (
                <List dense disablePadding>
                  {statusMessages.map((status, index) => (
                    <ListItem key={index} disableGutters>
                      <ListItemIcon sx={{ minWidth: 36 }}>
                        {status.type === 'success' && <CheckCircleIcon color="success" />}
                        {status.type === 'error' && <ErrorIcon color="error" />}
                        {status.type === 'warning' && <WarningIcon color="warning" />}
                        {status.type === 'info' && <InfoIcon color="info" />}
                      </ListItemIcon>
                      <ListItemText
                        primary={status.message}
                        primaryTypographyProps={{ variant: 'body2' }}
                        secondary={new Date().toLocaleTimeString()}
                        secondaryTypographyProps={{ variant: 'caption' }}
                      />
                    </ListItem>
                  ))}
                </List>
              )}
            </Box>
          </Card>
        </Box>
        
        {/* Current Strategy Status */}
        {isActive ? (
          <Alert severity="success" sx={{ mt: 2 }}>
            <AlertTitle>Strategy Active</AlertTitle>
            The strategy is currently active and {mlEnabled ? 'using ML for confirmation' : 'executing without ML confirmation'}.
          </Alert>
        ) : (
          <Alert severity="info" sx={{ mt: 2 }}>
            <AlertTitle>Strategy Inactive</AlertTitle>
            Activate the strategy to begin trading with ML confirmation.
          </Alert>
        )}
      </CardContent>
    </Card>
  );
}
