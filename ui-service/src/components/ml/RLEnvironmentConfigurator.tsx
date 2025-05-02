import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  Grid, 
  TextField, 
  Button,
  Card,
  CardContent,
  Alert,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Divider,
  Switch,
  FormControlLabel,
  IconButton,
  Tooltip,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  FormGroup,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  ExpandMore as ExpandMoreIcon,
  InfoOutlined as InfoIcon,
  Save as SaveIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import { mlService } from '../../services/mlService';

interface TimeframeConfig {
  timeframe: string;
  lookbackPeriods: number;
}

interface FeatureConfig {
  name: string;
  category: string;
  enabled: boolean;
  parameters?: Record<string, any>;
}

interface RewardComponent {
  name: string;
  description: string;
  enabled: boolean;
  weight: number;
  parameters?: Record<string, any>;
}

interface CurriculumLevel {
  name: string;
  description: string;
  episodes: number;
  difficulty: number;
  parameters: Record<string, any>;
}

interface EnvironmentConfig {
  id?: string;
  name: string;
  description: string;
  symbol: string;
  baseDataTimeframe: string;
  episodeLength: number;
  initialBalance: number;
  maxDrawdown: number;
  timeframes: TimeframeConfig[];
  features: FeatureConfig[];
  positionSizing: {
    type: string;
    parameters: Record<string, any>;
  };
  rewardComponents: RewardComponent[];
  tradingFees: {
    commissionRate: number;
    slippageModel: string;
    spreadModel: string;
  };
  curriculumLearning: {
    enabled: boolean;
    levels: CurriculumLevel[];
  };
}

interface RLEnvironmentConfiguratorProps {
  onSave?: (config: EnvironmentConfig) => void;
  existingConfigId?: string;
}

const AVAILABLE_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'];

const AVAILABLE_FEATURES = [
  { name: 'price_open', category: 'price', description: 'Opening price of the candle' },
  { name: 'price_high', category: 'price', description: 'Highest price in the candle' },
  { name: 'price_low', category: 'price', description: 'Lowest price in the candle' },
  { name: 'price_close', category: 'price', description: 'Closing price of the candle' },
  { name: 'volume', category: 'price', description: 'Trading volume' },
  { name: 'bid_ask_spread', category: 'orderbook', description: 'Spread between bid and ask prices' },
  { name: 'order_book_imbalance', category: 'orderbook', description: 'Imbalance between buy and sell orders' },
  { name: 'market_depth', category: 'orderbook', description: 'Depth of the order book' },
  { name: 'sma_5', category: 'technical', description: 'Simple Moving Average (5 periods)' },
  { name: 'sma_10', category: 'technical', description: 'Simple Moving Average (10 periods)' },
  { name: 'sma_20', category: 'technical', description: 'Simple Moving Average (20 periods)' },
  { name: 'sma_50', category: 'technical', description: 'Simple Moving Average (50 periods)' },
  { name: 'sma_200', category: 'technical', description: 'Simple Moving Average (200 periods)' },
  { name: 'ema_5', category: 'technical', description: 'Exponential Moving Average (5 periods)' },
  { name: 'ema_10', category: 'technical', description: 'Exponential Moving Average (10 periods)' },
  { name: 'ema_20', category: 'technical', description: 'Exponential Moving Average (20 periods)' },
  { name: 'ema_50', category: 'technical', description: 'Exponential Moving Average (50 periods)' },
  { name: 'ema_200', category: 'technical', description: 'Exponential Moving Average (200 periods)' },
  { name: 'rsi_14', category: 'technical', description: 'Relative Strength Index (14 periods)' },
  { name: 'macd', category: 'technical', description: 'Moving Average Convergence Divergence' },
  { name: 'bollinger_bands', category: 'technical', description: 'Bollinger Bands' },
  { name: 'atr_14', category: 'technical', description: 'Average True Range (14 periods)' },
  { name: 'stochastic_k_14', category: 'technical', description: 'Stochastic Oscillator %K (14 periods)' },
  { name: 'stochastic_d_3', category: 'technical', description: 'Stochastic Oscillator %D (3 periods)' },
  { name: 'obv', category: 'technical', description: 'On-Balance Volume' },
  { name: 'adx_14', category: 'technical', description: 'Average Directional Index (14 periods)' },
  { name: 'market_regime', category: 'market_state', description: 'Current market regime (trending, ranging, etc.)' },
  { name: 'volatility_index', category: 'market_state', description: 'Volatility index' },
  { name: 'current_hour', category: 'market_state', description: 'Current hour of the day' },
  { name: 'current_day', category: 'market_state', description: 'Current day of the week' },
  { name: 'economic_calendar', category: 'fundamental', description: 'Economic calendar events' },
  { name: 'news_sentiment', category: 'sentiment', description: 'News sentiment score' },
  { name: 'social_sentiment', category: 'sentiment', description: 'Social media sentiment score' },
  { name: 'analyst_ratings', category: 'sentiment', description: 'Analyst ratings sentiment' },
  { name: 'account_balance', category: 'account', description: 'Current account balance' },
  { name: 'open_positions', category: 'account', description: 'Number of open positions' },
  { name: 'unrealized_pnl', category: 'account', description: 'Unrealized profit/loss' },
  { name: 'max_drawdown', category: 'account', description: 'Maximum drawdown so far' }
];

const AVAILABLE_REWARD_COMPONENTS = [
  { 
    name: 'pnl', 
    description: 'Raw profit and loss from trades',
    parameters: { scaling: 1.0 }
  },
  { 
    name: 'risk_adjusted_return', 
    description: 'Return adjusted by the risk taken',
    parameters: { risk_free_rate: 0.0, risk_aversion: 1.0 }
  },
  { 
    name: 'trade_duration', 
    description: 'Penalty for holding trades too long',
    parameters: { penalty_factor: 0.001 }
  },
  { 
    name: 'drawdown_penalty', 
    description: 'Penalty for large drawdowns',
    parameters: { drawdown_threshold: 0.05, penalty_factor: 1.0 }
  },
  { 
    name: 'trade_frequency', 
    description: 'Penalty for excessive trading frequency',
    parameters: { target_trades_per_day: 5, penalty_factor: 0.1 }
  },
  { 
    name: 'sharpe_ratio', 
    description: 'Reward for maintaining good Sharpe ratio',
    parameters: { window_size: 100 }
  },
  { 
    name: 'exploration_bonus', 
    description: 'Bonus for exploring uncommon states/actions',
    parameters: { bonus_factor: 0.01 }
  },
  { 
    name: 'consistency', 
    description: 'Reward for consistent returns',
    parameters: { window_size: 20, consistency_factor: 0.5 }
  }
];

const POSITION_SIZING_STRATEGIES = [
  { 
    name: 'fixed', 
    description: 'Fixed position size for all trades',
    parameters: { size: 1.0 }
  },
  { 
    name: 'percentage', 
    description: 'Percentage of account balance',
    parameters: { percentage: 2.0 }
  },
  { 
    name: 'kelly', 
    description: 'Kelly criterion for optimal position sizing',
    parameters: { fraction: 0.5, window_size: 100 }
  },
  { 
    name: 'risk_parity', 
    description: 'Risk parity approach to position sizing',
    parameters: { target_risk: 0.01, volatility_window: 20 }
  },
  { 
    name: 'volatility_adjusted', 
    description: 'Adjust position size based on market volatility',
    parameters: { base_size: 1.0, volatility_window: 20, inverse_volatility: true }
  },
  { 
    name: 'dynamic_rl', 
    description: 'Let the RL agent decide position size',
    parameters: { min_size: 0.1, max_size: 5.0, size_increment: 0.1 }
  }
];

/**
 * RL Environment Configurator Component
 * 
 * This component provides a comprehensive interface for creating and configuring
 * reinforcement learning environments for forex trading. It allows detailed
 * configuration of observation spaces, reward functions, and training parameters.
 */
const RLEnvironmentConfigurator: React.FC<RLEnvironmentConfiguratorProps> = ({
  onSave,
  existingConfigId
}) => {
  const [config, setConfig] = useState<EnvironmentConfig>({
    name: 'New Environment',
    description: 'RL environment for forex trading',
    symbol: 'EURUSD',
    baseDataTimeframe: '1m',
    episodeLength: 1440, // 1 day when base timeframe is 1m
    initialBalance: 10000,
    maxDrawdown: 0.1, // 10%
    timeframes: [
      { timeframe: '1m', lookbackPeriods: 60 },
      { timeframe: '5m', lookbackPeriods: 24 },
      { timeframe: '1h', lookbackPeriods: 24 }
    ],
    features: AVAILABLE_FEATURES.slice(0, 10).map(feature => ({
      ...feature,
      enabled: true
    })),
    positionSizing: {
      type: 'percentage',
      parameters: { percentage: 2.0 }
    },
    rewardComponents: AVAILABLE_REWARD_COMPONENTS.slice(0, 3).map(component => ({
      ...component,
      enabled: true,
      weight: component.name === 'pnl' ? 1.0 : 0.5
    })),
    tradingFees: {
      commissionRate: 0.001, // 0.1%
      slippageModel: 'proportional',
      spreadModel: 'variable'
    },
    curriculumLearning: {
      enabled: false,
      levels: [
        {
          name: 'Level 1 - Basics',
          description: 'Basic trading with simple patterns',
          episodes: 100,
          difficulty: 1,
          parameters: {
            volatility: 0.5,
            trend_strength: 0.5,
            noise_level: 0.2
          }
        },
        {
          name: 'Level 2 - Intermediate',
          description: 'Moderate complexity with various regimes',
          episodes: 200,
          difficulty: 2,
          parameters: {
            volatility: 1.0,
            trend_strength: 0.8,
            noise_level: 0.5
          }
        },
        {
          name: 'Level 3 - Advanced',
          description: 'Complex scenarios with realistic conditions',
          episodes: 300,
          difficulty: 3,
          parameters: {
            volatility: 1.5,
            trend_strength: 1.0,
            noise_level: 0.8
          }
        }
      ]
    }
  });
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [environmentSummary, setEnvironmentSummary] = useState<{
    observationDimensions: number;
    estimatedMemoryMb: number;
    activeFeatureCount: number;
  } | null>(null);

  // Load existing config if provided
  useEffect(() => {
    if (existingConfigId) {
      setLoading(true);
      mlService.getEnvironmentConfig(existingConfigId)
        .then(data => {
          setConfig(data);
          calculateEnvironmentSummary(data);
        })
        .catch(err => {
          console.error("Failed to load environment config:", err);
          setError("Failed to load configuration. Please try again.");
        })
        .finally(() => {
          setLoading(false);
        });
    } else {
      calculateEnvironmentSummary(config);
    }
  }, [existingConfigId]);

  // Recalculate environment summary when config changes
  const calculateEnvironmentSummary = (currentConfig: EnvironmentConfig) => {
    // Count active features across all timeframes
    const activeFeatures = currentConfig.features.filter(f => f.enabled).length;
    const totalTimeframes = currentConfig.timeframes.length;
    
    // Rough estimate of observation dimensions:
    // Each enabled feature gets multiplied by the number of lookback periods for each timeframe
    let totalDimensions = 0;
    currentConfig.timeframes.forEach(tf => {
      totalDimensions += activeFeatures * tf.lookbackPeriods;
    });
    
    // Also add account state features (roughly 5 dimensions)
    totalDimensions += 5;
    
    // Very rough estimation of memory required (in MB)
    // Assuming float32 (4 bytes) for each dimension and some overhead
    const estimatedMemoryBytes = totalDimensions * 4 * 1.5; // 50% overhead
    const estimatedMemoryMb = estimatedMemoryBytes / (1024 * 1024);
    
    setEnvironmentSummary({
      observationDimensions: totalDimensions,
      estimatedMemoryMb: estimatedMemoryMb,
      activeFeatureCount: activeFeatures * totalTimeframes
    });
  };

  useEffect(() => {
    calculateEnvironmentSummary(config);
  }, [
    config.timeframes, 
    config.features
  ]);

  // Basic info handlers
  const handleBasicInfoChange = (field: keyof EnvironmentConfig) => (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    setConfig({
      ...config,
      [field]: event.target.value
    });
  };

  // Timeframe handlers
  const handleAddTimeframe = () => {
    const availableTimeframes = AVAILABLE_TIMEFRAMES.filter(
      tf => !config.timeframes.some(configTf => configTf.timeframe === tf)
    );
    
    if (availableTimeframes.length > 0) {
      setConfig({
        ...config,
        timeframes: [
          ...config.timeframes,
          { timeframe: availableTimeframes[0], lookbackPeriods: 10 }
        ]
      });
    }
  };

  const handleRemoveTimeframe = (index: number) => {
    const newTimeframes = [...config.timeframes];
    newTimeframes.splice(index, 1);
    setConfig({
      ...config,
      timeframes: newTimeframes
    });
  };

  const handleTimeframeChange = (index: number, field: keyof TimeframeConfig) => (
    event: React.ChangeEvent<HTMLInputElement | { value: unknown }>
  ) => {
    const newTimeframes = [...config.timeframes];
    newTimeframes[index] = {
      ...newTimeframes[index],
      [field]: field === 'lookbackPeriods' ? Number(event.target.value) : event.target.value
    };
    setConfig({
      ...config,
      timeframes: newTimeframes
    });
  };
  
  const handleLookbackSliderChange = (index: number) => (
    event: Event, newValue: number | number[]
  ) => {
    const newTimeframes = [...config.timeframes];
    newTimeframes[index] = {
      ...newTimeframes[index],
      lookbackPeriods: newValue as number
    };
    setConfig({
      ...config,
      timeframes: newTimeframes
    });
  };

  // Feature handlers
  const handleFeatureToggle = (index: number) => (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const newFeatures = [...config.features];
    newFeatures[index] = {
      ...newFeatures[index],
      enabled: event.target.checked
    };
    setConfig({
      ...config,
      features: newFeatures
    });
  };

  // Position sizing handlers
  const handlePositionSizingTypeChange = (
    event: React.ChangeEvent<{ value: unknown }>
  ) => {
    const newType = event.target.value as string;
    const strategy = POSITION_SIZING_STRATEGIES.find(s => s.name === newType);
    
    if (strategy) {
      setConfig({
        ...config,
        positionSizing: {
          type: newType,
          parameters: { ...strategy.parameters }
        }
      });
    }
  };

  const handlePositionSizingParamChange = (param: string) => (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    setConfig({
      ...config,
      positionSizing: {
        ...config.positionSizing,
        parameters: {
          ...config.positionSizing.parameters,
          [param]: Number(event.target.value)
        }
      }
    });
  };

  // Reward component handlers
  const handleRewardComponentToggle = (index: number) => (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const newComponents = [...config.rewardComponents];
    newComponents[index] = {
      ...newComponents[index],
      enabled: event.target.checked
    };
    setConfig({
      ...config,
      rewardComponents: newComponents
    });
  };

  const handleRewardWeightChange = (index: number) => (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const newComponents = [...config.rewardComponents];
    newComponents[index] = {
      ...newComponents[index],
      weight: Number(event.target.value)
    };
    setConfig({
      ...config,
      rewardComponents: newComponents
    });
  };

  const handleRewardWeightSliderChange = (index: number) => (
    event: Event, newValue: number | number[]
  ) => {
    const newComponents = [...config.rewardComponents];
    newComponents[index] = {
      ...newComponents[index],
      weight: newValue as number
    };
    setConfig({
      ...config,
      rewardComponents: newComponents
    });
  };

  const handleRewardParamChange = (index: number, param: string) => (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const newComponents = [...config.rewardComponents];
    newComponents[index] = {
      ...newComponents[index],
      parameters: {
        ...newComponents[index].parameters,
        [param]: Number(event.target.value)
      }
    };
    setConfig({
      ...config,
      rewardComponents: newComponents
    });
  };

  const handleAddRewardComponent = () => {
    const unusedComponents = AVAILABLE_REWARD_COMPONENTS.filter(
      component => !config.rewardComponents.some(rc => rc.name === component.name)
    );
    
    if (unusedComponents.length > 0) {
      setConfig({
        ...config,
        rewardComponents: [
          ...config.rewardComponents,
          {
            ...unusedComponents[0],
            enabled: true,
            weight: 0.5
          }
        ]
      });
    }
  };

  const handleRemoveRewardComponent = (index: number) => {
    const newComponents = [...config.rewardComponents];
    newComponents.splice(index, 1);
    setConfig({
      ...config,
      rewardComponents: newComponents
    });
  };

  // Trading fees handlers
  const handleTradingFeesChange = (field: keyof typeof config.tradingFees) => (
    event: React.ChangeEvent<HTMLInputElement | { value: unknown }>
  ) => {
    setConfig({
      ...config,
      tradingFees: {
        ...config.tradingFees,
        [field]: field === 'commissionRate' 
          ? Number(event.target.value) 
          : event.target.value
      }
    });
  };

  // Curriculum learning handlers
  const handleCurriculumToggle = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    setConfig({
      ...config,
      curriculumLearning: {
        ...config.curriculumLearning,
        enabled: event.target.checked
      }
    });
  };

  const handleCurriculumLevelChange = (index: number, field: keyof CurriculumLevel) => (
    event: React.ChangeEvent<HTMLInputElement | { value: unknown }>
  ) => {
    const newLevels = [...config.curriculumLearning.levels];
    newLevels[index] = {
      ...newLevels[index],
      [field]: field === 'difficulty' || field === 'episodes'
        ? Number(event.target.value)
        : event.target.value
    };
    setConfig({
      ...config,
      curriculumLearning: {
        ...config.curriculumLearning,
        levels: newLevels
      }
    });
  };

  const handleCurriculumParamChange = (index: number, param: string) => (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const newLevels = [...config.curriculumLearning.levels];
    newLevels[index] = {
      ...newLevels[index],
      parameters: {
        ...newLevels[index].parameters,
        [param]: Number(event.target.value)
      }
    };
    setConfig({
      ...config,
      curriculumLearning: {
        ...config.curriculumLearning,
        levels: newLevels
      }
    });
  };

  const handleSave = () => {
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    // In a real implementation, this would send the config to your backend
    console.log("Saving configuration:", config);
    
    // Simulated backend call
    setTimeout(() => {
      setLoading(false);
      setSuccess("Environment configuration saved successfully!");
      
      if (onSave) {
        onSave(config);
      }
    }, 1000);
  };

  const getFeaturesByCategory = () => {
    const categories = Array.from(new Set(config.features.map(f => f.category)));
    
    return categories.map(category => ({
      category,
      features: config.features.filter(f => f.category === category)
    }));
  };

  if (loading && !config) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%' }}>
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h5" component="h2">
            RL Environment Configuration
          </Typography>
          <Button 
            variant="contained" 
            color="primary" 
            onClick={handleSave}
            disabled={loading}
            startIcon={<SaveIcon />}
          >
            Save Configuration
          </Button>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>
        )}

        {success && (
          <Alert severity="success" sx={{ mb: 2 }}>{success}</Alert>
        )}

        <Grid container spacing={3}>
          {/* Basic Settings */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" component="h3" gutterBottom>
                  Basic Settings
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={4}>
                    <TextField
                      label="Environment Name"
                      value={config.name}
                      onChange={handleBasicInfoChange('name')}
                      fullWidth
                      margin="normal"
                    />
                  </Grid>
                  <Grid item xs={12} md={8}>
                    <TextField
                      label="Description"
                      value={config.description}
                      onChange={handleBasicInfoChange('description')}
                      fullWidth
                      margin="normal"
                    />
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <FormControl fullWidth margin="normal">
                      <InputLabel id="symbol-select-label">Trading Symbol</InputLabel>
                      <Select
                        labelId="symbol-select-label"
                        value={config.symbol}
                        onChange={(e) => setConfig({...config, symbol: e.target.value as string})}
                        label="Trading Symbol"
                      >
                        <MenuItem value="EURUSD">EUR/USD</MenuItem>
                        <MenuItem value="GBPUSD">GBP/USD</MenuItem>
                        <MenuItem value="USDJPY">USD/JPY</MenuItem>
                        <MenuItem value="AUDUSD">AUD/USD</MenuItem>
                        <MenuItem value="USDCAD">USD/CAD</MenuItem>
                        <MenuItem value="NZDUSD">NZD/USD</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <FormControl fullWidth margin="normal">
                      <InputLabel id="base-timeframe-select-label">Base Data Timeframe</InputLabel>
                      <Select
                        labelId="base-timeframe-select-label"
                        value={config.baseDataTimeframe}
                        onChange={(e) => setConfig({...config, baseDataTimeframe: e.target.value as string})}
                        label="Base Data Timeframe"
                      >
                        {AVAILABLE_TIMEFRAMES.map(tf => (
                          <MenuItem key={tf} value={tf}>{tf}</MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <TextField
                      label="Episode Length (steps)"
                      type="number"
                      value={config.episodeLength}
                      onChange={(e) => setConfig({...config, episodeLength: Number(e.target.value)})}
                      fullWidth
                      margin="normal"
                      InputProps={{
                        endAdornment: (
                          <Tooltip title={`With ${config.baseDataTimeframe} base timeframe, this equals ${
                            config.baseDataTimeframe === '1m' ? config.episodeLength / 1440 : 
                            config.baseDataTimeframe === '5m' ? config.episodeLength / 288 :
                            config.baseDataTimeframe === '15m' ? config.episodeLength / 96 :
                            config.baseDataTimeframe === '30m' ? config.episodeLength / 48 :
                            config.baseDataTimeframe === '1h' ? config.episodeLength / 24 :
                            config.baseDataTimeframe === '4h' ? config.episodeLength / 6 :
                            1
                          } days`}>
                            <InfoIcon />
                          </Tooltip>
                        )
                      }}
                    />
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <TextField
                      label="Initial Balance"
                      type="number"
                      value={config.initialBalance}
                      onChange={(e) => setConfig({...config, initialBalance: Number(e.target.value)})}
                      fullWidth
                      margin="normal"
                    />
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <TextField
                      label="Max Drawdown"
                      type="number"
                      value={config.maxDrawdown}
                      onChange={(e) => setConfig({...config, maxDrawdown: Number(e.target.value)})}
                      fullWidth
                      margin="normal"
                      InputProps={{
                        endAdornment: (
                          <Typography color="textSecondary">
                            ({(config.maxDrawdown * 100).toFixed(0)}%)
                          </Typography>
                        )
                      }}
                    />
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          {/* Timeframes Configuration */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                  <Typography variant="h6" component="h3">
                    Timeframes
                  </Typography>
                  <Button 
                    onClick={handleAddTimeframe}
                    startIcon={<AddIcon />}
                    disabled={AVAILABLE_TIMEFRAMES.length === config.timeframes.length}
                  >
                    Add Timeframe
                  </Button>
                </Box>
                
                {config.timeframes.length === 0 ? (
                  <Alert severity="info">Add at least one timeframe for your environment.</Alert>
                ) : (
                  <List>
                    {config.timeframes.map((tf, index) => (
                      <ListItem 
                        key={index}
                        divider={index < config.timeframes.length - 1}
                        sx={{ flexDirection: { xs: 'column', sm: 'row' }, alignItems: { xs: 'flex-start', sm: 'center' } }}
                      >
                        <FormControl sx={{ minWidth: 120, mr: 2, mb: { xs: 2, sm: 0 } }}>
                          <InputLabel id={`timeframe-select-label-${index}`}>Timeframe</InputLabel>
                          <Select
                            labelId={`timeframe-select-label-${index}`}
                            value={tf.timeframe}
                            label="Timeframe"
                            onChange={handleTimeframeChange(index, 'timeframe')}
                            size="small"
                          >
                            {AVAILABLE_TIMEFRAMES.map(availableTf => (
                              <MenuItem 
                                key={availableTf} 
                                value={availableTf}
                                disabled={config.timeframes.some(
                                  (configTf, i) => i !== index && configTf.timeframe === availableTf
                                )}
                              >
                                {availableTf}
                              </MenuItem>
                            ))}
                          </Select>
                        </FormControl>
                        
                        <Box sx={{ 
                          display: 'flex', 
                          flexGrow: 1, 
                          alignItems: 'center',
                          width: { xs: '100%', sm: 'auto' }
                        }}>
                          <Typography variant="body2" sx={{ mr: 2 }}>
                            Lookback:
                          </Typography>
                          <Slider
                            value={tf.lookbackPeriods}
                            min={1}
                            max={100}
                            step={1}
                            onChange={handleLookbackSliderChange(index)}
                            sx={{ flexGrow: 1, maxWidth: 200 }}
                            valueLabelDisplay="auto"
                          />
                          <TextField
                            value={tf.lookbackPeriods}
                            onChange={handleTimeframeChange(index, 'lookbackPeriods')}
                            type="number"
                            size="small"
                            sx={{ width: 70, ml: 2 }}
                            InputProps={{ inputProps: { min: 1, max: 100 } }}
                          />
                        </Box>
                        
                        <ListItemSecondaryAction>
                          <IconButton 
                            edge="end" 
                            onClick={() => handleRemoveTimeframe(index)}
                            disabled={config.timeframes.length <= 1}
                          >
                            <DeleteIcon />
                          </IconButton>
                        </ListItemSecondaryAction>
                      </ListItem>
                    ))}
                  </List>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Features Selection */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" component="h3" gutterBottom>
                  Features Selection
                </Typography>
                
                {getFeaturesByCategory().map(categoryGroup => (
                  <Accordion key={categoryGroup.category} defaultExpanded>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography>{categoryGroup.category.charAt(0).toUpperCase() + categoryGroup.category.slice(1)}</Typography>
                      <Chip 
                        label={`${categoryGroup.features.filter(f => f.enabled).length}/${categoryGroup.features.length}`} 
                        size="small" 
                        sx={{ ml: 1 }}
                        color={categoryGroup.features.some(f => f.enabled) ? "primary" : "default"}
                      />
                    </AccordionSummary>
                    <AccordionDetails>
                      <FormGroup>
                        {categoryGroup.features.map((feature, featureIndex) => {
                          const globalIndex = config.features.findIndex(f => f.name === feature.name);
                          return (
                            <FormControlLabel
                              key={feature.name}
                              control={
                                <Switch
                                  checked={feature.enabled}
                                  onChange={handleFeatureToggle(globalIndex)}
                                  name={feature.name}
                                />
                              }
                              label={
                                <Tooltip title={feature.description || ''}>
                                  <Typography variant="body2">{feature.name}</Typography>
                                </Tooltip>
                              }
                            />
                          );
                        })}
                      </FormGroup>
                    </AccordionDetails>
                  </Accordion>
                ))}
              </CardContent>
            </Card>
          </Grid>

          {/* Position Sizing */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" component="h3" gutterBottom>
                  Position Sizing
                </Typography>
                <FormControl fullWidth margin="normal">
                  <InputLabel id="position-sizing-select-label">Position Sizing Strategy</InputLabel>
                  <Select
                    labelId="position-sizing-select-label"
                    value={config.positionSizing.type}
                    onChange={handlePositionSizingTypeChange}
                    label="Position Sizing Strategy"
                  >
                    {POSITION_SIZING_STRATEGIES.map(strategy => (
                      <MenuItem key={strategy.name} value={strategy.name}>
                        {strategy.name.charAt(0).toUpperCase() + strategy.name.slice(1)} - {strategy.description}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                
                <Box sx={{ mt: 3 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Parameters
                  </Typography>
                  
                  <Grid container spacing={2}>
                    {Object.entries(config.positionSizing.parameters).map(([param, value]) => (
                      <Grid item xs={12} sm={6} key={param}>
                        <TextField
                          label={param.replace('_', ' ').charAt(0).toUpperCase() + param.replace('_', ' ').slice(1)}
                          value={value}
                          onChange={handlePositionSizingParamChange(param)}
                          type="number"
                          fullWidth
                        />
                      </Grid>
                    ))}
                  </Grid>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Trading Fees */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" component="h3" gutterBottom>
                  Trading Fees & Execution
                </Typography>
                
                <TextField
                  label="Commission Rate"
                  type="number"
                  value={config.tradingFees.commissionRate}
                  onChange={handleTradingFeesChange('commissionRate')}
                  fullWidth
                  margin="normal"
                  InputProps={{
                    endAdornment: (
                      <Typography color="textSecondary">
                        ({(config.tradingFees.commissionRate * 100).toFixed(2)}%)
                      </Typography>
                    ),
                    inputProps: {
                      step: 0.0001,
                      min: 0
                    }
                  }}
                />
                
                <FormControl fullWidth margin="normal">
                  <InputLabel id="slippage-model-select-label">Slippage Model</InputLabel>
                  <Select
                    labelId="slippage-model-select-label"
                    value={config.tradingFees.slippageModel}
                    onChange={handleTradingFeesChange('slippageModel')}
                    label="Slippage Model"
                  >
                    <MenuItem value="none">None - Perfect Execution</MenuItem>
                    <MenuItem value="fixed">Fixed - Constant Amount</MenuItem>
                    <MenuItem value="proportional">Proportional - Based on Order Size</MenuItem>
                    <MenuItem value="volume_based">Volume-Based - Depends on Market Liquidity</MenuItem>
                    <MenuItem value="realistic">Realistic - Complex Model with Multiple Factors</MenuItem>
                  </Select>
                </FormControl>
                
                <FormControl fullWidth margin="normal">
                  <InputLabel id="spread-model-select-label">Spread Model</InputLabel>
                  <Select
                    labelId="spread-model-select-label"
                    value={config.tradingFees.spreadModel}
                    onChange={handleTradingFeesChange('spreadModel')}
                    label="Spread Model"
                  >
                    <MenuItem value="fixed">Fixed - Constant Spread</MenuItem>
                    <MenuItem value="variable">Variable - Time-Based Variation</MenuItem>
                    <MenuItem value="volatility_based">Volatility-Based - Widens During Volatility</MenuItem>
                    <MenuItem value="realistic">Realistic - Based on Historical Data</MenuItem>
                  </Select>
                </FormControl>
              </CardContent>
            </Card>
          </Grid>

          {/* Reward Function */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                  <Typography variant="h6" component="h3">
                    Reward Function
                  </Typography>
                  <Button 
                    onClick={handleAddRewardComponent}
                    startIcon={<AddIcon />}
                    disabled={config.rewardComponents.length === AVAILABLE_REWARD_COMPONENTS.length}
                  >
                    Add Component
                  </Button>
                </Box>
                
                {config.rewardComponents.length === 0 ? (
                  <Alert severity="warning">Add at least one reward component.</Alert>
                ) : (
                  <List>
                    {config.rewardComponents.map((component, index) => (
                      <ListItem 
                        key={index}
                        divider={index < config.rewardComponents.length - 1}
                        sx={{ flexDirection: { xs: 'column', sm: 'row' }, alignItems: { xs: 'flex-start', sm: 'center' }, py: 2 }}
                      >
                        <FormControlLabel
                          control={
                            <Switch
                              checked={component.enabled}
                              onChange={handleRewardComponentToggle(index)}
                            />
                          }
                          label={
                            <Box>
                              <Typography variant="subtitle1">
                                {component.name.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
                              </Typography>
                              <Typography variant="body2" color="textSecondary">
                                {component.description}
                              </Typography>
                            </Box>
                          }
                          sx={{ flexGrow: 1, mb: { xs: 2, sm: 0 }, mr: 2 }}
                        />
                        
                        <Box sx={{ 
                          display: 'flex', 
                          alignItems: 'center',
                          width: { xs: '100%', sm: '30%' }
                        }}>
                          <Typography variant="body2" sx={{ mr: 2 }}>
                            Weight:
                          </Typography>
                          <Slider
                            value={component.weight}
                            min={0}
                            max={2}
                            step={0.1}
                            disabled={!component.enabled}
                            onChange={handleRewardWeightSliderChange(index)}
                            sx={{ flexGrow: 1 }}
                            valueLabelDisplay="auto"
                          />
                          <TextField
                            value={component.weight}
                            onChange={handleRewardWeightChange(index)}
                            type="number"
                            size="small"
                            disabled={!component.enabled}
                            sx={{ width: 70, ml: 2 }}
                            InputProps={{ inputProps: { min: 0, max: 2, step: 0.1 } }}
                          />
                        </Box>
                        
                        <IconButton 
                          onClick={() => handleRemoveRewardComponent(index)}
                          disabled={config.rewardComponents.length <= 1}
                          sx={{ ml: 1 }}
                        >
                          <DeleteIcon />
                        </IconButton>
                      </ListItem>
                    ))}
                  </List>
                )}
                
                <Divider sx={{ my: 2 }} />
                
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Parameter Configuration
                  </Typography>
                  
                  {config.rewardComponents.filter(c => c.enabled && c.parameters).map((component, componentIndex) => (
                    <Box key={componentIndex} sx={{ mb: 3 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        {component.name.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')} Parameters
                      </Typography>
                      
                      <Grid container spacing={2}>
                        {component.parameters && Object.entries(component.parameters).map(([param, value]) => {
                          const index = config.rewardComponents.findIndex(c => c.name === component.name);
                          return (
                            <Grid item xs={12} sm={6} md={4} key={param}>
                              <TextField
                                label={param.replace('_', ' ').charAt(0).toUpperCase() + param.replace('_', ' ').slice(1)}
                                value={value}
                                onChange={handleRewardParamChange(index, param)}
                                type="number"
                                fullWidth
                                size="small"
                              />
                            </Grid>
                          );
                        })}
                      </Grid>
                    </Box>
                  ))}
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Curriculum Learning */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                  <Typography variant="h6" component="h3">
                    Curriculum Learning
                  </Typography>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={config.curriculumLearning.enabled}
                        onChange={handleCurriculumToggle}
                      />
                    }
                    label="Enable Curriculum Learning"
                  />
                </Box>
                
                {config.curriculumLearning.enabled ? (
                  <>
                    {config.curriculumLearning.levels.map((level, index) => (
                      <Accordion key={index} defaultExpanded={index === 0}>
                        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                          <Typography>{level.name}</Typography>
                          <Chip 
                            label={`Difficulty: ${level.difficulty}`} 
                            size="small" 
                            sx={{ ml: 1 }}
                            color={level.difficulty <= 2 ? "success" : level.difficulty <= 3 ? "warning" : "error"}
                          />
                        </AccordionSummary>
                        <AccordionDetails>
                          <Grid container spacing={2}>
                            <Grid item xs={12} md={6}>
                              <TextField
                                label="Level Name"
                                value={level.name}
                                onChange={handleCurriculumLevelChange(index, 'name')}
                                fullWidth
                                margin="normal"
                              />
                            </Grid>
                            <Grid item xs={12} md={6}>
                              <TextField
                                label="Description"
                                value={level.description}
                                onChange={handleCurriculumLevelChange(index, 'description')}
                                fullWidth
                                margin="normal"
                              />
                            </Grid>
                            <Grid item xs={12} md={6}>
                              <TextField
                                label="Episodes"
                                type="number"
                                value={level.episodes}
                                onChange={handleCurriculumLevelChange(index, 'episodes')}
                                fullWidth
                                margin="normal"
                              />
                            </Grid>
                            <Grid item xs={12} md={6}>
                              <TextField
                                label="Difficulty (1-5)"
                                type="number"
                                value={level.difficulty}
                                onChange={handleCurriculumLevelChange(index, 'difficulty')}
                                fullWidth
                                margin="normal"
                                InputProps={{ inputProps: { min: 1, max: 5 } }}
                              />
                            </Grid>
                            
                            <Grid item xs={12}>
                              <Typography variant="subtitle2" gutterBottom>
                                Parameters
                              </Typography>
                              <Grid container spacing={2}>
                                {Object.entries(level.parameters).map(([param, value]) => (
                                  <Grid item xs={12} sm={6} md={4} key={param}>
                                    <TextField
                                      label={param.replace('_', ' ').charAt(0).toUpperCase() + param.replace('_', ' ').slice(1)}
                                      value={value}
                                      onChange={handleCurriculumParamChange(index, param)}
                                      type="number"
                                      fullWidth
                                    />
                                  </Grid>
                                ))}
                              </Grid>
                            </Grid>
                          </Grid>
                        </AccordionDetails>
                      </Accordion>
                    ))}
                  </>
                ) : (
                  <Alert severity="info">
                    Enable curriculum learning to progressively increase training difficulty.
                  </Alert>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Environment Summary */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" component="h3" gutterBottom>
                  Environment Summary
                </Typography>
                
                {environmentSummary && (
                  <Grid container spacing={3}>
                    <Grid item xs={12} md={4}>
                      <Card variant="outlined" sx={{ height: '100%' }}>
                        <CardContent>
                          <Typography variant="subtitle1" gutterBottom>
                            Observation Space
                          </Typography>
                          <Typography variant="h4" component="div">
                            {environmentSummary.observationDimensions}
                          </Typography>
                          <Typography variant="body2" color="textSecondary">
                            dimensions
                          </Typography>
                          
                          <Box sx={{ mt: 2 }}>
                            <Typography variant="body2">
                              <strong>Active features:</strong> {environmentSummary.activeFeatureCount}
                            </Typography>
                            <Typography variant="body2">
                              <strong>Timeframes:</strong> {config.timeframes.length}
                            </Typography>
                          </Box>
                        </CardContent>
                      </Card>
                    </Grid>
                    
                    <Grid item xs={12} md={4}>
                      <Card variant="outlined" sx={{ height: '100%' }}>
                        <CardContent>
                          <Typography variant="subtitle1" gutterBottom>
                            Memory Estimate
                          </Typography>
                          <Typography variant="h4" component="div">
                            {environmentSummary.estimatedMemoryMb.toFixed(1)} MB
                          </Typography>
                          <Typography variant="body2" color="textSecondary">
                            per environment instance
                          </Typography>
                          
                          <Box sx={{ mt: 2 }}>
                            <Typography variant="body2">
                              <strong>Batch size impact:</strong> {(environmentSummary.estimatedMemoryMb * 32).toFixed(0)} MB for batch size 32
                            </Typography>
                          </Box>
                        </CardContent>
                      </Card>
                    </Grid>
                    
                    <Grid item xs={12} md={4}>
                      <Card variant="outlined" sx={{ height: '100%' }}>
                        <CardContent>
                          <Typography variant="subtitle1" gutterBottom>
                            Reward Components
                          </Typography>
                          <Typography variant="h4" component="div">
                            {config.rewardComponents.filter(c => c.enabled).length}
                          </Typography>
                          <Typography variant="body2" color="textSecondary">
                            active components
                          </Typography>
                          
                          <Box sx={{ mt: 2 }}>
                            <Typography variant="body2">
                              <strong>Primary component:</strong> {
                                config.rewardComponents
                                  .filter(c => c.enabled)
                                  .sort((a, b) => b.weight - a.weight)[0]?.name.replace('_', ' ') || 'None'
                              }
                            </Typography>
                          </Box>
                        </CardContent>
                      </Card>
                    </Grid>
                  </Grid>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Paper>
    </Box>
  );
};

export default RLEnvironmentConfigurator;
