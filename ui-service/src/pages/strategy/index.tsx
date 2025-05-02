import React from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  TextField,
  Button,
  Stepper,
  Step,
  StepLabel,
  MenuItem,
  Divider,
  Chip,
  IconButton,
  List,
  ListItem,
  ListItemText,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import SaveIcon from '@mui/icons-material/Save';
import DashboardLayout from '../../components/layout/DashboardLayout';

const steps = ['Basic Configuration', 'Indicators & Rules', 'Risk Management', 'Review & Test'];

interface StrategyConfig {
  name: string;
  description: string;
  timeframe: string;
  symbols: string[];
  indicators: {
    type: string;
    params: Record<string, number>;
  }[];
  rules: {
    condition: string;
    action: string;
    parameters: Record<string, number>;
  }[];
  riskManagement: {
    maxPositionSize: number;
    stopLoss: number;
    takeProfit: number;
    maxDrawdown: number;
  };
}

const StrategyBuilder: React.FC = () => {
  const [activeStep, setActiveStep] = React.useState(0);
  const [config, setConfig] = React.useState<StrategyConfig>({
    name: '',
    description: '',
    timeframe: '1h',
    symbols: ['EUR/USD'],
    indicators: [],
    rules: [],
    riskManagement: {
      maxPositionSize: 100000,
      stopLoss: 50,
      takeProfit: 100,
      maxDrawdown: 10,
    },
  });

  const handleNext = () => {
    setActiveStep((prevStep) => prevStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  const handleSaveStrategy = () => {
    // Implement strategy saving logic
    console.log('Saving strategy:', config);
  };

  const handleBacktest = () => {
    // Implement backtesting logic
    console.log('Running backtest for strategy:', config);
  };

  const renderBasicConfig = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <TextField
          fullWidth
          label="Strategy Name"
          value={config.name}
          onChange={(e) => setConfig({ ...config, name: e.target.value })}
        />
      </Grid>
      <Grid item xs={12}>
        <TextField
          fullWidth
          multiline
          rows={3}
          label="Description"
          value={config.description}
          onChange={(e) => setConfig({ ...config, description: e.target.value })}
        />
      </Grid>
      <Grid item xs={12} md={6}>
        <TextField
          fullWidth
          select
          label="Timeframe"
          value={config.timeframe}
          onChange={(e) => setConfig({ ...config, timeframe: e.target.value })}
        >
          {['1m', '5m', '15m', '1h', '4h', '1d'].map((tf) => (
            <MenuItem key={tf} value={tf}>{tf}</MenuItem>
          ))}
        </TextField>
      </Grid>
      <Grid item xs={12} md={6}>
        <TextField
          fullWidth
          select
          label="Symbol"
          value={config.symbols[0]}
          onChange={(e) => setConfig({ ...config, symbols: [e.target.value] })}
        >
          {['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD'].map((symbol) => (
            <MenuItem key={symbol} value={symbol}>{symbol}</MenuItem>
          ))}
        </TextField>
      </Grid>
    </Grid>
  );

  const renderIndicatorsAndRules = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>Technical Indicators</Typography>
          <List>
            {config.indicators.map((indicator, index) => (
              <ListItem
                key={index}
                secondaryAction={
                  <IconButton edge="end" onClick={() => {
                    const newIndicators = [...config.indicators];
                    newIndicators.splice(index, 1);
                    setConfig({ ...config, indicators: newIndicators });
                  }}>
                    <DeleteIcon />
                  </IconButton>
                }
              >
                <ListItemText
                  primary={indicator.type}
                  secondary={Object.entries(indicator.params)
                    .map(([key, value]) => `${key}: ${value}`)
                    .join(', ')}
                />
              </ListItem>
            ))}
          </List>
          <Button
            startIcon={<AddIcon />}
            onClick={() => {
              setConfig({
                ...config,
                indicators: [
                  ...config.indicators,
                  { type: 'MA', params: { period: 14 } },
                ],
              });
            }}
          >
            Add Indicator
          </Button>
        </Paper>
      </Grid>
      
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>Trading Rules</Typography>
          <List>
            {config.rules.map((rule, index) => (
              <ListItem
                key={index}
                secondaryAction={
                  <IconButton edge="end" onClick={() => {
                    const newRules = [...config.rules];
                    newRules.splice(index, 1);
                    setConfig({ ...config, rules: newRules });
                  }}>
                    <DeleteIcon />
                  </IconButton>
                }
              >
                <ListItemText
                  primary={`${rule.condition} → ${rule.action}`}
                  secondary={Object.entries(rule.parameters)
                    .map(([key, value]) => `${key}: ${value}`)
                    .join(', ')}
                />
              </ListItem>
            ))}
          </List>
          <Button
            startIcon={<AddIcon />}
            onClick={() => {
              setConfig({
                ...config,
                rules: [
                  ...config.rules,
                  {
                    condition: 'MA Cross',
                    action: 'BUY',
                    parameters: { threshold: 0 },
                  },
                ],
              });
            }}
          >
            Add Rule
          </Button>
        </Paper>
      </Grid>
    </Grid>
  );

  const renderRiskManagement = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <TextField
          fullWidth
          type="number"
          label="Max Position Size"
          value={config.riskManagement.maxPositionSize}
          onChange={(e) => setConfig({
            ...config,
            riskManagement: {
              ...config.riskManagement,
              maxPositionSize: Number(e.target.value),
            },
          })}
        />
      </Grid>
      <Grid item xs={12} md={6}>
        <TextField
          fullWidth
          type="number"
          label="Stop Loss (pips)"
          value={config.riskManagement.stopLoss}
          onChange={(e) => setConfig({
            ...config,
            riskManagement: {
              ...config.riskManagement,
              stopLoss: Number(e.target.value),
            },
          })}
        />
      </Grid>
      <Grid item xs={12} md={6}>
        <TextField
          fullWidth
          type="number"
          label="Take Profit (pips)"
          value={config.riskManagement.takeProfit}
          onChange={(e) => setConfig({
            ...config,
            riskManagement: {
              ...config.riskManagement,
              takeProfit: Number(e.target.value),
            },
          })}
        />
      </Grid>
      <Grid item xs={12} md={6}>
        <TextField
          fullWidth
          type="number"
          label="Max Drawdown (%)"
          value={config.riskManagement.maxDrawdown}
          onChange={(e) => setConfig({
            ...config,
            riskManagement: {
              ...config.riskManagement,
              maxDrawdown: Number(e.target.value),
            },
          })}
        />
      </Grid>
    </Grid>
  );

  const renderReview = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>Strategy Summary</Typography>
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2">Basic Configuration</Typography>
            <Typography variant="body2" color="text.secondary">
              Name: {config.name}<br />
              Timeframe: {config.timeframe}<br />
              Symbols: {config.symbols.join(', ')}
            </Typography>
          </Box>
          
          <Divider sx={{ my: 2 }} />
          
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2">Indicators</Typography>
            {config.indicators.map((indicator, index) => (
              <Chip
                key={index}
                label={`${indicator.type} (${Object.entries(indicator.params)
                  .map(([key, value]) => `${key}: ${value}`)
                  .join(', ')})`}
                sx={{ m: 0.5 }}
              />
            ))}
          </Box>
          
          <Divider sx={{ my: 2 }} />
          
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2">Risk Management</Typography>
            <Typography variant="body2" color="text.secondary">
              Max Position: {config.riskManagement.maxPositionSize}<br />
              Stop Loss: {config.riskManagement.stopLoss} pips<br />
              Take Profit: {config.riskManagement.takeProfit} pips<br />
              Max Drawdown: {config.riskManagement.maxDrawdown}%
            </Typography>
          </Box>
        </Paper>
      </Grid>
    </Grid>
  );

  const stepContent = [
    renderBasicConfig,
    renderIndicatorsAndRules,
    renderRiskManagement,
    renderReview,
  ];

  return (
    <DashboardLayout>
      <Box sx={{ mb: 4 }}>
        <Stepper activeStep={activeStep}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>
      </Box>

      {stepContent[activeStep]()}

      <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
        <Button
          disabled={activeStep === 0}
          onClick={handleBack}
        >
          Back
        </Button>
        <Box>
          {activeStep === steps.length - 1 ? (
            <>
              <Button
                variant="contained"
                startIcon={<PlayArrowIcon />}
                onClick={handleBacktest}
                sx={{ mr: 1 }}
              >
                Run Backtest
              </Button>
              <Button
                variant="contained"
                startIcon={<SaveIcon />}
                onClick={handleSaveStrategy}
              >
                Save Strategy
              </Button>
            </>
          ) : (
            <Button
              variant="contained"
              onClick={handleNext}
            >
              Next
            </Button>
          )}
        </Box>
      </Box>
    </DashboardLayout>
  );
};

export default StrategyBuilder;
