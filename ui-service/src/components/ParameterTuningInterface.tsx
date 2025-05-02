import React from 'react';
import {
  Grid,
  Paper,
  Typography,
  Slider,
  TextField,
  Button,
  Box,
  Switch,
  FormControlLabel,
  Divider,
  Alert,
  CircularProgress
} from '@mui/material';

export interface Parameter {
  id: string;
  name: string;
  type: 'number' | 'boolean' | 'string' | 'select';
  value: number | boolean | string;
  options?: { label: string; value: string | number }[];
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  description?: string;
}

export interface StrategyConfig {
  id: string;
  name: string;
  parameters: Parameter[];
}

interface ParameterTuningInterfaceProps {
  strategyId: string;
  onSave: (config: StrategyConfig) => Promise<void>;
  onParameterChange?: (parameterId: string, value: any) => void;
}

const ParameterTuningInterface: React.FC<ParameterTuningInterfaceProps> = ({
  strategyId,
  onSave,
  onParameterChange
}) => {
  const [config, setConfig] = React.useState<StrategyConfig | null>(null);
  const [isLoading, setIsLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [isSaving, setIsSaving] = React.useState(false);

  React.useEffect(() => {
    const fetchParameters = async () => {
      setIsLoading(true);
      setError(null);
      try {
        // In a real implementation, this would be an API call
        // const response = await api.get(`/strategies/${strategyId}/parameters`);
        await new Promise(resolve => setTimeout(resolve, 500)); // Simulate API call
        
        // Mock data
        setConfig({
          id: strategyId,
          name: 'Enhanced Trend Following',
          parameters: [
            {
              id: 'trend_period',
              name: 'Trend Period',
              type: 'number',
              value: 20,
              min: 5,
              max: 50,
              step: 1,
              description: 'Number of periods used to calculate trend'
            },
            {
              id: 'volatility_lookback',
              name: 'Volatility Lookback',
              type: 'number',
              value: 14,
              min: 5,
              max: 30,
              step: 1,
              description: 'Periods for volatility calculation'
            },
            {
              id: 'use_adaptive_stops',
              name: 'Use Adaptive Stops',
              type: 'boolean',
              value: true,
              description: 'Dynamically adjust stop losses based on volatility'
            },
            {
              id: 'risk_mode',
              name: 'Risk Management Mode',
              type: 'select',
              value: 'dynamic',
              options: [
                { label: 'Fixed', value: 'fixed' },
                { label: 'Dynamic', value: 'dynamic' },
                { label: 'Adaptive', value: 'adaptive' }
              ],
              description: 'Method for calculating position sizes'
            }
          ]
        });
      } catch (err) {
        setError('Failed to load strategy parameters');
        console.error('Parameter fetch error:', err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchParameters();
  }, [strategyId]);

  const handleParameterChange = (parameter: Parameter) => (
    event: React.ChangeEvent<HTMLInputElement> | Event,
    newValue: number | number[] | null
  ) => {
    if (!config) return;

    const updatedValue = 
      parameter.type === 'boolean' 
        ? (event as React.ChangeEvent<HTMLInputElement>).target.checked
        : parameter.type === 'select' || parameter.type === 'string'
          ? (event as React.ChangeEvent<HTMLInputElement>).target.value
          : newValue;

    const updatedConfig = {
      ...config,
      parameters: config.parameters.map(p => 
        p.id === parameter.id 
          ? { ...p, value: updatedValue }
          : p
      )
    };

    setConfig(updatedConfig);
    onParameterChange?.(parameter.id, updatedValue);
  };

  const handleSave = async () => {
    if (!config) return;
    
    setIsSaving(true);
    setError(null);
    try {
      await onSave(config);
      // Show success message or trigger callback
    } catch (err) {
      setError('Failed to save parameters');
      console.error('Save error:', err);
    } finally {
      setIsSaving(false);
    }
  };

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" p={3}>
        <CircularProgress />
      </Box>
    );
  }

  if (!config) {
    return (
      <Alert severity="error">
        Could not load strategy parameters
      </Alert>
    );
  }

  return (
    <Paper sx={{ p: 3 }}>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          {config.name} Parameters
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Adjust the parameters below to optimize the strategy performance.
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {config.parameters.map((parameter) => (
          <Grid item xs={12} key={parameter.id}>
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                {parameter.name}
              </Typography>
              {parameter.description && (
                <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
                  {parameter.description}
                </Typography>
              )}

              {parameter.type === 'number' && (
                <Box sx={{ px: 1 }}>
                  <Slider
                    value={parameter.value as number}
                    min={parameter.min}
                    max={parameter.max}
                    step={parameter.step}
                    valueLabelDisplay="auto"
                    onChange={handleParameterChange(parameter)}
                    marks={[
                      { value: parameter.min!, label: parameter.min!.toString() },
                      { value: parameter.max!, label: parameter.max!.toString() }
                    ]}
                  />
                  <TextField
                    type="number"
                    value={parameter.value}
                    onChange={handleParameterChange(parameter)}
                    size="small"
                    inputProps={{
                      min: parameter.min,
                      max: parameter.max,
                      step: parameter.step
                    }}
                    sx={{ mt: 1, width: 100 }}
                  />
                  {parameter.unit && (
                    <Typography variant="body2" color="text.secondary" display="inline" sx={{ ml: 1 }}>
                      {parameter.unit}
                    </Typography>
                  )}
                </Box>
              )}

              {parameter.type === 'boolean' && (
                <FormControlLabel
                  control={
                    <Switch
                      checked={parameter.value as boolean}
                      onChange={handleParameterChange(parameter)}
                    />
                  }
                  label={parameter.value ? 'Enabled' : 'Disabled'}
                />
              )}

              {parameter.type === 'select' && parameter.options && (
                <TextField
                  select
                  value={parameter.value}
                  onChange={handleParameterChange(parameter)}
                  fullWidth
                  SelectProps={{ native: true }}
                >
                  {parameter.options.map(option => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </TextField>
              )}
            </Box>
            <Divider />
          </Grid>
        ))}
      </Grid>

      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}

      <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
        <Button
          variant="contained"
          onClick={handleSave}
          disabled={isSaving}
        >
          {isSaving ? 'Saving...' : 'Save Parameters'}
        </Button>
      </Box>
    </Paper>
  );
};

export default ParameterTuningInterface;
