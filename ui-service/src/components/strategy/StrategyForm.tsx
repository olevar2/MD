/**
 * StrategyForm component - For creating and editing trading strategies
 */
import { useState, useEffect } from 'react';
import {
  Box,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormHelperText,
  Grid,
  Typography,
  Divider,
  Tabs,
  Tab,
  Chip,
  Autocomplete,
} from '@mui/material';
import { Strategy, MarketRegime, TimeFrame, StrategyParameter } from '@/types/strategy';
import { strategyApi } from '@/api/strategy-api';
import ParameterConfigurator from './ParameterConfigurator';

interface StrategyFormProps {
  initialData?: Strategy | null;
  onSubmit: (data: Partial<Strategy>) => void;
}

export default function StrategyForm({ initialData, onSubmit }: StrategyFormProps) {
  const [formData, setFormData] = useState<Partial<Strategy>>({
    name: '',
    description: '',
    type: 'adaptive_ma',
    symbols: [],
    timeframes: [],
    primaryTimeframe: TimeFrame.H1,
    parameters: {},
    isActive: false
  });
  
  const [availableSymbols, setAvailableSymbols] = useState<string[]>([
    'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 'NZD/USD', 'USD/CHF', 
    'EUR/GBP', 'EUR/JPY', 'GBP/JPY'
  ]);
  
  const [tabIndex, setTabIndex] = useState(0);
  const [templates, setTemplates] = useState<any[]>([]);
  const [parameterTemplates, setParameterTemplates] = useState<StrategyParameter[]>([]);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);

  // Load strategy templates on mount
  useEffect(() => {
    const loadTemplates = async () => {
      try {
        const data = await strategyApi.getStrategyTemplates();
        setTemplates(data);
      } catch (error) {
        console.error('Failed to load strategy templates', error);
      }
    };

    loadTemplates();
  }, []);

  // Initialize form with initial data if provided
  useEffect(() => {
    if (initialData) {
      setFormData({
        ...initialData,
      });
      
      if (initialData.parameterTemplates) {
        setParameterTemplates(initialData.parameterTemplates);
      }
    }
  }, [initialData]);

  // Load parameter templates when strategy type changes
  useEffect(() => {
    const loadParameterTemplates = async () => {
      if (!formData.type) return;
      
      // Find matching template
      const matchingTemplate = templates.find(t => t.id === formData.type);
      
      if (matchingTemplate) {
        setParameterTemplates(matchingTemplate.parameterTemplates);
        
        // Initialize parameter values with defaults if this is a new strategy
        if (!initialData) {
          const defaultParams: Record<string, any> = {};
          matchingTemplate.parameterTemplates.forEach((param: StrategyParameter) => {
            defaultParams[param.id] = param.defaultValue;
          });
          
          setFormData(prev => ({
            ...prev,
            parameters: defaultParams
          }));
        }
      }
    };
    
    if (templates.length > 0) {
      loadParameterTemplates();
    }
  }, [formData.type, templates, initialData]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | { name?: string; value: unknown }>) => {
    const { name, value } = e.target;
    if (name) {
      setFormData(prev => ({
        ...prev,
        [name]: value
      }));
      
      // Clear error for this field if exists
      if (errors[name]) {
        setErrors(prev => ({
          ...prev,
          [name]: ''
        }));
      }
    }
  };

  const handleSymbolsChange = (_event: any, values: string[]) => {
    setFormData(prev => ({
      ...prev,
      symbols: values
    }));
  };

  const handleTimeframesChange = (_event: any, values: string[]) => {
    setFormData(prev => ({
      ...prev,
      timeframes: values,
      // Set primary timeframe to first selected timeframe if current primary not in selection
      primaryTimeframe: values.includes(formData.primaryTimeframe as string) ? 
        formData.primaryTimeframe : 
        values[0]
    }));
  };

  const handlePrimaryTfChange = (e: React.ChangeEvent<{ name?: string; value: unknown }>) => {
    const { value } = e.target;
    setFormData(prev => ({
      ...prev,
      primaryTimeframe: value as TimeFrame
    }));
  };

  const handleParameterChange = (parameterId: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      parameters: {
        ...prev.parameters,
        [parameterId]: value
      }
    }));
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabIndex(newValue);
  };

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};
    
    if (!formData.name?.trim()) {
      newErrors.name = 'Strategy name is required';
    }
    
    if (!formData.type) {
      newErrors.type = 'Strategy type is required';
    }
    
    if (!formData.symbols || formData.symbols.length === 0) {
      newErrors.symbols = 'At least one symbol must be selected';
    }
    
    if (!formData.timeframes || formData.timeframes.length === 0) {
      newErrors.timeframes = 'At least one timeframe must be selected';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (validateForm()) {
      onSubmit(formData);
    }
  };

  return (
    <Box component="form" onSubmit={handleSubmit} sx={{ p: 2 }}>
      <Tabs value={tabIndex} onChange={handleTabChange} sx={{ mb: 3 }}>
        <Tab label="Basic Settings" />
        <Tab label="Parameters" />
        <Tab label="Regime-Specific" />
        <Tab label="Indicators" />
      </Tabs>
      
      {/* Basic Settings Tab */}
      {tabIndex === 0 && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Strategy Name"
              name="name"
              value={formData.name}
              onChange={handleChange}
              error={!!errors.name}
              helperText={errors.name}
              required
            />
          </Grid>
          
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Description"
              name="description"
              value={formData.description || ''}
              onChange={handleChange}
              multiline
              rows={3}
            />
          </Grid>
          
          <Grid item xs={12} md={6}>
            <FormControl fullWidth error={!!errors.type}>
              <InputLabel>Strategy Type</InputLabel>
              <Select
                name="type"
                value={formData.type || ''}
                onChange={handleChange}
                label="Strategy Type"
              >
                <MenuItem value="adaptive_ma">Adaptive MA Strategy</MenuItem>
                <MenuItem value="elliott_wave">Elliott Wave Strategy</MenuItem>
                <MenuItem value="multi_timeframe_confluence">Multi-Timeframe Confluence</MenuItem>
                <MenuItem value="harmonic_pattern">Harmonic Pattern</MenuItem>
                <MenuItem value="advanced_breakout">Advanced Breakout</MenuItem>
                <MenuItem value="custom">Custom Strategy</MenuItem>
              </Select>
              {errors.type && <FormHelperText>{errors.type}</FormHelperText>}
            </FormControl>
          </Grid>
          
          <Grid item xs={12}>
            <Autocomplete
              multiple
              options={availableSymbols}
              value={formData.symbols || []}
              onChange={handleSymbolsChange}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Symbols"
                  error={!!errors.symbols}
                  helperText={errors.symbols}
                />
              )}
              renderTags={(selected, getTagProps) =>
                selected.map((option, index) => (
                  <Chip
                    label={option}
                    size="small"
                    {...getTagProps({ index })}
                  />
                ))
              }
            />
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Autocomplete
              multiple
              options={Object.values(TimeFrame)}
              value={formData.timeframes || []}
              onChange={handleTimeframesChange}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Timeframes"
                  error={!!errors.timeframes}
                  helperText={errors.timeframes}
                />
              )}
              renderTags={(selected, getTagProps) =>
                selected.map((option, index) => (
                  <Chip
                    label={option}
                    size="small"
                    {...getTagProps({ index })}
                  />
                ))
              }
            />
          </Grid>
          
          <Grid item xs={12} md={6}>
            <FormControl fullWidth disabled={!formData.timeframes?.length}>
              <InputLabel>Primary Timeframe</InputLabel>
              <Select
                name="primaryTimeframe"
                value={formData.primaryTimeframe || ''}
                onChange={handlePrimaryTfChange}
                label="Primary Timeframe"
              >
                {formData.timeframes?.map((tf) => (
                  <MenuItem key={tf} value={tf}>{tf}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
        </Grid>
      )}
      
      {/* Parameters Tab */}
      {tabIndex === 1 && (
        <Box>
          <Typography variant="h6" gutterBottom>
            Strategy Parameters
          </Typography>
          <Typography variant="body2" color="textSecondary" paragraph>
            Configure parameters specific to the {formData.type} strategy
          </Typography>
          
          {parameterTemplates.length === 0 ? (
            <Typography color="textSecondary">
              No parameters available for this strategy type
            </Typography>
          ) : (
            <Grid container spacing={3}>
              {/* Group parameters by category */}
              {['main', 'risk', 'adaptive', 'advanced'].map((category) => {
                const categoryParams = parameterTemplates.filter(
                  p => p.category === category && !p.regimeSpecific
                );
                
                if (categoryParams.length === 0) return null;
                
                return (
                  <Grid item xs={12} key={category}>
                    <Typography variant="subtitle1" sx={{ mb: 1, textTransform: 'capitalize' }}>
                      {category} Parameters
                    </Typography>
                    <Divider sx={{ mb: 2 }} />
                    
                    <Grid container spacing={2}>
                      {categoryParams.map((param) => (
                        <Grid item xs={12} sm={6} md={4} key={param.id}>
                          <ParameterConfigurator
                            parameter={param}
                            value={formData.parameters?.[param.id]}
                            onChange={(value) => handleParameterChange(param.id, value)}
                          />
                        </Grid>
                      ))}
                    </Grid>
                  </Grid>
                );
              })}
            </Grid>
          )}
        </Box>
      )}
      
      {/* Regime-Specific Tab */}
      {tabIndex === 2 && (
        <Box>
          <Typography variant="h6" gutterBottom>
            Regime-Specific Parameters
          </Typography>
          <Typography variant="body2" color="textSecondary" paragraph>
            Configure how parameters adapt to different market regimes
          </Typography>
          
          <Tabs value={0} sx={{ mb: 2 }}>
            {Object.values(MarketRegime).map((regime, index) => (
              <Tab key={regime} label={regime} value={index} />
            ))}
          </Tabs>
          
          {/* For simplicity, just showing trending regime parameters */}
          <Grid container spacing={3}>
            {parameterTemplates.filter(p => p.regimeSpecific).map((param) => (
              <Grid item xs={12} sm={6} md={4} key={param.id}>
                <ParameterConfigurator
                  parameter={param}
                  value={
                    formData.regimeParameters?.[MarketRegime.TRENDING]?.[param.id] ||
                    formData.parameters?.[param.id]
                  }
                  onChange={(value) => {
                    setFormData(prev => ({
                      ...prev,
                      regimeParameters: {
                        ...prev.regimeParameters,
                        [MarketRegime.TRENDING]: {
                          ...(prev.regimeParameters?.[MarketRegime.TRENDING] || {}),
                          [param.id]: value
                        }
                      }
                    }));
                  }}
                />
              </Grid>
            ))}
          </Grid>
        </Box>
      )}
      
      {/* Indicators Tab */}
      {tabIndex === 3 && (
        <Box>
          <Typography variant="h6" gutterBottom>
            Technical Indicators
          </Typography>
          <Typography variant="body2" color="textSecondary" paragraph>
            Configure indicators used by this strategy
          </Typography>
          
          <Typography color="textSecondary">
            Indicator configuration will be implemented in the next phase
          </Typography>
        </Box>
      )}
      
      <Box sx={{ mt: 4, display: 'flex', justifyContent: 'flex-end' }}>
        <Button
          type="submit"
          variant="contained"
          color="primary"
          disabled={loading}
        >
          {initialData ? 'Update Strategy' : 'Create Strategy'}
        </Button>
      </Box>
    </Box>
  );
}
