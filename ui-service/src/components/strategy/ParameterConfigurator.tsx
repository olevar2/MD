/**
 * Parameter Configurator Component - Renders different input types based on parameter definition
 */
import { useState } from 'react';
import {
  TextField,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Typography,
  Box,
  Tooltip,
  IconButton,
} from '@mui/material';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import { StrategyParameter } from '@/types/strategy';

interface ParameterConfiguratorProps {
  parameter: StrategyParameter;
  value: any;
  onChange: (value: any) => void;
}

export default function ParameterConfigurator({ parameter, value, onChange }: ParameterConfiguratorProps) {
  // Use local state to handle numeric inputs before committing to parent
  const [localValue, setLocalValue] = useState<string>('');
  
  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = event.target.value;
    setLocalValue(newValue);
  };

  const handleBlur = () => {
    // Only update if valid number for number type
    if (parameter.type === 'number') {
      const numValue = parseFloat(localValue);
      if (!isNaN(numValue)) {
        // Apply min/max constraints
        let constrainedValue = numValue;
        if (parameter.min !== undefined) {
          constrainedValue = Math.max(parameter.min, constrainedValue);
        }
        if (parameter.max !== undefined) {
          constrainedValue = Math.min(parameter.max, constrainedValue);
        }
        onChange(constrainedValue);
        setLocalValue(''); // Reset local value
      }
    }
  };

  const handleSliderChange = (_event: Event, newValue: number | number[]) => {
    onChange(newValue);
  };

  const handleSwitchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    onChange(event.target.checked);
  };

  const handleSelectChange = (event: React.ChangeEvent<{ value: unknown }>) => {
    onChange(event.target.value);
  };

  // Render the appropriate input component based on parameter type
  const renderInputComponent = () => {
    switch (parameter.type) {
      case 'number':
        // For number types, decide if slider or text input is more appropriate
        if (parameter.min !== undefined && parameter.max !== undefined && parameter.step) {
          return (
            <Box sx={{ width: '100%' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Typography variant="body2">{value}</Typography>
              </Box>
              <Slider
                value={value ?? parameter.defaultValue}
                min={parameter.min}
                max={parameter.max}
                step={parameter.step}
                onChange={handleSliderChange}
                valueLabelDisplay="auto"
                sx={{ mt: 1 }}
              />
            </Box>
          );
        } else {
          return (
            <TextField
              fullWidth
              type="number"
              value={localValue !== '' ? localValue : (value ?? parameter.defaultValue)}
              onChange={handleChange}
              onBlur={handleBlur}
              inputProps={{
                min: parameter.min,
                max: parameter.max,
                step: parameter.step || 1,
              }}
            />
          );
        }

      case 'boolean':
        return (
          <FormControlLabel
            control={
              <Switch
                checked={value ?? parameter.defaultValue}
                onChange={handleSwitchChange}
              />
            }
            label=""
          />
        );

      case 'select':
        return (
          <FormControl fullWidth>
            <Select
              value={value ?? parameter.defaultValue}
              onChange={handleSelectChange}
            >
              {parameter.options?.map(option => (
                <MenuItem key={option.value} value={option.value}>
                  {option.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        );

      case 'string':
      default:
        return (
          <TextField
            fullWidth
            value={value ?? parameter.defaultValue}
            onChange={(e) => onChange(e.target.value)}
          />
        );
    }
  };

  return (
    <Box sx={{ mb: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
        <Typography variant="body2" fontWeight="medium">
          {parameter.name}
        </Typography>
        <Tooltip title={parameter.description} placement="top">
          <IconButton size="small" sx={{ ml: 0.5 }}>
            <HelpOutlineIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>
      {renderInputComponent()}
    </Box>
  );
}
