import React from 'react';
import { Chip, TextField, Autocomplete } from '@mui/material';

interface IndicatorInput {
  name: string;
  defaultPeriod?: number;
}

interface IndicatorSelectorProps {
  value: string[];
  onChange: (newValue: string[]) => void;
  availableIndicators?: IndicatorInput[];
}

const defaultIndicators: IndicatorInput[] = [
  { name: 'SMA', defaultPeriod: 20 },
  { name: 'EMA', defaultPeriod: 50 },
  { name: 'RSI', defaultPeriod: 14 },
  { name: 'MACD', defaultPeriod: 12 },
  { name: 'Bollinger Bands', defaultPeriod: 20 },
  { name: 'Stochastic', defaultPeriod: 14 },
  { name: 'ATR', defaultPeriod: 14 },
  { name: 'ADX', defaultPeriod: 14 }
];

const IndicatorSelector: React.FC<IndicatorSelectorProps> = ({
  value,
  onChange,
  availableIndicators = defaultIndicators
}) => {
  const formatIndicator = (indicator: string) => {
    const [name, period] = indicator.split('-');
    const defaultPeriod = availableIndicators.find(i => i.name === name)?.defaultPeriod;
    return period ? indicator : `${name}-${defaultPeriod}`;
  };

  const handleChange = (_event: React.SyntheticEvent, newValues: string[]) => {
    onChange(newValues.map(formatIndicator));
  };

  return (
    <Autocomplete
      multiple
      id="indicator-selector"
      options={availableIndicators.map(i => i.name)}
      value={value.map(v => v.split('-')[0])}
      onChange={handleChange}
      renderTags={(values: string[], getTagProps) =>
        values.map((option: string, index: number) => {
          const indicator = value[index];
          return (
            <Chip
              {...getTagProps({ index })}
              key={indicator}
              label={indicator}
              variant="outlined"
            />
          );
        })
      }
      renderInput={(params) => (
        <TextField
          {...params}
          variant="outlined"
          placeholder="Add indicators"
          helperText="Select technical indicators to show by default on charts"
        />
      )}
    />
  );
};

export default IndicatorSelector;
