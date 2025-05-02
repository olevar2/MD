/**
 * ChartControlPanel Component
 * 
 * Control panel for the trading chart that allows configuring chart options,
 * including toggling different types of confluence highlighting.
 */
// @ts-ignore - Adding ts-ignore to bypass missing module declarations
import React from 'react';
import {
  Box,
  Paper,
  Typography,
  FormControlLabel,
  Switch,
  Slider,
  Divider,
  Chip,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip
// @ts-ignore
} from '@mui/material';
// @ts-ignore
import { ConfluenceType, ChartOptions, TimeFrame } from '../../types/chart';

// Icon components would normally be imported from a library like Material-UI
const SupportIcon = () => <span style={{ color: '#089981' }}>▲</span>;
const ResistanceIcon = () => <span style={{ color: '#F23645' }}>▼</span>;
const PatternIcon = () => <span style={{ color: '#9C27B0' }}>◆</span>;
const MAIcon = () => <span style={{ color: '#1E88E5' }}>■</span>;
const MTFIcon = () => <span style={{ color: '#FF9800' }}>△</span>;

interface ChartControlPanelProps {
  options: ChartOptions;
  timeframe: TimeFrame;
  onOptionsChange: (options: Partial<ChartOptions>) => void;
  onTimeframeChange: (timeframe: TimeFrame) => void;
  activeConfluenceTypes: ConfluenceType[];
  onConfluenceToggle: (type: ConfluenceType[]) => void;
}

const ChartControlPanel: React.FC<ChartControlPanelProps> = ({
  options,
  timeframe,
  onOptionsChange,
  onTimeframeChange,
  activeConfluenceTypes,
  onConfluenceToggle,
}) => {
  // Handle options toggle
  const handleOptionToggle = (option: keyof ChartOptions, value?: any) => {
    onOptionsChange({ [option]: value !== undefined ? value : !options[option] });
  };

  // Handle confluence threshold change
  const handleThresholdChange = (_: Event, value: number | number[]) => {
    onOptionsChange({ confluenceThreshold: value as number });
  };

  // Handle timeframe change
  const handleTimeframeChange = (_: React.MouseEvent<HTMLElement>, newTimeframe: TimeFrame) => {
    if (newTimeframe !== null) {
      onTimeframeChange(newTimeframe);
    }
  };

  // Handle confluence type toggle
  const handleConfluenceTypeToggle = (_: React.MouseEvent<HTMLElement>, newTypes: ConfluenceType[]) => {
    onConfluenceToggle(newTypes);
  };

  return (
    <Paper elevation={2} sx={{ p: 2, mb: 2, borderRadius: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">Chart Controls</Typography>
        
        <ToggleButtonGroup
          size="small"
          value={timeframe}
          exclusive
          onChange={handleTimeframeChange}
          aria-label="timeframe selection"
        >
          {Object.values(TimeFrame).map((tf) => (
            <ToggleButton key={tf} value={tf} aria-label={tf}>
              {tf}
            </ToggleButton>
          ))}
        </ToggleButtonGroup>
      </Box>
      
      <Divider sx={{ my: 1.5 }} />
      
      <Typography variant="subtitle2" sx={{ mb: 1 }}>Chart Display</Typography>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
        <FormControlLabel
          control={
            <Switch 
              checked={options.darkMode} 
              onChange={() => handleOptionToggle('darkMode')} 
              size="small"
            />
          }
          label="Dark Mode"
        />
        <FormControlLabel
          control={
            <Switch 
              checked={options.showVolume} 
              onChange={() => handleOptionToggle('showVolume')} 
              size="small"
            />
          }
          label="Show Volume"
        />
        <FormControlLabel
          control={
            <Switch 
              checked={options.showGrid} 
              onChange={() => handleOptionToggle('showGrid')} 
              size="small"
            />
          }
          label="Show Grid"
        />
      </Box>
      
      <Divider sx={{ my: 1.5 }} />
      
      <Box sx={{ mb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
          <Typography variant="subtitle2">Confluence Highlighting</Typography>
          <FormControlLabel
            control={
              <Switch 
                checked={options.showConfluence} 
                onChange={() => handleOptionToggle('showConfluence')} 
                size="small"
              />
            }
            label="Show"
          />
        </Box>
        
        {options.showConfluence && (
          <>
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" gutterBottom>
                Minimum Confluence Strength: {(options.confluenceThreshold * 100).toFixed(0)}%
              </Typography>
              <Slider
                value={options.confluenceThreshold}
                min={0}
                max={1}
                step={0.05}
                onChange={handleThresholdChange}
                valueLabelDisplay="auto"
                valueLabelFormat={(v: number) => `${(v * 100).toFixed(0)}%`}
                disabled={!options.showConfluence}
              />
            </Box>
            
            <Typography variant="body2" gutterBottom>Confluence Types:</Typography>
            <ToggleButtonGroup
              size="small"
              value={activeConfluenceTypes}
              onChange={handleConfluenceTypeToggle}
              aria-label="confluence types"
              disabled={!options.showConfluence}
              sx={{ flexWrap: 'wrap' }}
              multiple
            >
              <Tooltip title="Support Levels">
                <ToggleButton value="support" aria-label="support">
                  <SupportIcon /> Support
                </ToggleButton>
              </Tooltip>
              <Tooltip title="Resistance Levels">
                <ToggleButton value="resistance" aria-label="resistance">
                  <ResistanceIcon /> Resistance
                </ToggleButton>
              </Tooltip>
              <Tooltip title="Harmonic Patterns">
                <ToggleButton value="harmonic_pattern" aria-label="harmonic patterns">
                  <PatternIcon /> Patterns
                </ToggleButton>
              </Tooltip>
              <Tooltip title="Moving Average Confluence">
                <ToggleButton value="ma_confluence" aria-label="ma confluence">
                  <MAIcon /> MA Confluence
                </ToggleButton>
              </Tooltip>
              <Tooltip title="Multi-Timeframe Confluence">
                <ToggleButton value="multi_timeframe" aria-label="multi-timeframe">
                  <MTFIcon /> MTF Confluence
                </ToggleButton>
              </Tooltip>
              <Tooltip title="Fibonacci Confluence">
                <ToggleButton value="fibonacci_confluence" aria-label="fibonacci">
                  Fibonacci
                </ToggleButton>
              </Tooltip>
            </ToggleButtonGroup>
          </>
        )}
      </Box>
      
      {options.showConfluence && activeConfluenceTypes.length > 0 && (
        <Box sx={{ mt: 1 }}>
          <Typography variant="body2" color="text.secondary">
            Current confluences:
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 0.5 }}>
            {activeConfluenceTypes.map(type => (
              <Chip
                key={type}
                label={type.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
                size="small"
                color={
                  type === 'support' ? 'success' :
                  type === 'resistance' ? 'error' :
                  type === 'harmonic_pattern' ? 'secondary' :
                  type === 'ma_confluence' ? 'primary' :
                  type === 'multi_timeframe' ? 'warning' : 'default'
                }
                variant="outlined"
              />
            ))}
          </Box>
        </Box>
      )}
    </Paper>
  );
};

export default ChartControlPanel;
