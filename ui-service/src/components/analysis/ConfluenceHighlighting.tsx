/**
 * ConfluenceHighlighting Component - Highlights zones where multiple indicators and patterns align
 */
import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  Slider,
  FormGroup,
  FormControlLabel,
  Switch,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Divider,
  styled,
  Grid,
  Tooltip,
  Badge,
  Card,
  CardContent,
  IconButton
} from '@mui/material';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import { alpha, Theme } from '@mui/material/styles';

interface ConfluenceHighlightingProps {
  chartId: string;
  symbol: string;
  timeframe: string;
  confluenceData?: ConfluenceData;
  isEnabled: boolean;
  onToggle: (enabled: boolean) => void;
  onIntensityChange: (intensity: number) => void;
}

interface ConfluenceZone {
  id: string;
  price: number;
  strength: number;
  type: string;
  sources: string[];
  startTime?: string;
  endTime?: string;
  description?: string;
  color?: string;
}

interface ConfluenceData {
  zones: ConfluenceZone[];
  supportResistance?: {
    levels: { price: number; strength: number; type: string }[];
  };
  movingAverages?: {
    confluences: { price: number; timeframes: string[]; type: string }[];
  };
  fibonacci?: {
    levels: { price: number; ratio: string; strength: number }[];
  };
  patterns?: {
    harmonic: { price: number; patternType: string; completion: number }[];
    elliott: { price: number; waveType: string; confidence: number }[];
  };
}

const StyledPaper = styled(Paper)(({ theme }: { theme: Theme }) => ({
  padding: theme.spacing(2),
  backgroundColor: alpha(theme.palette.background.paper, 0.8),
  backdropFilter: 'blur(8px)',
  border: `1px solid ${theme.palette.divider}`,
}));

const StrengthIndicator = styled(Box)<{ strength: number }>(({ theme, strength }) => {
  // Calculate color based on strength (0-1)
  const getColorForStrength = () => {
    if (strength > 0.8) return theme.palette.success.main;
    if (strength > 0.6) return theme.palette.success.light;
    if (strength > 0.4) return theme.palette.warning.main;
    if (strength > 0.2) return theme.palette.warning.light;
    return theme.palette.error.light;
  };

  return {
    width: '100%',
    height: 8,
    borderRadius: 4,
    backgroundColor: theme.palette.grey[300],
    position: 'relative',
    overflow: 'hidden',
    '&::after': {
      content: '""',
      position: 'absolute',
      top: 0,
      left: 0,
      width: `${strength * 100}%`,
      height: '100%',
      backgroundColor: getColorForStrength(),
      transition: 'width 0.3s ease'
    }
  };
});

export default function ConfluenceHighlighting({
  chartId,
  symbol,
  timeframe,
  confluenceData,
  isEnabled,
  onToggle,
  onIntensityChange
}: ConfluenceHighlightingProps) {
  const [intensity, setIntensity] = useState<number>(50);
  const [confluenceTypeFilter, setConfluenceTypeFilter] = useState<string>('all');
  const [highlightedZones, setHighlightedZones] = useState<ConfluenceZone[]>([]);
  const [visibleSourceTypes, setVisibleSourceTypes] = useState<string[]>([
    'support-resistance', 'fibonacci', 'pattern', 'moving-average', 'volume'
  ]);
  const [selectedZone, setSelectedZone] = useState<ConfluenceZone | null>(null);
  
  const chartRef = useRef<HTMLElement | null>(null);
  const highlightsRef = useRef<Map<string, any>>(new Map());
  
  // Connect to chart and apply highlights when enabled
  useEffect(() => {
    try {
      // In a real implementation, this would use a proper chart library API
      // This is simplified for demonstration
      chartRef.current = document.getElementById(chartId);
      
      // Clean up old highlights if any
      removeAllHighlights();
      
      // Apply new highlights if enabled
      if (isEnabled && confluenceData) {
        applyHighlights();
      }
    } catch (error) {
      console.error('Error connecting to chart:', error);
    }
    
    return () => {
      // Clean up highlights when component unmounts
      removeAllHighlights();
    };
  }, [chartId, isEnabled, confluenceData, intensity, confluenceTypeFilter, visibleSourceTypes]);
  
  // Filter and process confluence zones based on intensity threshold and type filter
  useEffect(() => {
    if (!confluenceData?.zones) {
      setHighlightedZones([]);
      return;
    }
    
    // Filter zones based on strength threshold (intensity)
    const strengthThreshold = intensity / 100;
    const filteredZones = confluenceData.zones.filter(zone => {
      // Filter by strength
      if (zone.strength < strengthThreshold) return false;
      
      // Filter by type if not 'all'
      if (confluenceTypeFilter !== 'all' && zone.type !== confluenceTypeFilter) return false;
      
      // Filter by source types
      if (visibleSourceTypes.length > 0) {
        // Check if any of the zone's sources match the visible types
        // This is a simplified check - in real implementation you'd have proper source categorization
        const hasVisibleSource = zone.sources.some(source => {
          if (visibleSourceTypes.includes('support-resistance') && 
              source.toLowerCase().includes('support') || source.toLowerCase().includes('resistance')) {
            return true;
          }
          if (visibleSourceTypes.includes('fibonacci') && source.toLowerCase().includes('fib')) {
            return true;
          }
          if (visibleSourceTypes.includes('pattern') && 
              (source.toLowerCase().includes('pattern') || source.toLowerCase().includes('harmonic'))) {
            return true;
          }
          if (visibleSourceTypes.includes('moving-average') && source.toLowerCase().includes('ma')) {
            return true;
          }
          if (visibleSourceTypes.includes('volume') && source.toLowerCase().includes('volume')) {
            return true;
          }
          return false;
        });
        
        if (!hasVisibleSource) return false;
      }
      
      return true;
    });
    
    setHighlightedZones(filteredZones);
  }, [confluenceData, intensity, confluenceTypeFilter, visibleSourceTypes]);
  
  // Apply visual highlights to the chart
  const applyHighlights = () => {
    // In a real implementation, this would use the chart library's API to draw the highlights
    // For this demo, we'll log what would be highlighted
    console.log(`Applying ${highlightedZones.length} confluence zones to chart ${chartId}`);
    
    // Clear existing highlights first
    removeAllHighlights();
    
    // Apply new highlights
    highlightedZones.forEach(zone => {
      // Create a highlight element for the zone
      const highlight = createHighlightElement(zone);
      highlightsRef.current.set(zone.id, highlight);
    });
  };
  
  // Create a visual highlight element for a confluence zone
  const createHighlightElement = (zone: ConfluenceZone) => {
    // In a real implementation, this would create an actual chart overlay
    // For this demo, we just return the zone data that would be used
    return {
      zoneId: zone.id,
      price: zone.price,
      strength: zone.strength,
      color: getColorForZone(zone),
      // This would be an actual DOM element or chart object in production
      remove: () => console.log(`Removing highlight for zone ${zone.id}`)
    };
  };
  
  // Remove all highlights from the chart
  const removeAllHighlights = () => {
    highlightsRef.current.forEach(highlight => {
      if (highlight.remove) {
        highlight.remove();
      }
    });
    highlightsRef.current.clear();
  };
  
  // Get the appropriate color for a confluence zone based on type and strength
  const getColorForZone = (zone: ConfluenceZone) => {
    const isSupport = zone.type === 'support';
    
    // Base colors for support and resistance
    const baseColor = isSupport ? '#4caf50' : '#f44336';
    
    // Adjust opacity based on strength
    const opacity = 0.3 + (zone.strength * 0.7);
    
    // Convert to rgba
    const rgb = hexToRgb(zone.color || baseColor);
    return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${opacity})`;
  };
  
  // Helper function to convert hex color to RGB
  const hexToRgb = (hex: string) => {
    // Default to a neutral color if conversion fails
    const defaultRgb = { r: 100, g: 100, b: 100 };
    
    // Check if the hex color is valid
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    if (!result) return defaultRgb;
    
    return {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16)
    };
  };
  
  // Handle intensity slider change
  const handleIntensityChange = (_event: Event, newValue: number | number[]) => {
    const newIntensity = Array.isArray(newValue) ? newValue[0] : newValue;
    setIntensity(newIntensity);
    onIntensityChange(newIntensity);
  };
  
  // Handle toggle change
  const handleToggle = (event: React.ChangeEvent<HTMLInputElement>) => {
    onToggle(event.target.checked);
  };
  
  // Handle confluence type filter change
  const handleTypeFilterChange = (event: React.ChangeEvent<{ value: unknown }>) => {
    setConfluenceTypeFilter(event.target.value as string);
  };
  
  // Toggle visibility of a source type
  const toggleSourceTypeVisibility = (sourceType: string) => {
    if (visibleSourceTypes.includes(sourceType)) {
      setVisibleSourceTypes(visibleSourceTypes.filter(type => type !== sourceType));
    } else {
      setVisibleSourceTypes([...visibleSourceTypes, sourceType]);
    }
  };
  
  // Handle zone selection
  const handleZoneClick = (zone: ConfluenceZone) => {
    setSelectedZone(zone === selectedZone ? null : zone);
  };

  return (
    <StyledPaper elevation={3}>
      <Typography variant="subtitle1" gutterBottom fontWeight="bold">
        Confluence Highlighting
      </Typography>
      
      <FormGroup>
        <FormControlLabel
          control={<Switch checked={isEnabled} onChange={handleToggle} />}
          label="Enable Confluence Highlighting"
        />
      </FormGroup>
      
      {isEnabled && (
        <>
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2" gutterBottom>
              Highlight Intensity: {intensity}%
            </Typography>
            <Slider
              value={intensity}
              onChange={handleIntensityChange}
              aria-label="Confluence Intensity"
              valueLabelDisplay="auto"
            />
          </Box>
          
          <Box sx={{ mt: 2 }}>
            <FormControl fullWidth size="small">
              <InputLabel>Confluence Type</InputLabel>
              <Select
                value={confluenceTypeFilter}
                onChange={handleTypeFilterChange}
                label="Confluence Type"
              >
                <MenuItem value="all">All Types</MenuItem>
                <MenuItem value="support">Support</MenuItem>
                <MenuItem value="resistance">Resistance</MenuItem>
                <MenuItem value="decision">Decision</MenuItem>
                <MenuItem value="reversal">Reversal</MenuItem>
                <MenuItem value="continuation">Continuation</MenuItem>
              </Select>
            </FormControl>
          </Box>
          
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2" gutterBottom>
              Source Types:
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              <Chip
                icon={visibleSourceTypes.includes('support-resistance') ? <VisibilityIcon /> : <VisibilityOffIcon />}
                label="S/R Levels"
                onClick={() => toggleSourceTypeVisibility('support-resistance')}
                color={visibleSourceTypes.includes('support-resistance') ? 'primary' : 'default'}
                variant={visibleSourceTypes.includes('support-resistance') ? 'filled' : 'outlined'}
              />
              <Chip
                icon={visibleSourceTypes.includes('fibonacci') ? <VisibilityIcon /> : <VisibilityOffIcon />}
                label="Fibonacci"
                onClick={() => toggleSourceTypeVisibility('fibonacci')}
                color={visibleSourceTypes.includes('fibonacci') ? 'primary' : 'default'}
                variant={visibleSourceTypes.includes('fibonacci') ? 'filled' : 'outlined'}
              />
              <Chip
                icon={visibleSourceTypes.includes('pattern') ? <VisibilityIcon /> : <VisibilityOffIcon />}
                label="Patterns"
                onClick={() => toggleSourceTypeVisibility('pattern')}
                color={visibleSourceTypes.includes('pattern') ? 'primary' : 'default'}
                variant={visibleSourceTypes.includes('pattern') ? 'filled' : 'outlined'}
              />
              <Chip
                icon={visibleSourceTypes.includes('moving-average') ? <VisibilityIcon /> : <VisibilityOffIcon />}
                label="Moving Avgs"
                onClick={() => toggleSourceTypeVisibility('moving-average')}
                color={visibleSourceTypes.includes('moving-average') ? 'primary' : 'default'}
                variant={visibleSourceTypes.includes('moving-average') ? 'filled' : 'outlined'}
              />
              <Chip
                icon={visibleSourceTypes.includes('volume') ? <VisibilityIcon /> : <VisibilityOffIcon />}
                label="Volume"
                onClick={() => toggleSourceTypeVisibility('volume')}
                color={visibleSourceTypes.includes('volume') ? 'primary' : 'default'}
                variant={visibleSourceTypes.includes('volume') ? 'filled' : 'outlined'}
              />
            </Box>
          </Box>
          
          <Divider sx={{ my: 2 }} />
          
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2" gutterBottom>
              Active Confluence Zones:
            </Typography>
            
            {highlightedZones.length === 0 ? (
              <Typography variant="body2" color="text.secondary">
                No confluence zones match the current filters
              </Typography>
            ) : (
              <Grid container spacing={1}>
                {highlightedZones.map(zone => (
                  <Grid item xs={12} key={zone.id}>
                    <Card 
                      variant="outlined" 
                      onClick={() => handleZoneClick(zone)}
                      sx={{ 
                        cursor: 'pointer',
                        mb: 1,
                        borderLeft: `4px solid ${getColorForZone(zone)}`,
                        backgroundColor: zone === selectedZone ? 'action.hover' : 'inherit'
                      }}
                    >
                      <CardContent sx={{ p: 1, '&:last-child': { pb: 1 } }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Box>
                            <Typography variant="body2" fontWeight="medium">
                              {zone.price.toFixed(5)} - {zone.type.toUpperCase()}
                            </Typography>
                            
                            <Box sx={{ mt: 0.5 }}>
                              <StrengthIndicator strength={zone.strength} />
                            </Box>
                          </Box>
                          
                          <Badge badgeContent={zone.sources.length} color="primary">
                            <InfoOutlinedIcon color="action" />
                          </Badge>
                        </Box>
                        
                        {zone === selectedZone && (
                          <Box sx={{ mt: 1 }}>
                            <Typography variant="caption" color="text.secondary">
                              Sources:
                            </Typography>
                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                              {zone.sources.map((source, i) => (
                                <Chip key={i} label={source} size="small" />
                              ))}
                            </Box>
                            
                            {zone.description && (
                              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
                                {zone.description}
                              </Typography>
                            )}
                          </Box>
                        )}
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            )}
          </Box>
        </>
      )}
    </StyledPaper>
  );
}
