/**
 * Advanced Chart component with pattern visualization
 * Provides multi-timeframe analysis and confluence highlighting
 * Optimized version with WebGL acceleration and memory management
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  ButtonGroup,
  Button,
  CircularProgress,
  ToggleButtonGroup,
  ToggleButton,
  Tooltip,
  FormControlLabel,
  Switch,
  Grid,
} from '@mui/material';
import { createChart, IChartApi, ISeriesApi, LineStyle, Time } from 'lightweight-charts';
import { TimeFrame } from '@/types/strategy';
import { useChartData, OHLCData } from '@/context/ChartDataContext';
import {
  useWebGLRenderer,
  useChartMemoryManagement,
  useChartResize,
  useChartVisibility,
  PatternMarker,
  useOptimizedPatternRendering
} from '@/hooks/useChartOptimization';

interface PatternMarker {
  id: string;
  type: string;
  startTime: Time;
  endTime: Time;
  startPrice: number;
  endPrice: number;
  description: string;
  confidence: number;
  color: string;
}

interface IndicatorData {
  id: string;
  name: string;
  data: Array<{
    time: Time;
    value: number;
  }>;
  color: string;
}

interface SupportResistanceLevel {
  id: string;
  price: number;
  strength: number;
  description: string;
  startTime: Time;
  endTime: Time;
}

interface ChartAnnotation {
  id: string;
  type: 'text' | 'arrow' | 'highlight';
  text?: string;
  time: Time;
  price: number;
  color: string;
}

interface ConfluenceZone {
  id: string;
  startPrice: number;
  endPrice: number;
  time: Time;
  strength: number;
  description: string;
  tags: string[];
}

interface AdvancedChartProps {
  symbol: string;
  onTimeframeChange?: (timeframe: TimeFrame) => void;
  height?: number;
  initialTimeframe?: TimeFrame;
  showVolume?: boolean;
  enableMultiTimeframe?: boolean;
  enablePatternDetection?: boolean;
  enableConfluenceHighlighting?: boolean;
  enableElliottWaveOverlays?: boolean;
}

export default function AdvancedChart({
  symbol,
  onTimeframeChange,
  height = 500,
  initialTimeframe = TimeFrame.H1,
  showVolume = true,
  enableMultiTimeframe = true,
  enablePatternDetection = true,
  enableConfluenceHighlighting = true,
  enableElliottWaveOverlays = true,
}: AdvancedChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const patternMarkersRef = useRef<Map<string, any>>(new Map());
  const supportResistanceRef = useRef<Map<string, any>>(new Map());
  const indicatorSeriesRef = useRef<Map<string, ISeriesApi<'Line'>>>(new Map());
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);

  const [loading, setLoading] = useState<boolean>(true);
  const [timeframe, setTimeframe] = useState<TimeFrame>(initialTimeframe);
  const [patternMarkers, setPatternMarkers] = useState<PatternMarker[]>([]);
  const [indicators, setIndicators] = useState<IndicatorData[]>([]);
  const [supportResistanceLevels, setSupportResistanceLevels] = useState<SupportResistanceLevel[]>([]);
  const [annotations, setAnnotations] = useState<ChartAnnotation[]>([]);
  const [confluenceZones, setConfluenceZones] = useState<ConfluenceZone[]>([]);
  const [selectedPatternTypes, setSelectedPatternTypes] = useState<string[]>(['all']);
  const [showAnnotations, setShowAnnotations] = useState<boolean>(true);
  const [legendValues, setLegendValues] = useState({ price: 0, time: '' });

  // Selected comparative timeframes for multi-timeframe analysis
  const [compareTimeframes, setCompareTimeframes] = useState<TimeFrame[]>([]);

  // Use chart data context
  const { loadData, dataByTimeframe } = useChartData();

  // Check WebGL support
  const isWebGLSupported = useWebGLRenderer();

  // Check if chart is visible in viewport
  const isVisible = useChartVisibility(chartContainerRef);

  // Memory management
  const { releaseMemory, restoreChart } = useChartMemoryManagement(chartRef);

  // Handle resize
  useChartResize(chartRef, chartContainerRef);

  // Mock price data for demonstration
  const generateMockPriceData = (count: number): OHLCData[] => {
    const data: OHLCData[] = [];
    const basePrice = 1.2000;
    let lastClose = basePrice;

    const now = new Date();

    for (let i = count - 1; i >= 0; i--) {
      const time = new Date(now);

      // Adjust time based on timeframe
      switch (timeframe) {
        case TimeFrame.M1:
          time.setMinutes(now.getMinutes() - i);
          break;
        case TimeFrame.M5:
          time.setMinutes(now.getMinutes() - i * 5);
          break;
        case TimeFrame.M15:
          time.setMinutes(now.getMinutes() - i * 15);
          break;
        case TimeFrame.M30:
          time.setMinutes(now.getMinutes() - i * 30);
          break;
        case TimeFrame.H1:
          time.setHours(now.getHours() - i);
          break;
        case TimeFrame.H4:
          time.setHours(now.getHours() - i * 4);
          break;
        case TimeFrame.D1:
          time.setDate(now.getDate() - i);
          break;
        case TimeFrame.W1:
          time.setDate(now.getDate() - i * 7);
          break;
      }

      // Generate random price action
      const change = (Math.random() - 0.5) * 0.005; // Random price change
      const open = lastClose;
      const close = open + change;
      const high = Math.max(open, close) + Math.random() * 0.002;
      const low = Math.min(open, close) - Math.random() * 0.002;
      const volume = Math.floor(Math.random() * 1000) + 500;

      data.push({
        time: Math.floor(time.getTime() / 1000) as Time,
        open,
        high,
        low,
        close,
        volume
      });

      lastClose = close;
    }

    return data;
  };

  // Load data when component mounts or timeframe changes
  useEffect(() => {
    if (!isVisible) return;

    const loadChartData = async () => {
      setLoading(true);

      try {
        // Use shared data context to load data
        const data = await loadData(symbol, timeframe);

        // Generate pattern data
        if (enablePatternDetection) {
          setPatternMarkers(generateMockPatternMarkers(data));
        }

        if (enableConfluenceHighlighting) {
          setConfluenceZones(generateMockConfluenceZones(data));
        }

        setSupportResistanceLevels(generateMockSupportResistance(data));
        setIndicators(generateMockIndicators(data));
      } catch (error) {
        console.error('Failed to load chart data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadChartData();
  }, [symbol, timeframe, enablePatternDetection, enableConfluenceHighlighting, isVisible, loadData]);

  // Initialize chart when component mounts
  useEffect(() => {
    if (!chartContainerRef.current || !isVisible) return;

    // Create chart with WebGL acceleration if supported
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: height,
      layout: {
        background: { color: '#1E222D' },
        textColor: '#DDD',
      },
      grid: {
        vertLines: { color: '#2B2B43' },
        horzLines: { color: '#2B2B43' },
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        borderColor: '#2B2B43',
      },
      crosshair: {
        mode: 1,
        vertLine: {
          labelVisible: false,
        },
      },
      rightPriceScale: {
        borderColor: '#2B2B43',
      },
      // Use WebGL for rendering if supported
      renderer: isWebGLSupported ? 'webgl' : 'canvas',
    });

    // Add candlestick series
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
    });

    // Add volume series if enabled
    let volumeSeries = null;
    if (showVolume) {
      volumeSeries = chart.addHistogramSeries({
        color: '#182233',
        lineWidth: 2,
        priceFormat: {
          type: 'volume',
        },
        priceScaleId: 'volume',
        scaleMargins: {
          top: 0.8,
          bottom: 0,
        },
      });
      volumeSeriesRef.current = volumeSeries;
    }

    // Setup crosshair move handler for legend
    chart.subscribeCrosshairMove((param) => {
      if (param.time && param.point) {
        const data = param.seriesData.get(candlestickSeries) as OHLCData;
        if (data) {
          const dateStr = new Date((data.time as number) * 1000).toLocaleString();
          setLegendValues({
            price: data.close,
            time: dateStr,
          });
        }
      }
    });

    // Store refs for later use
    chartRef.current = chart;
    candlestickSeriesRef.current = candlestickSeries;

    // Cleanup function
    return () => {
      chart.remove();
      chartRef.current = null;
      candlestickSeriesRef.current = null;
      volumeSeriesRef.current = null;
      indicatorSeriesRef.current.clear();
    };
  }, [height, showVolume, isVisible, isWebGLSupported]);

  // Update chart when data changes in the context
  useEffect(() => {
    if (!candlestickSeriesRef.current || !isVisible) return;

    const data = dataByTimeframe[timeframe];
    if (!data || data.length === 0) return;

    // Update candlesticks
    candlestickSeriesRef.current.setData(data);

    // Update volume if available
    if (volumeSeriesRef.current) {
      const volumeData = data.map(d => ({
        time: d.time,
        value: d.volume || 0,
        color: d.close >= d.open ? 'rgba(38, 166, 154, 0.5)' : 'rgba(239, 83, 80, 0.5)',
      }));
      volumeSeriesRef.current.setData(volumeData);
    }

    // Fit chart content
    if (chartRef.current) {
      chartRef.current.timeScale().fitContent();
    }
  }, [dataByTimeframe, timeframe, isVisible]);

  // Helper function to draw pattern on chart
  const drawPatternOnChart = useCallback((chart: IChartApi, pattern: PatternMarker) => {
    // Create a simple line
    const lineSeries = chart.addLineSeries({
      color: pattern.color,
      lineWidth: 2,
      lineStyle: LineStyle.Dashed,
    });

    lineSeries.setData([
      { time: pattern.startTime, value: pattern.startPrice },
      { time: pattern.endTime, value: pattern.endPrice }
    ]);

    return lineSeries;
  }, []);

  // Use optimized pattern rendering
  const renderedPatterns = useOptimizedPatternRendering(
    chartRef,
    selectedPatternTypes.includes('all')
      ? patternMarkers
      : patternMarkers.filter(p => selectedPatternTypes.includes(p.type)),
    isVisible,
    drawPatternOnChart
  );

  // Update support/resistance levels with optimization
  useEffect(() => {
    if (!chartRef.current || !isVisible) return;

    // Clear existing levels
    supportResistanceRef.current.forEach(level => {
      if (level.remove) {
        level.remove();
      }
    });
    supportResistanceRef.current.clear();

    // Add support/resistance levels
    supportResistanceLevels.forEach(level => {
      const srLine = drawSupportResistanceLine(level);
      if (srLine) {
        supportResistanceRef.current.set(level.id, srLine);
      }
    });

    return () => {
      supportResistanceRef.current.forEach(level => {
        if (level.remove) {
          level.remove();
        }
      });
    };
  }, [supportResistanceLevels, isVisible, drawSupportResistanceLine]);

  // Update indicators with optimization
  useEffect(() => {
    if (!chartRef.current || !isVisible) return;

    // Remove old indicators
    indicatorSeriesRef.current.forEach((series) => {
      chartRef.current?.removeSeries(series);
    });
    indicatorSeriesRef.current.clear();

    // Add new indicators
    indicators.forEach(indicator => {
      const series = chartRef.current?.addLineSeries({
        color: indicator.color,
        lineWidth: 1,
        priceLineVisible: false,
      });

      if (series) {
        series.setData(indicator.data);
        indicatorSeriesRef.current.set(indicator.id, series);
      }
    });

    return () => {
      indicatorSeriesRef.current.forEach((series) => {
        chartRef.current?.removeSeries(series);
      });
    };
  }, [indicators, isVisible]);

  // Update confluence zones with optimization
  useEffect(() => {
    if (!chartRef.current || !isVisible || confluenceZones.length === 0) return;

    // Render confluence zones
    // In a real implementation, would use advanced rendering
    // For this demo, just log the zones
    console.log('Rendering confluence zones:', confluenceZones);
  }, [confluenceZones, isVisible]);



  // Helper function to draw support/resistance
  const drawSupportResistanceLine = useCallback((level: SupportResistanceLevel) => {
    if (!chartRef.current || !isVisible) return null;

    // Create horizontal line for S/R level
    const lineSeries = chartRef.current.addLineSeries({
      color: level.strength > 0.7 ? '#f48fb1' : '#81c784',
      lineWidth: Math.max(1, Math.floor(level.strength * 3)),
      lineStyle: LineStyle.Solid,
      lastValueVisible: false,
      priceLineVisible: false,
    });

    // Create data for horizontal line
    const data = [];

    // Add points from start to end time
    data.push({ time: level.startTime, value: level.price });
    data.push({ time: level.endTime, value: level.price });

    lineSeries.setData(data);

    return lineSeries;
  }, [chartRef, isVisible]);

  // Generate mock pattern markers
  const generateMockPatternMarkers = (data: OHLCData[]): PatternMarker[] => {
    if (data.length < 20) return [];

    const patterns = [];
    const patternTypes = ['Head and Shoulders', 'Double Top', 'Triangle', 'Flag', 'Elliott Wave'];
    const colors = ['#f48fb1', '#81c784', '#64b5f6', '#ffb74d', '#ba68c8'];

    // Generate 3-5 random patterns
    const patternCount = 3 + Math.floor(Math.random() * 3);

    for (let i = 0; i < patternCount; i++) {
      const startIndex = 10 + Math.floor(Math.random() * (data.length - 30));
      const endIndex = startIndex + 5 + Math.floor(Math.random() * 10);

      if (endIndex >= data.length) continue;

      const patternTypeIndex = Math.floor(Math.random() * patternTypes.length);

      patterns.push({
        id: `pattern-${i}`,
        type: patternTypes[patternTypeIndex],
        startTime: data[startIndex].time,
        endTime: data[endIndex].time,
        startPrice: data[startIndex].close,
        endPrice: data[endIndex].close,
        description: `${patternTypes[patternTypeIndex]} pattern detected`,
        confidence: 0.6 + Math.random() * 0.3,
        color: colors[patternTypeIndex],
      });
    }

    return patterns;
  };

  // Generate mock support/resistance levels
  const generateMockSupportResistance = (data: OHLCData[]): SupportResistanceLevel[] => {
    if (data.length < 10) return [];

    const levels = [];
    const minPrice = Math.min(...data.map(d => d.low));
    const maxPrice = Math.max(...data.map(d => d.high));
    const priceRange = maxPrice - minPrice;

    // Generate 2-4 S/R levels
    const levelCount = 2 + Math.floor(Math.random() * 3);

    for (let i = 0; i < levelCount; i++) {
      const levelPrice = minPrice + Math.random() * priceRange;
      const strength = 0.5 + Math.random() * 0.5;

      levels.push({
        id: `sr-${i}`,
        price: levelPrice,
        strength: strength,
        description: strength > 0.7 ? 'Strong resistance' : 'Support level',
        startTime: data[0].time,
        endTime: data[data.length - 1].time,
      });
    }

    return levels;
  };

  // Generate mock indicators
  const generateMockIndicators = (data: OHLCData[]): IndicatorData[] => {
    if (data.length === 0) return [];

    const indicators = [];

    // Generate MA indicator
    const ma1Data = calculateMovingAverage(data, 20);
    const ma2Data = calculateMovingAverage(data, 50);

    indicators.push({
      id: 'ma-20',
      name: 'MA (20)',
      data: ma1Data,
      color: '#64b5f6',
    });

    indicators.push({
      id: 'ma-50',
      name: 'MA (50)',
      data: ma2Data,
      color: '#ff8a65',
    });

    return indicators;
  };

  // Helper function to calculate moving average
  const calculateMovingAverage = (data: OHLCData[], period: number) => {
    const result = [];

    for (let i = period - 1; i < data.length; i++) {
      let sum = 0;
      for (let j = 0; j < period; j++) {
        sum += data[i - j].close;
      }

      result.push({
        time: data[i].time,
        value: sum / period,
      });
    }

    return result;
  };

  // Generate mock confluence zones
  const generateMockConfluenceZones = (data: OHLCData[]): ConfluenceZone[] => {
    if (data.length < 20) return [];

    const zones = [];
    const minPrice = Math.min(...data.map(d => d.low));
    const maxPrice = Math.max(...data.map(d => d.high));
    const priceRange = maxPrice - minPrice;

    // Generate 2-3 confluence zones
    const zoneCount = 2 + Math.floor(Math.random() * 2);

    for (let i = 0; i < zoneCount; i++) {
      const timeIndex = 20 + Math.floor(Math.random() * (data.length - 40));
      const zoneWidth = priceRange * (0.01 + Math.random() * 0.02);
      const centerPrice = minPrice + Math.random() * priceRange;

      zones.push({
        id: `zone-${i}`,
        startPrice: centerPrice - zoneWidth/2,
        endPrice: centerPrice + zoneWidth/2,
        time: data[timeIndex].time,
        strength: 0.6 + Math.random() * 0.4,
        description: 'Multiple indicators confirming this zone',
        tags: ['Fibonacci', 'Support/Resistance', 'Pattern'],
      });
    }

    return zones;
  };

  const handleTimeframeChange = (newTimeframe: TimeFrame) => {
    setTimeframe(newTimeframe);
    if (onTimeframeChange) {
      onTimeframeChange(newTimeframe);
    }
  };

  const handlePatternFilterChange = (_event: React.MouseEvent<HTMLElement>, newPatternTypes: string[]) => {
    // Prevent deselecting all patterns
    if (newPatternTypes.length === 0) {
      return;
    }
    setSelectedPatternTypes(newPatternTypes);
  };

  const handleCompareTimeframeChange = (_event: React.MouseEvent<HTMLElement>, newTimeframes: TimeFrame[]) => {
    setCompareTimeframes(newTimeframes);
    // In a real implementation, would fetch data for these timeframes
    console.log('Comparing timeframes:', newTimeframes);
  };

  return (
    <Paper sx={{ p: 0, height: height + 80, position: 'relative' }}>
      {/* Chart header with controls */}
      <Box sx={{ px: 2, py: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h6" component="h2">
          {symbol} - {timeframe}
        </Typography>

        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          {/* Legend values */}
          <Box sx={{ mr: 2, fontSize: '0.875rem' }}>
            <Typography component="span" variant="body2" mr={1}>
              Price: <strong>{legendValues.price.toFixed(4)}</strong>
            </Typography>
            <Typography component="span" variant="body2">
              Time: <strong>{legendValues.time}</strong>
            </Typography>
          </Box>

          {/* Timeframe selector */}
          <ButtonGroup size="small" aria-label="timeframe selector">
            {Object.values(TimeFrame).map((tf) => (
              <Button
                key={tf}
                variant={timeframe === tf ? 'contained' : 'outlined'}
                onClick={() => handleTimeframeChange(tf)}
              >
                {tf}
              </Button>
            ))}
          </ButtonGroup>
        </Box>
      </Box>

      {/* Chart Container */}
      <Box sx={{ position: 'relative' }}>
        <Box ref={chartContainerRef} style={{ height: `${height}px` }} />

        {/* Loading indicator */}
        {loading && (
          <Box sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: 'rgba(0,0,0,0.5)',
          }}>
            <CircularProgress />
          </Box>
        )}
      </Box>

      {/* Bottom controls */}
      <Box sx={{ p: 1, borderTop: '1px solid rgba(255,255,255,0.1)' }}>
        <Grid container spacing={2}>
          {/* Pattern type filter */}
          {enablePatternDetection && (
            <Grid item xs={12} md={6}>
              <Typography variant="body2" gutterBottom>
                Pattern Types:
              </Typography>
              <ToggleButtonGroup
                size="small"
                value={selectedPatternTypes}
                onChange={handlePatternFilterChange}
                aria-label="pattern types"
              >
                <ToggleButton value="all">All</ToggleButton>
                <ToggleButton value="Head and Shoulders">H&S</ToggleButton>
                <ToggleButton value="Double Top">Double Top</ToggleButton>
                <ToggleButton value="Triangle">Triangle</ToggleButton>
                <ToggleButton value="Flag">Flag</ToggleButton>
                <ToggleButton value="Elliott Wave">Elliott Wave</ToggleButton>
              </ToggleButtonGroup>
            </Grid>
          )}

          {/* Multi-timeframe comparison */}
          {enableMultiTimeframe && (
            <Grid item xs={12} md={6}>
              <Typography variant="body2" gutterBottom>
                Compare Timeframes:
              </Typography>
              <ToggleButtonGroup
                size="small"
                value={compareTimeframes}
                onChange={handleCompareTimeframeChange}
                aria-label="compare timeframes"
              >
                <ToggleButton value={TimeFrame.M15}>M15</ToggleButton>
                <ToggleButton value={TimeFrame.H1}>H1</ToggleButton>
                <ToggleButton value={TimeFrame.H4}>H4</ToggleButton>
                <ToggleButton value={TimeFrame.D1}>D1</ToggleButton>
              </ToggleButtonGroup>
            </Grid>
          )}

          {/* Other controls */}
          <Grid item xs={12}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Box>
                <FormControlLabel
                  control={
                    <Switch
                      checked={showAnnotations}
                      onChange={(e) => setShowAnnotations(e.target.checked)}
                      size="small"
                    />
                  }
                  label="Show Annotations"
                />
              </Box>

              {enableElliottWaveOverlays && (
                <Tooltip title="Enable Elliott Wave Overlays">
                  <Button size="small" variant="outlined">
                    Elliott Wave
                  </Button>
                </Tooltip>
              )}
            </Box>
          </Grid>
        </Grid>
      </Box>
    </Paper>
  );
}
