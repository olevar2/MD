/**
 * VirtualizedChart.tsx
 * Optimized chart component that only renders when visible
 * Uses shared data context and WebGL acceleration
 */
import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Box,
  CircularProgress,
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

interface SupportResistanceLevel {
  id: string;
  price: number;
  strength: number;
  description: string;
  startTime: Time;
  endTime: Time;
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

interface VirtualizedChartProps {
  symbol: string;
  timeframe: TimeFrame;
  height?: number;
  isVisible?: boolean;
  showVolume?: boolean;
  enablePatternDetection?: boolean;
  enableConfluenceHighlighting?: boolean;
  enableMultiTimeframe?: boolean;
  enableElliottWaveOverlays?: boolean;
  onTimeframeChange?: (timeframe: TimeFrame) => void;
  onChartReady?: (chart: IChartApi) => void;
  onCrosshairMove?: (param: any) => void;
}

export default function VirtualizedChart({
  symbol,
  timeframe,
  height = 500,
  isVisible = true,
  showVolume = true,
  enablePatternDetection = true,
  enableConfluenceHighlighting = true,
  enableMultiTimeframe = false,
  enableElliottWaveOverlays = false,
  onTimeframeChange,
  onChartReady,
  onCrosshairMove,
}: VirtualizedChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const indicatorSeriesRef = useRef<Map<string, ISeriesApi<'Line'>>>(new Map());
  
  const { loadData, isLoading } = useChartData();
  const [chartData, setChartData] = useState<OHLCData[]>([]);
  const [patternMarkers, setPatternMarkers] = useState<PatternMarker[]>([]);
  const [supportResistanceLevels, setSupportResistanceLevels] = useState<SupportResistanceLevel[]>([]);
  const [indicators, setIndicators] = useState<IndicatorData[]>([]);
  
  // Check if container is visible in viewport
  const containerIsVisible = useChartVisibility(chartContainerRef, (visible) => {
    if (visible && chartRef.current) {
      // Restore chart when becoming visible
      chartRef.current.timeScale().fitContent();
    }
  });
  
  // Determine if chart should be rendered
  const shouldRender = isVisible && (containerIsVisible || chartData.length > 0);
  
  // Check WebGL support
  const isWebGLSupported = useWebGLRenderer();
  
  // Memory management
  const { releaseMemory, restoreChart } = useChartMemoryManagement(chartRef);
  
  // Handle resize
  useChartResize(chartRef, chartContainerRef);
  
  // Load data when component mounts or timeframe changes
  useEffect(() => {
    if (!isVisible) return;
    
    const fetchData = async () => {
      const data = await loadData(symbol, timeframe);
      setChartData(data);
      
      // Generate mock data for patterns, indicators, etc.
      if (enablePatternDetection) {
        setPatternMarkers(generateMockPatternMarkers(data));
      }
      
      setSupportResistanceLevels(generateMockSupportResistance(data));
      setIndicators(generateMockIndicators(data));
    };
    
    fetchData();
  }, [symbol, timeframe, isVisible, loadData, enablePatternDetection]);
  
  // Initialize chart when component mounts
  useEffect(() => {
    if (!chartContainerRef.current || !shouldRender) return;
    
    // Create chart with WebGL acceleration if supported
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height,
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
    
    // Setup crosshair move handler
    if (onCrosshairMove) {
      chart.subscribeCrosshairMove(onCrosshairMove);
    }
    
    // Store refs for later use
    chartRef.current = chart;
    candlestickSeriesRef.current = candlestickSeries;
    
    // Notify parent when chart is ready
    if (onChartReady) {
      onChartReady(chart);
    }
    
    return () => {
      if (onCrosshairMove) {
        chart.unsubscribeCrosshairMove(onCrosshairMove);
      }
      chart.remove();
      chartRef.current = null;
      candlestickSeriesRef.current = null;
      volumeSeriesRef.current = null;
      indicatorSeriesRef.current.clear();
    };
  }, [height, shouldRender, showVolume, isWebGLSupported, onCrosshairMove, onChartReady]);
  
  // Update chart when data changes
  useEffect(() => {
    if (!candlestickSeriesRef.current || chartData.length === 0) return;
    
    // Update candlesticks
    candlestickSeriesRef.current.setData(chartData);
    
    // Update volume if available
    if (volumeSeriesRef.current) {
      const volumeData = chartData.map(d => ({
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
  }, [chartData]);
  
  // Update indicators
  useEffect(() => {
    if (!chartRef.current || indicators.length === 0) return;
    
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
  }, [indicators]);
  
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
  useOptimizedPatternRendering(chartRef, patternMarkers, shouldRender, drawPatternOnChart);
  
  // Helper function to draw support/resistance
  const drawSupportResistanceLine = useCallback((level: SupportResistanceLevel) => {
    if (!chartRef.current) return null;
    
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
  }, []);
  
  // Update support/resistance levels
  useEffect(() => {
    if (!chartRef.current || !shouldRender) return;
    
    // Clear existing levels
    const supportResistanceRef = new Map();
    
    // Add support/resistance levels
    supportResistanceLevels.forEach(level => {
      const srLine = drawSupportResistanceLine(level);
      if (srLine) {
        supportResistanceRef.set(level.id, srLine);
      }
    });
    
    return () => {
      supportResistanceRef.forEach(level => {
        if (level && chartRef.current) {
          chartRef.current.removeSeries(level);
        }
      });
    };
  }, [supportResistanceLevels, shouldRender, drawSupportResistanceLine]);
  
  // Render placeholder when not visible
  if (!isVisible) {
    return <Box style={{ height, width: '100%' }} />;
  }
  
  return (
    <Box sx={{ position: 'relative', height, width: '100%' }}>
      <Box ref={chartContainerRef} style={{ height: `${height}px`, width: '100%' }} />
      
      {/* Loading indicator */}
      {isLoading[timeframe] && (
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
  );
}

// Mock data generation functions
// These would be replaced with real data in a production environment

// Generate mock pattern markers
function generateMockPatternMarkers(data: OHLCData[]): PatternMarker[] {
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
}

// Generate mock support/resistance levels
function generateMockSupportResistance(data: OHLCData[]): SupportResistanceLevel[] {
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
}

// Generate mock indicators
function generateMockIndicators(data: OHLCData[]): IndicatorData[] {
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
}

// Helper function to calculate moving average
function calculateMovingAverage(data: OHLCData[], period: number) {
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
}
