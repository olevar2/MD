/**
 * TradingChart Component
 * 
 * Advanced chart visualization that supports technical analysis, pattern recognition,
 * and confluence highlighting with enhanced visualization capabilities.
 */
import React, { useEffect, useRef, useState } from 'react';
// @ts-ignore - Adding ts-ignore to bypass missing module declarations
import { Box, CircularProgress, Paper, Tooltip, IconButton } from '@mui/material';
// @ts-ignore
import ZoomInIcon from '@mui/icons-material/ZoomIn';
// @ts-ignore
import ZoomOutIcon from '@mui/icons-material/ZoomOut';
// @ts-ignore
import FullscreenIcon from '@mui/icons-material/Fullscreen';
// @ts-ignore
import { createChart, IChartApi, ISeriesApi, LineData, Time, LineStyle, PriceScaleMode } from 'lightweight-charts';
// @ts-ignore
import ConfluenceHighlighter from './ConfluenceHighlighter';
// @ts-ignore
import PatternVisualization from './PatternVisualization';
// @ts-ignore
import { ChartData, ConfluencePoint, ChartOptions, PatternData } from '../../types/chart';

interface TradingChartProps {
  symbol: string;
  timeframe: string;
  data: ChartData;
  confluencePoints?: ConfluencePoint[];
  patterns?: PatternData[];
  options?: Partial<ChartOptions>;
  height?: number;
  width?: string | number;
  loading?: boolean;
  onTimeRangeChange?: (from: Time, to: Time) => void;
}

const defaultOptions: ChartOptions = {
  darkMode: false,
  showVolume: true,
  showGrid: true,
  showConfluence: true,
  confluenceThreshold: 0.5,
  chartOptions: {}
};

const TradingChart: React.FC<TradingChartProps> = ({
  symbol,
  timeframe,
  data,
  confluencePoints = [],
  patterns = [],
  options = {},
  height = 500,
  width = '100%',
  loading = false,
  onTimeRangeChange,
}) => {
  // Merge default options with provided options
  const mergedOptions = { ...defaultOptions, ...options };
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const [chart, setChart] = useState<IChartApi | null>(null);
  const [candlestickSeries, setCandlestickSeries] = useState<ISeriesApi<'Candlestick'> | null>(null);
  const resizeObserver = useRef<ResizeObserver | null>(null);
  // Create chart on component mount
  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chartOptions = {
      layout: {
        background: { color: mergedOptions.darkMode ? '#1E1E1E' : '#FFFFFF' },
        textColor: mergedOptions.darkMode ? '#D9D9D9' : '#191919',
      },
      grid: {
        vertLines: { color: mergedOptions.darkMode ? '#2B2B43' : '#E6E6E6' },
        horzLines: { color: mergedOptions.darkMode ? '#2B2B43' : '#E6E6E6' },
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
      },
      crosshair: {
        mode: 1,
      },
      ...mergedOptions.chartOptions,
    };    const newChart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height,
      ...chartOptions,
    });

    const newCandlestickSeries = newChart.addCandlestickSeries({
      upColor: mergedOptions.darkMode ? '#26A69A' : '#089981',
      downColor: mergedOptions.darkMode ? '#EF5350' : '#F23645',
      borderVisible: false,
      wickUpColor: mergedOptions.darkMode ? '#26A69A' : '#089981',
      wickDownColor: mergedOptions.darkMode ? '#EF5350' : '#F23645',
    });

    setChart(newChart);
    setCandlestickSeries(newCandlestickSeries);

    // Handle time range changes
    if (onTimeRangeChange) {
      newChart.timeScale().subscribeVisibleTimeRangeChange(onTimeRangeChange);
    }

    // Cleanup
    return () => {
      if (newChart) {
        newChart.remove();
        setChart(null);
        setCandlestickSeries(null);
      }
      if (resizeObserver.current) {
        resizeObserver.current.disconnect();
      }
    };
  }, [height, mergedOptions.darkMode, mergedOptions.chartOptions, onTimeRangeChange]);

  // Set up resize observer for chart
  useEffect(() => {
    if (!chartContainerRef.current || !chart) return;

    resizeObserver.current = new ResizeObserver(entries => {
      if (entries[0].contentRect) {
        chart.applyOptions({ width: entries[0].contentRect.width });
      }
    });

    resizeObserver.current.observe(chartContainerRef.current);

    return () => {
      if (resizeObserver.current) {
        resizeObserver.current.disconnect();
      }
    };
  }, [chart]);

  // Update chart data when data changes
  useEffect(() => {
    if (!candlestickSeries || !data.candles || data.candles.length === 0) return;
    
    candlestickSeries.setData(data.candles);
    
    if (chart) {
      chart.timeScale().fitContent();
    }
  }, [candlestickSeries, chart, data]);

  return (
    <Paper 
      elevation={3}
      sx={{ 
        position: 'relative', 
        height: height, 
        width: width,
        borderRadius: 2,
        overflow: 'hidden'
      }}
    >
      {loading && (
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: 'rgba(0,0,0,0.3)',
            zIndex: 5,
          }}
        >
          <CircularProgress />
        </Box>
      )}

      <Box ref={chartContainerRef} sx={{ height: '100%', width: '100%' }} />      {chart && candlestickSeries && confluencePoints.length > 0 && (
        <ConfluenceHighlighter 
          chart={chart}
          series={candlestickSeries}
          confluencePoints={confluencePoints}
          darkMode={mergedOptions.darkMode}
        />
      )}
      
      {chart && candlestickSeries && patterns && patterns.length > 0 && (
        <PatternVisualization
          chart={chart}
          series={candlestickSeries}
          patterns={patterns}
          darkMode={mergedOptions.darkMode}
        />
      )}
    </Paper>
  );
};

export default TradingChart;
