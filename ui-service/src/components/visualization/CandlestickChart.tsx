import React, { useEffect, useRef } from 'react';
import { createChart, IChartApi } from 'lightweight-charts';
import { Box, useTheme } from '@mui/material';

export interface CandleData {
  time: string | number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface CandlestickChartProps {
  data: CandleData[];
  width?: number;
  height?: number;
  onCrosshairMove?: (price: number, time: string) => void;
  indicators?: {
    name: string;
    data: Array<{ time: string | number; value: number }>;
    color: string;
  }[];
}

const CandlestickChart: React.FC<CandlestickChartProps> = ({
  data,
  width = 800,
  height = 400,
  onCrosshairMove,
  indicators = []
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const theme = useTheme();

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Chart configuration
    const chartOptions = {
      width,
      height,
      layout: {
        background: { color: theme.palette.background.paper },
        textColor: theme.palette.text.primary,
      },
      grid: {
        vertLines: { color: theme.palette.divider },
        horzLines: { color: theme.palette.divider },
      },
      crosshair: {
        mode: 1,
        vertLine: {
          width: 1,
          color: theme.palette.primary.main,
          style: 2,
        },
        horzLine: {
          width: 1,
          color: theme.palette.primary.main,
          style: 2,
        },
      },
      timeScale: {
        borderColor: theme.palette.divider,
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: theme.palette.divider,
      },
    };

    // Create chart
    const chart = createChart(chartContainerRef.current, chartOptions);
    chartRef.current = chart;

    // Add candlestick series
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: theme.palette.success.main,
      downColor: theme.palette.error.main,
      borderVisible: false,
      wickUpColor: theme.palette.success.main,
      wickDownColor: theme.palette.error.main,
    });

    // Add volume series
    const volumeSeries = chart.addHistogramSeries({
      color: theme.palette.primary.main,
      priceFormat: { type: 'volume' },
      priceScaleId: '', // Set to empty string to overlay
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
    });

    // Add indicator series
    const indicatorSeries = indicators.map(indicator => {
      return chart.addLineSeries({
        color: indicator.color,
        lineWidth: 2,
        priceLineVisible: false,
      });
    });

    // Set data
    candlestickSeries.setData(data);
    if (data[0]?.volume) {
      volumeSeries.setData(
        data.map(candle => ({
          time: candle.time,
          value: candle.volume || 0,
          color: candle.close >= candle.open ? 
            theme.palette.success.main : 
            theme.palette.error.main
        }))
      );
    }

    // Set indicator data
    indicators.forEach((indicator, index) => {
      indicatorSeries[index].setData(indicator.data);
    });

    // Set up crosshair move handler
    if (onCrosshairMove) {
      chart.subscribeCrosshairMove(param => {
        if (
          param.point === undefined ||
          !param.time ||
          param.point.x < 0 ||
          param.point.x > width ||
          param.point.y < 0 ||
          param.point.y > height
        ) {
          return;
        }

        const price = candlestickSeries.coordinateToPrice(param.point.y);
        onCrosshairMove(price, param.time.toString());
      });
    }

    // Fit content
    chart.timeScale().fitContent();

    // Cleanup
    return () => {
      chart.remove();
      chartRef.current = null;
    };
  }, [
    data,
    width,
    height,
    onCrosshairMove,
    indicators,
    theme.palette.background.paper,
    theme.palette.text.primary,
    theme.palette.divider,
    theme.palette.primary.main,
    theme.palette.success.main,
    theme.palette.error.main,
  ]);

  return (
    <Box
      ref={chartContainerRef}
      sx={{
        width,
        height,
        '& .tv-lightweight-charts': {
          border: `1px solid ${theme.palette.divider}`,
          borderRadius: 1,
        },
      }}
    />
  );
};

export default CandlestickChart;
