import React, { useEffect, useRef } from 'react';
import { createChart, ColorType, CrosshairMode } from 'lightweight-charts';
import { Box, Paper, ToggleButton, ToggleButtonGroup } from '@mui/material';
import { styled } from '@mui/material/styles';

const ChartContainer = styled(Box)(({ theme }) => ({
  width: '100%',
  height: 'calc(100vh - 200px)',
  position: 'relative',
}));

const ControlsOverlay = styled(Paper)(({ theme }) => ({
  position: 'absolute',
  top: theme.spacing(1),
  right: theme.spacing(1),
  padding: theme.spacing(1),
  zIndex: 1,
}));

const TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d'];
const INDICATORS = ['MA', 'EMA', 'RSI', 'MACD', 'BB'];

interface ChartProps {
  symbol?: string;
  onTimeframeChange?: (timeframe: string) => void;
  onIndicatorChange?: (indicators: string[]) => void;
}

const Chart: React.FC<ChartProps> = ({
  symbol = 'EUR/USD',
  onTimeframeChange,
  onIndicatorChange,
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);
  const [timeframe, setTimeframe] = React.useState('15m');
  const [selectedIndicators, setSelectedIndicators] = React.useState<string[]>(['MA']);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#ffffff' },
        textColor: '#333',
      },
      grid: {
        vertLines: { color: '#f0f0f0' },
        horzLines: { color: '#f0f0f0' },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
      },
      rightPriceScale: {
        borderColor: '#d6dcde',
      },
      timeScale: {
        borderColor: '#d6dcde',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
    });

    // Example data - in production, this would come from your WebSocket connection
    const initialData = [
      { time: '2023-01-01', open: 1.0500, high: 1.0550, low: 1.0480, close: 1.0520 },
      { time: '2023-01-02', open: 1.0520, high: 1.0580, low: 1.0510, close: 1.0560 },
      // ... more data
    ];

    candleSeries.setData(initialData);
    chart.timeScale().fitContent();

    chartRef.current = chart;

    return () => {
      chart.remove();
    };
  }, []);

  useEffect(() => {
    if (!chartRef.current) return;

    // Handle indicator changes
    selectedIndicators.forEach(indicator => {
      switch (indicator) {
        case 'MA':
          // Add Moving Average
          const ma = chartRef.current.addLineSeries({
            color: '#2962FF',
            lineWidth: 2,
          });
          // Set MA data
          break;
        // Add other indicators similarly
      }
    });
  }, [selectedIndicators]);

  const handleTimeframeChange = (event: React.MouseEvent<HTMLElement>, newTimeframe: string) => {
    if (newTimeframe !== null) {
      setTimeframe(newTimeframe);
      onTimeframeChange?.(newTimeframe);
    }
  };

  const handleIndicatorChange = (event: React.MouseEvent<HTMLElement>, newIndicators: string[]) => {
    setSelectedIndicators(newIndicators);
    onIndicatorChange?.(newIndicators);
  };

  return (
    <ChartContainer>
      <ControlsOverlay elevation={2}>
        <Box sx={{ mb: 1 }}>
          <ToggleButtonGroup
            value={timeframe}
            exclusive
            onChange={handleTimeframeChange}
            size="small"
          >
            {TIMEFRAMES.map((tf) => (
              <ToggleButton key={tf} value={tf}>
                {tf}
              </ToggleButton>
            ))}
          </ToggleButtonGroup>
        </Box>
        <ToggleButtonGroup
          value={selectedIndicators}
          onChange={handleIndicatorChange}
          size="small"
          multiple
        >
          {INDICATORS.map((indicator) => (
            <ToggleButton key={indicator} value={indicator}>
              {indicator}
            </ToggleButton>
          ))}
        </ToggleButtonGroup>
      </ControlsOverlay>
      <div ref={chartContainerRef} style={{ width: '100%', height: '100%' }} />
    </ChartContainer>
  );
};

export default Chart;
