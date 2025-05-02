import React from 'react';
import { createChart, IChartApi, ISeriesApi, LineData, Time } from 'lightweight-charts';
import { Box, useTheme } from '@mui/material';

export interface ChartProps {
  data: LineData<Time>[];
  width?: number | string;
  height?: number | string;
  tooltipEnabled?: boolean;
  title?: string;
  onCrosshairMove?: (price: number, time: Time) => void;
  chartType?: 'line' | 'area' | 'bar' | 'candlestick';
}

export const Chart: React.FC<ChartProps> = ({
  data,
  width = '100%',
  height = 300,
  tooltipEnabled = true,
  title,
  onCrosshairMove,
  chartType = 'line',
}) => {
  const theme = useTheme();
  const chartContainerRef = React.useRef<HTMLDivElement>(null);
  const [chart, setChart] = React.useState<IChartApi | null>(null);
  const [series, setSeries] = React.useState<ISeriesApi<'Line'> | null>(null);

  React.useEffect(() => {
    if (chartContainerRef.current) {
      const newChart = createChart(chartContainerRef.current, {
        width: chartContainerRef.current.clientWidth,
        height: typeof height === 'number' ? height : 300,
        layout: {
          background: { color: theme.palette.background.paper },
          textColor: theme.palette.text.primary,
        },
        grid: {
          vertLines: { color: theme.palette.divider },
          horzLines: { color: theme.palette.divider },
        },
        timeScale: {
          borderColor: theme.palette.divider,
        },
        rightPriceScale: {
          borderColor: theme.palette.divider,
        },
        crosshair: {
          mode: tooltipEnabled ? 0 : 1,
        },
      });

      let newSeries;
      switch (chartType) {
        case 'area':
          newSeries = newChart.addAreaSeries({
            topColor: theme.palette.primary.main,
            bottomColor: theme.palette.primary.light,
            lineColor: theme.palette.primary.dark,
            lineWidth: 2,
          });
          break;
        case 'bar':
          newSeries = newChart.addBarSeries({
            upColor: theme.palette.success.main,
            downColor: theme.palette.error.main,
          });
          break;
        case 'candlestick':
          newSeries = newChart.addCandlestickSeries({
            upColor: theme.palette.success.main,
            downColor: theme.palette.error.main,
            borderVisible: false,
            wickUpColor: theme.palette.success.main,
            wickDownColor: theme.palette.error.main,
          });
          break;
        case 'line':
        default:
          newSeries = newChart.addLineSeries({
            color: theme.palette.primary.main,
            lineWidth: 2,
          });
          break;
      }

      if (newSeries) {
        newSeries.setData(data);
        setSeries(newSeries as any);
      }

      if (onCrosshairMove) {
        newChart.subscribeCrosshairMove((param) => {
          if (param.point && param.time && param.seriesPrices.size) {
            const price = param.seriesPrices.get(newSeries);
            if (price !== undefined) {
              onCrosshairMove(price as number, param.time);
            }
          }
        });
      }

      setChart(newChart);

      const handleResize = () => {
        if (chartContainerRef.current && newChart) {
          newChart.applyOptions({ width: chartContainerRef.current.clientWidth });
        }
      };

      window.addEventListener('resize', handleResize);
      return () => {
        window.removeEventListener('resize', handleResize);
        if (newChart) {
          newChart.remove();
        }
      };
    }
  }, [chartType, theme]);

  React.useEffect(() => {
    if (series && data) {
      series.setData(data);
    }
  }, [data, series]);

  return (
    <Box
      sx={{
        width: width,
        height: height,
        position: 'relative',
      }}
    >
      {title && (
        <Box sx={{ position: 'absolute', top: 8, left: 8, zIndex: 5, color: theme.palette.text.secondary, fontSize: '0.875rem' }}>
          {title}
        </Box>
      )}
      <Box ref={chartContainerRef} sx={{ width: '100%', height: '100%' }} />
    </Box>
  );
};

export default Chart;
