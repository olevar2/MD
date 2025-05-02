import React from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend
} from 'recharts';
import { format } from 'date-fns';

export interface Candle {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface CandlestickChartProps {
  data: Candle[];
  symbol: string;
  interval?: string;
  height?: number;
  onIntervalChange?: (interval: string) => void;
  onZoom?: (start: number, end: number) => void;
}

const CandlestickChart: React.FC<CandlestickChartProps> = ({
  data,
  symbol,
  interval = '1h',
  height = 400,
  onIntervalChange,
  onZoom
}) => {
  const formattedData = data.map(candle => ({
    ...candle,
    time: format(new Date(candle.timestamp), 'HH:mm'),
    range: [candle.low, candle.high],
    body: [candle.open, candle.close],
    color: candle.close >= candle.open ? '#4CAF50' : '#f44336'
  }));

  return (
    <div className="candlestick-chart" role="figure" aria-label={`${symbol} Price Chart`}>
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart
          data={formattedData}
          margin={{ top: 10, right: 30, left: 10, bottom: 0 }}
        >
          <defs>
            <linearGradient id="price" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#2196F3" stopOpacity={0.1}/>
              <stop offset="95%" stopColor="#2196F3" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <XAxis
            dataKey="time"
            tickLine={false}
            axisLine={false}
            style={{ fontSize: '12px' }}
          />
          <YAxis
            type="number"
            domain={['dataMin', 'dataMax']}
            orientation="right"
            axisLine={false}
            tickLine={false}
            style={{ fontSize: '12px' }}
          />
          <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(255, 255, 255, 0.9)',
              border: 'none',
              borderRadius: '4px',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
            }}
            formatter={(value: number) => [`$${value.toFixed(5)}`, 'Price']}
            labelFormatter={(label) => `Time: ${label}`}
          />
          <Legend verticalAlign="top" align="right" />
          
          {/* Price Range */}
          <Area
            type="monotone"
            dataKey="range"
            stroke="none"
            fill="url(#price)"
            opacity={0.3}
          />
          
          {/* Price Body */}
          <Area
            type="linear"
            dataKey="body"
            stroke="#2196F3"
            fill="none"
            strokeWidth={2}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export default CandlestickChart;
