import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend
} from 'recharts';
import { format } from 'date-fns';

export interface SignalDataPoint {
  timestamp: number;
  signal: string;
  confidence: number;
  direction?: 'buy' | 'sell' | 'neutral';
}

interface SignalConfidenceChartProps {
  signalData: SignalDataPoint[];
  timeRange?: 'day' | 'week' | 'month';
  confidenceThreshold?: number;
  height?: number;
}

const SignalConfidenceChart: React.FC<SignalConfidenceChartProps> = ({
  signalData,
  timeRange = 'day',
  confidenceThreshold = 0.7,
  height = 300
}) => {
  // Process data for charting
  const processedData = signalData.map(point => ({
    ...point,
    // Format timestamp for display
    time: format(new Date(point.timestamp), 'HH:mm'),
    // Color based on direction and confidence
    color: point.direction === 'buy' ? '#4CAF50' :
           point.direction === 'sell' ? '#f44336' : '#9e9e9e'
  }));

  return (
    <div className="signal-confidence-chart" role="figure" aria-label="Signal Confidence Chart">
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={processedData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="time"
            label={{ value: 'Time', position: 'insideBottom', offset: -10 }}
          />
          <YAxis
            domain={[0, 1]}
            label={{ value: 'Confidence', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip
            formatter={(value: number, name: string) => [
              `${(value * 100).toFixed(1)}%`,
              name === 'confidence' ? 'Signal Confidence' : name
            ]}
          />
          <Legend />
          {/* Reference line for confidence threshold */}
          <Line
            type="monotone"
            dataKey={() => confidenceThreshold}
            stroke="#757575"
            strokeDasharray="5 5"
            name="Threshold"
            isAnimationActive={false}
          />
          {/* Main confidence line */}
          <Line
            type="monotone"
            dataKey="confidence"
            stroke="#2196F3"
            strokeWidth={2}
            name="Signal Confidence"
            dot={{ fill: d => d.color, stroke: d => d.color }}
            activeDot={{ r: 8 }}
            animationDuration={300}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default SignalConfidenceChart;
