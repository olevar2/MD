import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine
} from 'recharts';
import { useTheme } from '@mui/material';

export interface MetricDataPoint {
  timestamp: string;
  value: number;
}

interface MetricChartProps {
  data: MetricDataPoint[];
  name: string;
  unit?: string;
  thresholds?: {
    warning?: number;
    critical?: number;
  };
  height?: number;
  showGrid?: boolean;
  areaChart?: boolean;
}

const MetricChart: React.FC<MetricChartProps> = ({
  data,
  name,
  unit = '',
  thresholds,
  height = 200,
  showGrid = true,
  areaChart = false
}) => {
  const theme = useTheme();

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart
        data={data}
        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
      >
        {showGrid && (
          <CartesianGrid
            strokeDasharray="3 3"
            stroke={theme.palette.divider}
          />
        )}
        
        <XAxis
          dataKey="timestamp"
          tick={{ fontSize: 12 }}
          stroke={theme.palette.text.secondary}
        />
        
        <YAxis
          tick={{ fontSize: 12 }}
          stroke={theme.palette.text.secondary}
          domain={['auto', 'auto']}
        />
        
        <Tooltip
          contentStyle={{
            backgroundColor: theme.palette.background.paper,
            border: `1px solid ${theme.palette.divider}`,
            borderRadius: 4
          }}
          formatter={(value: number) => [`${value}${unit}`, name]}
        />
        
        <Legend />
        
        {/* Warning threshold line */}
        {thresholds?.warning && (
          <ReferenceLine
            y={thresholds.warning}
            label="Warning"
            stroke={theme.palette.warning.main}
            strokeDasharray="3 3"
          />
        )}
        
        {/* Critical threshold line */}
        {thresholds?.critical && (
          <ReferenceLine
            y={thresholds.critical}
            label="Critical"
            stroke={theme.palette.error.main}
            strokeDasharray="3 3"
          />
        )}
        
        {/* Main metric line */}
        <Line
          type="monotone"
          name={name}
          dataKey="value"
          stroke={theme.palette.primary.main}
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 6 }}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default MetricChart;
