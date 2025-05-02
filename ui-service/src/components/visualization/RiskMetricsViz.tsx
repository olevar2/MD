import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Tooltip,
  LinearProgress
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip as RechartsTooltip
} from 'recharts';

interface RiskMetric {
  name: string;
  value: number;
  target?: number;
  warning?: number;
  critical?: number;
  unit?: string;
}

interface RiskMetricsVizProps {
  metrics: RiskMetric[];
}

const getRiskColor = (value: number, warning?: number, critical?: number) => {
  if (critical !== undefined && value >= critical) return 'error.main';
  if (warning !== undefined && value >= warning) return 'warning.main';
  return 'success.main';
};

const RiskMetricsViz: React.FC<RiskMetricsVizProps> = ({ metrics }) => {
  // Transform data for the bar chart
  const chartData = metrics.map(metric => ({
    name: metric.name,
    value: metric.value,
    target: metric.target,
    fill: getRiskColor(metric.value, metric.warning, metric.critical)
  }));

  return (
    <Grid container spacing={3}>
      {/* Gauges for individual metrics */}
      <Grid item xs={12}>
        <Grid container spacing={2}>
          {metrics.map((metric) => (
            <Grid item xs={12} sm={6} md={4} key={metric.name}>
              <Paper sx={{ p: 2 }}>
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    {metric.name}
                  </Typography>
                  
                  <Box sx={{ position: 'relative', mb: 1 }}>
                    <LinearProgress
                      variant="determinate"
                      value={100}
                      sx={{
                        height: 10,
                        backgroundColor: 'grey.200',
                        '& .MuiLinearProgress-bar': {
                          backgroundColor: 'grey.300'
                        }
                      }}
                    />
                    {metric.warning && (
                      <Tooltip title="Warning threshold">
                        <Box
                          sx={{
                            position: 'absolute',
                            left: `${metric.warning}%`,
                            top: 0,
                            bottom: 0,
                            width: 2,
                            backgroundColor: 'warning.main'
                          }}
                        />
                      </Tooltip>
                    )}
                    {metric.critical && (
                      <Tooltip title="Critical threshold">
                        <Box
                          sx={{
                            position: 'absolute',
                            left: `${metric.critical}%`,
                            top: 0,
                            bottom: 0,
                            width: 2,
                            backgroundColor: 'error.main'
                          }}
                        />
                      </Tooltip>
                    )}
                    <LinearProgress
                      variant="determinate"
                      value={metric.value}
                      sx={{
                        height: 10,
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        backgroundColor: 'transparent',
                        '& .MuiLinearProgress-bar': {
                          backgroundColor: getRiskColor(
                            metric.value,
                            metric.warning,
                            metric.critical
                          )
                        }
                      }}
                    />
                  </Box>

                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography
                      variant="body2"
                      color={getRiskColor(
                        metric.value,
                        metric.warning,
                        metric.critical
                      )}
                      fontWeight="bold"
                    >
                      {metric.value}
                      {metric.unit}
                    </Typography>
                    {metric.target && (
                      <Typography variant="caption" color="text.secondary">
                        Target: {metric.target}
                        {metric.unit}
                      </Typography>
                    )}
                  </Box>
                </Box>
              </Paper>
            </Grid>
          ))}
        </Grid>
      </Grid>

      {/* Bar chart comparison */}
      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Risk Metrics Comparison
          </Typography>
          <Box height={200}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <XAxis
                  dataKey="name"
                  tick={{ fontSize: 12 }}
                  interval={0}
                  angle={-45}
                  textAnchor="end"
                />
                <YAxis />
                <RechartsTooltip
                  formatter={(value: number, name: string, props: any) => [
                    `${value}${metrics.find(m => m.name === props.payload.name)?.unit || ''}`,
                    props.payload.name
                  ]}
                />
                <Bar
                  dataKey="value"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </Box>
        </Paper>
      </Grid>
    </Grid>
  );
};

export default RiskMetricsViz;
