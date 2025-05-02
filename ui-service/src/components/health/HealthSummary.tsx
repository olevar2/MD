import React from 'react';
import { Grid, Paper, Typography, Box, Divider, CircularProgress } from '@mui/material';
import StatusIndicator, { StatusType } from './StatusIndicator';
import MetricChart, { MetricDataPoint } from './MetricChart';
import { format } from 'date-fns';

export interface Alert {
  id: string;
  severity: 'critical' | 'warning' | 'info';
  message: string;
  service: string;
  timestamp: string;
}

export interface HealthMetric {
  name: string;
  value: number;
  unit: string;
  history: MetricDataPoint[];
  thresholds?: {
    warning?: number;
    critical?: number;
  };
}

export interface ServiceHealth {
  id: string;
  name: string;
  status: StatusType;
  metrics?: HealthMetric[];
  lastCheck: string;
}

interface HealthSummaryProps {
  services: ServiceHealth[];
  alerts: Alert[];
  isLoading?: boolean;
  error?: string | null;
}

const HealthSummary: React.FC<HealthSummaryProps> = ({
  services,
  alerts,
  isLoading,
  error
}) => {
  const getOverallStatus = (): StatusType => {
    if (services.some(s => s.status === 'Outage')) return 'Outage';
    if (services.some(s => s.status === 'Degraded')) return 'Degraded';
    if (services.every(s => s.status === 'Operational')) return 'Operational';
    return 'Unknown';
  };

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }

  return (
    <Grid container spacing={3}>
      {/* Overall Status */}
      <Grid item xs={12}>
        <Paper sx={{ p: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <StatusIndicator
              status={getOverallStatus()}
              label="Platform Status"
              size="large"
            />
            <Typography variant="caption" color="text.secondary">
              Last updated: {format(new Date(), 'HH:mm:ss')}
            </Typography>
          </Box>
        </Paper>
      </Grid>

      {/* Services Status */}
      <Grid item xs={12} md={8}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Service Status
          </Typography>
          <Grid container spacing={2}>
            {services.map((service) => (
              <Grid item xs={12} key={service.id}>
                <Box sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="subtitle1">
                      {service.name}
                    </Typography>
                    <StatusIndicator status={service.status} />
                  </Box>
                  {service.metrics && service.metrics.map((metric) => (
                    <Box key={metric.name} sx={{ mt: 2 }}>
                      <Typography variant="caption" color="text.secondary">
                        {metric.name}: {metric.value}{metric.unit}
                      </Typography>
                      <MetricChart
                        data={metric.history}
                        name={metric.name}
                        unit={metric.unit}
                        thresholds={metric.thresholds}
                        height={100}
                      />
                    </Box>
                  ))}
                </Box>
                <Divider />
              </Grid>
            ))}
          </Grid>
        </Paper>
      </Grid>

      {/* Recent Alerts */}
      <Grid item xs={12} md={4}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Recent Alerts
          </Typography>
          {alerts.length === 0 ? (
            <Typography variant="body2" color="text.secondary">
              No recent alerts
            </Typography>
          ) : (
            alerts.map((alert) => (
              <Box
                key={alert.id}
                sx={{
                  mb: 2,
                  p: 2,
                  borderRadius: 1,
                  bgcolor: (theme) => 
                    alert.severity === 'critical' ? theme.palette.error.light :
                    alert.severity === 'warning' ? theme.palette.warning.light :
                    theme.palette.info.light
                }}
              >
                <Typography variant="subtitle2">
                  {alert.service}
                </Typography>
                <Typography variant="body2">
                  {alert.message}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {format(new Date(alert.timestamp), 'HH:mm:ss')}
                </Typography>
              </Box>
            ))
          )}
        </Paper>
      </Grid>
    </Grid>
  );
};

export default HealthSummary;
