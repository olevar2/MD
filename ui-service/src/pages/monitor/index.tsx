import React from 'react';
import { Grid, Paper, Typography, Box, LinearProgress } from '@mui/material';
import { ResponsiveLine } from '@nivo/line';
import DashboardLayout from '../../components/layout/DashboardLayout';

interface ServiceHealth {
  name: string;
  status: 'healthy' | 'degraded' | 'down';
  latency: number;
  uptime: number;
  metrics: {
    timestamp: string;
    value: number;
  }[];
}

interface SystemMetrics {
  cpu: number;
  memory: number;
  network: number;
  messageQueue: number;
  services: ServiceHealth[];
}

const Monitor: React.FC = () => {
  const [metrics, setMetrics] = React.useState<SystemMetrics>({
    cpu: 45,
    memory: 62,
    network: 28,
    messageQueue: 15,
    services: [
      {
        name: 'Analysis Engine',
        status: 'healthy',
        latency: 45,
        uptime: 99.99,
        metrics: Array.from({ length: 24 }, (_, i) => ({
          timestamp: `${i}:00`,
          value: 45 + Math.random() * 10,
        })),
      },
      {
        name: 'Trading Gateway',
        status: 'healthy',
        latency: 12,
        uptime: 99.95,
        metrics: Array.from({ length: 24 }, (_, i) => ({
          timestamp: `${i}:00`,
          value: 12 + Math.random() * 5,
        })),
      },
      // Add more services as needed
    ],
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'success.main';
      case 'degraded':
        return 'warning.main';
      case 'down':
        return 'error.main';
      default:
        return 'text.primary';
    }
  };

  return (
    <DashboardLayout>
      <Grid container spacing={3}>
        {/* System Resource Usage */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>System Resources</Typography>
            
            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle2" gutterBottom>CPU Usage</Typography>
              <LinearProgress 
                variant="determinate" 
                value={metrics.cpu} 
                color={metrics.cpu > 80 ? 'error' : metrics.cpu > 60 ? 'warning' : 'primary'}
                sx={{ height: 8, borderRadius: 1 }}
              />
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                {metrics.cpu}%
              </Typography>
            </Box>

            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle2" gutterBottom>Memory Usage</Typography>
              <LinearProgress 
                variant="determinate" 
                value={metrics.memory}
                color={metrics.memory > 80 ? 'error' : metrics.memory > 60 ? 'warning' : 'primary'}
                sx={{ height: 8, borderRadius: 1 }}
              />
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                {metrics.memory}%
              </Typography>
            </Box>

            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle2" gutterBottom>Network Load</Typography>
              <LinearProgress 
                variant="determinate" 
                value={metrics.network}
                color={metrics.network > 80 ? 'error' : metrics.network > 60 ? 'warning' : 'primary'}
                sx={{ height: 8, borderRadius: 1 }}
              />
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                {metrics.network}%
              </Typography>
            </Box>

            <Box>
              <Typography variant="subtitle2" gutterBottom>Message Queue Load</Typography>
              <LinearProgress 
                variant="determinate" 
                value={metrics.messageQueue}
                color={metrics.messageQueue > 80 ? 'error' : metrics.messageQueue > 60 ? 'warning' : 'primary'}
                sx={{ height: 8, borderRadius: 1 }}
              />
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                {metrics.messageQueue}%
              </Typography>
            </Box>
          </Paper>
        </Grid>

        {/* Service Health */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>Service Health</Typography>
            {metrics.services.map((service) => (
              <Box key={service.name} sx={{ mb: 3 }}>
                <Grid container alignItems="center" spacing={2}>
                  <Grid item xs={12}>
                    <Typography variant="subtitle1">{service.name}</Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                      <Box
                        sx={{
                          width: 12,
                          height: 12,
                          borderRadius: '50%',
                          bgcolor: getStatusColor(service.status),
                          mr: 1,
                        }}
                      />
                      <Typography variant="body2" sx={{ mr: 2 }}>
                        {service.status.charAt(0).toUpperCase() + service.status.slice(1)}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Latency: {service.latency}ms | Uptime: {service.uptime}%
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} sx={{ height: 100 }}>
                    <ResponsiveLine
                      data={[{
                        id: `${service.name}-metrics`,
                        data: service.metrics.map(({ timestamp, value }) => ({
                          x: timestamp,
                          y: value,
                        })),
                      }]}
                      margin={{ top: 10, right: 10, bottom: 20, left: 40 }}
                      xScale={{ type: 'point' }}
                      yScale={{ type: 'linear', min: 'auto', max: 'auto' }}
                      curve="monotoneX"
                      enablePoints={false}
                      enableArea={true}
                      areaBaselineValue={0}
                      axisBottom={{ tickRotation: -45 }}
                      colors={[getStatusColor(service.status)]}
                    />
                  </Grid>
                </Grid>
              </Box>
            ))}
          </Paper>
        </Grid>
      </Grid>
    </DashboardLayout>
  );
};

export default Monitor;
