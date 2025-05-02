import React, { useState, useEffect } from 'react';
import { Container, Box, Typography } from '@mui/material';
import HealthSummary, { ServiceHealth, Alert } from '../components/health/HealthSummary';
import useWebSocket from '../hooks/useWebSocket';

const SystemHealth: React.FC = () => {
  const [healthStatus, setHealthStatus] = useState<ServiceHealth[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const ws = useWebSocket('ws://localhost:3001/health');

  useEffect(() => {
    const fetchInitialData = async () => {
      setIsLoading(true);
      try {
        // Fetch initial health status
        const healthResponse = await fetch('/api/health/status');
        const healthData = await healthResponse.json();
        setHealthStatus(healthData);

        // Fetch recent alerts
        const alertsResponse = await fetch('/api/health/alerts');
        const alertsData = await alertsResponse.json();
        setAlerts(alertsData);

        setError(null);
      } catch (err) {
        setError('Failed to fetch system health data');
        console.error('Health data fetch error:', err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchInitialData();

    return () => {
      // Cleanup WebSocket connection if needed
      ws.disconnect();
    };
  }, []);

  // Handle real-time updates from WebSocket
  useEffect(() => {
    if (!ws.data) return;

    try {
      const update = JSON.parse(ws.data);
      
      if (update.type === 'health') {
        setHealthStatus(prevStatus => {
          const updatedStatus = [...prevStatus];
          const index = updatedStatus.findIndex(s => s.id === update.service.id);
          
          if (index >= 0) {
            updatedStatus[index] = {
              ...updatedStatus[index],
              ...update.service
            };
          } else {
            updatedStatus.push(update.service);
          }
          
          return updatedStatus;
        });
      }
      
      if (update.type === 'alert') {
        setAlerts(prevAlerts => [update.alert, ...prevAlerts].slice(0, 10));
      }
    } catch (err) {
      console.error('WebSocket data parse error:', err);
    }
  }, [ws.data]);

  return (
    <Container maxWidth="xl">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          System Health
        </Typography>
        
        <HealthSummary
          services={healthStatus}
          alerts={alerts}
          isLoading={isLoading}
          error={error}
        />
      </Box>
    </Container>
  );
};

export default SystemHealth;
