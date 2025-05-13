import React, { useState, useEffect } from 'react';
import {
  Box,
  CircularProgress,
  Alert,
  Grid,
  Paper,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Button
} from '@mui/material';
import { DatePicker } from '@mui/x-date-pickers';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';

interface FeedbackItem {
  id: string;
  timestamp: number;
  source: string;
  category: string;
  message: string;
  severity: 'info' | 'warning' | 'error';
  related_signals?: string[];
}

interface FeedbackFilters {
  startDate: Date | null;
  endDate: Date | null;
  source: string;
  category: string;
  severity: string;
}

interface ChartData {
  date: string;
  count: number;
  warnings: number;
  errors: number;
}

const FeedbackDashboard: React.FC = () => {
  const [feedbackData, setFeedbackData] = useState<FeedbackItem[]>([]);
  const [filters, setFilters] = useState<FeedbackFilters>({
    startDate: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), // Last 7 days
    endDate: new Date(),
    source: 'all',
    category: 'all',
    severity: 'all'
  });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [chartData, setChartData] = useState<ChartData[]>([]);

  useEffect(() => {
    const fetchFeedbackData = async () => {
      try {
        const response = await fetch('/api/feedback', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(filters),
        });
        
        if (!response.ok) {
          throw new Error('Failed to fetch feedback data');
        }

        const data = await response.json();
        setFeedbackData(data);
        processChartData(data);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setIsLoading(false);
      }
    };

    fetchFeedbackData();
  }, [filters]);

  const processChartData = (data: FeedbackItem[]) => {
    const dateGroups = data.reduce((acc, item) => {
      const date = new Date(item.timestamp).toLocaleDateString();
      if (!acc[date]) {
        acc[date] = { total: 0, warnings: 0, errors: 0 };
      }
      acc[date].total++;
      if (item.severity === 'warning') acc[date].warnings++;
      if (item.severity === 'error') acc[date].errors++;
      return acc;
    }, {} as Record<string, { total: number; warnings: number; errors: number }>);

    const chartData = Object.entries(dateGroups).map(([date, counts]) => ({
      date,
      count: counts.total,
      warnings: counts.warnings,
      errors: counts.errors
    }));

    setChartData(chartData);
  };

  const handleFilterChange = (
    field: keyof FeedbackFilters,
    value: string | Date | null
  ) => {
    setFilters(prev => ({
      ...prev,
      [field]: value
    }));
  };

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mt: 2 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        System Feedback Dashboard
      </Typography>

      {/* Filter Controls */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={3}>
            <DatePicker
              label="Start Date"
              value={filters.startDate}
              onChange={(date) => handleFilterChange('startDate', date)}
            />
          </Grid>
          <Grid item xs={12} md={3}>
            <DatePicker
              label="End Date"
              value={filters.endDate}
              onChange={(date) => handleFilterChange('endDate', date)}
            />
          </Grid>
          <Grid item xs={12} md={2}>
            <FormControl fullWidth>
              <InputLabel>Source</InputLabel>
              <Select
                value={filters.source}
                label="Source"
                onChange={(e) => handleFilterChange('source', e.target.value)}
              >
                <MenuItem value="all">All Sources</MenuItem>
                <MenuItem value="trading">Trading Engine</MenuItem>
                <MenuItem value="analysis">Analysis Engine</MenuItem>
                <MenuItem value="data">Data Pipeline</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={2}>
            <FormControl fullWidth>
              <InputLabel>Category</InputLabel>
              <Select
                value={filters.category}
                label="Category"
                onChange={(e) => handleFilterChange('category', e.target.value)}
              >
                <MenuItem value="all">All Categories</MenuItem>
                <MenuItem value="performance">Performance</MenuItem>
                <MenuItem value="accuracy">Accuracy</MenuItem>
                <MenuItem value="system">System</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={2}>
            <FormControl fullWidth>
              <InputLabel>Severity</InputLabel>
              <Select
                value={filters.severity}
                label="Severity"
                onChange={(e) => handleFilterChange('severity', e.target.value)}
              >
                <MenuItem value="all">All Severities</MenuItem>
                <MenuItem value="info">Info</MenuItem>
                <MenuItem value="warning">Warning</MenuItem>
                <MenuItem value="error">Error</MenuItem>
              </Select>
            </FormControl>
          </Grid>
        </Grid>
      </Paper>

      {/* Data Visualization */}
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={{ p: 2, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Feedback Trend
            </Typography>
            <ResponsiveContainer>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Area 
                  type="monotone" 
                  dataKey="count" 
                  stackId="1"
                  stroke="#8884d8" 
                  fill="#8884d8" 
                />
                <Area 
                  type="monotone" 
                  dataKey="warnings" 
                  stackId="2"
                  stroke="#ffc658" 
                  fill="#ffc658" 
                />
                <Area 
                  type="monotone" 
                  dataKey="errors" 
                  stackId="3"
                  stroke="#ff7300" 
                  fill="#ff7300" 
                />
              </AreaChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Feedback List */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Recent Feedback
            </Typography>
            {feedbackData.map((item) => (
              <Alert 
                key={item.id}
                severity={item.severity}
                sx={{ mb: 1 }}
              >
                <Typography variant="subtitle2">
                  {new Date(item.timestamp).toLocaleString()} - {item.source}
                </Typography>
                <Typography>
                  {item.message}
                </Typography>
                {item.related_signals && (
                  <Typography variant="caption" display="block">
                    Related Signals: {item.related_signals.join(', ')}
                  </Typography>
                )}
              </Alert>
            ))}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default FeedbackDashboard;
