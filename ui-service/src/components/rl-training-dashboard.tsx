import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  CircularProgress,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell
} from 'recharts';

interface TrainingMetrics {
  episode: number;
  totalReward: number;
  averageReward: number;
  epsilon: number;
  lossValue: number;
  actionDistribution: Record<string, number>;
  rewardBreakdown: {
    profitReward: number;
    riskReward: number;
    timeDecay: number;
  };
}

interface EpisodeComparison {
  episode1: TrainingMetrics;
  episode2: TrainingMetrics;
}

const RLTrainingDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<TrainingMetrics[]>([]);
  const [selectedEpisodes, setSelectedEpisodes] = useState<EpisodeComparison | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const ws = useWebSocket('ws://localhost:3001/rl-training');

  useEffect(() => {
    if (ws.data) {
      try {
        const update = JSON.parse(ws.data);
        setMetrics(prev => [...prev, update].slice(-100)); // Keep last 100 episodes
      } catch (err) {
        console.error('Failed to parse training data:', err);
      }
    }
  }, [ws.data]);

  useEffect(() => {
    return () => {
      ws.disconnect();
    };
  }, []);

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        {error}
      </Alert>
    );
  }

  const latestMetrics = metrics[metrics.length - 1];

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        RL Training Progress
      </Typography>

      <Grid container spacing={3}>
        {/* Main Metrics */}
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 2, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Training Progress
            </Typography>
            <ResponsiveContainer>
              <LineChart data={metrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="episode" />
                <YAxis />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey="totalReward" 
                  stroke="#8884d8" 
                  name="Total Reward"
                />
                <Line 
                  type="monotone" 
                  dataKey="averageReward" 
                  stroke="#82ca9d" 
                  name="Avg Reward"
                />
                <Line 
                  type="monotone" 
                  dataKey="epsilon" 
                  stroke="#ffc658" 
                  name="Epsilon"
                />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Action Distribution */}
        <Grid item xs={12} lg={4}>
          <Paper sx={{ p: 2, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Action Distribution
            </Typography>
            {latestMetrics && (
              <ResponsiveContainer>
                <PieChart>
                  <Pie
                    data={Object.entries(latestMetrics.actionDistribution).map(
                      ([action, value]) => ({
                        name: action,
                        value
                      })
                    )}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    fill="#8884d8"
                    paddingAngle={5}
                    dataKey="value"
                    label
                  >
                    {Object.keys(latestMetrics.actionDistribution).map((_, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            )}
          </Paper>
        </Grid>

        {/* Reward Breakdown */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Reward Breakdown
            </Typography>
            <ResponsiveContainer height={200}>
              <BarChart data={metrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="episode" />
                <YAxis />
                <Tooltip />
                <Bar 
                  dataKey="rewardBreakdown.profitReward" 
                  stackId="a" 
                  fill="#8884d8" 
                  name="Profit"
                />
                <Bar 
                  dataKey="rewardBreakdown.riskReward" 
                  stackId="a" 
                  fill="#82ca9d" 
                  name="Risk"
                />
                <Bar 
                  dataKey="rewardBreakdown.timeDecay" 
                  stackId="a" 
                  fill="#ffc658" 
                  name="Time Decay"
                />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Episode Comparison */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Episode Comparison
            </Typography>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} md={4}>
                <FormControl fullWidth>
                  <InputLabel>Episode 1</InputLabel>
                  <Select
                    value={selectedEpisodes?.episode1?.episode || ''}
                    onChange={(e) => {
                      const episode = metrics.find(m => m.episode === e.target.value);
                      if (episode) {
                        setSelectedEpisodes(prev => ({
                          episode1: episode,
                          episode2: prev?.episode2 || episode
                        }));
                      }
                    }}
                  >
                    {metrics.map(m => (
                      <MenuItem key={m.episode} value={m.episode}>
                        Episode {m.episode}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={4}>
                <FormControl fullWidth>
                  <InputLabel>Episode 2</InputLabel>
                  <Select
                    value={selectedEpisodes?.episode2?.episode || ''}
                    onChange={(e) => {
                      const episode = metrics.find(m => m.episode === e.target.value);
                      if (episode) {
                        setSelectedEpisodes(prev => ({
                          episode1: prev?.episode1 || episode,
                          episode2: episode
                        }));
                      }
                    }}
                  >
                    {metrics.map(m => (
                      <MenuItem key={m.episode} value={m.episode}>
                        Episode {m.episode}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
            {selectedEpisodes && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2">
                  Comparison Results:
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Typography>
                      Episode {selectedEpisodes.episode1.episode}:
                      Total Reward: {selectedEpisodes.episode1.totalReward.toFixed(2)}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Typography>
                      Episode {selectedEpisodes.episode2.episode}:
                      Total Reward: {selectedEpisodes.episode2.totalReward.toFixed(2)}
                    </Typography>
                  </Grid>
                </Grid>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default RLTrainingDashboard;
