// RLTrainingDashboard.tsx
// Advanced Learning UI Component for visualizing RL training progress and model explainability

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell, Scatter, ScatterChart, ZAxis
} from 'recharts';
import {
  Box, Tabs, Tab, Typography, Grid, Paper, Button, CircularProgress,
  FormControl, InputLabel, MenuItem, Select, Slider, Switch, FormControlLabel
} from '@mui/material';

// Custom color palette for charts
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

interface TrainingMetrics {
  timestep: number;
  reward: number;
  episode_length: number;
  loss: number | null;
  explained_variance: number | null;
}

interface ModelComparison {
  modelName: string;
  meanReward: number;
  successRate: number;
  trainingTime: number;
}

interface ActionDistribution {
  action: string;
  count: number;
}

interface FeatureImportance {
  feature: string;
  importance: number;
}

interface RLTrainingDashboardProps {
  trainingId?: string;
  modelId?: string;
}

const RLTrainingDashboard: React.FC<RLTrainingDashboardProps> = ({ trainingId, modelId }) => {
  // State for dashboard data
  const [activeTab, setActiveTab] = useState(0);
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetrics[]>([]);
  const [modelComparisons, setModelComparisons] = useState<ModelComparison[]>([]);
  const [actionDistribution, setActionDistribution] = useState<ActionDistribution[]>([]);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [rewardComponents, setRewardComponents] = useState<any[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [autoRefresh, setAutoRefresh] = useState<boolean>(false);
  
  // Fetch data when component mounts or when selectedModel changes
  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      try {
        // Get available models
        const modelsResponse = await axios.get('/api/rl/models');
        setAvailableModels(modelsResponse.data.models);
        
        if (modelId || selectedModel) {
          const currentModelId = modelId || selectedModel;
          
          // Get training metrics
          const metricsResponse = await axios.get(`/api/rl/training/${currentModelId}/metrics`);
          setTrainingMetrics(metricsResponse.data.metrics);
          
          // Get action distribution
          const actionsResponse = await axios.get(`/api/rl/model/${currentModelId}/actions`);
          setActionDistribution(actionsResponse.data.distribution);
          
          // Get feature importance
          const featuresResponse = await axios.get(`/api/rl/model/${currentModelId}/features`);
          setFeatureImportance(featuresResponse.data.importance);
          
          // Get reward components
          const rewardsResponse = await axios.get(`/api/rl/model/${currentModelId}/rewards`);
          setRewardComponents(rewardsResponse.data.components);
          
          // If no model was previously selected, set it
          if (!selectedModel) {
            setSelectedModel(currentModelId);
          }
        }
        
        // Get model comparisons (always fetch this)
        const comparisonsResponse = await axios.get('/api/rl/models/comparison');
        setModelComparisons(comparisonsResponse.data.comparisons);
        
      } catch (error) {
        console.error("Error fetching RL data:", error);
        // Use sample data in case of error or for development
        loadSampleData();
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchData();
    
    // Set up auto-refresh if enabled
    let intervalId: number | undefined;
    if (autoRefresh) {
      intervalId = window.setInterval(fetchData, 30000); // Refresh every 30 seconds
    }
    
    // Clean up interval on unmount or when autoRefresh changes
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [modelId, selectedModel, autoRefresh]);
  
  // Load sample data for development purposes
  const loadSampleData = () => {
    // Sample training metrics
    setTrainingMetrics([
      { timestep: 1000, reward: 10.5, episode_length: 120, loss: 2.3, explained_variance: 0.6 },
      { timestep: 2000, reward: 15.2, episode_length: 130, loss: 2.1, explained_variance: 0.65 },
      { timestep: 3000, reward: 22.8, episode_length: 145, loss: 1.8, explained_variance: 0.7 },
      { timestep: 4000, reward: 28.4, episode_length: 150, loss: 1.5, explained_variance: 0.73 },
      { timestep: 5000, reward: 32.1, episode_length: 155, loss: 1.3, explained_variance: 0.75 },
      { timestep: 6000, reward: 42.6, episode_length: 160, loss: 1.1, explained_variance: 0.78 },
      { timestep: 7000, reward: 48.9, episode_length: 165, loss: 0.9, explained_variance: 0.81 },
      { timestep: 8000, reward: 55.3, episode_length: 170, loss: 0.85, explained_variance: 0.83 },
      { timestep: 9000, reward: 61.7, episode_length: 175, loss: 0.8, explained_variance: 0.85 },
      { timestep: 10000, reward: 68.2, episode_length: 180, loss: 0.75, explained_variance: 0.87 },
    ]);
    
    // Sample model comparisons
    setModelComparisons([
      { modelName: "PPO_v1", meanReward: 45.3, successRate: 0.68, trainingTime: 2450 },
      { modelName: "A2C_v1", meanReward: 38.7, successRate: 0.62, trainingTime: 1820 },
      { modelName: "SAC_v1", meanReward: 52.1, successRate: 0.72, trainingTime: 3100 },
      { modelName: "DQN_v1", meanReward: 41.5, successRate: 0.65, trainingTime: 2200 },
    ]);
    
    // Sample action distribution
    setActionDistribution([
      { action: "Buy", count: 320 },
      { action: "Sell", count: 285 },
      { action: "Hold", count: 450 },
      { action: "Close", count: 145 },
    ]);
    
    // Sample feature importance
    setFeatureImportance([
      { feature: "Close Price", importance: 0.28 },
      { feature: "SMA_20", importance: 0.22 },
      { feature: "RSI", importance: 0.18 },
      { feature: "MACD", importance: 0.15 },
      { feature: "Volume", importance: 0.12 },
      { feature: "ATR", importance: 0.05 },
    ]);
    
    // Sample reward components
    setRewardComponents([
      { name: "PnL", value: 0.65 },
      { name: "Risk Penalty", value: 0.18 },
      { name: "Transaction Cost", value: 0.12 },
      { name: "Time Decay", value: 0.05 },
    ]);
    
    // Sample available models
    setAvailableModels(["PPO_v1", "A2C_v1", "SAC_v1", "DQN_v1"]);
    
    // Set a default selected model if none
    if (!selectedModel) {
      setSelectedModel("PPO_v1");
    }
  };
  
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };
  
  const handleModelChange = (event: React.ChangeEvent<{ value: unknown }>) => {
    setSelectedModel(event.target.value as string);
  };
  
  const handleAutoRefreshChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setAutoRefresh(event.target.checked);
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
        <Tabs value={activeTab} onChange={handleTabChange} aria-label="RL dashboard tabs">
          <Tab label="Training Progress" />
          <Tab label="Agent Behavior" />
          <Tab label="Reward Analysis" />
          <Tab label="Model Comparison" />
          <Tab label="Explainability" />
        </Tabs>
      </Box>
      
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
        <FormControl variant="outlined" sx={{ minWidth: 200 }}>
          <InputLabel id="model-select-label">Select Model</InputLabel>
          <Select
            labelId="model-select-label"
            value={selectedModel}
            onChange={handleModelChange}
            label="Select Model"
          >
            {availableModels.map(model => (
              <MenuItem key={model} value={model}>{model}</MenuItem>
            ))}
          </Select>
        </FormControl>
        
        <FormControlLabel
          control={<Switch checked={autoRefresh} onChange={handleAutoRefreshChange} />}
          label="Auto Refresh"
        />
      </Box>
      
      {isLoading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress />
        </Box>
      ) : (
        <>
          {/* Training Progress Tab */}
          {activeTab === 0 && (
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  Training Metrics Over Time
                </Typography>
                <Paper sx={{ p: 2, height: 400 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={trainingMetrics}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="timestep" 
                        label={{ value: 'Timesteps', position: 'insideBottom', offset: -5 }} 
                      />
                      <YAxis yAxisId="left" label={{ value: 'Reward', angle: -90, position: 'insideLeft' }} />
                      <YAxis yAxisId="right" orientation="right" label={{ value: 'Loss', angle: 90, position: 'insideRight' }} />
                      <Tooltip />
                      <Legend />
                      <Line yAxisId="left" type="monotone" dataKey="reward" stroke={COLORS[0]} activeDot={{ r: 8 }} name="Mean Reward" />
                      <Line yAxisId="right" type="monotone" dataKey="loss" stroke={COLORS[1]} name="Loss Value" />
                      <Line yAxisId="left" type="monotone" dataKey="explained_variance" stroke={COLORS[2]} name="Explained Variance" />
                    </LineChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  Episode Length Trend
                </Typography>
                <Paper sx={{ p: 2, height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={trainingMetrics}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestep" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="episode_length" stroke={COLORS[3]} name="Episode Length" />
                    </LineChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  Training Statistics
                </Typography>
                <Paper sx={{ p: 2, height: 300 }}>
                  <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'space-around' }}>
                    {trainingMetrics.length > 0 && (
                      <>
                        <Typography variant="body1">
                          Total Training Steps: {trainingMetrics[trainingMetrics.length - 1].timestep}
                        </Typography>
                        <Typography variant="body1">
                          Latest Reward: {trainingMetrics[trainingMetrics.length - 1].reward.toFixed(2)}
                        </Typography>
                        <Typography variant="body1">
                          Best Reward: {Math.max(...trainingMetrics.map(m => m.reward)).toFixed(2)}
                        </Typography>
                        <Typography variant="body1">
                          Latest Loss: {trainingMetrics[trainingMetrics.length - 1].loss?.toFixed(3) || 'N/A'}
                        </Typography>
                        <Typography variant="body1">
                          Current Explained Variance: {
                            trainingMetrics[trainingMetrics.length - 1].explained_variance?.toFixed(3) || 'N/A'
                          }
                        </Typography>
                      </>
                    )}
                  </Box>
                </Paper>
              </Grid>
            </Grid>
          )}
          
          {/* Agent Behavior Tab */}
          {activeTab === 1 && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  Action Distribution
                </Typography>
                <Paper sx={{ p: 2, height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={actionDistribution}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="count"
                        nameKey="action"
                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                      >
                        {actionDistribution.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  Action Over Time
                </Typography>
                <Paper sx={{ p: 2, height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={[
                        { timestep: '1k', buy: 45, sell: 30, hold: 70, close: 15 },
                        { timestep: '2k', buy: 50, sell: 40, hold: 65, close: 20 },
                        { timestep: '3k', buy: 55, sell: 45, hold: 55, close: 25 },
                        { timestep: '4k', buy: 60, sell: 50, hold: 45, close: 30 },
                        { timestep: '5k', buy: 65, sell: 55, hold: 40, close: 35 },
                      ]}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestep" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="buy" fill={COLORS[0]} name="Buy" />
                      <Bar dataKey="sell" fill={COLORS[1]} name="Sell" />
                      <Bar dataKey="hold" fill={COLORS[2]} name="Hold" />
                      <Bar dataKey="close" fill={COLORS[3]} name="Close" />
                    </BarChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  Action Confidence
                </Typography>
                <Paper sx={{ p: 2, height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart
                      margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                    >
                      <CartesianGrid />
                      <XAxis type="number" dataKey="confidence" name="Confidence" unit="%" />
                      <YAxis type="number" dataKey="reward" name="Reward" />
                      <ZAxis type="number" dataKey="count" range={[50, 500]} />
                      <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                      <Legend />
                      <Scatter name="Buy" data={[
                        { confidence: 85, reward: 12, count: 85 },
                        { confidence: 90, reward: 15, count: 120 },
                        { confidence: 95, reward: 18, count: 100 },
                      ]} fill={COLORS[0]} />
                      <Scatter name="Sell" data={[
                        { confidence: 80, reward: 10, count: 70 },
                        { confidence: 85, reward: 13, count: 90 },
                        { confidence: 90, reward: 16, count: 110 },
                      ]} fill={COLORS[1]} />
                      <Scatter name="Hold" data={[
                        { confidence: 60, reward: 3, count: 120 },
                        { confidence: 70, reward: 5, count: 140 },
                        { confidence: 80, reward: 8, count: 170 },
                      ]} fill={COLORS[2]} />
                    </ScatterChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
            </Grid>
          )}
          
          {/* Reward Analysis Tab */}
          {activeTab === 2 && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  Reward Components
                </Typography>
                <Paper sx={{ p: 2, height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={rewardComponents}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                        nameKey="name"
                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                      >
                        {rewardComponents.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  Reward Distribution
                </Typography>
                <Paper sx={{ p: 2, height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={[
                        { range: '-10 to -5', count: 15 },
                        { range: '-5 to 0', count: 30 },
                        { range: '0 to 5', count: 120 },
                        { range: '5 to 10', count: 85 },
                        { range: '10 to 15', count: 45 },
                        { range: '15 to 20', count: 25 },
                        { range: '20+', count: 10 },
                      ]}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="range" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="count" fill={COLORS[0]} name="Episode Count" />
                    </BarChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  Reward Over Time by Market Regime
                </Typography>
                <Paper sx={{ p: 2, height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={[
                        { timestep: '1k', trending: 12, ranging: 8, volatile: 5, normal: 10 },
                        { timestep: '2k', trending: 15, ranging: 7, volatile: 3, normal: 12 },
                        { timestep: '3k', trending: 18, ranging: 9, volatile: 2, normal: 14 },
                        { timestep: '4k', trending: 20, ranging: 11, volatile: 4, normal: 16 },
                        { timestep: '5k', trending: 22, ranging: 13, volatile: 6, normal: 18 },
                      ]}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestep" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="trending" stroke={COLORS[0]} name="Trending Market" />
                      <Line type="monotone" dataKey="ranging" stroke={COLORS[1]} name="Ranging Market" />
                      <Line type="monotone" dataKey="volatile" stroke={COLORS[2]} name="Volatile Market" />
                      <Line type="monotone" dataKey="normal" stroke={COLORS[3]} name="Normal Market" />
                    </LineChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
            </Grid>
          )}
          
          {/* Model Comparison Tab */}
          {activeTab === 3 && (
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  Model Performance Comparison
                </Typography>
                <Paper sx={{ p: 2, height: 400 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={modelComparisons}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="modelName" />
                      <YAxis yAxisId="left" orientation="left" stroke={COLORS[0]} />
                      <YAxis yAxisId="right" orientation="right" stroke={COLORS[1]} />
                      <Tooltip />
                      <Legend />
                      <Bar yAxisId="left" dataKey="meanReward" fill={COLORS[0]} name="Mean Reward" />
                      <Bar yAxisId="right" dataKey="successRate" fill={COLORS[1]} name="Success Rate" />
                    </BarChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  Training Time Comparison
                </Typography>
                <Paper sx={{ p: 2, height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={modelComparisons}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="modelName" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="trainingTime" fill={COLORS[2]} name="Training Time (s)" />
                    </BarChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  Performance Matrix
                </Typography>
                <Paper sx={{ p: 2, height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart
                      margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                    >
                      <CartesianGrid />
                      <XAxis type="number" dataKey="trainingTime" name="Training Time" unit="s" />
                      <YAxis type="number" dataKey="meanReward" name="Mean Reward" />
                      <ZAxis type="number" dataKey="successRate" range={[40, 400]} name="Success Rate" />
                      <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                      <Legend />
                      <Scatter name="Models" data={modelComparisons} fill={COLORS[4]}>
                        {modelComparisons.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
            </Grid>
          )}
          
          {/* Explainability Tab */}
          {activeTab === 4 && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  Feature Importance
                </Typography>
                <Paper sx={{ p: 2, height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={featureImportance}
                      layout="vertical"
                      margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis type="category" dataKey="feature" />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="importance" fill={COLORS[0]} name="Importance Score" />
                    </BarChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  State-Action Heatmap
                </Typography>
                <Paper sx={{ p: 2, height: 300 }}>
                  <Typography variant="body1" align="center" sx={{ mt: 10 }}>
                    State-Action visualization will be displayed here
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  Critical Decision Analysis
                </Typography>
                <Paper sx={{ p: 2, minHeight: 200 }}>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle1">Example Critical Decision #1:</Typography>
                    <Typography variant="body2">
                      At timestep 4523, the agent decided to sell EUR/USD despite an upward trend.
                      Key factors in this decision were the RSI (overbought at 78.2) and volume decline (-15.3%).
                      This resulted in a profit of +12.4 pips before the price reversed.
                    </Typography>
                  </Box>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle1">Example Critical Decision #2:</Typography>
                    <Typography variant="body2">
                      At timestep 6812, the agent held position during high volatility news event.
                      The model assigned 68% importance to the news sentiment feature and 22% to recent price action.
                      This saved potential losses of -18.7 pips during the volatile price spike.
                    </Typography>
                  </Box>
                </Paper>
              </Grid>
            </Grid>
          )}
        </>
      )}
    </Box>
  );
};

export default RLTrainingDashboard;
