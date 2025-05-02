import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  Grid, 
  Tab, 
  Tabs,
  CircularProgress,
  Button,
  Card,
  CardContent,
  Alert 
} from '@mui/material';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  BarChart,
  Bar,
  Scatter,
  ScatterChart,
  ZAxis 
} from 'recharts';
import { mlService } from '../../services/mlService';
import { RLTrainingData, AgentAction, RewardComponent } from '../../types/rlTypes';

interface RLTrainingDashboardProps {
  agentId: string;
  symbol: string;
  timeframe: string;
  onCompare?: () => void;
}

/**
 * Advanced RL Training Dashboard Component
 * 
 * This component provides comprehensive visualization for reinforcement learning
 * agent training, including training progress, agent behavior analysis, reward
 * component breakdown, and agent comparison capabilities.
 */
const RLTrainingDashboard: React.FC<RLTrainingDashboardProps> = ({ 
  agentId, 
  symbol, 
  timeframe,
  onCompare 
}) => {
  const [activeTab, setActiveTab] = useState(0);
  const [trainingData, setTrainingData] = useState<RLTrainingData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedEpisode, setSelectedEpisode] = useState<number | null>(null);

  // Fetch training data for the specified agent
  useEffect(() => {
    setLoading(true);
    setError(null);
    
    mlService.getRLTrainingData(agentId)
      .then(data => {
        setTrainingData(data);
        setSelectedEpisode(data.episodes.length > 0 ? data.episodes.length - 1 : null);
      })
      .catch(err => {
        console.error("Failed to load RL training data:", err);
        setError("Failed to load training data. Please try again.");
      })
      .finally(() => {
        setLoading(false);
      });
  }, [agentId]);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error">{error}</Alert>
    );
  }

  if (!trainingData) {
    return (
      <Alert severity="info">No training data available for this agent.</Alert>
    );
  }

  return (
    <Box sx={{ width: '100%' }}>
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h5" component="h2">
            RL Training Dashboard - {agentId} ({symbol} {timeframe})
          </Typography>
          {onCompare && (
            <Button 
              variant="outlined" 
              color="primary" 
              onClick={onCompare}
            >
              Compare Agents
            </Button>
          )}
        </Box>
        
        <Tabs value={activeTab} onChange={handleTabChange} aria-label="RL training tabs">
          <Tab label="Training Progress" id="tab-0" />
          <Tab label="Agent Behavior" id="tab-1" />
          <Tab label="Reward Components" id="tab-2" />
          <Tab label="Parameter Evolution" id="tab-3" />
        </Tabs>

        <Box sx={{ mt: 2 }}>
          {/* Training Progress Tab */}
          {activeTab === 0 && (
            <TrainingProgressPanel data={trainingData} />
          )}

          {/* Agent Behavior Tab */}
          {activeTab === 1 && (
            <AgentBehaviorPanel 
              data={trainingData} 
              selectedEpisode={selectedEpisode}
              onEpisodeChange={setSelectedEpisode}
            />
          )}

          {/* Reward Components Tab */}
          {activeTab === 2 && (
            <RewardComponentsPanel 
              data={trainingData}
              selectedEpisode={selectedEpisode}
              onEpisodeChange={setSelectedEpisode}
            />
          )}

          {/* Parameter Evolution Tab */}
          {activeTab === 3 && (
            <ParameterEvolutionPanel data={trainingData} />
          )}
        </Box>
      </Paper>
    </Box>
  );
};

/**
 * Training Progress Panel
 * 
 * Visualizes the agent's training metrics over time, including
 * rewards, balances, Sharpe ratios, and win rates.
 */
const TrainingProgressPanel: React.FC<{ data: RLTrainingData }> = ({ data }) => {
  return (
    <Grid container spacing={2}>
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" component="h3">Rewards Over Time</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart
                data={data.episodeMetrics}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="episode" label={{ value: 'Episode', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Reward', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="reward" stroke="#8884d8" name="Episode Reward" />
                <Line type="monotone" dataKey="avgReward" stroke="#82ca9d" name="Moving Avg Reward" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" component="h3">Account Balance</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart
                data={data.episodeMetrics}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="episode" label={{ value: 'Episode', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Balance', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="finalBalance" stroke="#ff7300" name="Final Balance" />
                <Line type="monotone" dataKey="maxBalance" stroke="#387908" name="Max Balance" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" component="h3">Risk-Adjusted Returns</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart
                data={data.episodeMetrics}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="episode" label={{ value: 'Episode', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Ratio', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="sharpeRatio" stroke="#8884d8" name="Sharpe Ratio" />
                <Line type="monotone" dataKey="sortinoRatio" stroke="#82ca9d" name="Sortino Ratio" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" component="h3">Trading Performance</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart
                data={data.episodeMetrics}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="episode" label={{ value: 'Episode', position: 'insideBottom', offset: -5 }} />
                <YAxis yAxisId="left" label={{ value: 'Win Rate (%)', angle: -90, position: 'insideLeft' }} />
                <YAxis yAxisId="right" orientation="right" label={{ value: '# Trades', angle: 90, position: 'insideRight' }} />
                <Tooltip />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="winRate" stroke="#ff7300" name="Win Rate (%)" />
                <Line yAxisId="right" type="monotone" dataKey="numTrades" stroke="#387908" name="Number of Trades" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

/**
 * Agent Behavior Panel
 * 
 * Visualizes the agent's action distribution and decision patterns
 * for specific episodes.
 */
const AgentBehaviorPanel: React.FC<{ 
  data: RLTrainingData; 
  selectedEpisode: number | null;
  onEpisodeChange: (episode: number) => void;
}> = ({ data, selectedEpisode, onEpisodeChange }) => {
  
  // Filter actions for the selected episode
  const episodeActions = selectedEpisode !== null 
    ? (data.actions?.filter(action => action.episode === selectedEpisode) || []) 
    : [];

  // Calculate action distribution
  const actionCounts: Record<string, number> = {};
  episodeActions.forEach(action => {
    const actionType = action.actionType;
    actionCounts[actionType] = (actionCounts[actionType] || 0) + 1;
  });
  
  const actionDistribution = Object.entries(actionCounts).map(([actionType, count]) => ({
    actionType,
    count
  }));

  // Calculate action timing (when during the episode certain actions were taken)
  const actionTiming = episodeActions.map(action => ({
    step: action.step,
    actionType: action.actionType,
    marketRegime: action.marketState.regime,
    reward: action.reward
  }));

  return (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" component="h3">Episode Selection</Typography>
            <Box sx={{ display: 'flex', overflowX: 'auto', py: 1 }}>
              {data.episodes.map((episode) => (
                <Box 
                  key={episode} 
                  sx={{ 
                    px: 2, 
                    py: 1, 
                    mx: 0.5, 
                    borderRadius: 1,
                    cursor: 'pointer',
                    backgroundColor: selectedEpisode === episode ? 'primary.main' : 'grey.200',
                    color: selectedEpisode === episode ? 'white' : 'text.primary',
                  }}
                  onClick={() => onEpisodeChange(episode)}
                >
                  {episode}
                </Box>
              ))}
            </Box>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" component="h3">Action Distribution</Typography>
            {actionDistribution.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={actionDistribution}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="actionType" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="count" fill="#8884d8" name="Action Count" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Alert severity="info">No action data available for the selected episode</Alert>
            )}
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" component="h3">Action Timing</Typography>
            {actionTiming.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid />
                  <XAxis dataKey="step" name="Step" label={{ value: 'Step', position: 'insideBottom', offset: -5 }} />
                  <YAxis dataKey="reward" name="Reward" label={{ value: 'Reward', angle: -90, position: 'insideLeft' }} />
                  <ZAxis dataKey="actionType" name="Action" />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  <Legend />
                  <Scatter
                    name="Actions"
                    data={actionTiming}
                    fill="#8884d8"
                    shape="circle"
                  />
                </ScatterChart>
              </ResponsiveContainer>
            ) : (
              <Alert severity="info">No action data available for the selected episode</Alert>
            )}
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" component="h3">Actions by Market Regime</Typography>
            {actionTiming.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={actionTiming.reduce((acc: any[], action) => {
                    const existingEntry = acc.find(
                      entry => entry.actionType === action.actionType && entry.marketRegime === action.marketRegime
                    );
                    
                    if (existingEntry) {
                      existingEntry.count += 1;
                    } else {
                      acc.push({
                        actionType: action.actionType,
                        marketRegime: action.marketRegime,
                        count: 1
                      });
                    }
                    return acc;
                  }, [])}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="actionType" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="count" fill="#8884d8" name="Action Count" stackId="a" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Alert severity="info">No action data available for the selected episode</Alert>
            )}
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

/**
 * Reward Components Panel
 * 
 * Breaks down the different components of the reward function
 * to understand what is incentivizing the agent's behavior.
 */
const RewardComponentsPanel: React.FC<{ 
  data: RLTrainingData;
  selectedEpisode: number | null;
  onEpisodeChange: (episode: number) => void;
}> = ({ data, selectedEpisode, onEpisodeChange }) => {

  // Filter reward components for the selected episode
  const episodeRewards = selectedEpisode !== null 
    ? (data.rewardComponents?.filter(component => component.episode === selectedEpisode) || []) 
    : [];

  // Group rewards by component type
  const rewardsByComponent: Record<string, {step: number, value: number}[]> = {};
  
  episodeRewards.forEach(reward => {
    if (!rewardsByComponent[reward.componentName]) {
      rewardsByComponent[reward.componentName] = [];
    }
    
    rewardsByComponent[reward.componentName].push({
      step: reward.step,
      value: reward.value
    });
  });

  // Convert to format suitable for stacked area chart
  const rewardSteps = Array.from(new Set(episodeRewards.map(r => r.step))).sort((a, b) => a - b);
  const stackedRewardData = rewardSteps.map(step => {
    const dataPoint: any = { step };
    
    Object.keys(rewardsByComponent).forEach(component => {
      const componentData = rewardsByComponent[component].find(r => r.step === step);
      dataPoint[component] = componentData ? componentData.value : 0;
    });
    
    return dataPoint;
  });

  // Calculate cumulative reward by component
  const cumulativeByComponent = Object.keys(rewardsByComponent).map(component => {
    const totalValue = rewardsByComponent[component].reduce((sum, item) => sum + item.value, 0);
    return {
      component,
      totalValue
    };
  }).sort((a, b) => b.totalValue - a.totalValue); // Sort by descending total value

  return (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" component="h3">Episode Selection</Typography>
            <Box sx={{ display: 'flex', overflowX: 'auto', py: 1 }}>
              {data.episodes.map((episode) => (
                <Box 
                  key={episode} 
                  sx={{ 
                    px: 2, 
                    py: 1, 
                    mx: 0.5, 
                    borderRadius: 1,
                    cursor: 'pointer',
                    backgroundColor: selectedEpisode === episode ? 'primary.main' : 'grey.200',
                    color: selectedEpisode === episode ? 'white' : 'text.primary',
                  }}
                  onClick={() => onEpisodeChange(episode)}
                >
                  {episode}
                </Box>
              ))}
            </Box>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={8}>
        <Card>
          <CardContent>
            <Typography variant="h6" component="h3">Reward Components Over Time</Typography>
            {stackedRewardData.length > 0 ? (
              <ResponsiveContainer width="100%" height={400}>
                <LineChart
                  data={stackedRewardData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="step" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  {Object.keys(rewardsByComponent).map((component, index) => (
                    <Line 
                      key={component}
                      type="monotone"
                      dataKey={component}
                      stroke={`#${Math.floor(Math.random()*16777215).toString(16)}`}
                      dot={false}
                      name={component}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <Alert severity="info">No reward component data available for the selected episode</Alert>
            )}
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={4}>
        <Card>
          <CardContent>
            <Typography variant="h6" component="h3">Cumulative Impact by Component</Typography>
            {cumulativeByComponent.length > 0 ? (
              <ResponsiveContainer width="100%" height={400}>
                <BarChart
                  data={cumulativeByComponent}
                  layout="vertical"
                  margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="component" />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="totalValue" fill="#8884d8" name="Total Value" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Alert severity="info">No reward component data available for the selected episode</Alert>
            )}
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

/**
 * Parameter Evolution Panel
 * 
 * Shows how model hyperparameters and internal values change during training.
 */
const ParameterEvolutionPanel: React.FC<{ data: RLTrainingData }> = ({ data }) => {
  return (
    <Grid container spacing={2}>
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" component="h3">Exploration Rate (Epsilon)</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart
                data={data.parameterHistory?.filter(p => p.paramName === 'epsilon') || []}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="episode" label={{ value: 'Episode', position: 'insideBottom', offset: -5 }} />
                <YAxis domain={[0, 1]} label={{ value: 'Epsilon', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="value" stroke="#8884d8" name="Epsilon" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" component="h3">Learning Rate</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart
                data={data.parameterHistory?.filter(p => p.paramName === 'learning_rate') || []}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="episode" label={{ value: 'Episode', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Learning Rate', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="value" stroke="#82ca9d" name="Learning Rate" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" component="h3">Loss Values</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart
                data={data.parameterHistory?.filter(p => p.paramName === 'loss') || []}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="episode" label={{ value: 'Episode', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="value" stroke="#ff7300" name="Loss" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" component="h3">Network Gradients (Norm)</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart
                data={data.parameterHistory?.filter(p => p.paramName === 'gradient_norm') || []}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="episode" label={{ value: 'Episode', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Gradient Norm', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="value" stroke="#387908" name="Gradient Norm" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default RLTrainingDashboard;
