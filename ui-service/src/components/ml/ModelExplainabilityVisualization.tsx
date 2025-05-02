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
  Alert,
  TextField,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip
} from '@mui/material';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  LineChart,
  Line,
  Scatter,
  ScatterChart,
  Cell,
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Treemap
} from 'recharts';
import { mlService } from '../../services/mlService';

interface Feature {
  name: string;
  importance: number;
  category: string;
  timeframe?: string;
}

interface StateActionValue {
  action: string;
  value: number;
  probability: number;
}

interface AttentionPoint {
  featureName: string;
  attentionWeight: number;
}

interface CriticalState {
  id: string;
  timestamp: string;
  description: string;
  marketRegime: string;
  features: Feature[];
  actions: StateActionValue[];
  attentionWeights: AttentionPoint[];
  selectedAction: string;
  reward: number;
}

interface ModelExplainabilityData {
  featureImportance: Feature[];
  stateActionAnalysis: {
    currentStateValues: StateActionValue[];
    attentionWeights: AttentionPoint[];
  };
  criticalStates: CriticalState[];
}

interface ModelExplainabilityVisualizationProps {
  modelId: string;
  symbol: string;
  timeframe: string;
}

/**
 * Model Explainability Visualization Component
 * 
 * This component provides comprehensive visualization tools for understanding
 * RL model decision-making processes, feature importance, and critical state analysis.
 */
const ModelExplainabilityVisualization: React.FC<ModelExplainabilityVisualizationProps> = ({
  modelId,
  symbol,
  timeframe
}) => {
  const [activeTab, setActiveTab] = useState(0);
  const [explainabilityData, setExplainabilityData] = useState<ModelExplainabilityData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedCriticalState, setSelectedCriticalState] = useState<string | null>(null);
  const [featureCategoryFilter, setFeatureCategoryFilter] = useState<string>('all');
  const [counterfactualFeatures, setCounterfactualFeatures] = useState<Feature[]>([]);

  // Fetch explainability data for the specified model
  useEffect(() => {
    setLoading(true);
    setError(null);
    
    mlService.getModelExplainabilityData(modelId)
      .then(data => {
        setExplainabilityData(data);
        if (data.criticalStates.length > 0) {
          setSelectedCriticalState(data.criticalStates[0].id);
          setCounterfactualFeatures([...data.criticalStates[0].features]);
        }
      })
      .catch(err => {
        console.error("Failed to load model explainability data:", err);
        setError("Failed to load explainability data. Please try again.");
      })
      .finally(() => {
        setLoading(false);
      });
  }, [modelId]);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleFeatureCategoryFilterChange = (event: React.ChangeEvent<{ value: unknown }>) => {
    setFeatureCategoryFilter(event.target.value as string);
  };

  const handleCounterfactualFeatureChange = (name: string, value: number) => {
    setCounterfactualFeatures(prev => 
      prev.map(feature => 
        feature.name === name ? { ...feature, importance: value } : feature
      )
    );
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

  if (!explainabilityData) {
    return (
      <Alert severity="info">No explainability data available for this model.</Alert>
    );
  }

  const criticalState = explainabilityData.criticalStates.find(
    state => state.id === selectedCriticalState
  );

  return (
    <Box sx={{ width: '100%' }}>
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h5" component="h2">
            Model Explainability - {modelId} ({symbol} {timeframe})
          </Typography>
        </Box>
        
        <Tabs value={activeTab} onChange={handleTabChange} aria-label="Model explainability tabs">
          <Tab label="Feature Importance" id="tab-0" />
          <Tab label="State-Action Analysis" id="tab-1" />
          <Tab label="Critical States" id="tab-2" />
          <Tab label="Counterfactual Analysis" id="tab-3" />
        </Tabs>

        <Box sx={{ mt: 2 }}>
          {/* Feature Importance Tab */}
          {activeTab === 0 && (
            <FeatureImportancePanel 
              features={explainabilityData.featureImportance} 
              categoryFilter={featureCategoryFilter}
              onCategoryFilterChange={handleFeatureCategoryFilterChange}
            />
          )}

          {/* State-Action Analysis Tab */}
          {activeTab === 1 && (
            <StateActionPanel 
              stateActionValues={explainabilityData.stateActionAnalysis.currentStateValues}
              attentionWeights={explainabilityData.stateActionAnalysis.attentionWeights}
            />
          )}

          {/* Critical States Tab */}
          {activeTab === 2 && (
            <CriticalStatesPanel 
              criticalStates={explainabilityData.criticalStates}
              selectedStateId={selectedCriticalState}
              onStateSelect={setSelectedCriticalState}
            />
          )}

          {/* Counterfactual Analysis Tab */}
          {activeTab === 3 && criticalState && (
            <CounterfactualPanel 
              originalState={criticalState}
              modifiedFeatures={counterfactualFeatures}
              onFeatureChange={handleCounterfactualFeatureChange}
            />
          )}
        </Box>
      </Paper>
    </Box>
  );
};

/**
 * Feature Importance Panel
 * 
 * Visualizes which features have the most influence on the RL model's
 * decision-making process.
 */
const FeatureImportancePanel: React.FC<{ 
  features: Feature[]; 
  categoryFilter: string;
  onCategoryFilterChange: (event: React.ChangeEvent<{ value: unknown }>) => void;
}> = ({ features, categoryFilter, onCategoryFilterChange }) => {
  // Get unique feature categories
  const categories = ['all', ...Array.from(new Set(features.map(f => f.category)))];
  
  // Filter features by category
  const filteredFeatures = categoryFilter === 'all' 
    ? features 
    : features.filter(f => f.category === categoryFilter);
  
  // Sort features by importance
  const sortedFeatures = [...filteredFeatures].sort((a, b) => b.importance - a.importance);
  
  // Top 10 features for visualization
  const topFeatures = sortedFeatures.slice(0, 10);
  
  // Calculate importance by category
  const importanceByCategory = categories
    .filter(cat => cat !== 'all')
    .map(category => ({
      category,
      totalImportance: features
        .filter(f => f.category === category)
        .reduce((sum, feature) => sum + feature.importance, 0)
    }))
    .sort((a, b) => b.totalImportance - a.totalImportance);

  return (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <FormControl variant="outlined" sx={{ minWidth: 200, mr: 2 }}>
            <InputLabel id="category-filter-label">Feature Category</InputLabel>
            <Select
              labelId="category-filter-label"
              value={categoryFilter}
              onChange={onCategoryFilterChange}
              label="Feature Category"
            >
              {categories.map(category => (
                <MenuItem key={category} value={category}>
                  {category.charAt(0).toUpperCase() + category.slice(1)}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Typography variant="body2" color="textSecondary">
            {filteredFeatures.length} features shown
          </Typography>
        </Box>
      </Grid>

      <Grid item xs={12} md={7}>
        <Card>
          <CardContent>
            <Typography variant="h6" component="h3">Top Feature Importance</Typography>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart
                data={topFeatures}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 120, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis 
                  type="category" 
                  dataKey="name" 
                  tick={{ fontSize: 12 }} 
                />
                <Tooltip formatter={(value) => [`${(value as number * 100).toFixed(2)}%`, 'Importance']} />
                <Legend />
                <Bar 
                  dataKey="importance" 
                  fill="#8884d8" 
                  name="Importance" 
                  radius={[0, 4, 4, 0]}
                  label={{ 
                    position: 'right',
                    formatter: (value: number) => `${(value * 100).toFixed(1)}%`
                  }}
                >
                  {topFeatures.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={
                      entry.category === 'price' ? '#8884d8' :
                      entry.category === 'technical' ? '#82ca9d' :
                      entry.category === 'fundamental' ? '#ffc658' :
                      entry.category === 'market_state' ? '#ff8042' :
                      '#8884d8'
                    } />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={5}>
        <Card>
          <CardContent>
            <Typography variant="h6" component="h3">Importance by Category</Typography>
            <ResponsiveContainer width="100%" height={400}>
              <PieChart>
                <Pie
                  data={importanceByCategory}
                  cx="50%"
                  cy="50%"
                  labelLine={true}
                  label={({ category, percent }) => `${category}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="totalImportance"
                >
                  {importanceByCategory.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={
                      entry.category === 'price' ? '#8884d8' :
                      entry.category === 'technical' ? '#82ca9d' :
                      entry.category === 'fundamental' ? '#ffc658' :
                      entry.category === 'market_state' ? '#ff8042' :
                      entry.category === 'orderbook' ? '#00C49F' :
                      entry.category === 'sentiment' ? '#FFBB28' :
                      `#${Math.floor(Math.random()*16777215).toString(16)}`
                    } />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => [`${(value as number * 100).toFixed(2)}%`, 'Importance']} />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" component="h3">Feature Importance Table</Typography>
            <Box sx={{ maxHeight: '400px', overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ backgroundColor: '#f5f5f5' }}>
                    <th style={{ padding: '8px', textAlign: 'left' }}>Feature</th>
                    <th style={{ padding: '8px', textAlign: 'left' }}>Category</th>
                    <th style={{ padding: '8px', textAlign: 'left' }}>Timeframe</th>
                    <th style={{ padding: '8px', textAlign: 'right' }}>Importance (%)</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedFeatures.map((feature, index) => (
                    <tr 
                      key={index}
                      style={{ backgroundColor: index % 2 === 0 ? 'white' : '#fafafa' }}
                    >
                      <td style={{ padding: '8px' }}>{feature.name}</td>
                      <td style={{ padding: '8px' }}>
                        <Chip 
                          label={feature.category} 
                          size="small"
                          style={{ 
                            backgroundColor: 
                              feature.category === 'price' ? '#e3f2fd' :
                              feature.category === 'technical' ? '#e8f5e9' :
                              feature.category === 'fundamental' ? '#fff8e1' :
                              feature.category === 'market_state' ? '#fbe9e7' :
                              feature.category === 'orderbook' ? '#e0f7fa' :
                              feature.category === 'sentiment' ? '#fff3e0' :
                              '#f5f5f5'
                          }}
                        />
                      </td>
                      <td style={{ padding: '8px' }}>{feature.timeframe || 'N/A'}</td>
                      <td style={{ padding: '8px', textAlign: 'right' }}>
                        {(feature.importance * 100).toFixed(2)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Box>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

/**
 * State-Action Analysis Panel
 * 
 * Visualizes how the model evaluates different actions in the current state.
 */
const StateActionPanel: React.FC<{ 
  stateActionValues: StateActionValue[];
  attentionWeights: AttentionPoint[];
}> = ({ stateActionValues, attentionWeights }) => {
  const sortedValues = [...stateActionValues].sort((a, b) => b.value - a.value);
  const sortedAttention = [...attentionWeights].sort((a, b) => b.attentionWeight - a.attentionWeight);
  const topAttentionFeatures = sortedAttention.slice(0, 10);

  // Define color scale for heatmap
  const getValueColor = (value: number) => {
    // Normalize to [0, 1] range assuming values are between -1 and 1
    const normalized = (value + 1) / 2; 
    
    // Color scale from red (-1) through white (0) to green (1)
    if (value < 0) {
      // Red to white gradient
      const intensity = Math.floor(255 * (1 + value));
      return `rgb(255, ${intensity}, ${intensity})`;
    } else {
      // White to green gradient
      const intensity = Math.floor(255 * (1 - value));
      return `rgb(${intensity}, 255, ${intensity})`;
    }
  };

  return (
    <Grid container spacing={2}>
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" component="h3">Action Value Heatmap</Typography>
            <Box sx={{ 
              display: 'flex', 
              flexDirection: 'column',
              alignItems: 'center',
              height: '300px',
              justifyContent: 'center'
            }}>
              {sortedValues.map((action, idx) => (
                <Box 
                  key={idx} 
                  sx={{ 
                    display: 'flex',
                    width: '100%',
                    mb: 1,
                    alignItems: 'center'
                  }}
                >
                  <Typography sx={{ width: '100px', mr: 2 }}>
                    {action.action}
                  </Typography>
                  <Box 
                    sx={{ 
                      flex: 1,
                      height: '30px',
                      backgroundColor: getValueColor(action.value),
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      borderRadius: 1,
                      position: 'relative'
                    }}
                  >
                    <Typography 
                      sx={{ 
                        color: action.value > 0.3 || action.value < -0.3 ? 'white' : 'black',
                        fontWeight: 'bold'
                      }}
                    >
                      {action.value.toFixed(2)}
                    </Typography>
                    <Box 
                      sx={{ 
                        position: 'absolute',
                        top: 0,
                        bottom: 0,
                        left: 0,
                        width: `${action.probability * 100}%`,
                        borderRight: '2px dashed rgba(0,0,0,0.3)',
                        pointerEvents: 'none'
                      }}
                    />
                  </Box>
                  <Typography sx={{ width: '60px', ml: 1, textAlign: 'right' }}>
                    {(action.probability * 100).toFixed(0)}%
                  </Typography>
                </Box>
              ))}
            </Box>
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" color="textSecondary">
                Color represents action value. Dashed line shows action probability.
              </Typography>
            </Box>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" component="h3">Attention Visualization</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={topAttentionFeatures}
                margin={{ top: 5, right: 30, left: 120, bottom: 5 }}
                layout="vertical"
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 1]} />
                <YAxis 
                  type="category" 
                  dataKey="featureName" 
                  tick={{ fontSize: 12 }} 
                />
                <Tooltip formatter={(value) => [`${(value as number * 100).toFixed(2)}%`, 'Attention']} />
                <Bar 
                  dataKey="attentionWeight" 
                  fill="#8884d8" 
                  name="Attention Weight"
                  radius={[0, 4, 4, 0]}
                  label={{ 
                    position: 'right', 
                    formatter: (value: number) => `${(value * 100).toFixed(0)}%` 
                  }}
                />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" component="h3">Action Value Distribution</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart cx="50%" cy="50%" outerRadius="80%" data={stateActionValues}>
                <PolarGrid />
                <PolarAngleAxis dataKey="action" />
                <PolarRadiusAxis angle={30} domain={[-1, 1]} />
                <Radar
                  name="Action Value"
                  dataKey="value"
                  stroke="#8884d8"
                  fill="#8884d8"
                  fillOpacity={0.6}
                />
                <Radar
                  name="Probability"
                  dataKey="probability"
                  stroke="#82ca9d"
                  fill="#82ca9d"
                  fillOpacity={0.6}
                />
                <Legend />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

/**
 * Critical States Panel
 * 
 * Allows browsing and analyzing key decision points where the model
 * made significant or interesting choices.
 */
const CriticalStatesPanel: React.FC<{ 
  criticalStates: CriticalState[];
  selectedStateId: string | null;
  onStateSelect: (stateId: string) => void;
}> = ({ criticalStates, selectedStateId, onStateSelect }) => {
  const selectedState = criticalStates.find(state => state.id === selectedStateId);
  
  if (!selectedState) {
    return (
      <Alert severity="info">No critical state selected.</Alert>
    );
  }

  return (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <Box sx={{ display: 'flex', overflowX: 'auto', py: 1 }}>
          {criticalStates.map((state) => (
            <Box 
              key={state.id} 
              sx={{ 
                px: 2, 
                py: 1, 
                mx: 0.5, 
                borderRadius: 1,
                cursor: 'pointer',
                backgroundColor: selectedStateId === state.id ? 'primary.main' : 'grey.200',
                color: selectedStateId === state.id ? 'white' : 'text.primary',
              }}
              onClick={() => onStateSelect(state.id)}
            >
              {state.marketRegime} - {state.timestamp.split('T')[0]}
            </Box>
          ))}
        </Box>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" component="h3">State Information</Typography>
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle1">
                <strong>Description:</strong> {selectedState.description}
              </Typography>
              <Typography variant="body2" sx={{ mt: 1 }}>
                <strong>Time:</strong> {new Date(selectedState.timestamp).toLocaleString()}
              </Typography>
              <Typography variant="body2">
                <strong>Market Regime:</strong> {selectedState.marketRegime}
              </Typography>
              <Typography variant="body2">
                <strong>Selected Action:</strong> {selectedState.selectedAction}
              </Typography>
              <Typography variant="body2">
                <strong>Reward:</strong> {selectedState.reward.toFixed(4)}
              </Typography>
            </Box>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" component="h3">Action Evaluation</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={selectedState.actions}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="action" />
                <YAxis domain={[-1, 1]} />
                <Tooltip />
                <Legend />
                <Bar 
                  dataKey="value" 
                  fill="#8884d8" 
                  name="Action Value"
                >
                  {selectedState.actions.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={entry.action === selectedState.selectedAction ? '#ff7300' : '#8884d8'} 
                    />
                  ))}
                </Bar>
                <Bar 
                  dataKey="probability" 
                  fill="#82ca9d" 
                  name="Probability"
                />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" component="h3">Key Features</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <TreeMap
                data={selectedState.features.map(f => ({
                  name: f.name,
                  size: Math.abs(f.importance),
                  value: f.importance,
                  category: f.category
                }))}
                dataKey="size"
                aspectRatio={4 / 3}
                stroke="#fff"
                fill="#8884d8"
              >
                <Tooltip 
                  formatter={(value, name, props) => [
                    `${(props.payload.value * 100).toFixed(2)}%`, 
                    `${props.payload.name} (${props.payload.category})`
                  ]}
                />
              </TreeMap>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

/**
 * Counterfactual Analysis Panel
 * 
 * Enables "what-if" exploration by modifying input features and
 * observing how the model's decisions would change.
 */
const CounterfactualPanel: React.FC<{ 
  originalState: CriticalState;
  modifiedFeatures: Feature[];
  onFeatureChange: (name: string, value: number) => void;
}> = ({ originalState, modifiedFeatures, onFeatureChange }) => {
  const [actionPredictions, setActionPredictions] = useState<StateActionValue[]>([]);
  const [loading, setLoading] = useState(false);

  // Simulate fetching new predictions when features change
  useEffect(() => {
    setLoading(true);
    
    // This would normally be an API call to get new predictions
    const timer = setTimeout(() => {
      // Simulate modified predictions based on feature changes
      const predictions = originalState.actions.map(action => {
        const featureChanges = modifiedFeatures.map(f => {
          const originalFeature = originalState.features.find(of => of.name === f.name);
          return originalFeature ? (f.importance - originalFeature.importance) : 0;
        });
        
        // Simple simulation of how changes might affect predictions
        const changeImpact = featureChanges.reduce((sum, change) => sum + change, 0);
        
        return {
          action: action.action,
          value: Math.max(-1, Math.min(1, action.value + changeImpact * 0.5)),
          probability: Math.max(0, Math.min(1, action.probability + changeImpact * 0.3))
        };
      });
      
      setActionPredictions(predictions);
      setLoading(false);
    }, 500);
    
    return () => clearTimeout(timer);
  }, [modifiedFeatures, originalState]);

  // Select top features for modification
  const topFeatures = [...originalState.features]
    .sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance))
    .slice(0, 5);

  return (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" component="h3">Feature Modification</Typography>
            <Typography variant="body2" color="textSecondary" sx={{ mb: 3 }}>
              Adjust feature values to see how they would affect the model's decision
            </Typography>
            
            {topFeatures.map((feature) => {
              const modifiedFeature = modifiedFeatures.find(f => f.name === feature.name);
              const value = modifiedFeature ? modifiedFeature.importance : feature.importance;
              
              return (
                <Box key={feature.name} sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body1">{feature.name}</Typography>
                    <Typography variant="body2" color="textSecondary">
                      {feature.category}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Typography variant="body2" sx={{ mr: 2, minWidth: '40px' }}>
                      {value.toFixed(2)}
                    </Typography>
                    <Slider
                      value={value}
                      min={-1}
                      max={1}
                      step={0.01}
                      onChange={(e, newValue) => onFeatureChange(feature.name, newValue as number)}
                      sx={{ flex: 1, mx: 2 }}
                      valueLabelDisplay="auto"
                      valueLabelFormat={val => val.toFixed(2)}
                    />
                    <Typography 
                      variant="body2" 
                      color={value !== feature.importance ? "primary" : "textSecondary"}
                      sx={{ ml: 2, minWidth: '100px' }}
                    >
                      {value > feature.importance ? `+${((value - feature.importance) * 100).toFixed(0)}%` : 
                      value < feature.importance ? `${((value - feature.importance) * 100).toFixed(0)}%` : 
                      'Original'}
                    </Typography>
                  </Box>
                </Box>
              );
            })}
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" component="h3">Predicted Outcome Changes</Typography>
            
            {loading ? (
              <Box display="flex" justifyContent="center" alignItems="center" height="200px">
                <CircularProgress size={30} />
              </Box>
            ) : (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={actionPredictions.map(prediction => {
                    const original = originalState.actions.find(a => a.action === prediction.action);
                    return {
                      action: prediction.action,
                      original: original ? original.value : 0,
                      modified: prediction.value,
                      originalProb: original ? original.probability : 0,
                      modifiedProb: prediction.probability,
                    };
                  })}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="action" />
                  <YAxis domain={[-1, 1]} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="original" fill="#8884d8" name="Original Value" />
                  <Bar dataKey="modified" fill="#82ca9d" name="Modified Value" />
                </BarChart>
              </ResponsiveContainer>
            )}
            
            <Box sx={{ mt: 3 }}>
              <Typography variant="subtitle1">Impact Analysis</Typography>
              <Box sx={{ mt: 1 }}>
                {!loading && actionPredictions.length > 0 && (
                  <>
                    <Typography variant="body2">
                      <strong>Original Best Action:</strong> {originalState.selectedAction} (
                      {(originalState.actions.find(a => a.action === originalState.selectedAction)?.probability || 0 * 100).toFixed(0)}% confidence)
                    </Typography>
                    
                    <Typography variant="body2" sx={{ mt: 1 }}>
                      <strong>New Predicted Action:</strong> {
                        actionPredictions.reduce((best, current) => 
                          current.probability > best.probability ? current : best
                        , actionPredictions[0]).action
                      } (
                      {(actionPredictions.reduce((best, current) => 
                        current.probability > best.probability ? current : best
                      , actionPredictions[0]).probability * 100).toFixed(0)}% confidence)
                    </Typography>
                    
                    <Typography variant="body2" sx={{ mt: 2 }}>
                      The most significant feature changes that influenced this prediction were:
                    </Typography>
                    
                    <ul>
                      {modifiedFeatures
                        .filter(f => {
                          const original = originalState.features.find(of => of.name === f.name);
                          return original && Math.abs(f.importance - original.importance) > 0.01;
                        })
                        .sort((a, b) => {
                          const aOriginal = originalState.features.find(of => of.name === a.name);
                          const bOriginal = originalState.features.find(of => of.name === b.name);
                          const aDiff = aOriginal ? Math.abs(a.importance - aOriginal.importance) : 0;
                          const bDiff = bOriginal ? Math.abs(b.importance - bOriginal.importance) : 0;
                          return bDiff - aDiff;
                        })
                        .slice(0, 3)
                        .map(f => {
                          const original = originalState.features.find(of => of.name === f.name);
                          const diff = original ? f.importance - original.importance : 0;
                          return (
                            <li key={f.name}>
                              <Typography variant="body2">
                                <strong>{f.name}</strong>: {diff > 0 ? 'Increased' : 'Decreased'} by {Math.abs(diff * 100).toFixed(0)}%
                              </Typography>
                            </li>
                          );
                        })}
                    </ul>
                  </>
                )}
              </Box>
            </Box>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

// Temporary workaround for missing PieChart in the imports
// In a real implementation, you would import this from recharts
const PieChart: React.FC<any> = ({ children }) => {
  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      {children}
    </div>
  );
};

const Pie: React.FC<any> = ({ 
  data, 
  cx, 
  cy, 
  labelLine, 
  label, 
  outerRadius, 
  fill,
  dataKey,
  children 
}) => {
  // This is a simplified placeholder, in a real implementation
  // you would use the actual Pie component from recharts
  return (
    <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <Typography>
        (Pie Chart Visualization of Categories)
      </Typography>
      {children}
    </div>
  );
};

export default ModelExplainabilityVisualization;
