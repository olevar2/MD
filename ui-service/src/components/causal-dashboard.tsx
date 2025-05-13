import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Button, 
  Card, 
  CardContent, 
  CircularProgress, 
  FormControl, 
  Grid, 
  InputLabel, 
  MenuItem, 
  Select, 
  TextField, 
  Typography, 
  Paper, 
  Divider,
  Chip
} from '@mui/material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import Plot from 'react-plotly.js';
import { format, subDays } from 'date-fns';
import NetworkGraph from './NetworkGraph';

interface CausalDashboardProps {
  // Props can be expanded as needed
}

const CausalDashboard: React.FC<CausalDashboardProps> = () => {
  // State for form inputs
  const [symbols, setSymbols] = useState<string[]>([]);
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>([]);
  const [timeframes, setTimeframes] = useState<string[]>([]);
  const [selectedTimeframe, setSelectedTimeframe] = useState('1h');
  const [startDate, setStartDate] = useState<Date>(subDays(new Date(), 30));
  const [endDate, setEndDate] = useState<Date>(new Date());
  const [method, setMethod] = useState('granger');

  // State for intervention form
  const [targetSymbol, setTargetSymbol] = useState('');
  const [interventions, setInterventions] = useState<{ [key: string]: number }>({});

  // State for visualizations
  const [causalGraph, setCausalGraph] = useState<any>(null);
  const [graphVisualization, setGraphVisualization] = useState<any>(null);
  const [interventionVisualization, setInterventionVisualization] = useState<any>(null);
  const [counterfactualVisualization, setCounterfactualVisualization] = useState<any>(null);

  // Loading states
  const [loadingSymbols, setLoadingSymbols] = useState(false);
  const [loadingGraph, setLoadingGraph] = useState(false);
  const [loadingIntervention, setLoadingIntervention] = useState(false);
  const [loadingCounterfactual, setLoadingCounterfactual] = useState(false);

  // Error states
  const [error, setError] = useState('');

  // Fetch available symbols and timeframes on component mount
  useEffect(() => {
    const fetchSymbolsAndTimeframes = async () => {
      setLoadingSymbols(true);
      try {
        const [symbolsResponse, timeframesResponse] = await Promise.all([
          fetch('/api/v1/causal-visualization/available-symbols'),
          fetch('/api/v1/causal-visualization/available-timeframes')
        ]);
        
        if (!symbolsResponse.ok || !timeframesResponse.ok) {
          throw new Error('Failed to fetch symbols or timeframes');
        }
        
        const symbolsData = await symbolsResponse.json();
        const timeframesData = await timeframesResponse.json();
        
        setSymbols(symbolsData.symbols);
        setTimeframes(timeframesData.timeframes);
        
        // Set default selections
        if (symbolsData.symbols.length > 0) {
          setSelectedSymbols([symbolsData.symbols[0]]);
          setTargetSymbol(symbolsData.symbols[0]);
        }
      } catch (error) {
        console.error('Error fetching data:', error);
        setError('Failed to load symbols and timeframes');
      } finally {
        setLoadingSymbols(false);
      }
    };
    
    fetchSymbolsAndTimeframes();
  }, []);

  // Handle form submissions
  const handleGenerateCausalGraph = async () => {
    setLoadingGraph(true);
    setError('');
    
    try {
      const response = await fetch('/api/v1/causal-visualization/causal-graph', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbols: selectedSymbols,
          timeframe: selectedTimeframe,
          start_date: format(startDate, 'yyyy-MM-dd'),
          end_date: format(endDate, 'yyyy-MM-dd'),
          method: method
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to generate causal graph');
      }
      
      const data = await response.json();
      setCausalGraph(data.graph);
      setGraphVisualization(JSON.parse(data.visualization));
    } catch (error) {
      console.error('Error generating causal graph:', error);
      setError('Failed to generate causal graph');
    } finally {
      setLoadingGraph(false);
    }
  };

  const handleGenerateInterventionEffect = async () => {
    if (!targetSymbol || Object.keys(interventions).length === 0) {
      setError('Please select a target symbol and define at least one intervention');
      return;
    }
    
    setLoadingIntervention(true);
    setError('');
    
    try {
      const response = await fetch('/api/v1/causal-visualization/intervention-effect', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbols: selectedSymbols,
          timeframe: selectedTimeframe,
          target_symbol: targetSymbol,
          interventions: interventions,
          start_date: format(startDate, 'yyyy-MM-dd'),
          end_date: format(endDate, 'yyyy-MM-dd'),
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to generate intervention effect');
      }
      
      const data = await response.json();
      setInterventionVisualization(JSON.parse(data.visualization));
    } catch (error) {
      console.error('Error generating intervention effect:', error);
      setError('Failed to generate intervention effect');
    } finally {
      setLoadingIntervention(false);
    }
  };

  const handleGenerateCounterfactual = async () => {
    if (!targetSymbol || Object.keys(interventions).length === 0) {
      setError('Please select a target symbol and define at least one intervention');
      return;
    }
    
    setLoadingCounterfactual(true);
    setError('');
    
    try {
      const response = await fetch('/api/v1/causal-visualization/counterfactual-scenario', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          scenario_name: `Scenario_${targetSymbol}`,
          symbols: selectedSymbols,
          timeframe: selectedTimeframe,
          target_symbol: targetSymbol,
          interventions: interventions,
          start_date: format(startDate, 'yyyy-MM-dd'),
          end_date: format(endDate, 'yyyy-MM-dd'),
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to generate counterfactual scenario');
      }
      
      const data = await response.json();
      setCounterfactualVisualization({
        radarChart: JSON.parse(data.radar_chart),
        pathChart: data.path_chart
      });
    } catch (error) {
      console.error('Error generating counterfactual scenario:', error);
      setError('Failed to generate counterfactual scenario');
    } finally {
      setLoadingCounterfactual(false);
    }
  };

  // Handle adding an intervention
  const handleAddIntervention = (symbol: string, value: number) => {
    setInterventions({
      ...interventions,
      [`${symbol}_close`]: value
    });
  };

  // Handle removing an intervention
  const handleRemoveIntervention = (key: string) => {
    const updatedInterventions = { ...interventions };
    delete updatedInterventions[key];
    setInterventions(updatedInterventions);
  };

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Causal Inference Dashboard
      </Typography>
      
      {error && (
        <Paper sx={{ p: 2, mb: 3, bgcolor: '#FFF4E5' }}>
          <Typography color="error">{error}</Typography>
        </Paper>
      )}
      
      <Grid container spacing={3}>
        {/* Input Parameters Section */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Analysis Parameters
              </Typography>
              
              <FormControl fullWidth margin="normal">
                <InputLabel id="symbols-select-label">Symbols</InputLabel>
                <Select
                  labelId="symbols-select-label"
                  multiple
                  value={selectedSymbols}
                  onChange={(e) => setSelectedSymbols(e.target.value as string[])}
                  disabled={loadingSymbols}
                  label="Symbols"
                >
                  {symbols.map((symbol) => (
                    <MenuItem key={symbol} value={symbol}>
                      {symbol}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              
              <FormControl fullWidth margin="normal">
                <InputLabel id="timeframe-select-label">Timeframe</InputLabel>
                <Select
                  labelId="timeframe-select-label"
                  value={selectedTimeframe}
                  onChange={(e) => setSelectedTimeframe(e.target.value)}
                  disabled={loadingSymbols}
                  label="Timeframe"
                >
                  {timeframes.map((tf) => (
                    <MenuItem key={tf} value={tf}>
                      {tf}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              
              <FormControl fullWidth margin="normal">
                <InputLabel id="method-select-label">Causal Method</InputLabel>
                <Select
                  labelId="method-select-label"
                  value={method}
                  onChange={(e) => setMethod(e.target.value)}
                  label="Causal Method"
                >
                  <MenuItem value="granger">Granger Causality</MenuItem>
                  <MenuItem value="pc">PC Algorithm</MenuItem>
                </Select>
              </FormControl>
              
              <LocalizationProvider dateAdapter={AdapterDateFns}>
                <FormControl fullWidth margin="normal">
                  <DatePicker
                    label="Start Date"
                    value={startDate}
                    onChange={(newValue) => newValue && setStartDate(newValue)}
                  />
                </FormControl>
                
                <FormControl fullWidth margin="normal">
                  <DatePicker
                    label="End Date"
                    value={endDate}
                    onChange={(newValue) => newValue && setEndDate(newValue)}
                  />
                </FormControl>
              </LocalizationProvider>
              
              <Button 
                variant="contained" 
                color="primary" 
                fullWidth 
                onClick={handleGenerateCausalGraph}
                disabled={loadingGraph || selectedSymbols.length === 0}
                sx={{ mt: 2 }}
              >
                {loadingGraph ? <CircularProgress size={24} /> : 'Generate Causal Graph'}
              </Button>
            </CardContent>
          </Card>
          
          {/* Intervention Section */}
          <Card sx={{ mt: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Intervention Settings
              </Typography>
              
              <FormControl fullWidth margin="normal">
                <InputLabel id="target-symbol-label">Target Symbol</InputLabel>
                <Select
                  labelId="target-symbol-label"
                  value={targetSymbol}
                  onChange={(e) => setTargetSymbol(e.target.value)}
                  label="Target Symbol"
                >
                  {selectedSymbols.map((symbol) => (
                    <MenuItem key={symbol} value={symbol}>
                      {symbol}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              
              <Typography variant="subtitle2" sx={{ mt: 2 }}>
                Add Interventions:
              </Typography>
              
              {selectedSymbols.filter(s => s !== targetSymbol).map((symbol) => (
                <Grid container spacing={2} alignItems="center" key={symbol} sx={{ mt: 1 }}>
                  <Grid item xs={6}>
                    <Typography variant="body2">{symbol}</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      type="number"
                      size="small"
                      label="Value"
                      onChange={(e) => {
                        const value = parseFloat(e.target.value);
                        if (!isNaN(value)) {
                          handleAddIntervention(symbol, value);
                        }
                      }}
                    />
                  </Grid>
                </Grid>
              ))}
              
              <Typography variant="subtitle2" sx={{ mt: 2 }}>
                Current Interventions:
              </Typography>
              
              <Box sx={{ mt: 1, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {Object.entries(interventions).map(([key, value]) => (
                  <Chip
                    key={key}
                    label={`${key}: ${value}`}
                    onDelete={() => handleRemoveIntervention(key)}
                    color="primary"
                    variant="outlined"
                  />
                ))}
              </Box>
              
              <Button 
                variant="contained" 
                color="secondary" 
                fullWidth 
                onClick={handleGenerateInterventionEffect}
                disabled={loadingIntervention || !targetSymbol || Object.keys(interventions).length === 0}
                sx={{ mt: 2 }}
              >
                {loadingIntervention ? <CircularProgress size={24} /> : 'Generate Intervention Effect'}
              </Button>
              
              <Button 
                variant="outlined" 
                color="secondary" 
                fullWidth 
                onClick={handleGenerateCounterfactual}
                disabled={loadingCounterfactual || !targetSymbol || Object.keys(interventions).length === 0}
                sx={{ mt: 2 }}
              >
                {loadingCounterfactual ? <CircularProgress size={24} /> : 'Generate Counterfactual'}
              </Button>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Visualization Section */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Causal Graph
            </Typography>
            <Divider sx={{ mb: 2 }} />
            
            {loadingGraph ? (
              <Box display="flex" justifyContent="center" p={4}>
                <CircularProgress />
              </Box>
            ) : graphVisualization ? (
              <Box sx={{ height: 500 }}>
                {causalGraph && <NetworkGraph graph={causalGraph} />}
              </Box>
            ) : (
              <Typography color="textSecondary" align="center">
                Generate a causal graph to visualize relationships
              </Typography>
            )}
          </Paper>
          
          <Paper sx={{ p: 2, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Intervention Effect Analysis
            </Typography>
            <Divider sx={{ mb: 2 }} />
            
            {loadingIntervention ? (
              <Box display="flex" justifyContent="center" p={4}>
                <CircularProgress />
              </Box>
            ) : interventionVisualization ? (
              <Box>
                <Plot
                  data={interventionVisualization.data}
                  layout={interventionVisualization.layout}
                  config={{ responsive: true }}
                  style={{ width: '100%', height: '500px' }}
                />
              </Box>
            ) : (
              <Typography color="textSecondary" align="center">
                Generate intervention effects to visualize impacts
              </Typography>
            )}
          </Paper>
          
          {counterfactualVisualization && (
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Counterfactual Scenario Analysis
              </Typography>
              <Divider sx={{ mb: 2 }} />
              
              <Box>
                <Plot
                  data={counterfactualVisualization.radarChart.data}
                  layout={counterfactualVisualization.radarChart.layout}
                  config={{ responsive: true }}
                  style={{ width: '100%', height: '400px' }}
                />
              </Box>
            </Paper>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default CausalDashboard;
