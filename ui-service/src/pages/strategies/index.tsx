/**
 * Strategy Management Page - Main interface for creating, editing, and managing strategies
 */
import { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Button, 
  Grid, 
  Card, 
  CardContent, 
  CardActions,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tabs,
  Tab
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import SettingsIcon from '@mui/icons-material/Settings';
import { useRouter } from 'next/router';

import { strategyApi } from '@/api/strategy-api';
import { Strategy, MarketRegime } from '@/types/strategy';
import StrategyForm from '@/components/strategy/StrategyForm';
import PerformanceMetricsCard from '@/components/strategy/PerformanceMetricsCard';
import ConfirmDialog from '@/components/common/ConfirmDialog';
import PageHeader from '@/components/common/PageHeader';

export default function StrategyManagement() {
  const router = useRouter();
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [loading, setLoading] = useState(true);
  const [openCreateDialog, setOpenCreateDialog] = useState(false);
  const [openDeleteDialog, setOpenDeleteDialog] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState<Strategy | null>(null);
  const [strategyToDelete, setStrategyToDelete] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState(0);
  const [tabFilters] = useState(['all', 'active', 'inactive']);

  useEffect(() => {
    loadStrategies();
  }, []);

  async function loadStrategies() {
    try {
      setLoading(true);
      const data = await strategyApi.getStrategies();
      setStrategies(data);
    } catch (error) {
      console.error('Failed to load strategies', error);
      // Here you would typically show an error notification
    } finally {
      setLoading(false);
    }
  }

  const handleCreateStrategy = () => {
    setSelectedStrategy(null);
    setOpenCreateDialog(true);
  };

  const handleEditStrategy = (strategy: Strategy) => {
    setSelectedStrategy(strategy);
    setOpenCreateDialog(true);
  };

  const handleDialogClose = () => {
    setOpenCreateDialog(false);
    setSelectedStrategy(null);
  };

  const handleDeleteClick = (id: string) => {
    setStrategyToDelete(id);
    setOpenDeleteDialog(true);
  };

  const handleDeleteConfirm = async () => {
    if (!strategyToDelete) return;
    
    try {
      await strategyApi.deleteStrategy(strategyToDelete);
      setStrategies(strategies.filter(s => s.id !== strategyToDelete));
      setOpenDeleteDialog(false);
      setStrategyToDelete(null);
    } catch (error) {
      console.error('Failed to delete strategy', error);
      // Show error notification
    }
  };

  const handleDeleteCancel = () => {
    setOpenDeleteDialog(false);
    setStrategyToDelete(null);
  };

  const handleStrategyToggle = async (strategy: Strategy) => {
    try {
      const updatedStrategy = strategy.isActive 
        ? await strategyApi.deactivateStrategy(strategy.id)
        : await strategyApi.activateStrategy(strategy.id);
        
      setStrategies(
        strategies.map(s => s.id === updatedStrategy.id ? updatedStrategy : s)
      );
    } catch (error) {
      console.error(`Failed to ${strategy.isActive ? 'deactivate' : 'activate'} strategy`, error);
      // Show error notification
    }
  };

  const handleViewBacktests = (id: string) => {
    router.push(`/strategies/${id}/backtests`);
  };

  const handleViewDetails = (id: string) => {
    router.push(`/strategies/${id}`);
  };

  const handleSubmitStrategy = async (formData: Partial<Strategy>) => {
    try {
      let updatedStrategy: Strategy;
      
      if (selectedStrategy) {
        // Update existing strategy
        updatedStrategy = await strategyApi.updateStrategy(selectedStrategy.id, formData);
        setStrategies(strategies.map(s => s.id === updatedStrategy.id ? updatedStrategy : s));
      } else {
        // Create new strategy
        updatedStrategy = await strategyApi.createStrategy(formData);
        setStrategies([...strategies, updatedStrategy]);
      }
      
      setOpenCreateDialog(false);
      setSelectedStrategy(null);
    } catch (error) {
      console.error('Failed to save strategy', error);
      // Show error notification
    }
  };

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };
  
  // Filter strategies based on active tab
  const filteredStrategies = strategies.filter(strategy => {
    if (tabFilters[activeTab] === 'all') return true;
    if (tabFilters[activeTab] === 'active') return strategy.isActive;
    if (tabFilters[activeTab] === 'inactive') return !strategy.isActive;
    return true;
  });

  const getStrategyTypeLabel = (type: string): string => {
    switch (type) {
      case 'adaptive_ma': return 'Adaptive MA';
      case 'elliott_wave': return 'Elliott Wave';
      case 'multi_timeframe_confluence': return 'Multi-TF Confluence';
      case 'harmonic_pattern': return 'Harmonic Pattern';
      case 'advanced_breakout': return 'Advanced Breakout';
      case 'custom': return 'Custom';
      default: return type;
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <PageHeader 
        title="Strategy Management" 
        description="Create, edit, and manage your trading strategies"
        action={
          <Button 
            variant="contained" 
            color="primary" 
            startIcon={<AddIcon />}
            onClick={handleCreateStrategy}
          >
            Create Strategy
          </Button>
        }
      />
      
      <Box sx={{ mb: 3 }}>
        <Tabs value={activeTab} onChange={handleTabChange}>
          <Tab label="All Strategies" />
          <Tab label="Active" />
          <Tab label="Inactive" />
        </Tabs>
      </Box>

      {loading ? (
        <Typography>Loading strategies...</Typography>
      ) : filteredStrategies.length === 0 ? (
        <Card sx={{ textAlign: 'center', py: 6 }}>
          <CardContent>
            <Typography variant="h6" color="textSecondary" gutterBottom>
              No strategies found
            </Typography>
            <Typography variant="body2" color="textSecondary" paragraph>
              Get started by creating your first trading strategy
            </Typography>
            <Button 
              variant="contained" 
              color="primary" 
              startIcon={<AddIcon />}
              onClick={handleCreateStrategy}
            >
              Create Strategy
            </Button>
          </CardContent>
        </Card>
      ) : (
        <Grid container spacing={3}>
          {filteredStrategies.map((strategy) => (
            <Grid item xs={12} md={6} lg={4} key={strategy.id}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="h6" component="h2">
                      {strategy.name}
                    </Typography>
                    <Chip 
                      size="small"
                      label={strategy.isActive ? 'Active' : 'Inactive'}
                      color={strategy.isActive ? 'success' : 'default'}
                    />
                  </Box>
                  
                  <Chip 
                    size="small" 
                    label={getStrategyTypeLabel(strategy.type)}
                    sx={{ mb: 2 }}
                  />
                  
                  <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                    {strategy.description}
                  </Typography>
                  
                  <Typography variant="body2" gutterBottom>
                    <strong>Symbols:</strong> {strategy.symbols.join(', ')}
                  </Typography>
                  
                  <Typography variant="body2" gutterBottom>
                    <strong>Timeframes:</strong> {strategy.timeframes.join(', ')}
                  </Typography>
                  
                  {strategy.performance && (
                    <PerformanceMetricsCard 
                      winRate={strategy.performance.winRate}
                      profitFactor={strategy.performance.profitFactor}
                      totalTrades={strategy.performance.totalTrades}
                    />
                  )}
                </CardContent>
                
                <CardActions>
                  <IconButton 
                    size="small"
                    onClick={() => handleStrategyToggle(strategy)}
                    title={strategy.isActive ? 'Deactivate' : 'Activate'}
                  >
                    {strategy.isActive ? <StopIcon /> : <PlayArrowIcon />}
                  </IconButton>
                  <IconButton 
                    size="small"
                    onClick={() => handleViewDetails(strategy.id)} 
                    title="View Details"
                  >
                    <SettingsIcon />
                  </IconButton>
                  <IconButton 
                    size="small"
                    onClick={() => handleViewBacktests(strategy.id)}
                    title="View Backtests"
                  >
                    <ShowChartIcon />
                  </IconButton>
                  <IconButton 
                    size="small"
                    onClick={() => handleEditStrategy(strategy)}
                    title="Edit"
                  >
                    <EditIcon />
                  </IconButton>
                  <IconButton 
                    size="small"
                    onClick={() => handleDeleteClick(strategy.id)}
                    title="Delete"
                  >
                    <DeleteIcon />
                  </IconButton>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Create/Edit Strategy Dialog */}
      <Dialog 
        open={openCreateDialog} 
        onClose={handleDialogClose}
        fullWidth
        maxWidth="md"
      >
        <DialogTitle>
          {selectedStrategy ? 'Edit Strategy' : 'Create New Strategy'}
        </DialogTitle>
        <DialogContent dividers>
          <StrategyForm 
            initialData={selectedStrategy}
            onSubmit={handleSubmitStrategy}
          />
        </DialogContent>
      </Dialog>

      {/* Confirm Delete Dialog */}
      <ConfirmDialog
        open={openDeleteDialog}
        title="Delete Strategy"
        message="Are you sure you want to delete this strategy? This action cannot be undone."
        confirmButtonText="Delete"
        cancelButtonText="Cancel"
        onConfirm={handleDeleteConfirm}
        onCancel={handleDeleteCancel}
      />
    </Box>
  );
}
