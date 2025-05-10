/**
 * Analysis Dashboard
 * 
 * This component provides a dashboard for forex analysis, including
 * confluence detection, divergence analysis, and pattern recognition.
 */

import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Container, 
  Tab, 
  Tabs, 
  Typography, 
  AppBar, 
  Toolbar, 
  IconButton, 
  Menu, 
  MenuItem, 
  Drawer, 
  List, 
  ListItem, 
  ListItemIcon, 
  ListItemText, 
  Divider, 
  Paper, 
  Grid, 
  Card, 
  CardContent, 
  CardHeader, 
  Button 
} from '@mui/material';
import { 
  Menu as MenuIcon, 
  Dashboard, 
  TrendingUp, 
  CompareArrows, 
  Timeline, 
  Settings, 
  Refresh, 
  Save, 
  Share, 
  Notifications 
} from '@mui/icons-material';

import ConfluenceDetectionWidget from './ConfluenceDetectionWidget';
import DivergenceAnalysisWidget from './DivergenceAnalysisWidget';
import PatternRecognitionWidget from './PatternRecognitionWidget';
import CurrencyStrengthWidget from './CurrencyStrengthWidget';
import PerformanceMetricsWidget from './PerformanceMetricsWidget';
import RecentAnalysisWidget from './RecentAnalysisWidget';
import { getSystemStatus } from '../api/systemApi';

// TabPanel component for tab content
function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`analysis-tabpanel-${index}`}
      aria-labelledby={`analysis-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

// Tab props
function a11yProps(index) {
  return {
    id: `analysis-tab-${index}`,
    'aria-controls': `analysis-tabpanel-${index}`,
  };
}

const AnalysisDashboard = () => {
  // State
  const [tabValue, setTabValue] = useState(0);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [menuAnchorEl, setMenuAnchorEl] = useState(null);
  const [systemStatus, setSystemStatus] = useState({
    status: 'loading',
    components: {}
  });
  const [recentAnalyses, setRecentAnalyses] = useState([]);
  const [confluenceResults, setConfluenceResults] = useState(null);
  const [divergenceResults, setDivergenceResults] = useState(null);

  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  // Handle drawer toggle
  const toggleDrawer = () => {
    setDrawerOpen(!drawerOpen);
  };

  // Handle menu open
  const handleMenuOpen = (event) => {
    setMenuAnchorEl(event.currentTarget);
  };

  // Handle menu close
  const handleMenuClose = () => {
    setMenuAnchorEl(null);
  };

  // Fetch system status
  useEffect(() => {
    const fetchSystemStatus = async () => {
      try {
        const status = await getSystemStatus();
        setSystemStatus(status);
      } catch (error) {
        console.error('Error fetching system status:', error);
        setSystemStatus({
          status: 'error',
          message: error.message,
          components: {}
        });
      }
    };

    fetchSystemStatus();
    const interval = setInterval(fetchSystemStatus, 60000); // Refresh every minute

    return () => clearInterval(interval);
  }, []);

  // Handle confluence results
  const handleConfluenceResults = (results) => {
    setConfluenceResults(results);
    
    // Add to recent analyses
    if (results) {
      const analysis = {
        id: results.request_id,
        type: 'confluence',
        symbol: results.symbol,
        timeframe: results.timeframe,
        score: results.confluence_score,
        timestamp: new Date().toISOString(),
        details: `${results.signal_type} - ${results.signal_direction}`
      };
      
      setRecentAnalyses(prev => [analysis, ...prev].slice(0, 10));
    }
  };

  // Handle divergence results
  const handleDivergenceResults = (results) => {
    setDivergenceResults(results);
    
    // Add to recent analyses
    if (results) {
      const analysis = {
        id: results.request_id,
        type: 'divergence',
        symbol: results.symbol,
        timeframe: results.timeframe,
        score: results.divergence_score,
        timestamp: new Date().toISOString(),
        details: `${results.divergences_found} divergences found`
      };
      
      setRecentAnalyses(prev => [analysis, ...prev].slice(0, 10));
    }
  };

  return (
    <Box sx={{ display: 'flex' }}>
      {/* App Bar */}
      <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={toggleDrawer}
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>
          
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            Forex Analysis Dashboard
          </Typography>
          
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Typography variant="body2" sx={{ mr: 2 }}>
              System Status: 
              <span style={{ 
                color: systemStatus.status === 'healthy' ? '#4caf50' : 
                       systemStatus.status === 'degraded' ? '#ff9800' : 
                       systemStatus.status === 'error' ? '#f44336' : '#bdbdbd',
                marginLeft: '8px',
                fontWeight: 'bold'
              }}>
                {systemStatus.status === 'loading' ? 'Loading...' : 
                 systemStatus.status.charAt(0).toUpperCase() + systemStatus.status.slice(1)}
              </span>
            </Typography>
            
            <IconButton color="inherit" aria-label="refresh">
              <Refresh />
            </IconButton>
            
            <IconButton color="inherit" aria-label="notifications">
              <Notifications />
            </IconButton>
            
            <IconButton
              color="inherit"
              aria-label="settings"
              aria-controls="menu-appbar"
              aria-haspopup="true"
              onClick={handleMenuOpen}
            >
              <Settings />
            </IconButton>
            
            <Menu
              id="menu-appbar"
              anchorEl={menuAnchorEl}
              anchorOrigin={{
                vertical: 'top',
                horizontal: 'right',
              }}
              keepMounted
              transformOrigin={{
                vertical: 'top',
                horizontal: 'right',
              }}
              open={Boolean(menuAnchorEl)}
              onClose={handleMenuClose}
            >
              <MenuItem onClick={handleMenuClose}>Profile</MenuItem>
              <MenuItem onClick={handleMenuClose}>Settings</MenuItem>
              <MenuItem onClick={handleMenuClose}>Logout</MenuItem>
            </Menu>
          </Box>
        </Toolbar>
      </AppBar>
      
      {/* Drawer */}
      <Drawer
        variant="persistent"
        anchor="left"
        open={drawerOpen}
        sx={{
          width: 240,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: 240,
            boxSizing: 'border-box',
            top: '64px',
            height: 'calc(100% - 64px)'
          },
        }}
      >
        <List>
          <ListItem button onClick={() => setTabValue(0)}>
            <ListItemIcon>
              <Dashboard />
            </ListItemIcon>
            <ListItemText primary="Dashboard" />
          </ListItem>
          
          <ListItem button onClick={() => setTabValue(1)}>
            <ListItemIcon>
              <TrendingUp />
            </ListItemIcon>
            <ListItemText primary="Confluence Detection" />
          </ListItem>
          
          <ListItem button onClick={() => setTabValue(2)}>
            <ListItemIcon>
              <CompareArrows />
            </ListItemIcon>
            <ListItemText primary="Divergence Analysis" />
          </ListItem>
          
          <ListItem button onClick={() => setTabValue(3)}>
            <ListItemIcon>
              <Timeline />
            </ListItemIcon>
            <ListItemText primary="Pattern Recognition" />
          </ListItem>
        </List>
        
        <Divider />
        
        <List>
          <ListItem button>
            <ListItemIcon>
              <Settings />
            </ListItemIcon>
            <ListItemText primary="Settings" />
          </ListItem>
        </List>
      </Drawer>
      
      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { sm: `calc(100% - ${drawerOpen ? 240 : 0}px)` },
          ml: { sm: `${drawerOpen ? 240 : 0}px` },
          mt: '64px',
          transition: theme => theme.transitions.create(['margin', 'width'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
        }}
      >
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={handleTabChange} aria-label="analysis tabs">
            <Tab label="Dashboard" {...a11yProps(0)} />
            <Tab label="Confluence Detection" {...a11yProps(1)} />
            <Tab label="Divergence Analysis" {...a11yProps(2)} />
            <Tab label="Pattern Recognition" {...a11yProps(3)} />
          </Tabs>
        </Box>
        
        {/* Dashboard Tab */}
        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Paper elevation={3} sx={{ p: 2, mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Recent Analysis Results
                </Typography>
                <RecentAnalysisWidget analyses={recentAnalyses} />
              </Paper>
              
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Paper elevation={3} sx={{ p: 2 }}>
                    <Typography variant="h6" gutterBottom>
                      Currency Strength
                    </Typography>
                    <CurrencyStrengthWidget />
                  </Paper>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Paper elevation={3} sx={{ p: 2 }}>
                    <Typography variant="h6" gutterBottom>
                      Performance Metrics
                    </Typography>
                    <PerformanceMetricsWidget />
                  </Paper>
                </Grid>
              </Grid>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Paper elevation={3} sx={{ p: 2, mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                  System Status
                </Typography>
                <Box>
                  {Object.entries(systemStatus.components || {}).map(([name, status]) => (
                    <Box key={name} sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2">{name}</Typography>
                      <Typography 
                        variant="body2" 
                        sx={{ 
                          color: status === 'healthy' ? 'success.main' : 
                                 status === 'degraded' ? 'warning.main' : 
                                 'error.main'
                        }}
                      >
                        {status.charAt(0).toUpperCase() + status.slice(1)}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              </Paper>
              
              <Paper elevation={3} sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Quick Actions
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <Button variant="contained" startIcon={<TrendingUp />} onClick={() => setTabValue(1)}>
                    New Confluence Analysis
                  </Button>
                  <Button variant="contained" startIcon={<CompareArrows />} onClick={() => setTabValue(2)}>
                    New Divergence Analysis
                  </Button>
                  <Button variant="contained" startIcon={<Timeline />} onClick={() => setTabValue(3)}>
                    New Pattern Recognition
                  </Button>
                  <Button variant="outlined" startIcon={<Save />}>
                    Save Dashboard
                  </Button>
                  <Button variant="outlined" startIcon={<Share />}>
                    Share Results
                  </Button>
                </Box>
              </Paper>
            </Grid>
          </Grid>
        </TabPanel>
        
        {/* Confluence Detection Tab */}
        <TabPanel value={tabValue} index={1}>
          <ConfluenceDetectionWidget onResultsChange={handleConfluenceResults} />
        </TabPanel>
        
        {/* Divergence Analysis Tab */}
        <TabPanel value={tabValue} index={2}>
          <DivergenceAnalysisWidget onResultsChange={handleDivergenceResults} />
        </TabPanel>
        
        {/* Pattern Recognition Tab */}
        <TabPanel value={tabValue} index={3}>
          <PatternRecognitionWidget />
        </TabPanel>
      </Box>
    </Box>
  );
};

export default AnalysisDashboard;
