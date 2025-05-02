/**
 * Phase 4 Documentation Component - Provides documentation for the advanced features
 */
import { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tabs,
  Tab,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Link,
  Button,
  Grid,
  Card,
  CardContent,
  Alert
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import BookmarkIcon from '@mui/icons-material/Bookmark';
import TipsAndUpdatesIcon from '@mui/icons-material/TipsAndUpdates';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import CodeIcon from '@mui/icons-material/Code';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import PsychologyIcon from '@mui/icons-material/Psychology';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import TimelineIcon from '@mui/icons-material/Timeline';
import LayersIcon from '@mui/icons-material/Layers';
import InfoIcon from '@mui/icons-material/Info';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import LooksOneIcon from '@mui/icons-material/LooksOne';
import LooksTwoIcon from '@mui/icons-material/LooksTwo';
import Looks3Icon from '@mui/icons-material/Looks3';
import Looks4Icon from '@mui/icons-material/Looks4';
import Looks5Icon from '@mui/icons-material/Looks5';
import Looks6Icon from '@mui/icons-material/Looks6';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import InfoIcon from '@mui/icons-material/Info';
import LooksOneIcon from '@mui/icons-material/LooksOne';
import LooksTwoIcon from '@mui/icons-material/LooksTwo';
import Looks3Icon from '@mui/icons-material/Looks3';
import Looks4Icon from '@mui/icons-material/Looks4';
import Looks5Icon from '@mui/icons-material/Looks5';
import Looks6Icon from '@mui/icons-material/Looks6';

interface DocSection {
  id: string;
  title: string;
  content: JSX.Element;
  examples?: JSX.Element;
}

export default function PhaseDocumentation() {
  const [activeTab, setActiveTab] = useState(0);
  const [expanded, setExpanded] = useState<string | false>('overview');
  
  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };
  
  const handleAccordionChange = (panel: string) => (_event: React.SyntheticEvent, isExpanded: boolean) => {
    setExpanded(isExpanded ? panel : false);
  };
  
  const documentation: Record<string, DocSection[]> = {
    overview: [
      {
        id: 'introduction',
        title: 'Phase 4 Introduction',
        content: (
          <Box>
            <Typography paragraph>
              Phase 4 of the forex trading platform introduces advanced analytic capabilities, adaptive strategies, 
              and machine learning integration to enhance trading decision-making and performance.
            </Typography>
            
            <Typography paragraph>
              The key features of Phase 4 include:
            </Typography>
            
            <List dense>
              <ListItem>
                <ListItemIcon><ShowChartIcon color="primary" /></ListItemIcon>
                <ListItemText primary="Multi-timeframe analysis for comprehensive market understanding" />
              </ListItem>
              <ListItem>
                <ListItemIcon><LayersIcon color="primary" /></ListItemIcon>
                <ListItemText primary="Confluence highlighting to identify high-probability trading zones" />
              </ListItem>
              <ListItem>
                <ListItemIcon><PsychologyIcon color="primary" /></ListItemIcon>
                <ListItemText primary="ML model integration for signal confirmation and insights" />
              </ListItem>
              <ListItem>
                <ListItemIcon><AutoAwesomeIcon color="primary" /></ListItemIcon>
                <ListItemText primary="Adaptive strategies that adjust to changing market conditions" />
              </ListItem>
              <ListItem>
                <ListItemIcon><TimelineIcon color="primary" /></ListItemIcon>
                <ListItemText primary="Advanced pattern visualization and Elliott Wave overlays" />
              </ListItem>
            </List>
            
            <Typography paragraph>
              These features work together to provide traders with more accurate signals, better risk management,
              and improved strategy optimization capabilities.
            </Typography>
          </Box>
        )
      },
      {
        id: 'system-overview',
        title: 'System Architecture',
        content: (
          <Box>
            <Typography paragraph>
              The Phase 4 architecture adds new layers to the existing trading platform:
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" gutterBottom>UI Layer</Typography>
                    <List dense disablePadding>
                      <ListItem disableGutters>
                        <ListItemText primary="Advanced Chart Components" secondary="Enhanced visualization of patterns and timeframes" />
                      </ListItem>
                      <ListItem disableGutters>
                        <ListItemText primary="Strategy Management Interface" secondary="Creation and management of adaptive strategies" />
                      </ListItem>
                      <ListItem disableGutters>
                        <ListItemText primary="Confluence Highlighting" secondary="Visual representation of significant price zones" />
                      </ListItem>
                    </List>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" gutterBottom>ML Integration Layer</Typography>
                    <List dense disablePadding>
                      <ListItem disableGutters>
                        <ListItemText primary="Model Registry" secondary="Management and versioning of ML models" />
                      </ListItem>
                      <ListItem disableGutters>
                        <ListItemText primary="Prediction Service" secondary="Real-time inference and signal confirmation" />
                      </ListItem>
                      <ListItem disableGutters>
                        <ListItemText primary="Market Regime Detection" secondary="Automatic identification of market conditions" />
                      </ListItem>
                    </List>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" gutterBottom>Adaptive Strategy Layer</Typography>
                    <List dense disablePadding>
                      <ListItem disableGutters>
                        <ListItemText primary="Parameter Optimization" secondary="Automatic tuning based on market conditions" />
                      </ListItem>
                      <ListItem disableGutters>
                        <ListItemText primary="Multi-timeframe Signal Aggregation" secondary="Consolidation of signals from various timeframes" />
                      </ListItem>
                      <ListItem disableGutters>
                        <ListItemText primary="Performance Tracking" secondary="Regime-specific performance metrics" />
                      </ListItem>
                    </List>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" gutterBottom>Data Analysis Layer</Typography>
                    <List dense disablePadding>
                      <ListItem disableGutters>
                        <ListItemText primary="Pattern Recognition" secondary="Advanced detection of chart and harmonic patterns" />
                      </ListItem>
                      <ListItem disableGutters>
                        <ListItemText primary="Confluence Detection" secondary="Identification of overlapping indicators and levels" />
                      </ListItem>
                      <ListItem disableGutters>
                        <ListItemText primary="Historical Testing" secondary="Comprehensive backtesting across market regimes" />
                      </ListItem>
                    </List>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Box>
        )
      }
    ],
    
    features: [
      {
        id: 'multi-timeframe',
        title: 'Multi-Timeframe Analysis',
        content: (
          <Box>
            <Typography paragraph>
              The Multi-Timeframe Analysis feature allows traders to simultaneously analyze price action across 
              different timeframes to identify stronger signals and filter out noise.
            </Typography>
            
            <Typography variant="subtitle2" gutterBottom>
              Key Capabilities:
            </Typography>
            
            <List dense>
              <ListItem>
                <ListItemIcon><CheckCircleIcon color="success" /></ListItemIcon>
                <ListItemText 
                  primary="Comparative View" 
                  secondary="Side-by-side analysis of multiple timeframes" 
                />
              </ListItem>
              <ListItem>
                <ListItemIcon><CheckCircleIcon color="success" /></ListItemIcon>
                <ListItemText 
                  primary="Indicator Table" 
                  secondary="Consolidated view of technical indicators across timeframes" 
                />
              </ListItem>
              <ListItem>
                <ListItemIcon><CheckCircleIcon color="success" /></ListItemIcon>
                <ListItemText 
                  primary="Correlation Analysis" 
                  secondary="Heat map showing correlation between timeframes" 
                />
              </ListItem>
              <ListItem>
                <ListItemIcon><CheckCircleIcon color="success" /></ListItemIcon>
                <ListItemText 
                  primary="Trend Analysis" 
                  secondary="Short, medium, and long-term trend identification" 
                />
              </ListItem>
            </List>
            
            <Alert severity="info" sx={{ mt: 2 }}>
              Multi-timeframe analysis helps avoid false signals by confirming patterns across multiple timeframes,
              significantly improving trading decision quality.
            </Alert>
          </Box>
        ),
        examples: (
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Usage Examples:
            </Typography>
            
            <List>
              <ListItem>
                <ListItemIcon><PsychologyIcon color="primary" /></ListItemIcon>
                <ListItemText 
                  primary="Trend Confirmation" 
                  secondary="Use higher timeframes to confirm the primary trend direction before trading setups on lower timeframes." 
                />
              </ListItem>
              <ListItem>
                <ListItemIcon><PsychologyIcon color="primary" /></ListItemIcon>
                <ListItemText 
                  primary="Entry Timing" 
                  secondary="Identify the primary direction on higher timeframes, then use lower timeframes for more precise entry points." 
                />
              </ListItem>
              <ListItem>
                <ListItemIcon><PsychologyIcon color="primary" /></ListItemIcon>
                <ListItemText 
                  primary="Divergence Detection" 
                  secondary="Compare indicator readings across multiple timeframes to identify hidden divergences." 
                />
              </ListItem>
            </List>
          </Box>
        )
      },
      {
        id: 'confluence-highlighting',
        title: 'Confluence Highlighting',
        content: (
          <Box>
            <Typography paragraph>
              Confluence Highlighting automatically identifies and visualizes zones where multiple technical 
              factors align, indicating areas of higher probability for price reaction.
            </Typography>
            
            <Typography variant="subtitle2" gutterBottom>
              Key Features:
            </Typography>
            
            <List dense>
              <ListItem>
                <ListItemIcon><CheckCircleIcon color="success" /></ListItemIcon>
                <ListItemText 
                  primary="Zone Detection" 
                  secondary="Automatic identification of price levels with multiple confirmations" 
                />
              </ListItem>
              <ListItem>
                <ListItemIcon><CheckCircleIcon color="success" /></ListItemIcon>
                <ListItemText 
                  primary="Source Filtering" 
                  secondary="Toggle between different types of confluence sources" 
                />
              </ListItem>
              <ListItem>
                <ListItemIcon><CheckCircleIcon color="success" /></ListItemIcon>
                <ListItemText 
                  primary="Strength Indicators" 
                  secondary="Visual representation of confluence zone strength" 
                />
              </ListItem>
              <ListItem>
                <ListItemIcon><CheckCircleIcon color="success" /></ListItemIcon>
                <ListItemText 
                  primary="Interactive Exploration" 
                  secondary="Detailed information and sources for each zone" 
                />
              </ListItem>
            </List>
            
            <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
              Source Types:
            </Typography>
            
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
              <Chip label="Support/Resistance" color="primary" />
              <Chip label="Fibonacci Levels" color="secondary" />
              <Chip label="Moving Averages" color="success" />
              <Chip label="Chart Patterns" color="warning" />
              <Chip label="Volume Profiles" color="error" />
              <Chip label="Pivot Points" />
              <Chip label="Order Flow" />
            </Box>
            
            <Alert severity="info">
              Combining multiple independent technical factors increases the statistical probability of price
              respecting a level or zone, leading to higher-quality trading decisions.
            </Alert>
          </Box>
        ),
        examples: (
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Common Confluence Scenarios:
            </Typography>
            
            <List>
              <ListItem>
                <ListItemText 
                  primary="Support/Resistance + Fibonacci Retracement" 
                  secondary="A historical support level aligning with a key Fibonacci retracement level creates a stronger zone." 
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Multiple Moving Average Convergence" 
                  secondary="When several moving averages (e.g., 50, 100, 200) converge at a similar price level." 
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Pattern Completion + Key Level" 
                  secondary="A chart pattern completion (e.g., head and shoulders) coinciding with a major support/resistance zone." 
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Multi-timeframe Support/Resistance" 
                  secondary="The same price area showing significance across multiple timeframes." 
                />
              </ListItem>
            </List>
          </Box>
        )
      },
      {
        id: 'ml-integration',
        title: 'ML Model Integration',
        content: (
          <Box>
            <Typography paragraph>
              Machine Learning integration enhances trading strategies by providing data-driven confirmations, 
              insights, and pattern recognition capabilities that go beyond traditional technical analysis.
            </Typography>
            
            <Typography variant="subtitle2" gutterBottom>
              Key Capabilities:
            </Typography>
            
            <List dense>
              <ListItem>
                <ListItemIcon><CheckCircleIcon color="success" /></ListItemIcon>
                <ListItemText 
                  primary="Signal Confirmation" 
                  secondary="ML models validate trading signals before execution" 
                />
              </ListItem>
              <ListItem>
                <ListItemIcon><CheckCircleIcon color="success" /></ListItemIcon>
                <ListItemText 
                  primary="Ensemble Decision-making" 
                  secondary="Multiple models work together for more robust predictions" 
                />
              </ListItem>
              <ListItem>
                <ListItemIcon><CheckCircleIcon color="success" /></ListItemIcon>
                <ListItemText 
                  primary="Explanatory Insights" 
                  secondary="Transparent reasoning for all model decisions" 
                />
              </ListItem>
              <ListItem>
                <ListItemIcon><CheckCircleIcon color="success" /></ListItemIcon>
                <ListItemText 
                  primary="Parameter Recommendations" 
                  secondary="Data-driven suggestions for strategy optimization" 
                />
              </ListItem>
            </List>
            
            <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
              Supported ML Models:
            </Typography>
            
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
              <Chip label="Trend Classifier" color="primary" />
              <Chip label="Pattern Recognition CNN" color="secondary" />
              <Chip label="Volatility Predictor" color="success" />
              <Chip label="Market Regime Classifier" color="warning" />
              <Chip label="News Sentiment Analyzer" color="error" />
            </Box>
            
            <Alert severity="info">
              ML models continuously learn from new market data and trading results, gradually improving their
              accuracy and providing increasingly valuable insights.
            </Alert>
          </Box>
        ),
        examples: (
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              ML Integration Workflow:
            </Typography>
            
            <List>
              <ListItem>
                <ListItemIcon><LooksOneIcon /></ListItemIcon>
                <ListItemText 
                  primary="Strategy generates a trading signal" 
                />
              </ListItem>
              <ListItem>
                <ListItemIcon><LooksTwoIcon /></ListItemIcon>
                <ListItemText 
                  primary="Signal is sent to ML Confirmation Filter" 
                />
              </ListItem>
              <ListItem>
                <ListItemIcon><Looks3Icon /></ListItemIcon>
                <ListItemText 
                  primary="Multiple ML models evaluate market conditions" 
                />
              </ListItem>
              <ListItem>
                <ListItemIcon><Looks4Icon /></ListItemIcon>
                <ListItemText 
                  primary="Ensemble decision is calculated with weighted confidence" 
                />
              </ListItem>
              <ListItem>
                <ListItemIcon><Looks5Icon /></ListItemIcon>
                <ListItemText 
                  primary="Signal is confirmed or rejected based on confidence threshold" 
                />
              </ListItem>
              <ListItem>
                <ListItemIcon><Looks6Icon /></ListItemIcon>
                <ListItemText 
                  primary="Trading action is taken with ML-enhanced parameters" 
                />
              </ListItem>
            </List>
          </Box>
        )
      },
      {
        id: 'adaptive-strategies',
        title: 'Adaptive Strategies',
        content: (
          <Box>
            <Typography paragraph>
              Adaptive strategies automatically adjust their parameters and behavior based on changing market conditions,
              optimizing performance across different market regimes.
            </Typography>
            
            <Typography variant="subtitle2" gutterBottom>
              Key Features:
            </Typography>
            
            <List dense>
              <ListItem>
                <ListItemIcon><CheckCircleIcon color="success" /></ListItemIcon>
                <ListItemText 
                  primary="Market Regime Detection" 
                  secondary="Automatic identification of current market conditions" 
                />
              </ListItem>
              <ListItem>
                <ListItemIcon><CheckCircleIcon color="success" /></ListItemIcon>
                <ListItemText 
                  primary="Regime-specific Parameters" 
                  secondary="Tailored settings for each market environment" 
                />
              </ListItem>
              <ListItem>
                <ListItemIcon><CheckCircleIcon color="success" /></ListItemIcon>
                <ListItemText 
                  primary="Performance Analytics" 
                  secondary="Tracking strategy performance across different regimes" 
                />
              </ListItem>
              <ListItem>
                <ListItemIcon><CheckCircleIcon color="success" /></ListItemIcon>
                <ListItemText 
                  primary="ML-Enhanced Optimization" 
                  secondary="Machine learning recommendations for parameter adjustments" 
                />
              </ListItem>
            </List>
            
            <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
              Supported Market Regimes:
            </Typography>
            
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
              <Chip label="Trending Market" color="primary" />
              <Chip label="Range-bound Market" color="secondary" />
              <Chip label="Volatile Market" color="error" />
              <Chip label="Breakout Market" color="warning" />
            </Box>
            
            <Alert severity="info">
              Strategies that adapt to changing market conditions significantly outperform static strategies
              over the long term, especially in forex markets which regularly cycle through different regimes.
            </Alert>
          </Box>
        ),
        examples: (
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Adaptive Strategy Examples:
            </Typography>
            
            <List>
              <ListItem>
                <ListItemText 
                  primary="Adaptive Moving Average Strategy" 
                  secondary="Adjusts MA periods and signal thresholds based on market volatility and trend strength." 
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Multi-regime Pattern Strategy" 
                  secondary="Changes the pattern types it looks for based on detected market regime." 
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Volatility-adjusted Breakout Strategy" 
                  secondary="Scales breakout thresholds based on current market volatility." 
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Adaptive Risk Management" 
                  secondary="Adjusts position sizing and stop distance based on market conditions." 
                />
              </ListItem>
            </List>
          </Box>
        )
      }
    ],
    
    technical: [
      {
        id: 'tech-architecture',
        title: 'Technical Architecture',
        content: (
          <Box>
            <Typography paragraph>
              The Phase 4 implementation follows a modular architecture, integrating new components with the existing
              trading platform while maintaining clean separation of concerns.
            </Typography>
            
            <Typography variant="subtitle2" gutterBottom>
              Key Technical Components:
            </Typography>
            
            <List dense>
              <ListItem>
                <ListItemIcon><CodeIcon color="primary" /></ListItemIcon>
                <ListItemText 
                  primary="React Components" 
                  secondary="Custom UI components for visualization and interaction" 
                />
              </ListItem>
              <ListItem>
                <ListItemIcon><CodeIcon color="primary" /></ListItemIcon>
                <ListItemText 
                  primary="ML Service Integration" 
                  secondary="REST APIs for model inference and training" 
                />
              </ListItem>
              <ListItem>
                <ListItemIcon><CodeIcon color="primary" /></ListItemIcon>
                <ListItemText 
                  primary="Advanced Data Processing" 
                  secondary="Real-time calculation of indicators and patterns" 
                />
              </ListItem>
              <ListItem>
                <ListItemIcon><CodeIcon color="primary" /></ListItemIcon>
                <ListItemText 
                  primary="Strategy Parameter Management" 
                  secondary="Dynamic configuration based on market conditions" 
                />
              </ListItem>
            </List>
            
            <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
              Technical Diagram:
            </Typography>
            
            <Card variant="outlined" sx={{ p: 2, mb: 2, bgcolor: 'background.default' }}>
              <pre style={{ overflow: 'auto', fontSize: '0.75rem' }}>
{`                                             
┌─────────────────────────┐   ┌──────────────────────────┐
│                         │   │                          │
│   UI Service (React)    │◄──┤   ML Integration API     │
│                         │   │                          │
└───────────┬─────────────┘   └──────────────┬───────────┘
            │                                │
            ▼                                ▼
┌─────────────────────────┐   ┌──────────────────────────┐
│                         │   │                          │
│   Strategy Execution    │◄──┤   Analysis Engine        │
│                         │   │                          │
└───────────┬─────────────┘   └──────────────┬───────────┘
            │                                │
            ▼                                ▼
┌─────────────────────────┐   ┌──────────────────────────┐
│                         │   │                          │
│   Data Pipeline         │◄──┤   Feature Store          │
│                         │   │                          │
└─────────────────────────┘   └──────────────────────────┘
`}
              </pre>
            </Card>
            
            <Alert severity="info">
              The platform uses state-of-the-art technologies including React with TypeScript,
              Python-based ML services with FastAPI, and Redis for real-time data processing.
            </Alert>
          </Box>
        )
      },
      {
        id: 'implementation',
        title: 'Implementation Details',
        content: (
          <Box>
            <Typography paragraph>
              Phase 4 implements several complex technical features to enable its advanced functionality.
              Below are detailed explanations of key implementation aspects.
            </Typography>
            
            <Typography variant="subtitle2" gutterBottom>
              Pattern Visualization Implementation:
            </Typography>
            
            <Typography paragraph>
              The pattern visualization component uses a canvas-based approach for high-performance rendering
              of complex chart patterns. The component maintains its own data structure for patterns and
              uses optimization techniques to ensure smooth interaction even with large datasets.
            </Typography>
            
            <Typography variant="subtitle2" gutterBottom>
              ML Model Integration:
            </Typography>
            
            <Typography paragraph>
              ML models are integrated via a RESTful API that connects to a dedicated ML service. 
              Models are versioned and accessed through a registry system that ensures consistent 
              behavior and allows for A/B testing of new models.
            </Typography>
            
            <Typography variant="subtitle2" gutterBottom>
              Adaptive Strategy Mechanism:
            </Typography>
            
            <Typography paragraph>
              The adaptive strategy framework uses a combination of market state detection algorithms
              and parameter optimization. Market regimes are detected using a collection of indicators
              and machine learning classifiers that analyze recent price action and volatility patterns.
            </Typography>
            
            <Typography variant="body2" paragraph>
              Technical challenges addressed during implementation:
            </Typography>
            
            <List dense>
              <ListItem>
                <ListItemIcon><WarningAmberIcon color="warning" /></ListItemIcon>
                <ListItemText 
                  primary="Performance optimization for real-time chart rendering" 
                  secondary="Solved using canvas-based rendering and worker threads" 
                />
              </ListItem>
              <ListItem>
                <ListItemIcon><WarningAmberIcon color="warning" /></ListItemIcon>
                <ListItemText 
                  primary="Synchronization of data across multiple timeframes" 
                  secondary="Implemented using a central data store and subscription model" 
                />
              </ListItem>
              <ListItem>
                <ListItemIcon><WarningAmberIcon color="warning" /></ListItemIcon>
                <ListItemText 
                  primary="ML model latency for real-time trading" 
                  secondary="Addressed with response time optimization and caching mechanisms" 
                />
              </ListItem>
            </List>
          </Box>
        )
      },
      {
        id: 'api-reference',
        title: 'API Reference',
        content: (
          <Box>
            <Typography paragraph>
              The Phase 4 implementation exposes several APIs for integration with other parts of the platform
              and external systems.
            </Typography>
            
            <Typography variant="subtitle2" gutterBottom>
              ML Service API Endpoints:
            </Typography>
            
            <Card variant="outlined" sx={{ mb: 2 }}>
              <List dense>
                <ListItem divider>
                  <ListItemText
                    primary="POST /api/ml/predict"
                    secondary="Request predictions from ML models"
                  />
                </ListItem>
                <ListItem divider>
                  <ListItemText
                    primary="GET /api/ml/models"
                    secondary="List available ML models and their status"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="POST /api/ml/regime/detect"
                    secondary="Detect current market regime"
                  />
                </ListItem>
              </List>
            </Card>
            
            <Typography variant="subtitle2" gutterBottom>
              Strategy Management API:
            </Typography>
            
            <Card variant="outlined" sx={{ mb: 2 }}>
              <List dense>
                <ListItem divider>
                  <ListItemText
                    primary="PUT /api/strategies/:id/parameters"
                    secondary="Update strategy parameters"
                  />
                </ListItem>
                <ListItem divider>
                  <ListItemText
                    primary="POST /api/strategies/:id/optimize"
                    secondary="Request parameter optimization for a strategy"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="GET /api/strategies/:id/performance"
                    secondary="Get strategy performance metrics by market regime"
                  />
                </ListItem>
              </List>
            </Card>
            
            <Typography variant="subtitle2" gutterBottom>
              Pattern Analysis API:
            </Typography>
            
            <Card variant="outlined">
              <List dense>
                <ListItem divider>
                  <ListItemText
                    primary="POST /api/analysis/detect-patterns"
                    secondary="Detect chart patterns in price data"
                  />
                </ListItem>
                <ListItem divider>
                  <ListItemText
                    primary="GET /api/analysis/confluence"
                    secondary="Get confluence zones for a symbol and timeframe"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="POST /api/analysis/multi-timeframe"
                    secondary="Perform multi-timeframe analysis"
                  />
                </ListItem>
              </List>
            </Card>
          </Box>
        )
      }
    ]
  };
  
  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>
        Phase 4 Documentation
      </Typography>
      
      <Typography variant="body2" color="text.secondary" paragraph>
        This documentation provides a comprehensive guide to the Phase 4 features, 
        including the multi-timeframe analysis, ML integration, and adaptive strategies.
      </Typography>
      
      <Tabs value={activeTab} onChange={handleTabChange} sx={{ mb: 2 }}>
        <Tab label="Overview" />
        <Tab label="Features" />
        <Tab label="Technical Details" />
      </Tabs>
      
      <Box sx={{ mt: 3 }}>
        {activeTab === 0 && documentation.overview.map((section) => (
          <Accordion
            key={section.id}
            expanded={expanded === section.id}
            onChange={handleAccordionChange(section.id)}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography fontWeight="medium">{section.title}</Typography>
            </AccordionSummary>
            <AccordionDetails>
              {section.content}
              {section.examples && (
                <>
                  <Divider sx={{ my: 2 }} />
                  {section.examples}
                </>
              )}
            </AccordionDetails>
          </Accordion>
        ))}
        
        {activeTab === 1 && documentation.features.map((section) => (
          <Accordion
            key={section.id}
            expanded={expanded === section.id}
            onChange={handleAccordionChange(section.id)}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography fontWeight="medium">{section.title}</Typography>
            </AccordionSummary>
            <AccordionDetails>
              {section.content}
              {section.examples && (
                <>
                  <Divider sx={{ my: 2 }} />
                  {section.examples}
                </>
              )}
            </AccordionDetails>
          </Accordion>
        ))}
        
        {activeTab === 2 && documentation.technical.map((section) => (
          <Accordion
            key={section.id}
            expanded={expanded === section.id}
            onChange={handleAccordionChange(section.id)}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography fontWeight="medium">{section.title}</Typography>
            </AccordionSummary>
            <AccordionDetails>
              {section.content}
              {section.examples && (
                <>
                  <Divider sx={{ my: 2 }} />
                  {section.examples}
                </>
              )}
            </AccordionDetails>
          </Accordion>
        ))}
      </Box>
      
      <Divider sx={{ my: 3 }} />
      
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="body2" color="text.secondary">
          Last updated: April 14, 2025
        </Typography>
        
        <Button variant="contained" size="small">
          Download PDF Documentation
        </Button>
      </Box>
    </Paper>
  );
}
