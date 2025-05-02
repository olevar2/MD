/**
 * MultiTimeframeAnalysis component for comparing chart patterns across timeframes
 */
import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  CircularProgress,
  Card,
  CardHeader,
  CardContent,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Alert,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import SwapHorizIcon from '@mui/icons-material/SwapHoriz';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import InfoIcon from '@mui/icons-material/Info';
import AdvancedChart from './AdvancedChart';
import { TimeFrame } from '@/types/strategy';

// Extended interfaces with more comprehensive data
interface IndicatorValues {
  timeframe: TimeFrame;
  rsi: number;
  macd: {
    histogram: number;
    signal: number;
    value: number;
  };
  ma: {
    fast: number;
    slow: number;
    crossover: 'bullish' | 'bearish' | 'none';
  };
  atr: number;
  trend: 'bullish' | 'bearish' | 'neutral';
  patterns: Array<{
    type: string;
    confidence: number;
  }>;
  volume: {
    value: number;
    trend: 'increasing' | 'decreasing' | 'neutral';
    anomalies: boolean;
  };
  support_resistance: Array<{
    price: number;
    strength: number;
    type: 'support' | 'resistance';
  }>;
  divergences: Array<{
    type: string;
    indicator: string;
    significance: number;
  }>;
}

interface MTFSummary {
  overallBias: 'bullish' | 'bearish' | 'neutral';
  confidence: number;
  conflictLevel: 'low' | 'medium' | 'high';
  recommendedTimeframe: TimeFrame;
  keyLevels: Array<{
    price: number;
    type: 'support' | 'resistance';
    strength: number;
    timeframes: TimeFrame[];
  }>;
  confluenceZones: Array<{
    price: number;
    strength: number;
    description: string;
    sources: string[];
  }>;
  patterns: {
    aligned: boolean;
    dominant: string;
    conflicting: boolean;
  };
  trends: {
    shortTerm: 'bullish' | 'bearish' | 'neutral';
    mediumTerm: 'bullish' | 'bearish' | 'neutral';
    longTerm: 'bullish' | 'bearish' | 'neutral';
  };
}

interface MultiTimeframeAnalysisProps {
  symbol: string;
  primaryTimeframe: TimeFrame;
  comparisonTimeframes: TimeFrame[];
  height?: number;
}

export default function MultiTimeframeAnalysis({
  symbol,
  primaryTimeframe,
  comparisonTimeframes,
  height = 400
}: MultiTimeframeAnalysisProps) {
  const [loading, setLoading] = useState<boolean>(true);
  const [activeTab, setActiveTab] = useState<number>(0);
  const [indicatorData, setIndicatorData] = useState<Record<TimeFrame, IndicatorValues>>({} as Record<TimeFrame, IndicatorValues>);
  const [mtfSummary, setMtfSummary] = useState<MTFSummary | null>(null);
  const [selectedTimeframe, setSelectedTimeframe] = useState<TimeFrame>(primaryTimeframe);
  const [chartMode, setChartMode] = useState<'comparison' | 'detailed'>('comparison');
  const [correlationData, setCorrelationData] = useState<any>(null);

  // Load data whenever symbol or timeframes change
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      
      try {
        // In a real implementation, this would fetch data from an API
        await simulateDataLoading();
        
        // Generate mock indicator data
        const mockData = generateMockIndicatorData();
        setIndicatorData(mockData);
        
        // Generate MTF analysis summary
        const mockSummary = generateMockMTFSummary(mockData);
        setMtfSummary(mockSummary);
        
        // Generate mock correlation data
        setCorrelationData(generateMockCorrelationData(mockData));
        
      } catch (error) {
        console.error('Error loading MTF data:', error);
      } finally {
        setLoading(false);
      }
    };
    
    loadData();
  }, [symbol, primaryTimeframe, comparisonTimeframes]);
  
  // Simulate API data loading
  const simulateDataLoading = async () => {
    return new Promise(resolve => setTimeout(resolve, 1000));
  };
  
  // Generate mock correlation data
  const generateMockCorrelationData = (data: Record<TimeFrame, IndicatorValues>) => {
    const timeframes = Object.keys(data);
    const correlationMatrix: Record<string, Record<string, number>> = {};
    
    timeframes.forEach(tf1 => {
      correlationMatrix[tf1] = {};
      timeframes.forEach(tf2 => {
        // Generate correlation coefficient (1 for same timeframe, random for others)
        correlationMatrix[tf1][tf2] = tf1 === tf2 ? 
          1 : 
          0.3 + Math.random() * 0.7; // Random correlation between 0.3 and 1.0
      });
    });
    
    return {
      matrix: correlationMatrix,
      insights: [
        "Strong correlation between H1 and H4 price movements",
        "Divergence between D1 and lower timeframes indicates potential trend change",
        "M15 showing early signals not yet visible on higher timeframes"
      ]
    };
  };
  
  // Generate mock indicator data for demonstration
  const generateMockIndicatorData = (): Record<TimeFrame, IndicatorValues> => {
    const result: Record<string, IndicatorValues> = {};
    
    // Base trend bias for this symbol (random)
    const baseBias = Math.random() > 0.5;
    const allTimeframes = [primaryTimeframe, ...comparisonTimeframes];
    
    allTimeframes.forEach(timeframe => {
      // Add some randomization while keeping consistent bias across timeframes
      const isBullish = baseBias 
        ? Math.random() > 0.3  // 70% chance to follow base bias if bullish
        : Math.random() > 0.7; // 30% chance to be bullish if base bias is bearish
      
      // Adjust values based on timeframe (higher timeframes are more stable)
      const tfIndex = [TimeFrame.M1, TimeFrame.M5, TimeFrame.M15, TimeFrame.M30, 
                        TimeFrame.H1, TimeFrame.H4, TimeFrame.D1, TimeFrame.W1]
                        .indexOf(timeframe);
      
      const stability = tfIndex / 7; // 0 to 1, higher for higher timeframes
      
      // Create random adjustments
      const adj = {
        rsi: (Math.random() - 0.5) * 20,
        macd: (Math.random() - 0.5) * 0.5,
        ma: (Math.random() - 0.5) * 10,
        atr: Math.random() * 0.3
      };
      
      // Random pattern count (more patterns on higher timeframes)
      const patternCount = 1 + Math.floor(Math.random() * 4);
      const patternTypes = [
        'Double Top', 'Double Bottom', 'Head & Shoulders', 
        'Triangle', 'Flag', 'Channel', 'Elliott Wave', 
        'Fibonacci Retracement', 'ABCD Pattern'
      ];
      
      const patterns = Array(patternCount).fill(0).map(() => ({
        type: patternTypes[Math.floor(Math.random() * patternTypes.length)],
        confidence: 0.5 + Math.random() * 0.4
      }));
      
      // Generate support/resistance levels
      const srLevels = Array(2 + Math.floor(Math.random() * 3)).fill(0).map(() => ({
        price: 1.1000 + Math.random() * 0.1000,
        strength: 0.5 + Math.random() * 0.5,
        type: Math.random() > 0.5 ? 'support' : 'resistance' as ('support' | 'resistance')
      }));
      
      // Generate divergences
      const divergences = Math.random() > 0.7 ? [
        {
          type: Math.random() > 0.5 ? 'bullish' : 'bearish',
          indicator: ['RSI', 'MACD', 'OBV', 'Stochastic'][Math.floor(Math.random() * 4)],
          significance: 0.6 + Math.random() * 0.4
        }
      ] : [];
      
      data[timeframe] = {
        timeframe,
        rsi: isBullish ? 
          60 + Math.random() * 15 - adj.rsi : 
          40 - Math.random() * 15 - adj.rsi,
        macd: {
          histogram: isBullish ? 
            0.05 + Math.random() * 0.3 - adj.macd : 
            -0.05 - Math.random() * 0.3 - adj.macd,
          signal: isBullish ? -0.1 + Math.random() * 0.2 : 0.1 + Math.random() * 0.2,
          value: isBullish ? 0.1 + Math.random() * 0.3 : -0.1 - Math.random() * 0.3
        },
        ma: {
          fast: 1.1000 + Math.random() * 0.1000,
          slow: 1.1000 + Math.random() * 0.1000 - (isBullish ? 0.002 : -0.002),
          crossover: isBullish ? 'bullish' : 'bearish'
        },
        atr: 0.0020 + Math.random() * 0.0030 + adj.atr,
        trend: isBullish ? 'bullish' : Math.random() > 0.3 ? 'bearish' : 'neutral',
        patterns,
        volume: {
          value: 1000 + Math.random() * 9000,
          trend: isBullish ? 'increasing' : 'decreasing',
          anomalies: Math.random() > 0.8
        },
        support_resistance: srLevels,
        divergences
      };
    });
    
    return result;
  };
  
  // Generate mock MTF summary
  const generateMockMTFSummary = (data: Record<TimeFrame, IndicatorValues>): MTFSummary => {
    // Count bullish vs bearish trends across timeframes
    const trends = Object.values(data).map(tf => tf.trend);
    const bullishCount = trends.filter(t => t === 'bullish').length;
    const bearishCount = trends.filter(t => t === 'bearish').length;
    
    // Determine overall bias
    const overallBias = bullishCount > bearishCount ? 'bullish' :
                        bearishCount > bullishCount ? 'bearish' : 'neutral';
                        
    // Calculate confidence based on agreement between timeframes
    const totalTimeframes = trends.length;
    const dominantCount = Math.max(bullishCount, bearishCount);
    const confidence = dominantCount / totalTimeframes;
    
    // Determine conflict level
    const conflictLevel = confidence > 0.8 ? 'low' :
                          confidence > 0.6 ? 'medium' : 'high';
                          
    // Find recommended timeframe (simple logic - could be more sophisticated)
    const timeframeOptions = Object.keys(data) as TimeFrame[];
    const recommendedTimeframe = conflictLevel === 'high' 
      ? TimeFrame.D1  // Use higher timeframe when conflict is high
      : timeframeOptions[Math.floor(timeframeOptions.length / 2)]; // Middle timeframe
    
    // Collect key levels from all timeframes
    const allLevels = Object.entries(data).flatMap(([tf, tfData]) => 
      tfData.support_resistance.map(level => ({
        price: level.price,
        type: level.type,
        strength: level.strength,
        timeframes: [tf as TimeFrame]
      }))
    );
    
    // Merge similar levels (simplified algorithm)
    const keyLevels = [];
    const priceTolerance = 0.0010; // Consider levels within 10 pips as the same
    
    for (const level of allLevels) {
      const existingLevel = keyLevels.find(
        l => Math.abs(l.price - level.price) < priceTolerance && l.type === level.type
      );
      
      if (existingLevel) {
        // Merge into existing level
        existingLevel.strength = Math.max(existingLevel.strength, level.strength);
        if (!existingLevel.timeframes.includes(level.timeframes[0])) {
          existingLevel.timeframes.push(level.timeframes[0]);
        }
      } else {
        // Add as new level
        keyLevels.push(level);
      }
    }
    
    // Sort by strength
    keyLevels.sort((a, b) => b.strength - a.strength);
    
    // Generate confluence zones (areas where multiple indicators/levels align)
    const confluenceZones = Array(2 + Math.floor(Math.random() * 3)).fill(0).map(() => ({
      price: 1.1000 + Math.random() * 0.1000,
      strength: 0.6 + Math.random() * 0.4,
      description: `Strong ${Math.random() > 0.5 ? 'resistance' : 'support'} zone`,
      sources: [
        'Multiple timeframe S/R',
        'Fibonacci level',
        'Moving average convergence',
        'Volume profile'
      ].slice(0, 2 + Math.floor(Math.random() * 3))
    }));
    
    // Generate pattern analysis
    const patterns = {
      aligned: Math.random() > 0.3,
      dominant: ['Triangle', 'Channel', 'Double Top', 'Elliott Wave'][Math.floor(Math.random() * 4)],
      conflicting: Math.random() > 0.7
    };
    
    // Generate trend analysis for different timeframes
    const timeframeWeights = {
      [TimeFrame.M1]: 'shortTerm',
      [TimeFrame.M5]: 'shortTerm',
      [TimeFrame.M15]: 'shortTerm',
      [TimeFrame.M30]: 'shortTerm',
      [TimeFrame.H1]: 'mediumTerm',
      [TimeFrame.H4]: 'mediumTerm',
      [TimeFrame.D1]: 'longTerm',
      [TimeFrame.W1]: 'longTerm'
    };
    
    const trendCounts = {
      shortTerm: { bullish: 0, bearish: 0, neutral: 0 },
      mediumTerm: { bullish: 0, bearish: 0, neutral: 0 },
      longTerm: { bullish: 0, bearish: 0, neutral: 0 }
    };
    
    // Count trends by timeframe category
    Object.entries(data).forEach(([tf, tfData]) => {
      const category = timeframeWeights[tf as TimeFrame] || 'shortTerm';
      trendCounts[category][tfData.trend]++;
    });
    
    // Determine dominant trend for each category
    const trends = {
      shortTerm: trendCounts.shortTerm.bullish > trendCounts.shortTerm.bearish ? 'bullish' :
                 trendCounts.shortTerm.bearish > trendCounts.shortTerm.bullish ? 'bearish' : 'neutral',
      mediumTerm: trendCounts.mediumTerm.bullish > trendCounts.mediumTerm.bearish ? 'bullish' :
                 trendCounts.mediumTerm.bearish > trendCounts.mediumTerm.bullish ? 'bearish' : 'neutral',
      longTerm: trendCounts.longTerm.bullish > trendCounts.longTerm.bearish ? 'bullish' :
                trendCounts.longTerm.bearish > trendCounts.longTerm.bullish ? 'bearish' : 'neutral'
    };
    
    return {
      overallBias,
      confidence,
      conflictLevel,
      recommendedTimeframe,
      keyLevels: keyLevels.slice(0, 5), // Limit to top 5
      confluenceZones,
      patterns,
      trends
    };
  };
  
  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };
  
  const handleTimeframeSelect = (timeframe: TimeFrame) => {
    setSelectedTimeframe(timeframe);
  };
  
  const toggleChartMode = () => {
    setChartMode(chartMode === 'comparison' ? 'detailed' : 'comparison');
  };
  
  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'bullish':
        return <TrendingUpIcon fontSize="small" color="success" />;
      case 'bearish':
        return <TrendingDownIcon fontSize="small" color="error" />;
      default:
        return <SwapHorizIcon fontSize="small" color="disabled" />;
    }
  };
  
  const getTrendColor = (trend: string) => {
    switch (trend) {
      case 'bullish':
        return 'success';
      case 'bearish':
        return 'error';
      default:
        return 'default';
    }
  };

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom display="flex" alignItems="center">
        Multi-Timeframe Analysis - {symbol}
        <Button 
          variant="outlined" 
          size="small" 
          startIcon={chartMode === 'comparison' ? <ZoomInIcon /> : <CompareArrowsIcon />}
          onClick={toggleChartMode}
          sx={{ ml: 2 }}
        >
          {chartMode === 'comparison' ? 'Detailed View' : 'Comparison View'}
        </Button>
      </Typography>
      
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      ) : (
        <>
          <Tabs value={activeTab} onChange={handleTabChange} sx={{ mb: 2 }}>
            <Tab label="Chart Comparison" />
            <Tab label="Indicator Table" />
            <Tab label="Analysis Summary" />
            <Tab label="Correlation Analysis" />
          </Tabs>
          
          {/* Chart Comparison Tab */}
          {activeTab === 0 && (
            <>
              {chartMode === 'comparison' ? (
                <Grid container spacing={2}>
                  {/* Primary Chart (larger) */}
                  <Grid item xs={12}>
                    <Typography variant="subtitle1" gutterBottom>
                      Primary Timeframe: {primaryTimeframe}
                    </Typography>
                    <AdvancedChart 
                      symbol={symbol}
                      initialTimeframe={primaryTimeframe as TimeFrame}
                      height={height}
                      showVolume={true}
                      enablePatternDetection={true}
                      enableConfluenceHighlighting={true}
                    />
                  </Grid>
                  
                  {/* Comparison charts (smaller) */}
                  {comparisonTimeframes.map((tf) => (
                    <Grid item xs={12} md={6} key={tf}>
                      <Typography variant="subtitle2" gutterBottom>
                        {tf}
                      </Typography>
                      <AdvancedChart 
                        symbol={symbol}
                        initialTimeframe={tf as TimeFrame}
                        height={height / 2}
                        showVolume={false}
                        enablePatternDetection={true}
                        enableConfluenceHighlighting={true}
                      />
                    </Grid>
                  ))}
                </Grid>
              ) : (
                <Box>
                  <FormControl sx={{ mb: 2, minWidth: 120 }} size="small">
                    <InputLabel>Timeframe</InputLabel>
                    <Select
                      value={selectedTimeframe}
                      onChange={(e) => handleTimeframeSelect(e.target.value as TimeFrame)}
                      label="Timeframe"
                    >
                      <MenuItem value={primaryTimeframe}>{primaryTimeframe} (Primary)</MenuItem>
                      {comparisonTimeframes.map(tf => (
                        <MenuItem value={tf} key={tf}>{tf}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  
                  <AdvancedChart 
                    symbol={symbol}
                    initialTimeframe={selectedTimeframe}
                    height={height * 1.5}
                    showVolume={true}
                    enablePatternDetection={true}
                    enableConfluenceHighlighting={true}
                    enableMultiTimeframe={true}
                    enableElliottWaveOverlays={true}
                  />
                </Box>
              )}
            </>
          )}
          
          {/* Indicator Table Tab */}
          {activeTab === 1 && (
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Timeframe</TableCell>
                    <TableCell>Trend</TableCell>
                    <TableCell>RSI</TableCell>
                    <TableCell>MACD</TableCell>
                    <TableCell>MA Crossover</TableCell>
                    <TableCell>ATR</TableCell>
                    <TableCell>Volume</TableCell>
                    <TableCell>Patterns</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.entries(indicatorData).map(([timeframe, data]) => (
                    <TableRow key={timeframe} hover>
                      <TableCell>{timeframe}</TableCell>
                      <TableCell>
                        <Chip 
                          size="small"
                          label={data.trend.toUpperCase()}
                          color={getTrendColor(data.trend) as any}
                          icon={getTrendIcon(data.trend)}
                        />
                      </TableCell>
                      <TableCell>
                        <Typography 
                          color={data.rsi > 70 ? 'error' : data.rsi < 30 ? 'success' : 'inherit'}
                          fontWeight={data.rsi > 70 || data.rsi < 30 ? 'bold' : 'normal'}
                        >
                          {data.rsi.toFixed(1)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography 
                          color={data.macd.histogram > 0 ? 'success.main' : 'error.main'}
                          fontWeight="medium"
                        >
                          {data.macd.histogram.toFixed(3)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip 
                          size="small"
                          label={data.ma.crossover}
                          color={data.ma.crossover === 'bullish' ? 'success' : 
                                data.ma.crossover === 'bearish' ? 'error' : 'default'}
                        />
                      </TableCell>
                      <TableCell>{data.atr.toFixed(4)}</TableCell>
                      <TableCell>
                        <Typography 
                          color={data.volume.anomalies ? 'warning.main' : 'inherit'}
                          sx={{ display: 'flex', alignItems: 'center' }}
                        >
                          {data.volume.trend === 'increasing' && <TrendingUpIcon fontSize="small" sx={{ mr: 0.5 }} />}
                          {data.volume.trend === 'decreasing' && <TrendingDownIcon fontSize="small" sx={{ mr: 0.5 }} />}
                          {data.volume.value.toLocaleString()}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        {data.patterns.length ? (
                          data.patterns.map((pattern, i) => (
                            <Chip 
                              key={i}
                              size="small"
                              label={pattern.type}
                              sx={{ mr: 0.5, mb: 0.5 }}
                            />
                          ))
                        ) : (
                          <Typography variant="body2" color="text.secondary">None</Typography>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
          
          {/* Analysis Summary Tab */}
          {activeTab === 2 && mtfSummary && (
            <Grid container spacing={3}>
              {/* Overall analysis card */}
              <Grid item xs={12} md={6}>
                <Card>
                  <CardHeader 
                    title="Market Overview" 
                    titleTypographyProps={{ variant: 'h6' }} 
                  />
                  <CardContent>
                    <Grid container spacing={2}>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          Overall Bias:
                        </Typography>
                        <Chip 
                          label={mtfSummary.overallBias.toUpperCase()}
                          color={getTrendColor(mtfSummary.overallBias) as any}
                          icon={getTrendIcon(mtfSummary.overallBias)}
                        />
                      </Grid>
                      
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          Confidence:
                        </Typography>
                        <Typography variant="body1">
                          {(mtfSummary.confidence * 100).toFixed(0)}% 
                          ({mtfSummary.conflictLevel} conflict)
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={12}>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          Timeframe Trends:
                        </Typography>
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          <Chip 
                            size="small"
                            label={`Short: ${mtfSummary.trends.shortTerm}`}
                            color={getTrendColor(mtfSummary.trends.shortTerm) as any}
                          />
                          <Chip 
                            size="small"
                            label={`Medium: ${mtfSummary.trends.mediumTerm}`}
                            color={getTrendColor(mtfSummary.trends.mediumTerm) as any}
                          />
                          <Chip 
                            size="small"
                            label={`Long: ${mtfSummary.trends.longTerm}`}
                            color={getTrendColor(mtfSummary.trends.longTerm) as any}
                          />
                        </Box>
                      </Grid>
                      
                      <Grid item xs={12}>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          Recommended Timeframe:
                        </Typography>
                        <Chip label={mtfSummary.recommendedTimeframe} />
                      </Grid>
                      
                      {mtfSummary.patterns.conflicting && (
                        <Grid item xs={12}>
                          <Alert severity="warning" sx={{ mt: 1 }}>
                            Conflicting patterns detected across timeframes
                          </Alert>
                        </Grid>
                      )}
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
              
              {/* Key levels card */}
              <Grid item xs={12} md={6}>
                <Card>
                  <CardHeader 
                    title="Key Price Levels" 
                    titleTypographyProps={{ variant: 'h6' }} 
                  />
                  <CardContent>
                    <TableContainer>
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Price</TableCell>
                            <TableCell>Type</TableCell>
                            <TableCell>Strength</TableCell>
                            <TableCell>Timeframes</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {mtfSummary.keyLevels.map((level, index) => (
                            <TableRow key={index} hover>
                              <TableCell>{level.price.toFixed(4)}</TableCell>
                              <TableCell>
                                <Chip 
                                  size="small"
                                  label={level.type}
                                  color={level.type === 'resistance' ? 'error' : 'success'}
                                />
                              </TableCell>
                              <TableCell>
                                {(level.strength * 100).toFixed(0)}%
                              </TableCell>
                              <TableCell>
                                {level.timeframes.join(', ')}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </CardContent>
                </Card>
              </Grid>
              
              {/* Confluence zones card */}
              <Grid item xs={12}>
                <Card>
                  <CardHeader 
                    title="Confluence Zones" 
                    titleTypographyProps={{ variant: 'h6' }} 
                  />
                  <CardContent>
                    <Grid container spacing={2}>
                      {mtfSummary.confluenceZones.map((zone, index) => (
                        <Grid item xs={12} md={6} lg={4} key={index}>
                          <Card variant="outlined" sx={{ height: '100%' }}>
                            <CardContent>
                              <Typography variant="h6" gutterBottom>
                                {zone.price.toFixed(4)}
                              </Typography>
                              
                              <Typography variant="body2" gutterBottom>
                                {zone.description}
                              </Typography>
                              
                              <Typography variant="body2" color="text.secondary" gutterBottom>
                                Strength: {(zone.strength * 100).toFixed(0)}%
                              </Typography>
                              
                              <Divider sx={{ my: 1 }} />
                              
                              <Typography variant="caption" color="text.secondary" gutterBottom>
                                Supporting Evidence:
                              </Typography>
                              
                              <Box sx={{ mt: 1 }}>
                                {zone.sources.map((source, i) => (
                                  <Chip 
                                    key={i}
                                    size="small"
                                    label={source}
                                    sx={{ mr: 0.5, mb: 0.5 }}
                                  />
                                ))}
                              </Box>
                            </CardContent>
                          </Card>
                        </Grid>
                      ))}
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          )}
          
          {/* Correlation Analysis Tab */}
          {activeTab === 3 && correlationData && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={7}>
                <Card>
                  <CardHeader 
                    title="Timeframe Correlation Matrix" 
                    titleTypographyProps={{ variant: 'h6' }} 
                  />
                  <CardContent>
                    <TableContainer>
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Timeframe</TableCell>
                            {Object.keys(correlationData.matrix).map(tf => (
                              <TableCell key={tf}>{tf}</TableCell>
                            ))}
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {Object.entries(correlationData.matrix).map(([tf, correlations]) => (
                            <TableRow key={tf} hover>
                              <TableCell><strong>{tf}</strong></TableCell>
                              {Object.entries(correlations as Record<string, number>).map(([targetTf, value]) => (
                                <TableCell key={targetTf} 
                                  sx={{
                                    backgroundColor: `rgba(25, 118, 210, ${value.toFixed(2)})`,
                                    color: value > 0.7 ? 'white' : 'inherit'
                                  }}
                                >
                                  {value.toFixed(2)}
                                </TableCell>
                              ))}
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={5}>
                <Card sx={{ height: '100%' }}>
                  <CardHeader 
                    title="Correlation Insights" 
                    titleTypographyProps={{ variant: 'h6' }} 
                  />
                  <CardContent>
                    <List>
                      {correlationData.insights.map((insight, i) => (
                        <ListItem key={i}>
                          <ListItemIcon>
                            <InfoIcon color="primary" />
                          </ListItemIcon>
                          <ListItemText primary={insight} />
                        </ListItem>
                      ))}
                    </List>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          )}
        </>
      )}
    </Paper>
  );
}
