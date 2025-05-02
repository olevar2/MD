/**
 * ConfluenceChartContainer Component
 * 
 * Container component that integrates the TradingChart with backend services
 * to fetch market data and confluence analysis results.
 */
import React, { useEffect, useState, useCallback } from 'react';
import { Box, Typography, CircularProgress, Alert } from '@mui/material';
import TradingChart from './TradingChart';
import ChartControlPanel from './ChartControlPanel';
import { ChartData, ConfluencePoint, ChartOptions, TimeFrame, ConfluenceType } from '../../types/chart';
import { fetchMarketData, fetchConfluenceAnalysis } from '../../services/analysisService';

interface ConfluenceChartContainerProps {
  symbol: string;
  initialTimeframe?: TimeFrame;
  height?: number;
  width?: string | number;
}

const ConfluenceChartContainer: React.FC<ConfluenceChartContainerProps> = ({
  symbol,
  initialTimeframe = TimeFrame.H1,
  height = 600,
  width = '100%',
}) => {
  // State
  const [timeframe, setTimeframe] = useState<TimeFrame>(initialTimeframe);
  const [chartData, setChartData] = useState<ChartData>({ candles: [] });
  const [confluencePoints, setConfluencePoints] = useState<ConfluencePoint[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [activeConfluenceTypes, setActiveConfluenceTypes] = useState<ConfluenceType[]>([
    'support', 'resistance', 'harmonic_pattern', 'ma_confluence', 'multi_timeframe'
  ]);
  const [chartOptions, setChartOptions] = useState<ChartOptions>({
    darkMode: false,
    showVolume: true,
    showGrid: true,
    showConfluence: true,
    confluenceThreshold: 0.5, // 50% threshold by default
    chartOptions: {}
  });

  // Load market data and confluence analysis
  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      // Fetch market data
      const data = await fetchMarketData(symbol, timeframe);
      setChartData(data);
      
      // Only fetch confluence analysis if enabled
      if (chartOptions.showConfluence) {
        const confluenceData = await fetchConfluenceAnalysis(symbol, timeframe);
        
        // Filter confluences by selected types and threshold
        const filteredConfluences = confluenceData.confluencePoints.filter(point => 
          activeConfluenceTypes.includes(point.type) && 
          point.strength >= chartOptions.confluenceThreshold
        );
        
        setConfluencePoints(filteredConfluences);
      } else {
        setConfluencePoints([]);
      }
    } catch (err) {
      console.error('Error loading chart data:', err);
      setError(`Failed to load chart data: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setChartData({ candles: [] });
      setConfluencePoints([]);
    } finally {
      setLoading(false);
    }
  }, [symbol, timeframe, chartOptions.showConfluence, chartOptions.confluenceThreshold, activeConfluenceTypes]);

  // Load data on component mount and when dependencies change
  useEffect(() => {
    loadData();
  }, [loadData]);

  // Handle chart option changes
  const handleOptionsChange = (options: Partial<ChartOptions>) => {
    setChartOptions(prev => ({
      ...prev,
      ...options
    }));
  };

  // Handle timeframe change
  const handleTimeframeChange = (newTimeframe: TimeFrame) => {
    setTimeframe(newTimeframe);
  };

  // Handle confluence type toggle
  const handleConfluenceToggle = (types: ConfluenceType[]) => {
    setActiveConfluenceTypes(types);
  };

  // Filter confluences based on active types and threshold
  const visibleConfluencePoints = chartOptions.showConfluence
    ? confluencePoints.filter(point => 
        activeConfluenceTypes.includes(point.type) && 
        point.strength >= chartOptions.confluenceThreshold
      )
    : [];

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        {symbol} - {timeframe}
      </Typography>
      
      <ChartControlPanel
        options={chartOptions}
        timeframe={timeframe}
        onOptionsChange={handleOptionsChange}
        onTimeframeChange={handleTimeframeChange}
        activeConfluenceTypes={activeConfluenceTypes}
        onConfluenceToggle={handleConfluenceToggle}
      />
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      
      <TradingChart
        symbol={symbol}
        timeframe={timeframe}
        data={chartData}
        confluencePoints={visibleConfluencePoints}
        options={chartOptions}
        height={height}
        width={width}
        loading={loading}
      />
      
      {chartOptions.showConfluence && visibleConfluencePoints.length > 0 && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Confluence Analysis
          </Typography>
          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 2 }}>
            {visibleConfluencePoints.slice(0, 5).map((point, index) => (
              <Box key={`${point.time}-${index}`} sx={{ border: '1px solid #e0e0e0', borderRadius: 1, p: 1.5 }}>
                <Typography variant="body2">
                  <strong>{point.type.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}</strong>
                  {point.description && `: ${point.description}`}
                </Typography>
                <Typography variant="body2">
                  Price: {point.price.toFixed(5)}
                </Typography>
                <Typography variant="body2">
                  Strength: {(point.strength * 100).toFixed(0)}%
                </Typography>
                {point.sources && point.sources.length > 0 && (
                  <Typography variant="body2">
                    Sources: {point.sources.join(', ')}
                  </Typography>
                )}
              </Box>
            ))}
            {visibleConfluencePoints.length > 5 && (
              <Typography variant="body2" color="text.secondary">
                And {visibleConfluencePoints.length - 5} more confluence points...
              </Typography>
            )}
          </Box>
        </Box>
      )}
    </Box>
  );
};

export default ConfluenceChartContainer;
