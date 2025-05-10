/**
 * ChartDataContext.tsx
 * Provides shared data context for multi-timeframe chart visualization
 * Implements caching and progressive loading for improved performance
 */
import React, { createContext, useState, useRef, useContext, useCallback } from 'react';
import { TimeFrame } from '@/types/strategy';

// Types for chart data
export interface OHLCData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface ChartDataContextType {
  dataByTimeframe: Record<TimeFrame, OHLCData[]>;
  loadData: (symbol: string, timeframe: TimeFrame) => Promise<OHLCData[]>;
  loadDataProgressive: (
    symbol: string,
    timeframe: TimeFrame,
    startTime?: Date,
    endTime?: Date,
    resolution?: 'low' | 'medium' | 'high'
  ) => Promise<OHLCData[]>;
  isLoading: Record<TimeFrame, boolean>;
  clearCache: () => void;
}

// Create the context with default values
export const ChartDataContext = createContext<ChartDataContextType>({
  dataByTimeframe: {} as Record<TimeFrame, OHLCData[]>,
  loadData: async () => [],
  loadDataProgressive: async () => [],
  isLoading: {} as Record<TimeFrame, boolean>,
  clearCache: () => {},
});

// Cache entry type
interface CacheEntry {
  data: OHLCData[];
  timestamp: number;
}

// Provider props
interface ChartDataProviderProps {
  children: React.ReactNode;
  cacheTTL?: number; // Cache time-to-live in milliseconds
}

/**
 * Mock function to fetch chart data
 * In a real implementation, this would call an API
 */
const fetchChartData = async (symbol: string, timeframe: TimeFrame): Promise<OHLCData[]> => {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 300));
  
  const data: OHLCData[] = [];
  const basePrice = 1.2000;
  let lastClose = basePrice;
  
  const now = new Date();
  
  for (let i = 100 - 1; i >= 0; i--) {
    const time = new Date(now);
    
    // Adjust time based on timeframe
    switch (timeframe) {
      case TimeFrame.M1:
        time.setMinutes(now.getMinutes() - i);
        break;
      case TimeFrame.M5:
        time.setMinutes(now.getMinutes() - i * 5);
        break;
      case TimeFrame.M15:
        time.setMinutes(now.getMinutes() - i * 15);
        break;
      case TimeFrame.M30:
        time.setMinutes(now.getMinutes() - i * 30);
        break;
      case TimeFrame.H1:
        time.setHours(now.getHours() - i);
        break;
      case TimeFrame.H4:
        time.setHours(now.getHours() - i * 4);
        break;
      case TimeFrame.D1:
        time.setDate(now.getDate() - i);
        break;
      case TimeFrame.W1:
        time.setDate(now.getDate() - i * 7);
        break;
    }
    
    // Generate random price action
    const change = (Math.random() - 0.5) * 0.005; // Random price change
    const open = lastClose;
    const close = open + change;
    const high = Math.max(open, close) + Math.random() * 0.002;
    const low = Math.min(open, close) - Math.random() * 0.002;
    const volume = Math.floor(Math.random() * 1000) + 500;
    
    data.push({
      time: Math.floor(time.getTime() / 1000),
      open,
      high,
      low,
      close,
      volume
    });
    
    lastClose = close;
  }
  
  return data;
};

/**
 * Mock function to fetch chart data with specific resolution
 */
const fetchChartDataWithResolution = async (
  symbol: string,
  timeframe: TimeFrame,
  startTime?: Date,
  endTime?: Date,
  dataPoints: number = 100
): Promise<OHLCData[]> => {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 200));
  
  const data: OHLCData[] = [];
  const basePrice = 1.2000;
  let lastClose = basePrice;
  
  const start = startTime || new Date(Date.now() - 30 * 24 * 60 * 60 * 1000); // Default to 30 days ago
  const end = endTime || new Date();
  
  const timeRange = end.getTime() - start.getTime();
  const timeStep = timeRange / dataPoints;
  
  for (let i = 0; i < dataPoints; i++) {
    const time = new Date(start.getTime() + i * timeStep);
    
    // Generate random price action
    const change = (Math.random() - 0.5) * 0.005; // Random price change
    const open = lastClose;
    const close = open + change;
    const high = Math.max(open, close) + Math.random() * 0.002;
    const low = Math.min(open, close) - Math.random() * 0.002;
    const volume = Math.floor(Math.random() * 1000) + 500;
    
    data.push({
      time: Math.floor(time.getTime() / 1000),
      open,
      high,
      low,
      close,
      volume
    });
    
    lastClose = close;
  }
  
  return data;
};

/**
 * Provider component for chart data
 */
export function ChartDataProvider({ children, cacheTTL = 5 * 60 * 1000 }: ChartDataProviderProps) {
  const [dataByTimeframe, setDataByTimeframe] = useState<Record<TimeFrame, OHLCData[]>>({} as Record<TimeFrame, OHLCData[]>);
  const [isLoading, setIsLoading] = useState<Record<TimeFrame, boolean>>({} as Record<TimeFrame, boolean>);
  const dataCache = useRef<Record<string, CacheEntry>>({});
  
  // Clear the entire cache
  const clearCache = useCallback(() => {
    dataCache.current = {};
    setDataByTimeframe({} as Record<TimeFrame, OHLCData[]>);
  }, []);
  
  // Load data for a specific timeframe
  const loadData = useCallback(async (symbol: string, timeframe: TimeFrame): Promise<OHLCData[]> => {
    const cacheKey = `${symbol}-${timeframe}`;
    
    // Set loading state for this timeframe
    setIsLoading(prev => ({ ...prev, [timeframe]: true }));
    
    try {
      // Check cache first (with expiration)
      const cachedData = dataCache.current[cacheKey];
      const now = Date.now();
      if (cachedData && (now - cachedData.timestamp < cacheTTL)) {
        setDataByTimeframe(prev => ({ ...prev, [timeframe]: cachedData.data }));
        setIsLoading(prev => ({ ...prev, [timeframe]: false }));
        return cachedData.data;
      }
      
      // Fetch data
      const data = await fetchChartData(symbol, timeframe);
      
      // Update cache
      dataCache.current[cacheKey] = { data, timestamp: now };
      
      // Update state
      setDataByTimeframe(prev => ({ ...prev, [timeframe]: data }));
      
      return data;
    } catch (error) {
      console.error(`Error loading data for ${symbol} ${timeframe}:`, error);
      return [];
    } finally {
      setIsLoading(prev => ({ ...prev, [timeframe]: false }));
    }
  }, [cacheTTL]);
  
  // Load data progressively with specific resolution
  const loadDataProgressive = useCallback(async (
    symbol: string,
    timeframe: TimeFrame,
    startTime?: Date,
    endTime?: Date,
    resolution: 'low' | 'medium' | 'high' = 'medium'
  ): Promise<OHLCData[]> => {
    const cacheKey = `${symbol}-${timeframe}-${startTime?.getTime()}-${endTime?.getTime()}-${resolution}`;
    
    // Set loading state
    setIsLoading(prev => ({ ...prev, [timeframe]: true }));
    
    try {
      // Check cache first
      const cachedData = dataCache.current[cacheKey];
      const now = Date.now();
      if (cachedData && (now - cachedData.timestamp < cacheTTL)) {
        return cachedData.data;
      }
      
      // Determine appropriate data density based on resolution
      const dataPoints = resolution === 'low' ? 50 : 
                        resolution === 'medium' ? 200 : 500;
      
      // Fetch data with appropriate density
      const data = await fetchChartDataWithResolution(
        symbol,
        timeframe,
        startTime,
        endTime,
        dataPoints
      );
      
      // Cache the result
      dataCache.current[cacheKey] = { data, timestamp: now };
      
      return data;
    } catch (error) {
      console.error(`Error loading progressive data for ${symbol} ${timeframe}:`, error);
      return [];
    } finally {
      setIsLoading(prev => ({ ...prev, [timeframe]: false }));
    }
  }, [cacheTTL]);
  
  return (
    <ChartDataContext.Provider value={{ 
      dataByTimeframe, 
      loadData, 
      loadDataProgressive, 
      isLoading, 
      clearCache 
    }}>
      {children}
    </ChartDataContext.Provider>
  );
}

// Custom hook to use the chart data context
export function useChartData() {
  return useContext(ChartDataContext);
}
