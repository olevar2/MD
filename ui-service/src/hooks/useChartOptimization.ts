/**
 * useChartOptimization.ts
 * Custom hooks for optimizing chart rendering and performance
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import { IChartApi, ISeriesApi, Time } from 'lightweight-charts';
import { TimeFrame } from '@/types/strategy';
import { OHLCData } from '@/context/ChartDataContext';

/**
 * Hook to detect WebGL support
 * @returns boolean indicating if WebGL is supported
 */
export function useWebGLRenderer() {
  const [isWebGLSupported, setIsWebGLSupported] = useState<boolean>(false);
  
  useEffect(() => {
    // Check if WebGL is supported
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      setIsWebGLSupported(!!gl);
    } catch (e) {
      setIsWebGLSupported(false);
    }
  }, []);
  
  return isWebGLSupported;
}

/**
 * Hook to manage chart memory and resources
 * @param chartRef Reference to chart instance
 */
export function useChartMemoryManagement(chartRef: React.MutableRefObject<IChartApi | null>) {
  // Function to release memory when chart is not visible
  const releaseMemory = useCallback(() => {
    if (chartRef.current) {
      // Remove all series but keep chart instance
      const series = chartRef.current.series();
      series.forEach(s => chartRef.current?.removeSeries(s));
    }
  }, [chartRef]);
  
  // Function to restore chart when becoming visible again
  const restoreChart = useCallback(() => {
    // Recreate series when chart becomes visible again
    // Implementation depends on specific chart requirements
    console.log('Chart restored');
  }, [chartRef]);
  
  // Clean up resources when component unmounts
  useEffect(() => {
    return () => {
      if (chartRef.current) {
        // Remove all series
        const series = chartRef.current.series();
        series.forEach(s => chartRef.current?.removeSeries(s));
        
        // Remove chart
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [chartRef]);
  
  return { releaseMemory, restoreChart };
}

/**
 * Interface for pattern markers
 */
export interface PatternMarker {
  id: string;
  type: string;
  startTime: Time;
  endTime: Time;
  startPrice: number;
  endPrice: number;
  description: string;
  confidence: number;
  color: string;
}

/**
 * Hook for optimized pattern rendering
 * Only renders patterns that are in the visible range
 */
export function useOptimizedPatternRendering(
  chartRef: React.MutableRefObject<IChartApi | null>,
  patterns: PatternMarker[],
  isVisible: boolean,
  drawPatternOnChart: (chart: IChartApi, pattern: PatternMarker) => any
) {
  const [renderedPatterns, setRenderedPatterns] = useState<Map<string, any>>(new Map());
  
  useEffect(() => {
    if (!chartRef.current || !isVisible) return;
    
    // Clear existing patterns
    renderedPatterns.forEach(marker => {
      if (marker && typeof marker.remove === 'function') {
        marker.remove();
      }
    });
    
    // Only render patterns that are in the visible range
    const visibleRange = chartRef.current.timeScale().getVisibleRange();
    if (!visibleRange) return;
    
    const visiblePatterns = patterns.filter(pattern => {
      const patternTime = typeof pattern.startTime === 'number' ? pattern.startTime : parseInt(pattern.startTime as string);
      return patternTime >= visibleRange.from && patternTime <= visibleRange.to;
    });
    
    // Render only visible patterns
    const newRenderedPatterns = new Map();
    visiblePatterns.forEach(pattern => {
      const marker = drawPatternOnChart(chartRef.current!, pattern);
      if (marker) {
        newRenderedPatterns.set(pattern.id, marker);
      }
    });
    
    setRenderedPatterns(newRenderedPatterns);
    
    // Add listener for visible range changes
    const handleVisibleRangeChange = () => {
      // This would be implemented in a real application
      // For now, we'll just log that the range changed
      console.log('Visible range changed');
    };
    
    chartRef.current.timeScale().subscribeVisibleTimeRangeChange(handleVisibleRangeChange);
    
    return () => {
      chartRef.current?.timeScale().unsubscribeVisibleTimeRangeChange(handleVisibleRangeChange);
    };
  }, [chartRef, patterns, isVisible, drawPatternOnChart, renderedPatterns]);
  
  return renderedPatterns;
}

/**
 * Hook to manage chart resizing
 */
export function useChartResize(
  chartRef: React.MutableRefObject<IChartApi | null>,
  containerRef: React.RefObject<HTMLDivElement>
) {
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ 
          width: containerRef.current.clientWidth 
        });
      }
    };

    window.addEventListener('resize', handleResize);
    
    // Initial resize
    handleResize();
    
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [chartRef, containerRef]);
}

/**
 * Hook to track chart visibility using Intersection Observer
 */
export function useChartVisibility(
  ref: React.RefObject<HTMLDivElement>,
  onVisibilityChange?: (isVisible: boolean) => void
) {
  const [isVisible, setIsVisible] = useState(false);
  
  useEffect(() => {
    if (!ref.current) return;
    
    const observer = new IntersectionObserver(
      ([entry]) => {
        const visible = entry.isIntersecting;
        setIsVisible(visible);
        if (onVisibilityChange) {
          onVisibilityChange(visible);
        }
      },
      { threshold: 0.1 } // Consider visible when 10% is in view
    );
    
    observer.observe(ref.current);
    
    return () => {
      observer.disconnect();
    };
  }, [ref, onVisibilityChange]);
  
  return isVisible;
}

/**
 * Hook to manage data loading for charts
 */
export function useChartDataLoading(
  symbol: string,
  timeframe: TimeFrame,
  loadData: (symbol: string, timeframe: TimeFrame) => Promise<OHLCData[]>,
  isVisible: boolean = true
) {
  const [data, setData] = useState<OHLCData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  
  useEffect(() => {
    if (!isVisible) return;
    
    let isMounted = true;
    setLoading(true);
    
    loadData(symbol, timeframe)
      .then(result => {
        if (isMounted) {
          setData(result);
          setError(null);
        }
      })
      .catch(err => {
        if (isMounted) {
          setError(err);
        }
      })
      .finally(() => {
        if (isMounted) {
          setLoading(false);
        }
      });
    
    return () => {
      isMounted = false;
    };
  }, [symbol, timeframe, loadData, isVisible]);
  
  return { data, loading, error };
}
