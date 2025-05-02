/**
 * PatternVisualization Component
 * 
 * Advanced visualization component for chart patterns (harmonic patterns, Elliott waves, etc.)
 * that complements the confluence highlighting.
 */
import React, { useEffect, useRef } from 'react';
import { IChartApi, ISeriesApi, Time } from 'lightweight-charts';
import { PatternData, PatternType } from '../../types/chart';

interface PatternVisualizationProps {
  chart: IChartApi;
  series: ISeriesApi<'Candlestick'>;
  patterns: PatternData[];
  darkMode?: boolean;
}

const PatternVisualization: React.FC<PatternVisualizationProps> = ({
  chart,
  series,
  patterns,
  darkMode = false,
}) => {
  // Store created line series for cleanup
  const lineSeries = useRef<ISeriesApi<'Line'>[]>([]);
  
  // Get pattern style based on pattern type and completion status
  const getPatternStyle = (pattern: PatternData) => {
    const opacity = Math.floor((pattern.completion || 0.7) * 255).toString(16).padStart(2, '0');
    
    const styleMap: Record<PatternType, { color: string, lineWidth: number, lineStyle: number }> = {
      'gartley': {
        color: darkMode ? `#9C27B0${opacity}` : `#9C27B0${opacity}`,
        lineWidth: 2,
        lineStyle: 2, // Dashed
      },
      'butterfly': {
        color: darkMode ? `#E91E63${opacity}` : `#E91E63${opacity}`,
        lineWidth: 2,
        lineStyle: 2, // Dashed
      },
      'bat': {
        color: darkMode ? `#673AB7${opacity}` : `#673AB7${opacity}`,
        lineWidth: 2,
        lineStyle: 2, // Dashed
      },
      'crab': {
        color: darkMode ? `#3F51B5${opacity}` : `#3F51B5${opacity}`,
        lineWidth: 2,
        lineStyle: 2, // Dashed
      },
      'shark': {
        color: darkMode ? `#00BCD4${opacity}` : `#00BCD4${opacity}`,
        lineWidth: 2,
        lineStyle: 2, // Dashed
      },
      'cypher': {
        color: darkMode ? `#4CAF50${opacity}` : `#4CAF50${opacity}`,
        lineWidth: 2,
        lineStyle: 2, // Dashed
      },
      'abcd': {
        color: darkMode ? `#FFC107${opacity}` : `#FF9800${opacity}`,
        lineWidth: 2,
        lineStyle: 2, // Dashed
      },
      'triangle': {
        color: darkMode ? `#795548${opacity}` : `#795548${opacity}`,
        lineWidth: 2,
        lineStyle: 3, // Dotted
      },
      'elliott_wave': {
        color: darkMode ? `#2196F3${opacity}` : `#2196F3${opacity}`,
        lineWidth: 2,
        lineStyle: 0, // Solid
      },
      'head_shoulders': {
        color: darkMode ? `#FF5722${opacity}` : `#FF5722${opacity}`,
        lineWidth: 2,
        lineStyle: 1, // Solid with sparse dots
      },
      'double_top': {
        color: darkMode ? `#F44336${opacity}` : `#F44336${opacity}`,
        lineWidth: 2,
        lineStyle: 1, // Solid with sparse dots
      },
      'double_bottom': {
        color: darkMode ? `#4CAF50${opacity}` : `#4CAF50${opacity}`,
        lineWidth: 2,
        lineStyle: 1, // Solid with sparse dots
      },
      'flag': {
        color: darkMode ? `#FF9800${opacity}` : `#FF9800${opacity}`,
        lineWidth: 1,
        lineStyle: 0, // Solid
      },
      'wedge': {
        color: darkMode ? `#9E9E9E${opacity}` : `#9E9E9E${opacity}`,
        lineWidth: 1,
        lineStyle: 0, // Solid
      },
    };
    
    return styleMap[pattern.type] || {
      color: darkMode ? `#7986CB${opacity}` : `#3F51B5${opacity}`,
      lineWidth: 1,
      lineStyle: 0, // Solid
    };
  };
  
  // Draw a pattern on the chart
  const drawPattern = (pattern: PatternData) => {
    // Get style based on pattern type
    const { color, lineWidth, lineStyle } = getPatternStyle(pattern);
    
    // Create line series for pattern
    const patternSeries = chart.addLineSeries({
      color,
      lineWidth,
      lineStyle,
      lastValueVisible: false,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
      autoscaleInfoProvider: () => ({
        priceRange: {
          minValue: pattern.points[0].price * 0.99,
          maxValue: pattern.points[0].price * 1.01,
        },
      }),
    });
    
    // Add to ref for cleanup
    lineSeries.current.push(patternSeries);
    
    // Set pattern data points
    patternSeries.setData(pattern.points.map(p => ({
      time: p.time,
      value: p.price,
    })));
    
    // Add pattern labels if enabled
    if (pattern.labelPoints) {
      pattern.labelPoints.forEach((label) => {
        const coordinate = patternSeries.priceToCoordinate(label.price);
        if (coordinate) {
          const markerSeries = chart.addLineSeries({
            lastValueVisible: false,
            priceLineVisible: false,
          });
          lineSeries.current.push(markerSeries);
          
          markerSeries.setMarkers([
            {
              time: label.time,
              position: 'inBar',
              color,
              shape: 'circle',
              text: label.text,
            },
          ]);
        }
      });
    }
    
    // Add projected targets if available
    if (pattern.projectionPoints && pattern.projectionPoints.length > 0) {
      const projectionSeries = chart.addLineSeries({
        color: color.replace(opacity, '80'), // More transparent
        lineWidth: 1,
        lineStyle: 2, // Dashed
        lastValueVisible: false,
        priceLineVisible: false,
      });
      
      lineSeries.current.push(projectionSeries);
      
      projectionSeries.setData(pattern.projectionPoints.map(p => ({
        time: p.time,
        value: p.price,
      })));
    }
  };
  
  // Apply pattern visualization
  useEffect(() => {
    if (!chart || !series || patterns.length === 0) return;
    
    // Clean up previous pattern visualizations
    lineSeries.current.forEach(series => {
      chart.removeSeries(series);
    });
    lineSeries.current = [];
    
    // Draw each pattern
    patterns.forEach(drawPattern);
    
    // Cleanup
    return () => {
      lineSeries.current.forEach(series => {
        chart.removeSeries(series);
      });
      lineSeries.current = [];
    };
  }, [chart, series, patterns, darkMode]);
  
  // No actual DOM render, this component only manages chart visualizations
  return null;
};

export default PatternVisualization;
