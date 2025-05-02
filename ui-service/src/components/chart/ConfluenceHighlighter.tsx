/**
 * ConfluenceHighlighter Component
 * 
 * Renders confluence points as visual markers on the trading chart.
 * Highlights areas where multiple indicators or analysis methods agree.
 * Enhanced with interactive capabilities and additional visualization options.
 */
import React, { useEffect, useRef, useState, useCallback } from 'react';
import { IChartApi, ISeriesApi, SeriesMarker, Time, PriceLineOptions } from 'lightweight-charts';
import { ConfluencePoint, ConfluenceType } from '../../types/chart';

interface ConfluenceHighlighterProps {
  chart: IChartApi;
  series: ISeriesApi<'Candlestick'>;
  confluencePoints: ConfluencePoint[];
  darkMode?: boolean;
  interactive?: boolean;
  onConfluenceClick?: (point: ConfluencePoint) => void;
}

const ConfluenceHighlighter: React.FC<ConfluenceHighlighterProps> = ({
  chart,
  series,
  confluencePoints,
  darkMode = false,
  interactive = true,
  onConfluenceClick,
}) => {
  const markerSeries = useRef<ISeriesApi<'Line'> | null>(null);
  const priceLines = useRef<Map<string, { series: ISeriesApi<'Line'>, lineId: string }>>(new Map());
  const [hoveredConfluence, setHoveredConfluence] = useState<string | null>(null);
  
  // Get icon and color based on confluence type and strength
  const getConfluenceStyle = (type: ConfluenceType, strength: number, importance: number = 5) => {
    // Base colors with opacity based on strength
    const strengthFactor = Math.min(Math.max(strength, 0.3), 1);
    const opacity = Math.floor(strengthFactor * 255).toString(16).padStart(2, '0');
    
    // Size factor based on importance
    const sizeFactor = Math.max(1, Math.min(3, importance / 3));
    
    let color = '';
    let position = 'aboveBar';
    let shape = 'circle';
      switch (type) {
      case 'support':
        color = darkMode ? `#26A69A${opacity}` : `#089981${opacity}`;
        position = 'belowBar';
        break;
      case 'resistance':
        color = darkMode ? `#EF5350${opacity}` : `#F23645${opacity}`;
        position = 'aboveBar';
        break;
      case 'harmonic_pattern':
        color = darkMode ? `#9C27B0${opacity}` : `#9C27B0${opacity}`;
        shape = 'diamond';
        break;
      case 'ma_confluence':
        color = darkMode ? `#1E88E5${opacity}` : `#1E88E5${opacity}`;
        shape = 'square';
        break;
      case 'multi_timeframe':
        color = darkMode ? `#FFB74D${opacity}` : `#FF9800${opacity}`;
        shape = 'arrowUp';
        position = strength >= 0.7 ? 'aboveBar' : 'belowBar';
        break;
      case 'volume_profile':
        color = darkMode ? `#8BC34A${opacity}` : `#8BC34A${opacity}`;
        shape = 'square';
        break;
      case 'order_flow':
        color = darkMode ? `#FF5722${opacity}` : `#FF5722${opacity}`;
        shape = 'arrowDown'; 
        break;
      case 'volatility_contraction':
        color = darkMode ? `#9E9E9E${opacity}` : `#9E9E9E${opacity}`;
        shape = 'circle';
        break;
      case 'liquidity_zone':
        color = darkMode ? `#00BCD4${opacity}` : `#00BCD4${opacity}`;
        shape = 'square';
        position = 'inBar';
        break;
      case 'smart_money_concept':
        color = darkMode ? `#E91E63${opacity}` : `#E91E63${opacity}`;
        shape = 'diamond';
        position = 'inBar';
        break;
      default:
        color = darkMode ? `#7986CB${opacity}` : `#3F51B5${opacity}`;
    }
    
    return {
      color,
      position,
      shape,
      size: Math.round(sizeFactor + (strength * 2)) // Size based on importance and strength
    };
    
    return { color, position, shape };
  };
  // Handle marker click
  const handleMarkerClick = useCallback((point: ConfluencePoint) => {
    if (interactive && onConfluenceClick) {
      onConfluenceClick(point);
    }
  }, [interactive, onConfluenceClick]);

  // Create tooltip for a confluence point
  const createTooltipContent = (point: ConfluencePoint): string => {
    const typeFormatted = point.type.split('_').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
    
    let tooltip = `<div style="padding: 8px;">
      <div style="font-weight: bold; margin-bottom: 4px;">${typeFormatted}</div>
      <div>Strength: ${(point.strength * 100).toFixed(0)}%</div>
      <div>Price: ${point.price.toFixed(5)}</div>`;
    
    if (point.description) {
      tooltip += `<div style="margin-top: 4px;">${point.description}</div>`;
    }
    
    if (point.sources && point.sources.length > 0) {
      tooltip += `<div style="margin-top: 4px; font-style: italic;">Sources: ${point.sources.join(', ')}</div>`;
    }
    
    if (point.regime) {
      tooltip += `<div>Regime: ${point.regime}</div>`;
    }
    
    if (interactive) {
      tooltip += `<div style="font-size: 0.8em; margin-top: 8px; color: #0088ff;">Click for details</div>`;
    }
    
    tooltip += '</div>';
    return tooltip;
  };

  // Add interactive behavior
  const setupInteractivity = useCallback(() => {
    if (!interactive || !chart) return;
    
    // Add click handler to chart
    chart.subscribeCrosshairMove((param) => {
      if (!param.point || !param.seriesData.size) {
        setHoveredConfluence(null);
        return;
      }
      
      // Find closest confluence point to crosshair
      const time = param.time as Time;
      const coordsPrice = param.point.y;
      
      if (time && coordsPrice !== undefined) {
        // Find nearby confluence points
        const nearbyPoints = confluencePoints.filter(point => {
          const timeDiff = Math.abs(Number(point.time) - Number(time));
          const priceDiff = Math.abs(point.price - coordsPrice);
          return timeDiff < 5 && priceDiff < 0.005; // Adjust sensitivity as needed
        });
        
        if (nearbyPoints.length > 0) {
          // Sort by distance and get the closest one
          nearbyPoints.sort((a, b) => {
            const aDist = Math.abs(Number(a.time) - Number(time)) + Math.abs(a.price - coordsPrice);
            const bDist = Math.abs(Number(b.time) - Number(time)) + Math.abs(b.price - coordsPrice);
            return aDist - bDist;
          });
          
          setHoveredConfluence(nearbyPoints[0].id || null);
        } else {
          setHoveredConfluence(null);
        }
      }
    });
  }, [interactive, chart, confluencePoints]);

  // Apply confluence highlights
  useEffect(() => {
    if (!chart || !series || !confluencePoints.length) return;
    
    // Create markers for confluence points
    const markers: SeriesMarker<Time>[] = confluencePoints.map(point => {
      const { color, position, shape, size } = getConfluenceStyle(
        point.type, 
        point.strength,
        point.importance || 5
      );
      
      // Create tooltip
      const tooltipContent = createTooltipContent(point);
      
      return {
        time: point.time,
        position: position as any,
        color,
        shape: shape as any,
        size,
        text: point.description?.substring(0, 10) || `${point.type.substring(0, 1).toUpperCase()}`,
        id: point.id,
        tooltip: tooltipContent,
      };
    });
    
    // Add markers to chart
    series.setMarkers(markers);
    
    // Clean up any existing price lines
    for (const [id, { series: lineSeries }] of priceLines.current.entries()) {
      chart.removeSeries(lineSeries);
    }
    priceLines.current.clear();
    
    // Add highlight areas for strong confluence zones
    const significantConfluences = confluencePoints.filter(
      point => point.strength > 0.6 || (point.importance || 0) > 7
    );
    
    // Group confluences by price level to create zones
    const confluenceZones = significantConfluences.reduce((zones, point) => {
      const key = Math.round(point.price * 100) / 100; // Round to 2 decimal places as key
      if (!zones[key]) {
        zones[key] = [];
      }
      zones[key].push(point);
      return zones;
    }, {} as Record<number, ConfluencePoint[]>);
    
    // Create horizontal lines for strong confluence zones
    Object.entries(confluenceZones).forEach(([priceStr, points]) => {
      const price = parseFloat(priceStr);
      
      // Calculate zone strength based on combined points
      const totalPoints = points.length;
      const combinedStrength = points.reduce((sum, p) => sum + p.strength, 0) / totalPoints;
      const maxImportance = Math.max(...points.map(p => p.importance || 5));
      
      // Determine zone type based on contained points
      const types = points.map(p => p.type);
      let dominantType: ConfluenceType;
      
      if (types.includes('support')) {
        dominantType = 'support';
      } else if (types.includes('resistance')) {
        dominantType = 'resistance';
      } else if (types.includes('liquidity_zone')) {
        dominantType = 'liquidity_zone';
      } else {
        dominantType = points[0].type;
      }
      
      // Create style for this zone
      const { color } = getConfluenceStyle(dominantType, combinedStrength, maxImportance);
      
      // Create zone ID
      const zoneId = `zone-${priceStr}-${dominantType}`;
      
      // Create horizontal line series for this zone
      const zoneSeries = chart.addLineSeries({
        lineVisible: false,
        lastValueVisible: false,
        priceLineVisible: false,
      });
      
      // Build descriptive title for zone
      let typeNames = [...new Set(types)].map(type => 
        type.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')
      );
      
      // Create zone label with sources count and strength
      const zoneTitle = `${typeNames.join(' + ')} (${totalPoints} signals, ${(combinedStrength * 100).toFixed(0)}%)`;
      
      // Create the price line
      const priceLine = {
        price,
        color,
        lineWidth: 1 + Math.floor(combinedStrength * 3),
        lineStyle: 2, // Dashed line
        axisLabelVisible: true,
        title: zoneTitle,
      };
      
      const lineId = zoneSeries.createPriceLine(priceLine as PriceLineOptions);
      
      // Store the series and line ID for later cleanup
      priceLines.current.set(zoneId, { series: zoneSeries, lineId });
    });
    
    // Set up interactivity after rendering
    setupInteractivity();
    
    // Cleanup
    return () => {
      series.setMarkers([]);
      for (const [id, { series: lineSeries }] of priceLines.current.entries()) {
        chart.removeSeries(lineSeries);
      }
      priceLines.current.clear();
    };
  }, [chart, series, confluencePoints, darkMode, interactive, setupInteractivity, createTooltipContent]);
  
  // No actual DOM render, this component only manages chart markers
  return null;
};

export default ConfluenceHighlighter;
