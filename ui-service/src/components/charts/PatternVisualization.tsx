/**
 * PatternVisualization component - Renders forex pattern visualizations
 * Part of Phase 4 implementation for advanced chart visualization
 */
import React, { useState, useEffect, useRef } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  Chip, 
  ButtonGroup, 
  Button,
  Card,
  CardContent,
  Grid,
  Slider,
  FormControl,
  FormControlLabel,
  Switch,
  Tooltip
} from '@mui/material';

// Pattern type definitions
export interface Pattern {
  id: string;
  type: string;
  subType?: string;
  startIndex: number;
  endIndex: number;
  startPrice: number;
  endPrice: number;
  pivotPoints: Array<{x: number, y: number, role: string}>;
  confidence: number;
  completed: boolean;
  projection?: {
    targets: Array<{price: number, ratio: string}>;
    stopLoss: number;
  };
  notes?: string;
}

// Supported pattern types
export const PATTERN_TYPES = {
  ELLIOTT_WAVE: 'elliott_wave',
  HARMONIC: 'harmonic',
  CHART_PATTERN: 'chart_pattern',
  CANDLESTICK: 'candlestick',
  SUPPORT_RESISTANCE: 'support_resistance',
  FIBONACCI: 'fibonacci',
  TREND_LINE: 'trend_line',
  CHANNEL: 'channel'
};

interface PatternVisualizationProps {  canvasWidth?: number;
  canvasHeight?: number; 
  patterns: Pattern[];
  priceData: Array<{time: number, open: number, high: number, low: number, close: number}>;
  visibleRange?: {start: number, end: number};
  minPrice?: number;
  maxPrice?: number;
  onPatternClick?: (pattern: Pattern) => void;
  highlightedPatternId?: string;
  showLabels?: boolean;
  showConfidence?: boolean;
  showProjections?: boolean;
}

export default function PatternVisualization({
  canvasWidth = 800,
  canvasHeight = 300,
  patterns,
  priceData,
  visibleRange = {start: 0, end: 0},
  minPrice,
  maxPrice,
  onPatternClick,
  highlightedPatternId,
  showLabels = true,
  showConfidence = true,
  showProjections = true
}: PatternVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredPattern, setHoveredPattern] = useState<Pattern | null>(null);

  // Determine actual visible range
  const effectiveVisibleRange = {
    start: visibleRange.start || 0,
    end: visibleRange.end || priceData.length - 1
  };

  // Calculate price range if not provided
  const effectiveMinPrice = minPrice || Math.min(...priceData.slice(effectiveVisibleRange.start, effectiveVisibleRange.end + 1).map(d => d.low));
  const effectiveMaxPrice = maxPrice || Math.max(...priceData.slice(effectiveVisibleRange.start, effectiveVisibleRange.end + 1).map(d => d.high));
  
  // Draw patterns when component mounts or inputs change
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw patterns
    drawPatterns(ctx);
    
  }, [patterns, priceData, canvasWidth, canvasHeight, visibleRange, minPrice, maxPrice, highlightedPatternId, showLabels, showConfidence, showProjections]);
  
  // Convert price and time to canvas coordinates
  const priceToY = (price: number): number => {
    const priceRange = effectiveMaxPrice - effectiveMinPrice;
    if (priceRange === 0) return 0;
    
    const normalized = (price - effectiveMinPrice) / priceRange;
    // Invert Y so higher prices are at top
    return canvasHeight - (normalized * canvasHeight);
  };
  
  const indexToX = (index: number): number => {
    const timeRange = effectiveVisibleRange.end - effectiveVisibleRange.start;
    if (timeRange === 0) return 0;
    
    const normalized = (index - effectiveVisibleRange.start) / timeRange;
    return normalized * canvasWidth;
  };
  
  // Draw patterns on canvas
  const drawPatterns = (ctx: CanvasRenderingContext2D) => {
    // Filter patterns that are in visible range
    const visiblePatterns = patterns.filter(pattern => {
      return pattern.startIndex <= effectiveVisibleRange.end && 
             pattern.endIndex >= effectiveVisibleRange.start;
    });
    
    // Sort patterns so the highlighted one is drawn last (on top)
    const sortedPatterns = [...visiblePatterns].sort((a, b) => {
      if (a.id === highlightedPatternId) return 1;
      if (b.id === highlightedPatternId) return -1;
      return b.confidence - a.confidence;
    });
    
    // Draw each pattern
    sortedPatterns.forEach(pattern => {
      const isHighlighted = pattern.id === highlightedPatternId;
      
      switch (pattern.type) {
        case PATTERN_TYPES.ELLIOTT_WAVE:
          drawElliottWavePattern(ctx, pattern, isHighlighted);
          break;
        case PATTERN_TYPES.HARMONIC:
          drawHarmonicPattern(ctx, pattern, isHighlighted);
          break;
        case PATTERN_TYPES.CHART_PATTERN:
          drawChartPattern(ctx, pattern, isHighlighted);
          break;
        case PATTERN_TYPES.SUPPORT_RESISTANCE:
          drawSupportResistanceLine(ctx, pattern, isHighlighted);
          break;
        case PATTERN_TYPES.FIBONACCI:
          drawFibonacciPattern(ctx, pattern, isHighlighted);
          break;
        case PATTERN_TYPES.TREND_LINE:
          drawTrendLine(ctx, pattern, isHighlighted);
          break;
        case PATTERN_TYPES.CHANNEL:
          drawChannel(ctx, pattern, isHighlighted);
          break;
        default:
          // Generic pattern drawing
          drawGenericPattern(ctx, pattern, isHighlighted);
      }
      
      // Draw pattern label if enabled
      if (showLabels) {
        drawPatternLabel(ctx, pattern, isHighlighted);
      }
      
      // Draw confidence indicator if enabled
      if (showConfidence) {
        drawConfidenceIndicator(ctx, pattern, isHighlighted);
      }
      
      // Draw projections if enabled and available
      if (showProjections && pattern.projection) {
        drawPatternProjections(ctx, pattern, isHighlighted);
      }
    });
  };
  
  // Pattern-specific drawing functions
  const drawElliottWavePattern = (ctx: CanvasRenderingContext2D, pattern: Pattern, isHighlighted: boolean) => {
    ctx.save();
    
    // Set style for Elliott Wave pattern
    ctx.strokeStyle = isHighlighted ? '#f9a825' : '#42a5f5';
    ctx.lineWidth = isHighlighted ? 2.5 : 1.5;
    
    // Connect pivot points
    if (pattern.pivotPoints && pattern.pivotPoints.length > 0) {
      ctx.beginPath();
      
      // Move to first point
      const firstPoint = pattern.pivotPoints[0];
      ctx.moveTo(indexToX(firstPoint.x), priceToY(firstPoint.y));
      
      // Draw lines to subsequent points
      for (let i = 1; i < pattern.pivotPoints.length; i++) {
        const point = pattern.pivotPoints[i];
        ctx.lineTo(indexToX(point.x), priceToY(point.y));
      }
      
      ctx.stroke();
      
      // Draw wave labels (1, 2, 3, 4, 5 or A, B, C)
      if (showLabels) {
        pattern.pivotPoints.forEach((point, i) => {
          const x = indexToX(point.x);
          const y = priceToY(point.y);
          
          // Skip first point (0) which is usually the start point
          if (i > 0) {
            ctx.fillStyle = isHighlighted ? '#f9a825' : '#42a5f5';
            ctx.font = isHighlighted ? 'bold 14px Arial' : '12px Arial';
            ctx.fillText(point.role || i.toString(), x + 5, y - 5);
          }
        });
      }
    }
    
    ctx.restore();
  };
  
  const drawHarmonicPattern = (ctx: CanvasRenderingContext2D, pattern: Pattern, isHighlighted: boolean) => {
    ctx.save();
    
    // Set style for Harmonic pattern
    ctx.strokeStyle = isHighlighted ? '#e91e63' : '#9c27b0';
    ctx.lineWidth = isHighlighted ? 2.5 : 1.5;
    
    // Connect XABCD points
    if (pattern.pivotPoints && pattern.pivotPoints.length > 0) {
      ctx.beginPath();
      
      // Move to first point (X)
      const firstPoint = pattern.pivotPoints[0];
      ctx.moveTo(indexToX(firstPoint.x), priceToY(firstPoint.y));
      
      // Draw lines between points
      for (let i = 1; i < pattern.pivotPoints.length; i++) {
        const point = pattern.pivotPoints[i];
        ctx.lineTo(indexToX(point.x), priceToY(point.y));
      }
      
      ctx.stroke();
      
      // Draw harmonic pattern labels (X, A, B, C, D)
      if (showLabels) {
        const labels = ['X', 'A', 'B', 'C', 'D'];
        pattern.pivotPoints.forEach((point, i) => {
          if (i < labels.length) {
            const x = indexToX(point.x);
            const y = priceToY(point.y);
            
            ctx.fillStyle = isHighlighted ? '#e91e63' : '#9c27b0';
            ctx.font = isHighlighted ? 'bold 14px Arial' : '12px Arial';
            ctx.fillText(labels[i], x + 5, y - 5);
          }
        });
      }
    }
    
    ctx.restore();
  };
  
  const drawChartPattern = (ctx: CanvasRenderingContext2D, pattern: Pattern, isHighlighted: boolean) => {
    ctx.save();
    
    // Set style based on pattern subtype
    const getColorForChartPattern = (subType: string | undefined) => {
      switch (subType) {
        case 'head_and_shoulders':
        case 'inverse_head_and_shoulders':
          return isHighlighted ? '#ff5722' : '#ff9800';
        case 'double_top':
        case 'double_bottom':
          return isHighlighted ? '#673ab7' : '#9575cd';
        case 'triangle':
          return isHighlighted ? '#2196f3' : '#90caf9';
        case 'flag':
        case 'pennant':
          return isHighlighted ? '#4caf50' : '#a5d6a7';
        default:
          return isHighlighted ? '#607d8b' : '#90a4ae';
      }
    };
    
    ctx.strokeStyle = getColorForChartPattern(pattern.subType);
    ctx.lineWidth = isHighlighted ? 2.5 : 1.5;
    
    // Draw pattern outline
    if (pattern.pivotPoints && pattern.pivotPoints.length > 0) {
      ctx.beginPath();
      
      // Move to first point
      const firstPoint = pattern.pivotPoints[0];
      ctx.moveTo(indexToX(firstPoint.x), priceToY(firstPoint.y));
      
      // Draw lines to subsequent points
      for (let i = 1; i < pattern.pivotPoints.length; i++) {
        const point = pattern.pivotPoints[i];
        ctx.lineTo(indexToX(point.x), priceToY(point.y));
      }
      
      // For closed patterns, connect back to start
      if (['triangle', 'flag', 'pennant'].includes(pattern.subType || '')) {
        ctx.closePath();
      }
      
      ctx.stroke();
    }
    
    // Draw pattern name if labels are enabled
    if (showLabels) {
      const centerX = indexToX((pattern.startIndex + pattern.endIndex) / 2);
      const centerY = priceToY((pattern.startPrice + pattern.endPrice) / 2);
      
      ctx.fillStyle = getColorForChartPattern(pattern.subType);
      ctx.font = isHighlighted ? 'bold 14px Arial' : '12px Arial';
      ctx.fillText(pattern.subType || 'Pattern', centerX, centerY - 10);
    }
    
    ctx.restore();
  };
  
  const drawSupportResistanceLine = (ctx: CanvasRenderingContext2D, pattern: Pattern, isHighlighted: boolean) => {
    ctx.save();
    
    // Set style for support/resistance lines
    const isSupport = pattern.subType === 'support';
    ctx.strokeStyle = isSupport 
      ? (isHighlighted ? '#4caf50' : '#81c784')  // Green for support
      : (isHighlighted ? '#f44336' : '#e57373'); // Red for resistance
    
    ctx.lineWidth = isHighlighted ? 2.5 : 1.5;
    
    // Draw horizontal line
    const y = priceToY(pattern.startPrice);  // Support/resistance level
    const startX = indexToX(pattern.startIndex);
    const endX = indexToX(pattern.endIndex);
    
    ctx.beginPath();
    ctx.setLineDash([5, 3]); // Dashed line for S/R
    ctx.moveTo(startX, y);
    ctx.lineTo(endX, y);
    ctx.stroke();
    
    // Draw strength indicator (thicker line = stronger level)
    if (showConfidence) {
      // Use confidence as strength indicator
      const strengthWidth = 1 + (pattern.confidence * 4); // Scale 1-5px based on confidence
      ctx.lineWidth = strengthWidth;
      
      ctx.beginPath();
      ctx.setLineDash([]); // Solid line for strength indicator
      ctx.moveTo(startX, y);
      ctx.lineTo(startX + 20, y);
      ctx.stroke();
    }
    
    // Draw label if enabled
    if (showLabels) {
      ctx.fillStyle = isSupport 
        ? (isHighlighted ? '#4caf50' : '#81c784')
        : (isHighlighted ? '#f44336' : '#e57373');
      ctx.font = isHighlighted ? 'bold 14px Arial' : '12px Arial';
      ctx.fillText(isSupport ? 'Support' : 'Resistance', endX - 80, y - 5);
    }
    
    ctx.restore();
  };
  
  const drawFibonacciPattern = (ctx: CanvasRenderingContext2D, pattern: Pattern, isHighlighted: boolean) => {
    ctx.save();
    
    // Set style for Fibonacci pattern
    ctx.strokeStyle = isHighlighted ? '#ff9800' : '#ffc107';
    ctx.lineWidth = isHighlighted ? 2 : 1;
    
    // Draw the base retracement line
    if (pattern.pivotPoints && pattern.pivotPoints.length >= 2) {
      const startPoint = pattern.pivotPoints[0];
      const endPoint = pattern.pivotPoints[1];
      
      const startX = indexToX(startPoint.x);
      const startY = priceToY(startPoint.y);
      const endX = indexToX(endPoint.x);
      const endY = priceToY(endPoint.y);
      
      // Draw the base line
      ctx.beginPath();
      ctx.moveTo(startX, startY);
      ctx.lineTo(endX, endY);
      ctx.stroke();
      
      // Draw Fibonacci levels
      const isUptrend = startY > endY;
      const levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];
      const levelColors = ['rgba(255, 152, 0, 0.3)', 'rgba(255, 193, 7, 0.3)', 'rgba(255, 235, 59, 0.3)', 
                          'rgba(205, 220, 57, 0.3)', 'rgba(156, 39, 176, 0.3)', 'rgba(103, 58, 183, 0.3)'];
      
      // Calculate the full distance
      const fullDistance = endY - startY;
      
      // Draw each level
      for (let i = 0; i < levels.length - 1; i++) {
        const currentLevel = levels[i];
        const nextLevel = levels[i + 1];
        
        const currentY = startY + fullDistance * currentLevel;
        const nextY = startY + fullDistance * nextLevel;
        
        // Fill area between levels
        ctx.fillStyle = levelColors[i] || 'rgba(100, 100, 100, 0.2)';
        ctx.beginPath();
        ctx.moveTo(startX, currentY);
        ctx.lineTo(endX, currentY);
        ctx.lineTo(endX, nextY);
        ctx.lineTo(startX, nextY);
        ctx.closePath();
        ctx.fill();
        
        // Draw level line
        ctx.strokeStyle = isHighlighted ? '#ff9800' : '#ffc107';
        ctx.beginPath();
        ctx.setLineDash([2, 2]);
        ctx.moveTo(startX, currentY);
        ctx.lineTo(endX, currentY);
        ctx.stroke();
        
        // Draw level label
        if (showLabels) {
          ctx.fillStyle = '#333';
          ctx.font = '10px Arial';
          ctx.fillText(`${(currentLevel * 100).toFixed(1)}%`, endX + 5, currentY);
        }
      }
      
      // Draw final level line
      ctx.beginPath();
      ctx.setLineDash([2, 2]);
      ctx.moveTo(startX, startY + fullDistance);
      ctx.lineTo(endX, startY + fullDistance);
      ctx.stroke();
      
      if (showLabels) {
        ctx.fillStyle = '#333';
        ctx.font = '10px Arial';
        ctx.fillText('100%', endX + 5, startY + fullDistance);
      }
    }
    
    ctx.restore();
  };
  
  const drawTrendLine = (ctx: CanvasRenderingContext2D, pattern: Pattern, isHighlighted: boolean) => {
    ctx.save();
    
    // Set style for trend lines
    const isBullish = pattern.endPrice > pattern.startPrice;
    ctx.strokeStyle = isBullish 
      ? (isHighlighted ? '#00c853' : '#66bb6a')  // Green for bullish
      : (isHighlighted ? '#d50000' : '#ef5350'); // Red for bearish
    
    ctx.lineWidth = isHighlighted ? 2.5 : 1.5;
    
    // Draw line
    const startX = indexToX(pattern.startIndex);
    const startY = priceToY(pattern.startPrice);
    const endX = indexToX(pattern.endIndex);
    const endY = priceToY(pattern.endPrice);
    
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.stroke();
    
    // Extend the trend line if it's not completed
    if (!pattern.completed) {
      ctx.setLineDash([5, 5]);
      
      // Calculate slope
      const slope = (endY - startY) / (endX - startX);
      
      // Extend by 20% of visible width
      const extension = canvasWidth * 0.2;
      const extendedX = endX + extension;
      const extendedY = endY + slope * extension;
      
      ctx.beginPath();
      ctx.moveTo(endX, endY);
      ctx.lineTo(extendedX, extendedY);
      ctx.stroke();
    }
    
    ctx.restore();
  };
  
  const drawChannel = (ctx: CanvasRenderingContext2D, pattern: Pattern, isHighlighted: boolean) => {
    ctx.save();
    
    // Set style for channels
    ctx.strokeStyle = isHighlighted ? '#3f51b5' : '#7986cb';
    ctx.lineWidth = isHighlighted ? 2.5 : 1.5;
    
    // For channels, we expect pivotPoints to contain at least 4 points:
    // First two points define the base line, second two define the parallel line
    if (pattern.pivotPoints && pattern.pivotPoints.length >= 4) {
      const baseStart = pattern.pivotPoints[0];
      const baseEnd = pattern.pivotPoints[1];
      const parallelStart = pattern.pivotPoints[2];
      const parallelEnd = pattern.pivotPoints[3];
      
      // Draw base line
      ctx.beginPath();
      ctx.moveTo(indexToX(baseStart.x), priceToY(baseStart.y));
      ctx.lineTo(indexToX(baseEnd.x), priceToY(baseEnd.y));
      ctx.stroke();
      
      // Draw parallel line
      ctx.beginPath();
      ctx.moveTo(indexToX(parallelStart.x), priceToY(parallelStart.y));
      ctx.lineTo(indexToX(parallelEnd.x), priceToY(parallelEnd.y));
      ctx.stroke();
      
      // Fill channel with semi-transparent color
      ctx.beginPath();
      ctx.fillStyle = 'rgba(63, 81, 181, 0.1)';
      ctx.moveTo(indexToX(baseStart.x), priceToY(baseStart.y));
      ctx.lineTo(indexToX(baseEnd.x), priceToY(baseEnd.y));
      ctx.lineTo(indexToX(parallelEnd.x), priceToY(parallelEnd.y));
      ctx.lineTo(indexToX(parallelStart.x), priceToY(parallelStart.y));
      ctx.closePath();
      ctx.fill();
    }
    
    ctx.restore();
  };
  
  // Generic pattern drawing function
  const drawGenericPattern = (ctx: CanvasRenderingContext2D, pattern: Pattern, isHighlighted: boolean) => {
    ctx.save();
    
    // Set style for generic patterns
    ctx.strokeStyle = isHighlighted ? '#607d8b' : '#90a4ae';
    ctx.lineWidth = isHighlighted ? 2.5 : 1.5;
    
    // If pattern has pivot points, use them
    if (pattern.pivotPoints && pattern.pivotPoints.length > 0) {
      ctx.beginPath();
      const firstPoint = pattern.pivotPoints[0];
      ctx.moveTo(indexToX(firstPoint.x), priceToY(firstPoint.y));
      
      for (let i = 1; i < pattern.pivotPoints.length; i++) {
        const point = pattern.pivotPoints[i];
        ctx.lineTo(indexToX(point.x), priceToY(point.y));
      }
      
      ctx.stroke();
    } else {
      // Otherwise just draw a line from start to end
      ctx.beginPath();
      ctx.moveTo(indexToX(pattern.startIndex), priceToY(pattern.startPrice));
      ctx.lineTo(indexToX(pattern.endIndex), priceToY(pattern.endPrice));
      ctx.stroke();
    }
    
    ctx.restore();
  };
  
  // Additional drawing functions for annotations
  const drawPatternLabel = (ctx: CanvasRenderingContext2D, pattern: Pattern, isHighlighted: boolean) => {
    // Calculate position for label
    const centerX = indexToX((pattern.startIndex + pattern.endIndex) / 2);
    const centerY = priceToY((pattern.startPrice + pattern.endPrice) / 2) - 15;
    
    // Set label style
    ctx.fillStyle = isHighlighted ? '#263238' : '#455a64';
    ctx.font = isHighlighted ? 'bold 12px Arial' : '11px Arial';
    
    // Draw label
    const label = `${pattern.subType || pattern.type}`;
    ctx.fillText(label, centerX - ctx.measureText(label).width / 2, centerY);
  };
  
  const drawConfidenceIndicator = (ctx: CanvasRenderingContext2D, pattern: Pattern, isHighlighted: boolean) => {
    // Draw a small indicator showing confidence level
    const x = indexToX(pattern.endIndex) + 5;
    const y = priceToY(pattern.endPrice);
    const width = 30;
    const height = 4;
    
    // Draw confidence bar background
    ctx.fillStyle = '#e0e0e0';
    ctx.fillRect(x, y - height / 2, width, height);
    
    // Draw confidence level
    const confWidth = width * pattern.confidence;
    const confColor = pattern.confidence > 0.7 ? '#4caf50' : (pattern.confidence > 0.4 ? '#ff9800' : '#f44336');
    ctx.fillStyle = isHighlighted ? confColor : `${confColor}99`; // Add transparency if not highlighted
    ctx.fillRect(x, y - height / 2, confWidth, height);
  };
  
  const drawPatternProjections = (ctx: CanvasRenderingContext2D, pattern: Pattern, isHighlighted: boolean) => {
    if (!pattern.projection) return;
    
    ctx.save();
    
    // Draw targets
    pattern.projection.targets.forEach((target, i) => {
      const y = priceToY(target.price);
      
      // Draw target line
      ctx.strokeStyle = isHighlighted ? '#4caf50' : '#81c784'; // Green for targets
      ctx.setLineDash([2, 2]);
      ctx.lineWidth = 1;
      
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvasWidth, y);
      ctx.stroke();
      
      // Draw target label
      ctx.fillStyle = isHighlighted ? '#4caf50' : '#81c784';
      ctx.font = isHighlighted ? 'bold 12px Arial' : '11px Arial';
      ctx.fillText(`T${i+1} (${target.ratio})`, canvasWidth - 70, y - 5);
    });
    
    // Draw stop loss
    if (pattern.projection.stopLoss) {
      const y = priceToY(pattern.projection.stopLoss);
      
      // Draw stop loss line
      ctx.strokeStyle = isHighlighted ? '#f44336' : '#e57373'; // Red for stop loss
      ctx.setLineDash([2, 2]);
      ctx.lineWidth = 1;
      
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvasWidth, y);
      ctx.stroke();
      
      // Draw stop loss label
      ctx.fillStyle = isHighlighted ? '#f44336' : '#e57373';
      ctx.font = isHighlighted ? 'bold 12px Arial' : '11px Arial';
      ctx.fillText('Stop', canvasWidth - 50, y - 5);
    }
    
    ctx.restore();
  };
  
  // Handle mouse events for interactivity
  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Check if mouse is over any pattern
    let foundPattern = null;
    for (const pattern of patterns) {
      // Convert pattern bounds to canvas coordinates
      const startX = indexToX(pattern.startIndex);
      const endX = indexToX(pattern.endIndex);
      const startY = priceToY(pattern.startPrice);
      const endY = priceToY(pattern.endPrice);
      
      // Simple bounding box check
      if (x >= Math.min(startX, endX) && 
          x <= Math.max(startX, endX) && 
          y >= Math.min(startY, endY) - 10 && 
          y <= Math.max(startY, endY) + 10) {
        foundPattern = pattern;
        break;
      }
    }
    
    setHoveredPattern(foundPattern);
  };
  
  const handleCanvasMouseLeave = () => {
    setHoveredPattern(null);
  };
  
  const handleCanvasClick = () => {
    if (hoveredPattern && onPatternClick) {
      onPatternClick(hoveredPattern);
    }
  };
  
  return (
    <Box>
      <canvas 
        ref={canvasRef}
        width={canvasWidth}
        height={canvasHeight}
        onMouseMove={handleCanvasMouseMove}
        onMouseLeave={handleCanvasMouseLeave}
        onClick={handleCanvasClick}
        style={{ cursor: hoveredPattern ? 'pointer' : 'default' }}
      />
      
      {hoveredPattern && (
        <Card 
          sx={{ 
            position: 'absolute', 
            bottom: 10, 
            right: 10, 
            maxWidth: 300, 
            zIndex: 10, 
            backgroundColor: 'rgba(255, 255, 255, 0.9)' 
          }}
        >
          <CardContent sx={{ p: 1 }}>
            <Typography variant="subtitle2">
              {hoveredPattern.subType || hoveredPattern.type}
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
              <Typography variant="caption" sx={{ mr: 1 }}>
                Confidence:
              </Typography>
              <Box sx={{ flex: 1 }}>
                <Slider 
                  value={hoveredPattern.confidence * 100}
                  size="small"
                  disabled
                  valueLabelDisplay="auto"
                  sx={{ 
                    height: 8,
                    color: hoveredPattern.confidence > 0.7 ? 'success.main' : 
                           (hoveredPattern.confidence > 0.4 ? 'warning.main' : 'error.main')
                  }}
                />
              </Box>
            </Box>
            {hoveredPattern.notes && (
              <Typography variant="caption" color="text.secondary">
                {hoveredPattern.notes}
              </Typography>
            )}
          </CardContent>
        </Card>
      )}
    </Box>
  );
}
