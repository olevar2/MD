import React from 'react';

interface RegimeIndicatorProps {
  regime: string;
  size?: 'small' | 'medium' | 'large';
  showLabel?: boolean;
  variant?: 'default' | 'compact' | 'detailed';
}

/**
 * Component that visually represents different market regimes with appropriate colors and icons
 * 
 * @param regime - The market regime (trending, ranging, volatile, etc.)
 * @param size - The size of the indicator (small, medium, large)
 * @param showLabel - Whether to display the regime label text
 * @param variant - Display variant (default, compact, detailed)
 */
const RegimeIndicator: React.FC<RegimeIndicatorProps> = ({ 
  regime,
  size = 'medium',
  showLabel = true,
  variant = 'default'
}) => {
  // Define regime configurations
  const regimeConfigs: Record<string, { 
    color: string; 
    backgroundColor: string; 
    icon: string;
    description: string;
  }> = {
    'trending_bullish': {
      color: '#198754',
      backgroundColor: 'rgba(25, 135, 84, 0.15)',
      icon: '↗',
      description: 'Strong uptrend - favorable for long positions'
    },
    'trending_bearish': {
      color: '#dc3545',
      backgroundColor: 'rgba(220, 53, 69, 0.15)',
      icon: '↘',
      description: 'Strong downtrend - favorable for short positions'
    },
    'ranging': {
      color: '#6c757d',
      backgroundColor: 'rgba(108, 117, 125, 0.15)',
      icon: '↔',
      description: 'Price moving sideways within a channel'
    },
    'volatile': {
      color: '#fd7e14',
      backgroundColor: 'rgba(253, 126, 20, 0.15)',
      icon: '↕',
      description: 'High volatility - caution advised'
    },
    'breakout': {
      color: '#0dcaf0',
      backgroundColor: 'rgba(13, 202, 240, 0.15)',
      icon: '⇢',
      description: 'Potential breakout from a range'
    },
    'reversal': {
      color: '#9333ea',
      backgroundColor: 'rgba(147, 51, 234, 0.15)',
      icon: '↩',
      description: 'Potential trend reversal'
    },
    'low_liquidity': {
      color: '#664d03',
      backgroundColor: 'rgba(102, 77, 3, 0.15)',
      icon: '⚠',
      description: 'Low liquidity conditions - wider spreads'
    },
    'high_impact_news': {
      color: '#084298',
      backgroundColor: 'rgba(8, 66, 152, 0.15)',
      icon: '📰',
      description: 'High impact news - increased volatility expected'
    },
    'uncertain': {
      color: '#6c757d',
      backgroundColor: 'rgba(108, 117, 125, 0.15)',
      icon: '?',
      description: 'Uncertain market conditions'
    }
  };

  // Default to 'uncertain' if regime is not recognized
  const regimeConfig = regimeConfigs[regime.toLowerCase()] || regimeConfigs.uncertain;
  
  // Format regime text for display (replace underscores with spaces, capitalize)
  const formatRegimeText = (text: string): string => {
    return text
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ');
  };

  // Determine size class
  const getSizeClass = () => {
    switch (size) {
      case 'small': return 'regime-indicator-small';
      case 'large': return 'regime-indicator-large';
      default: return '';
    }
  };

  return (
    <div className={`regime-indicator ${getSizeClass()} variant-${variant}`}>
      <div 
        className="regime-icon" 
        style={{ 
          backgroundColor: regimeConfig.backgroundColor,
          color: regimeConfig.color 
        }}
      >
        {regimeConfig.icon}
      </div>
      
      {showLabel && (
        <div className="regime-label" style={{ color: regimeConfig.color }}>
          {formatRegimeText(regime)}
        </div>
      )}
      
      {variant === 'detailed' && (
        <div className="regime-description">
          {regimeConfig.description}
        </div>
      )}
      
      <style jsx>{`
        .regime-indicator {
          display: inline-flex;
          align-items: center;
          gap: 8px;
        }
        
        .regime-icon {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 24px;
          height: 24px;
          border-radius: 50%;
          font-weight: bold;
          font-size: 14px;
        }
        
        .regime-label {
          font-weight: 500;
          font-size: 14px;
        }
        
        .regime-description {
          font-size: 12px;
          color: #6c757d;
          margin-left: 4px;
        }
        
        /* Size variants */
        .regime-indicator-small .regime-icon {
          width: 18px;
          height: 18px;
          font-size: 12px;
        }
        
        .regime-indicator-small .regime-label {
          font-size: 12px;
        }
        
        .regime-indicator-small .regime-description {
          font-size: 10px;
        }
        
        .regime-indicator-large .regime-icon {
          width: 32px;
          height: 32px;
          font-size: 18px;
        }
        
        .regime-indicator-large .regime-label {
          font-size: 16px;
        }
        
        .regime-indicator-large .regime-description {
          font-size: 14px;
        }
        
        /* Variants */
        .variant-compact {
          gap: 4px;
        }
        
        .variant-compact .regime-label {
          font-size: 12px;
        }
        
        .variant-detailed {
          flex-direction: column;
          align-items: flex-start;
          gap: 4px;
        }
        
        .variant-detailed .regime-icon-label {
          display: flex;
          align-items: center;
          gap: 8px;
        }
      `}</style>
    </div>
  );
};

export default RegimeIndicator;
