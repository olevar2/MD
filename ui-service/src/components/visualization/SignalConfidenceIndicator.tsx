import React from 'react';
import {
  Box,
  Typography,
  CircularProgress,
  useTheme
} from '@mui/material';
import { useSpring, animated } from '@react-spring/web';

interface SignalConfidenceIndicatorProps {
  confidence: number;
  direction?: 'BUY' | 'SELL' | 'NEUTRAL';
  size?: 'small' | 'medium' | 'large';
  showLabel?: boolean;
  animated?: boolean;
}

const SignalConfidenceIndicator: React.FC<SignalConfidenceIndicatorProps> = ({
  confidence,
  direction = 'NEUTRAL',
  size = 'medium',
  showLabel = true,
  animated = true
}) => {
  const theme = useTheme();

  // Size configurations
  const sizes = {
    small: { width: 60, height: 60, fontSize: '0.875rem' },
    medium: { width: 100, height: 100, fontSize: '1rem' },
    large: { width: 150, height: 150, fontSize: '1.25rem' }
  };

  // Direction-based colors
  const getDirectionColor = () => {
    switch (direction) {
      case 'BUY':
        return theme.palette.success.main;
      case 'SELL':
        return theme.palette.error.main;
      default:
        return theme.palette.grey[500];
    }
  };

  // Animation configuration
  const props = useSpring({
    from: { value: 0 },
    to: { value: confidence },
    config: { tension: 120, friction: 14 },
    immediate: !animated
  });

  return (
    <Box
      sx={{
        position: 'relative',
        width: sizes[size].width,
        height: sizes[size].height,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center'
      }}
    >
      {/* Circular Progress */}
      <Box sx={{ position: 'relative', display: 'inline-flex' }}>
        <CircularProgress
          variant="determinate"
          value={100}
          size={sizes[size].width}
          thickness={4}
          sx={{ color: theme.palette.grey[200] }}
        />
        <animated.div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            transform: 'rotate(-90deg)',
            transformOrigin: '50% 50%'
          }}
        >
          <CircularProgress
            variant="determinate"
            value={props.value.to((v: number) => v * 100)}
            size={sizes[size].width}
            thickness={4}
            sx={{ color: getDirectionColor() }}
          />
        </animated.div>
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            bottom: 0,
            right: 0,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          <animated.div>
            {props.value.to((v: number) => (
              <Typography
                variant={size === 'small' ? 'body2' : 'h6'}
                color="text.primary"
                sx={{ fontWeight: 'bold' }}
              >
                {`${(v * 100).toFixed(0)}%`}
              </Typography>
            ))}
          </animated.div>
          {showLabel && (
            <Typography
              variant="caption"
              color={getDirectionColor()}
              sx={{
                fontWeight: 'medium',
                fontSize: sizes[size].fontSize
              }}
            >
              {direction}
            </Typography>
          )}
        </Box>
      </Box>
    </Box>
  );
};

export default SignalConfidenceIndicator;
