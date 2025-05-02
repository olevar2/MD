import React from 'react';
import { Box, Typography, CircularProgress, useTheme } from '@mui/material';
import { Check, Warning, Error, Help } from '@mui/icons-material';

export type StatusType = 'Operational' | 'Degraded' | 'Outage' | 'Unknown';

interface StatusIndicatorProps {
  status: StatusType;
  label?: string;
  showLabel?: boolean;
  size?: 'small' | 'medium' | 'large';
}

const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  status,
  label,
  showLabel = true,
  size = 'medium'
}) => {
  const theme = useTheme();

  const getStatusColor = () => {
    switch (status) {
      case 'Operational':
        return theme.palette.success.main;
      case 'Degraded':
        return theme.palette.warning.main;
      case 'Outage':
        return theme.palette.error.main;
      default:
        return theme.palette.grey[500];
    }
  };

  const getStatusIcon = () => {
    const iconProps = {
      fontSize: size,
      sx: { color: getStatusColor() }
    };

    switch (status) {
      case 'Operational':
        return <Check {...iconProps} />;
      case 'Degraded':
        return <Warning {...iconProps} />;
      case 'Outage':
        return <Error {...iconProps} />;
      default:
        return <Help {...iconProps} />;
    }
  };

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 1
      }}
    >
      {getStatusIcon()}
      {showLabel && (
        <Typography
          variant={size === 'small' ? 'caption' : 'body2'}
          color={getStatusColor()}
        >
          {label || status}
        </Typography>
      )}
    </Box>
  );
};

export default StatusIndicator;
