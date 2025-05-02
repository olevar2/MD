import React from 'react';
import { Box, Typography, Tooltip, styled } from '@mui/material';

export type StatusType = 'success' | 'warning' | 'error' | 'info' | 'neutral';

export interface StatusIndicatorProps {
  status: StatusType;
  label?: string;
  tooltip?: string;
  size?: 'small' | 'medium' | 'large';
  pulsating?: boolean;
}

const StatusDot = styled(Box, {
  shouldForwardProp: (prop) => !['status', 'size', 'pulsating'].includes(String(prop)),
})<{ status: StatusType; size: string; pulsating: boolean }>(
  ({ theme, status, size, pulsating }) => {
    const sizeMap = {
      small: 8,
      medium: 12,
      large: 16,
    };

    const colorMap = {
      success: theme.palette.success.main,
      warning: theme.palette.warning.main,
      error: theme.palette.error.main,
      info: theme.palette.info.main,
      neutral: theme.palette.grey[400],
    };

    return {
      width: sizeMap[size],
      height: sizeMap[size],
      backgroundColor: colorMap[status],
      borderRadius: '50%',
      display: 'inline-block',
      ...(pulsating && {
        animation: 'pulse 2s infinite',
        '@keyframes pulse': {
          '0%': {
            boxShadow: `0 0 0 0 rgba(${colorMap[status]}, 0.7)`,
          },
          '70%': {
            boxShadow: `0 0 0 10px rgba(${colorMap[status]}, 0)`,
          },
          '100%': {
            boxShadow: `0 0 0 0 rgba(${colorMap[status]}, 0)`,
          },
        },
      }),
    };
  }
);

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  status,
  label,
  tooltip,
  size = 'medium',
  pulsating = false,
}) => {
  const indicator = (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
      <StatusDot status={status} size={size} pulsating={pulsating} />
      {label && (
        <Typography variant="body2" component="span">
          {label}
        </Typography>
      )}
    </Box>
  );

  if (tooltip) {
    return (
      <Tooltip title={tooltip} arrow>
        {indicator}
      </Tooltip>
    );
  }

  return indicator;
};

export default StatusIndicator;
