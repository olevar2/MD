import React from 'react';
import { Card as MuiCard, CardProps as MuiCardProps, CardContent, CardHeader, styled } from '@mui/material';

export interface CardProps extends MuiCardProps {
  title?: string;
  subtitle?: string;
  headerAction?: React.ReactNode;
  variant?: 'default' | 'outlined' | 'elevated';
}

const StyledCard = styled(MuiCard, {
  shouldForwardProp: (prop) => !['variant'].includes(String(prop)),
})<CardProps>(({ theme, variant }) => ({
  borderRadius: '8px',
  ...(variant === 'elevated' && {
    boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
  }),
  ...(variant === 'outlined' && {
    border: `1px solid ${theme.palette.divider}`,
    boxShadow: 'none',
  }),
}));

export const Card: React.FC<CardProps> = ({ 
  children, 
  title,
  subtitle,
  headerAction,
  variant = 'default',
  ...props 
}) => {
  return (
    <StyledCard variant={variant} {...props}>
      {title && (
        <CardHeader
          title={title}
          subheader={subtitle}
          action={headerAction}
        />
      )}
      <CardContent>
        {children}
      </CardContent>
    </StyledCard>
  );
};

export default Card;
