import React from 'react';
import { Button as MuiButton, ButtonProps as MuiButtonProps, styled } from '@mui/material';

export interface ButtonProps extends MuiButtonProps {
  variant?: 'primary' | 'secondary' | 'success' | 'warning' | 'danger' | 'info';
  size?: 'small' | 'medium' | 'large';
}

const StyledButton = styled(MuiButton, {
  shouldForwardProp: (prop) => prop !== 'variant',
})<ButtonProps>(({ theme, variant }) => {
  const variantStyles = {
    primary: {
      backgroundColor: theme.palette.primary.main,
      color: theme.palette.primary.contrastText,
      '&:hover': {
        backgroundColor: theme.palette.primary.dark,
      },
    },
    secondary: {
      backgroundColor: theme.palette.secondary.main,
      color: theme.palette.secondary.contrastText,
      '&:hover': {
        backgroundColor: theme.palette.secondary.dark,
      },
    },
    success: {
      backgroundColor: theme.palette.success.main,
      color: theme.palette.success.contrastText,
      '&:hover': {
        backgroundColor: theme.palette.success.dark,
      },
    },
    warning: {
      backgroundColor: theme.palette.warning.main,
      color: theme.palette.warning.contrastText,
      '&:hover': {
        backgroundColor: theme.palette.warning.dark,
      },
    },
    danger: {
      backgroundColor: theme.palette.error.main,
      color: theme.palette.error.contrastText,
      '&:hover': {
        backgroundColor: theme.palette.error.dark,
      },
    },
    info: {
      backgroundColor: theme.palette.info.main,
      color: theme.palette.info.contrastText,
      '&:hover': {
        backgroundColor: theme.palette.info.dark,
      },
    },
  };

  return variant && variantStyles[variant] ? variantStyles[variant] : {};
});

export const Button: React.FC<ButtonProps> = ({ children, variant = 'primary', ...props }) => {
  const muiVariant = ['primary', 'secondary'].includes(variant) ? variant : 'contained';
  
  return (
    <StyledButton variant={muiVariant as any} variant-custom={variant} {...props}>
      {children}
    </StyledButton>
  );
};

export default Button;
