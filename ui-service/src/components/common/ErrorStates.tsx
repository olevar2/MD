import React from 'react';
import { Box, Typography, Button, Paper, CircularProgress, Alert } from '@mui/material';
import { ErrorType } from '../../utils/errorHandler';

interface ErrorStateProps {
  message?: string;
  onRetry?: () => void;
  onBack?: () => void;
}

/**
 * Loading Error component for displaying when data fails to load
 */
export const LoadingError: React.FC<ErrorStateProps> = ({ 
  message = 'Failed to load data. Please try again.',
  onRetry,
  onBack
}) => (
  <Paper
    elevation={2}
    sx={{
      p: 3,
      m: 2,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      textAlign: 'center'
    }}
  >
    <Alert severity="error" sx={{ mb: 2, width: '100%' }}>
      {message}
    </Alert>
    
    <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
      {onRetry && (
        <Button
          variant="contained"
          color="primary"
          onClick={onRetry}
        >
          Retry
        </Button>
      )}
      
      {onBack && (
        <Button
          variant="outlined"
          onClick={onBack}
        >
          Go Back
        </Button>
      )}
    </Box>
  </Paper>
);

/**
 * Network Error component for displaying when network issues occur
 */
export const NetworkError: React.FC<ErrorStateProps> = ({
  message = 'Network connection issue. Please check your internet connection.',
  onRetry
}) => (
  <Paper
    elevation={2}
    sx={{
      p: 3,
      m: 2,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      textAlign: 'center'
    }}
  >
    <Typography variant="h6" color="error" gutterBottom>
      Connection Error
    </Typography>
    
    <Alert severity="error" sx={{ mb: 2, width: '100%' }}>
      {message}
    </Alert>
    
    {onRetry && (
      <Button
        variant="contained"
        color="primary"
        onClick={onRetry}
        sx={{ mt: 2 }}
      >
        Retry Connection
      </Button>
    )}
  </Paper>
);

/**
 * Authentication Error component for displaying when authentication issues occur
 */
export const AuthenticationError: React.FC<ErrorStateProps> = ({
  message = 'You need to sign in to access this feature.',
  onRetry
}) => (
  <Paper
    elevation={2}
    sx={{
      p: 3,
      m: 2,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      textAlign: 'center'
    }}
  >
    <Typography variant="h6" color="warning.main" gutterBottom>
      Authentication Required
    </Typography>
    
    <Alert severity="warning" sx={{ mb: 2, width: '100%' }}>
      {message}
    </Alert>
    
    {onRetry && (
      <Button
        variant="contained"
        color="primary"
        onClick={onRetry}
        sx={{ mt: 2 }}
      >
        Sign In
      </Button>
    )}
  </Paper>
);

/**
 * Authorization Error component for displaying when permission issues occur
 */
export const AuthorizationError: React.FC<ErrorStateProps> = ({
  message = 'You do not have permission to access this feature.',
  onBack
}) => (
  <Paper
    elevation={2}
    sx={{
      p: 3,
      m: 2,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      textAlign: 'center'
    }}
  >
    <Typography variant="h6" color="warning.main" gutterBottom>
      Access Denied
    </Typography>
    
    <Alert severity="warning" sx={{ mb: 2, width: '100%' }}>
      {message}
    </Alert>
    
    {onBack && (
      <Button
        variant="outlined"
        onClick={onBack}
        sx={{ mt: 2 }}
      >
        Go Back
      </Button>
    )}
  </Paper>
);

/**
 * Service Error component for displaying when service issues occur
 */
export const ServiceError: React.FC<ErrorStateProps> = ({
  message = 'This service is currently unavailable. Please try again later.',
  onRetry,
  onBack
}) => (
  <Paper
    elevation={2}
    sx={{
      p: 3,
      m: 2,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      textAlign: 'center'
    }}
  >
    <Typography variant="h6" color="error" gutterBottom>
      Service Unavailable
    </Typography>
    
    <Alert severity="error" sx={{ mb: 2, width: '100%' }}>
      {message}
    </Alert>
    
    <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
      {onRetry && (
        <Button
          variant="contained"
          color="primary"
          onClick={onRetry}
        >
          Retry
        </Button>
      )}
      
      {onBack && (
        <Button
          variant="outlined"
          onClick={onBack}
        >
          Go Back
        </Button>
      )}
    </Box>
  </Paper>
);

/**
 * Empty State component for displaying when no data is available
 */
export const EmptyState: React.FC<{
  title?: string;
  message?: string;
  actionLabel?: string;
  onAction?: () => void;
}> = ({
  title = 'No Data Available',
  message = 'There is no data to display at this time.',
  actionLabel,
  onAction
}) => (
  <Paper
    elevation={1}
    sx={{
      p: 3,
      m: 2,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      textAlign: 'center'
    }}
  >
    <Typography variant="h6" gutterBottom>
      {title}
    </Typography>
    
    <Typography variant="body1" color="text.secondary" paragraph>
      {message}
    </Typography>
    
    {actionLabel && onAction && (
      <Button
        variant="outlined"
        onClick={onAction}
        sx={{ mt: 2 }}
      >
        {actionLabel}
      </Button>
    )}
  </Paper>
);

/**
 * Loading State component for displaying during data loading
 */
export const LoadingState: React.FC<{
  message?: string;
}> = ({
  message = 'Loading data...'
}) => (
  <Box
    sx={{
      p: 3,
      m: 2,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      textAlign: 'center'
    }}
  >
    <CircularProgress size={40} sx={{ mb: 2 }} />
    <Typography variant="body1" color="text.secondary">
      {message}
    </Typography>
  </Box>
);

/**
 * Error State Factory - returns the appropriate error component based on error type
 */
export const ErrorStateFactory: React.FC<{
  errorType: ErrorType;
  message?: string;
  onRetry?: () => void;
  onBack?: () => void;
}> = ({ errorType, message, onRetry, onBack }) => {
  switch (errorType) {
    case ErrorType.AUTHENTICATION_ERROR:
      return <AuthenticationError message={message} onRetry={onRetry} />;
      
    case ErrorType.AUTHORIZATION_ERROR:
      return <AuthorizationError message={message} onBack={onBack} />;
      
    case ErrorType.NETWORK_ERROR:
      return <NetworkError message={message} onRetry={onRetry} />;
      
    case ErrorType.SERVICE_ERROR:
    case ErrorType.SERVICE_UNAVAILABLE_ERROR:
    case ErrorType.SERVICE_TIMEOUT_ERROR:
      return <ServiceError message={message} onRetry={onRetry} onBack={onBack} />;
      
    case ErrorType.DATA_FETCH_ERROR:
    case ErrorType.DATA_VALIDATION_ERROR:
    case ErrorType.DATA_STORAGE_ERROR:
    case ErrorType.DATA_TRANSFORMATION_ERROR:
      return <LoadingError message={message} onRetry={onRetry} onBack={onBack} />;
      
    default:
      return <LoadingError message={message || 'An unexpected error occurred'} onRetry={onRetry} onBack={onBack} />;
  }
};
