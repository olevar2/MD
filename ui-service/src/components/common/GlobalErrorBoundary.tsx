import React from 'react';
import ErrorBoundary from './ErrorBoundary';
import { Box, Typography, Button } from '@mui/material';
import { ErrorType } from '../../utils/errorHandler';

/**
 * Global Error Boundary component that wraps the entire application.
 * This component provides a more comprehensive error handling for application-level errors.
 */
const GlobalErrorBoundary: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const handleGlobalError = (error: Error, errorInfo: React.ErrorInfo) => {
    // Here we could send the error to a monitoring service like Sentry
    console.error('Global error caught:', error, errorInfo);
    
    // You could also store the error in local storage to show it on reload
    // localStorage.setItem('lastGlobalError', JSON.stringify({
    //   message: error.message,
    //   stack: error.stack,
    //   time: new Date().toISOString()
    // }));
  };
  
  const globalFallbackUI = (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '100vh',
        p: 4,
        textAlign: 'center',
        bgcolor: 'background.default'
      }}
    >
      <Typography variant="h3" color="error" gutterBottom>
        Application Error
      </Typography>
      
      <Typography variant="h6" sx={{ mb: 4 }}>
        We're sorry, but something went wrong with the application.
      </Typography>
      
      <Typography variant="body1" sx={{ mb: 4, maxWidth: '600px' }}>
        Our team has been notified of this issue and we're working to fix it as soon as possible.
        In the meantime, you can try refreshing the page or coming back later.
      </Typography>
      
      <Box sx={{ display: 'flex', gap: 2 }}>
        <Button
          variant="contained"
          color="primary"
          size="large"
          onClick={() => window.location.reload()}
        >
          Refresh Page
        </Button>
        
        <Button
          variant="outlined"
          size="large"
          onClick={() => window.location.href = '/'}
        >
          Go to Home
        </Button>
      </Box>
    </Box>
  );
  
  return (
    <ErrorBoundary
      fallback={globalFallbackUI}
      onError={handleGlobalError}
      errorBoundaryName="GlobalErrorBoundary"
      resetOnPropsChange={false}
      showErrorDetails={true}
    >
      {children}
    </ErrorBoundary>
  );
};

export default GlobalErrorBoundary;
