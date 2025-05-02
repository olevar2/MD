import React, { Component, ErrorInfo, ReactNode } from 'react';
import { Box, Typography, Button, Paper } from '@mui/material';
import { logError } from '../../utils/errorHandler';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

/**
 * Error Boundary component to catch JavaScript errors anywhere in the child component tree,
 * log those errors, and display a fallback UI instead of the component tree that crashed.
 */
class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null
    };
  }

  static getDerivedStateFromError(error: Error): State {
    // Update state so the next render will show the fallback UI
    return {
      hasError: true,
      error,
      errorInfo: null
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // Log the error to the error reporting service
    logError(error, { componentStack: errorInfo.componentStack });
    
    this.setState({
      error,
      errorInfo
    });
  }

  handleReset = (): void => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null
    });
  };

  render(): ReactNode {
    if (this.state.hasError) {
      // Custom fallback UI
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Default fallback UI
      return (
        <Paper 
          elevation={3} 
          sx={{ 
            p: 4, 
            m: 2, 
            maxWidth: '800px', 
            mx: 'auto',
            backgroundColor: 'error.lighter',
            border: '1px solid',
            borderColor: 'error.light'
          }}
        >
          <Typography variant="h5" color="error" gutterBottom>
            Something went wrong
          </Typography>
          
          <Typography variant="body1" paragraph>
            We apologize for the inconvenience. An error has occurred in this component.
          </Typography>
          
          {process.env.NODE_ENV === 'development' && this.state.error && (
            <Box sx={{ my: 2, p: 2, bgcolor: 'background.paper', borderRadius: 1, overflowX: 'auto' }}>
              <Typography variant="subtitle2" fontWeight="bold">
                Error Details:
              </Typography>
              <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap' }}>
                {this.state.error.toString()}
              </Typography>
              
              {this.state.errorInfo && (
                <>
                  <Typography variant="subtitle2" fontWeight="bold" sx={{ mt: 2 }}>
                    Component Stack:
                  </Typography>
                  <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap' }}>
                    {this.state.errorInfo.componentStack}
                  </Typography>
                </>
              )}
            </Box>
          )}
          
          <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
            <Button 
              variant="contained" 
              color="primary" 
              onClick={this.handleReset}
            >
              Try Again
            </Button>
            
            <Button 
              variant="outlined"
              onClick={() => window.location.href = '/'}
            >
              Go to Home
            </Button>
          </Box>
        </Paper>
      );
    }

    // When there's no error, render children normally
    return this.props.children;
  }
}

export default ErrorBoundary;
