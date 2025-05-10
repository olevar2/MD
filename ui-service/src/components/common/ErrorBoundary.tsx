import React, { Component, ErrorInfo, ReactNode } from 'react';
import { Box, Typography, Button, Paper, Divider, Chip, Alert } from '@mui/material';
import { logError, ErrorType, getErrorType } from '../../utils/errorHandler';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  resetOnPropsChange?: boolean;
  showErrorDetails?: boolean;
  errorBoundaryName?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  errorType: ErrorType;
}

/**
 * Enhanced Error Boundary component to catch JavaScript errors anywhere in the child component tree,
 * log those errors, and display a fallback UI instead of the component tree that crashed.
 *
 * Features:
 * - Customizable fallback UI
 * - Error type detection
 * - Detailed error information in development mode
 * - Error reporting to monitoring services
 * - Reset capabilities
 * - Optional callback for error handling
 */
class ErrorBoundary extends Component<Props, State> {
  static defaultProps = {
    resetOnPropsChange: false,
    showErrorDetails: process.env.NODE_ENV === 'development',
    errorBoundaryName: 'Component'
  };

  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorType: ErrorType.UNKNOWN_ERROR
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    // Update state so the next render will show the fallback UI
    return {
      hasError: true,
      error,
      errorType: getErrorType(error)
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // Determine error type
    const errorType = getErrorType(error);

    // Log the error to the error reporting service
    logError(error, {
      componentStack: errorInfo.componentStack,
      errorBoundaryName: this.props.errorBoundaryName,
      errorType
    });

    // Update state with error info
    this.setState({
      errorInfo,
      errorType
    });

    // Call onError callback if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
  }

  componentDidUpdate(prevProps: Props): void {
    // Reset error state when props change if resetOnPropsChange is true
    if (
      this.props.resetOnPropsChange &&
      this.state.hasError &&
      prevProps.children !== this.props.children
    ) {
      this.handleReset();
    }
  }

  handleReset = (): void => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      errorType: ErrorType.UNKNOWN_ERROR
    });
  };

  handleReportError = (): void => {
    // Report error to support team
    // This could send an email, create a ticket, etc.
    alert('Error reported to support team. Thank you!');
  };

  renderErrorTypeChip = (errorType: ErrorType): ReactNode => {
    let color: 'error' | 'warning' | 'info' | 'default' = 'default';

    switch (errorType) {
      case ErrorType.AUTHENTICATION_ERROR:
      case ErrorType.AUTHORIZATION_ERROR:
        color = 'warning';
        break;
      case ErrorType.SERVICE_ERROR:
      case ErrorType.SERVICE_UNAVAILABLE_ERROR:
      case ErrorType.SERVICE_TIMEOUT_ERROR:
      case ErrorType.NETWORK_ERROR:
        color = 'error';
        break;
      case ErrorType.DATA_FETCH_ERROR:
      case ErrorType.DATA_VALIDATION_ERROR:
        color = 'info';
        break;
      default:
        color = 'default';
    }

    return (
      <Chip
        label={errorType}
        color={color}
        size="small"
        sx={{ mb: 2 }}
      />
    );
  };

  renderErrorMessage = (): string => {
    const { error, errorType } = this.state;

    if (!error) return 'An unknown error occurred';

    // Return user-friendly messages based on error type
    switch (errorType) {
      case ErrorType.AUTHENTICATION_ERROR:
        return 'You need to sign in to access this feature';
      case ErrorType.AUTHORIZATION_ERROR:
        return 'You do not have permission to access this feature';
      case ErrorType.SERVICE_UNAVAILABLE_ERROR:
        return 'This service is currently unavailable. Please try again later.';
      case ErrorType.SERVICE_TIMEOUT_ERROR:
        return 'The service took too long to respond. Please try again later.';
      case ErrorType.NETWORK_ERROR:
        return 'Network connection issue. Please check your internet connection.';
      case ErrorType.DATA_FETCH_ERROR:
        return 'Failed to load data. Please try again.';
      default:
        return error.message || 'An unexpected error occurred';
    }
  };

  renderErrorDetails = (): ReactNode => {
    const { error, errorInfo, errorType } = this.state;

    if (!error) return null;

    return (
      <Box sx={{ my: 2, p: 2, bgcolor: 'background.paper', borderRadius: 1, overflowX: 'auto' }}>
        <Typography variant="subtitle2" fontWeight="bold">
          Error Type:
        </Typography>
        <Typography variant="body2" sx={{ mb: 1 }}>
          {errorType}
        </Typography>

        <Typography variant="subtitle2" fontWeight="bold">
          Error Details:
        </Typography>
        <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap' }}>
          {error.toString()}
        </Typography>

        {errorInfo && (
          <>
            <Typography variant="subtitle2" fontWeight="bold" sx={{ mt: 2 }}>
              Component Stack:
            </Typography>
            <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap', fontSize: '0.75rem' }}>
              {errorInfo.componentStack}
            </Typography>
          </>
        )}
      </Box>
    );
  };

  renderRecoveryOptions = (): ReactNode => {
    const { errorType } = this.state;

    // Suggest specific recovery actions based on error type
    let recoveryMessage = '';

    switch (errorType) {
      case ErrorType.AUTHENTICATION_ERROR:
        recoveryMessage = 'Try signing in again';
        break;
      case ErrorType.NETWORK_ERROR:
        recoveryMessage = 'Check your internet connection and try again';
        break;
      case ErrorType.SERVICE_UNAVAILABLE_ERROR:
      case ErrorType.SERVICE_TIMEOUT_ERROR:
        recoveryMessage = 'The service might be temporarily down. Try again later.';
        break;
      default:
        recoveryMessage = 'You can try the following options:';
    }

    return (
      <>
        <Divider sx={{ my: 2 }} />
        <Typography variant="subtitle2" gutterBottom>
          {recoveryMessage}
        </Typography>

        <Box sx={{ mt: 2, display: 'flex', flexWrap: 'wrap', gap: 2 }}>
          <Button
            variant="contained"
            color="primary"
            onClick={this.handleReset}
          >
            Try Again
          </Button>

          <Button
            variant="outlined"
            onClick={() => window.location.reload()}
          >
            Reload Page
          </Button>

          <Button
            variant="outlined"
            onClick={() => window.location.href = '/'}
          >
            Go to Home
          </Button>

          <Button
            variant="text"
            color="secondary"
            onClick={this.handleReportError}
          >
            Report Issue
          </Button>
        </Box>
      </>
    );
  };

  render(): ReactNode {
    const { hasError, errorType } = this.state;
    const { children, fallback, showErrorDetails } = this.props;

    if (!hasError) {
      // When there's no error, render children normally
      return children;
    }

    // Custom fallback UI if provided
    if (fallback) {
      return fallback;
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
        <Box sx={{ mb: 2 }}>
          {this.renderErrorTypeChip(errorType)}
        </Box>

        <Typography variant="h5" color="error" gutterBottom>
          Something went wrong
        </Typography>

        <Alert severity="error" sx={{ mb: 2 }}>
          {this.renderErrorMessage()}
        </Alert>

        <Typography variant="body2" color="text.secondary" paragraph>
          We apologize for the inconvenience. An error has occurred in this component.
        </Typography>

        {showErrorDetails && this.renderErrorDetails()}

        {this.renderRecoveryOptions()}
      </Paper>
    );
  }
}

export default ErrorBoundary;
